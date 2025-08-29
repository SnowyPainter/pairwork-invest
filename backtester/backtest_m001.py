#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Top-K → Direction 필터 통합 백테스트 (안정화 리라이트)

- Step 1: EventDetector로 날짜별 이벤트 확률 상위 K 종목 선별
- Step 2: 선별된 후보만 DirectionClassifier로 상승 필터링
- Step 3: 최종 0/1 신호 생성 → 백테스트

주의:
- DirectionClassifier.predict(X) 는 pandas 입력 가정
- EventDetectorManager.predict(X) 는 polars 입력 가능 가정
- VOLATILITY_FEATURES / SELECTED_FEATURES 미존재 컬럼은 0으로 채움
"""

import os
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import polars as pl
import pandas as pd
import numpy as np

# 프로젝트 모듈 경로
sys.path.append(str(Path(__file__).parent.parent))

from models.M001_DirectionClassifier import (
    DirectionClassifierLGBM,
    create_direction_classifier_model,
    SELECTED_FEATURES,
)
from models.M001_EventDetector import (
    EventDetectorManager,
    create_event_detector_model,
    VOLATILITY_FEATURES,
)
from data.dataset_builder import build_dataset
from backtester.backtester import (
    BacktestConfig, UniverseRule, SignalRule, ExecutionRule, PortfolioRule,
    backtest, plot_equity, plot_drawdown, plot_monthly_heatmap,
    plot_rolling_sharpe, plot_contrib_by_ticker, plot_signals_per_ticker
)


# -----------------------------
# 유틸
# -----------------------------
def _ensure_cols(df: pl.DataFrame, cols: list[str], fill: float = 0.0) -> pl.DataFrame:
    """df에 cols가 모두 존재하도록 누락 열 추가(상수 fill), 열 순서 보전."""
    miss = [c for c in cols if c not in df.columns]
    if miss:
        df = df.with_columns([pl.lit(fill).alias(c) for c in miss])
    # 성능을 위해 굳이 select로 재정렬하지 않고 필요한 시점에만 select
    return df

def _align_pandas(df: pl.DataFrame, cols: list[str], fill: float = 0.0) -> pd.DataFrame:
    """Direction용: pandas DataFrame으로 변환 + 필요한 열 순서 고정 + 결측 0."""
    df2 = _ensure_cols(df, cols, fill)
    pdf = df2.select(cols).to_pandas()
    return pdf.fillna(fill)

def _predict_direction(direction_model, df: pl.DataFrame, feat_cols: list[str], prob_thresh: float = 0.5,
                       batch_size: int = 65536) -> tuple[np.ndarray, np.ndarray]:
    """Direction Classifier 배치 예측 (pandas 기반). return: (direction_int, prob_up)"""
    if not feat_cols:
        n = df.height
        return (np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.float64))

    probs_up_list = []
    n = df.height
    # 대용량 방지용 배치 분할
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X = _align_pandas(df.slice(start, end - start), feat_cols, 0.0)
        y_pred, proba = direction_model.predict(X)  # proba: (n,2) 혹은 (n,)
        if getattr(proba, "ndim", 1) == 2 and proba.shape[1] >= 2:
            p_up = proba[:, 1].astype("float64")
        else:
            p_up = np.asarray(proba, dtype="float64")
        probs_up_list.append(p_up)

    p_up_all = np.concatenate(probs_up_list, axis=0)
    direction = (p_up_all >= prob_thresh).astype("int64")

    # p_up_all 분포 확인 (디버깅용)
    print(f"p_up_all 분포: min={p_up_all.min():.4f}, max={p_up_all.max():.4f}, "
          f"mean={p_up_all.mean():.4f}, std={p_up_all.std():.4f}")
    print(f"p_up_all >= 0.5: {(p_up_all >= 0.5).sum()}/{len(p_up_all)} "
          f"({(p_up_all >= 0.5).mean()*100:.1f}%)")

    return direction, p_up_all

from typing import Literal

def create_event_topk_direction_signals(
    direction_model,
    event_model,
    df: pl.DataFrame,
    top_k: int = 5,
    dir_prob_thresh: float = 0.6,           # 방향 컷 임계(상승/하락 확률)
    *,
    side: Literal["up","down"] = "up",      # 저 up/down으로 1회 거르기
) -> pl.DataFrame:
    """
    1) 이벤트 확률로 날짜별 Top-K 후보 선별 (훈련 분포 정렬)
    2) 선별된 후보에만 방향 예측 수행 (분포 일치)
    3) 방향 필터 통과한 종목만 최종 신호 생성
    """
    assert "date" in df.columns, "`date` 컬럼이 필요합니다."
    base = df.with_row_count("rid", offset=0)

    # 모델 내부 daily_top_k는 끔 (외부에서 Top-K 수행)
    if hasattr(event_model, "daily_top_k"):
        try: event_model.daily_top_k = None
        except: pass

    # --- Step 1: 이벤트 확률 예측 (전체 데이터) ---
    _, ev_proba_raw = event_model.predict(df.fill_null(0.0))
    if getattr(ev_proba_raw, "ndim", 1) == 2 and ev_proba_raw.shape[1] >= 2:
        event_probs = ev_proba_raw[:, 1].astype("float64")
    else:
        event_probs = np.asarray(ev_proba_raw, dtype="float64")

    # 이벤트 확률로 날짜별 Top-K 후보 선별
    tmp = base.with_columns([
        pl.Series("event_prob", event_probs, dtype=pl.Float64).clip(0,1).fill_null(0.0),
    ]).with_columns([
        pl.col("event_prob").rank(method="dense", descending=True).over("date").alias("event_rank")
    ]).with_columns([
        (pl.col("event_rank") <= top_k).alias("is_event_candidate")
    ])

    # --- Step 2: 이벤트 후보에만 방향 예측 (분포 정렬) ---
    event_candidates = tmp.filter(pl.col("is_event_candidate"))
    
    if event_candidates.height == 0:
        # 후보가 없으면 모든 신호 0
        return base.with_columns(pl.lit(0).cast(pl.Int8).alias("final_signal"))
    
    # 이벤트 후보에만 방향 예측 수행
    dir_feats = [f for f in SELECTED_FEATURES if f in df.columns]
    _, dir_probs = _predict_direction(direction_model, event_candidates, dir_feats, prob_thresh=dir_prob_thresh)
    
    # 방향 예측 결과를 원본 인덱스에 매핑
    candidates_with_dir = event_candidates.with_columns([
        pl.Series("direction_prob", dir_probs, dtype=pl.Float64).clip(0,1).fill_null(0.0),
    ])

    # --- Step 3: 방향 필터 적용 ---
    if side == "up":
        candidates_with_dir = candidates_with_dir.with_columns(
            (pl.col("direction_prob") >= dir_prob_thresh).alias("pass_direction")
        )
    else:  # side == "down"
        # 하락 확률 = 1 - 상승 확률 로 취급
        candidates_with_dir = candidates_with_dir.with_columns(
            ((1.0 - pl.col("direction_prob")) >= dir_prob_thresh).alias("pass_direction")
        )

    # 최종 신호 생성 (이벤트 후보 + 방향 통과)
    final_candidates = candidates_with_dir.filter(pl.col("pass_direction"))

    # --- Step 4: 결과 매핑 ---
    # 원본 base에 최종 신호 매핑
    out = base.join(
        final_candidates.select([
            "rid",
            pl.lit(1).cast(pl.Int8).alias("final_signal"),
            pl.col("direction_prob").alias("signal_trigger_prob"),
            pl.col("event_prob").alias("signal_event_prob"),
            pl.col("event_rank").cast(pl.Int32).alias("signal_rank"),
            pl.col("is_event_candidate").alias("pass_event"),
            pl.col("pass_direction").alias("pass_dir"),
        ]),
        on="rid", how="left"
    ).drop("rid").with_columns([
        pl.col("final_signal").fill_null(0).cast(pl.Int8),
        pl.col("signal_trigger_prob").fill_null(0.0),
        pl.col("signal_event_prob").fill_null(0.0),
        pl.col("signal_rank").fill_null(0),
        pl.col("pass_event").fill_null(False),
        pl.col("pass_dir").fill_null(False),
    ])

    return out


def run_event_topk_direction_backtest(
    market: str = "KR",
    years_train: list[int] = [2018, 2019, 2020],
    years_test: list[int] = [2021],
    max_tickers: int = 50,
    top_k: int = 10,
    dir_prob_thresh: float = 0.6,
    move_exit_pct: float = 0.05,
) -> dict | None:

    print("[Event Detector Top-K + Direction 백테스트]")
    print("=" * 72)
    print(f"  시장         : {market}")
    print(f"  학습 연도    : {years_train}")
    print(f"  테스트 연도  : {years_test}")
    print(f"  최대 종목    : {max_tickers}")
    print(f"  Event 상위 K : {top_k}")
    print(f"  Direction p↑ : ≥ {dir_prob_thresh:.2f}")
    print(f"  Exit on ±n%: n={move_exit_pct:.2%}")
    print("=" * 72)

    t0 = time.time()

    # 1) Direction Classifier 준비
    print("\n[Direction Classifier 준비]")
    dir_model_path = f"models/saved/direction_classifier_{market}_{'_'.join(map(str, years_train))}.txt"
    if os.path.exists(dir_model_path):
        print(f"  기존 모델 로드: {dir_model_path}")
        direction_model = DirectionClassifierLGBM()
        direction_model.load_model(dir_model_path)
    else:
        print("  신규 학습 실행…")
        direction_model = create_direction_classifier_model(
            market=market, years=years_train, save_model=True
        )

    # 2) Event Detector 준비
    print("\n[Event Detector 준비]")
    ev_model_stub = f"models/saved/tcn_event_detector_{market}_{'_'.join(map(str, years_train))}_100pct_L60"
    if os.path.exists(f"{ev_model_stub}.pth"):
        print(f"  기존 TCN 모델 로드: {ev_model_stub}")
        event_model = EventDetectorManager(
            threshold=1.0,  # ATR 100%
            sequence_length=60,
            device="auto",
        )
        event_model.load_model(ev_model_stub)
    else:
        print("  신규 TCN 학습 실행…(참고: 내부 스케일링/보정 적용)")
        event_model = create_event_detector_model(
            market=market,
            years=years_train,
            threshold=1.0,
            target="big_move_event",
            max_tickers=max_tickers,
            save_model=True,
            sequence_length=60,
            batch_size=64,
            epochs=50,
            learning_rate=1e-3,
            apply_calibration=True,
            daily_top_k=5,
            target_precision=0.4,
        )

    # 3) 테스트 세트 로드
    print("\n[테스트 데이터 로드]")
    test_df = build_dataset(
        years=years_test,
        market=market,
        max_tickers=max_tickers,
        feature_set="v2",       # VOLATILITY_FEATURES 포함
        label_horizon=1,
        label_task="classification",
        normalize_features=False,  # TCN 내부 스케일링 가정
        verbose=True,
    )
    print(f"  테스트 데이터: {test_df.height:,} 행, {len(test_df.columns)} 열")
    
    # 사용 가능 피처 출력
    ev_avail = [c for c in VOLATILITY_FEATURES if c in test_df.columns]
    dir_avail = [c for c in SELECTED_FEATURES if c in test_df.columns]
    print(f"  Event 피처: {len(ev_avail)}/{len(VOLATILITY_FEATURES)} 사용")
    print(f"  Dir   피처: {len(dir_avail)}/{len(SELECTED_FEATURES)} 사용")

    miss_ev = [c for c in VOLATILITY_FEATURES if c not in test_df.columns]
    miss_dir = [c for c in SELECTED_FEATURES if c not in test_df.columns]
    if miss_ev:
        print(f"  [경고] Event 누락 피처: {miss_ev}")
    if miss_dir:
        print(f"  [경고] Direction 누락 피처: {miss_dir}")

    # 4) 신호 생성
    print("\n[신호 생성: Event Top-K → Direction]")
    with_signals = create_event_topk_direction_signals(
        direction_model=direction_model,
        event_model=event_model,
        df=test_df,
        top_k=top_k,
        dir_prob_thresh=dir_prob_thresh,
    )

    # 5) 백테스트 설정
    print("\n[백테스트 설정]")
    config = BacktestConfig(
        label_col="futret_1",
        signal_col="final_signal",
        universe=UniverseRule(
            top_k_per_day=max_tickers,
            min_turnover=1e3,
            min_price=5_000,
        ),
        signal=SignalRule(
            select_top_n=0,
            min_threshold=0.0,
            long_only=True,
        ),
        execution=ExecutionRule(mode="next_open_to_close_nmove", move_exit_pct=move_exit_pct),
        portfolio=PortfolioRule(
            weighting="equal",
            fee_bps=8.0,
            slippage_bps=5.0,
            capital_per_position=1_000_000,
        ),
        outdir=Path("reports/backtest_trigger_size"),
    )
    print(f"  신호 컬럼: {config.signal_col}")
    print(f"  날짜별 상위 선택: {config.signal.select_top_n}")
    print(f"  유니버스 top_k_per_day: {config.universe.top_k_per_day}")

    # 6) 백테스트 실행
    print("\n[백테스트 실행]")
    try:
        result = backtest(with_signals, config)
    except Exception as e:
        print(f"  [오류] 백테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 7) 결과 요약/디버깅
    print("\n[결과 요약]")
    if "summary" in result:
        s = result["summary"]
        print(f"  거래일수   : {s.get('trading_days')}")
        print(f"  연간 수익률: {s.get('ret_annual', 0.0):.2%}")
        print(f"  변동성     : {s.get('vol_annual', 0.0):.2%}")
        print(f"  샤프       : {s.get('sharpe', 0.0):.2f}")
        print(f"  MDD        : {s.get('max_drawdown', 0.0):.2%}")
        print(f"  승률       : {s.get('win_rate', 0.0):.1%}")
        print(f"  총 거래    : {s.get('total_trades', 0)}")
        
    # 8) 차트
    print("\n[차트 생성]")
    try:
        plot_equity(result, show=False)
        plot_drawdown(result, show=False)
        plot_monthly_heatmap(result, show=False)
        n_days = len(result.get("daily", pl.DataFrame()))
        window = min(30, max(10, n_days // 3)) if n_days else 10
        print(f"  Rolling Sharpe window={window}, days={n_days}")
        plot_rolling_sharpe(result, window=window, show=False)
        plot_contrib_by_ticker(result, show=False)
        # 종목별 시그널 디버깅 차트
        plot_signals_per_ticker(result, show=False)
    except Exception as e:
        print(f"  [경고] 차트 생성 실패: {e}")

    print(f"\n[완료] 총 {time.time() - t0:.2f}s")
    return result


def main():
    print("🎯 Event Top-K → Direction 필터 백테스트 (안정화 버전)")
    MARKET = "KR"
    TRAIN_YEARS = [2018, 2019, 2020]
    TEST_YEARS = [2020, 2021]
    MAX_TICKERS = 100
    TOP_K = 10
    DIR_PTH = 0.5  # 상승확률 임계
    MOVE_EXIT = 0.05

    try:
        result = run_event_topk_direction_backtest(
            market=MARKET,
            years_train=TRAIN_YEARS,
            years_test=TEST_YEARS,
            max_tickers=MAX_TICKERS,
            top_k=TOP_K,
            dir_prob_thresh=DIR_PTH,
            move_exit_pct=MOVE_EXIT,
        )
        if result:
            print("\n✅ 완료: reports/backtest_trigger_size/ 폴더를 확인하세요.")
        else:
            print("\n❌ 실패")
    except KeyboardInterrupt:
        print("\n⏹️ 중단됨")


if __name__ == "__main__":
    main()
