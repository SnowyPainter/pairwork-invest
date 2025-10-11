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
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import polars as pl
import pandas as pd
import numpy as np

# 프로젝트 모듈 경로 - 맨 앞에 추가해서 우선순위 높임
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.M001_DirectionClassifier import (
    DirectionClassifierLGBM,
    create_direction_classifier_model,
    SELECTED_FEATURES,
)
from models.M001_EventDetector import (
    EventDetectorManager,
    create_event_detector_model,
    VOLATILITY_FEATURES,
    TURNOVER_DERIVED_FEATURES,
    ENHANCED_FEATURES,
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
def load_model_metadata(model_path: str) -> dict:
    """모델 메타데이터를 JSON에서 로드"""
    # Direction Classifier의 경우: .txt -> _metadata.json
    if model_path.endswith('.txt'):
        json_path = model_path.replace('.txt', '_metadata.json')
    # Event Detector의 경우: .pth -> .json
    elif model_path.endswith('.pth'):
        json_path = model_path.replace('.pth', '.json')
    else:
        # 다른 경우는 직접 .json 파일로 가정
        json_path = model_path if model_path.endswith('.json') else f"{model_path}.json"

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except UnicodeDecodeError as e:
            print(f"메타데이터 파일 인코딩 오류: {json_path} - {e}")
            return {}
        except json.JSONDecodeError as e:
            print(f"메타데이터 JSON 파싱 오류: {json_path} - {e}")
            return {}
    else:
        print(f"메타데이터 파일을 찾을 수 없음: {json_path}")
        return {}

def display_model_info(metadata: dict, model_type: str):
    """모델 정보를 출력"""
    if not metadata:
        print(f"{model_type} 메타데이터를 찾을 수 없습니다.")
        return

    print(f"\n{model_type} 모델 정보:")
    print(f"  피처 수: {len(metadata.get('features', []))}")
    print(f"  주요 피처: {', '.join(metadata.get('features', [])[:5])}")

    if 'training_history' in metadata:
        hist = metadata['training_history']
        print(f"  학습 정확도: {hist.get('accuracy', 'N/A'):.4f}")
        print(f"  F1 스코어: {hist.get('f1_score', 'N/A'):.4f}")
        print(f"  ROC-AUC: {hist.get('roc_auc', 'N/A'):.4f}")

def create_extended_turnover_features(df: pl.DataFrame) -> pl.DataFrame:
    """모든 turnover 기반 피처들을 한 번에 생성 (통합 함수)"""
    if 'turnover' not in df.columns:
        print("turnover 컬럼이 없어 확장 피처 생성을 건너뜁니다.")
        return df

    print("turnover 기반 모든 피처 생성 중...")

    # 모든 turnover 피처들을 한 번에 생성 (단계별 적용으로 오류 방지)

    # 1단계: 기본 파생 피처들 + Direction Classifier용 피처들
    df = df.with_columns([
        # Event Detector & Direction Classifier 공통 피처들
        pl.when(pl.col("turnover").shift(1).is_null() | (pl.col("turnover").shift(1) == 0))
          .then(0.0)
          .otherwise((pl.col("turnover") / pl.col("turnover").shift(1)) - 1)
          .over("ticker")
          .alias("turnover_ratio_1d"),

        # 현재 순위 계산 (여러 용도로 사용)
        pl.col("turnover").rank(method="dense", descending=True).over(["date"]).alias("turnover_current_rank"),
    ])

    # 전일 순위 계산 (TURNOVER_DERIVED_FEATURES + Direction Classifier용)
    df = df.with_columns([
        pl.col("turnover_current_rank").shift(1).over("ticker").alias("turnover_rank_prev1"),
    ])

    # Direction Classifier용 추가 피처들
    df = df.with_columns([
        (pl.col("turnover_rank_prev1") <= 100).cast(pl.Int8).alias("is_top100_prev1"),
        (pl.col("turnover_rank_prev1") <= 300).cast(pl.Int8).alias("is_top300_prev1"),
    ])

    # 2단계: 이동평균 계열
    df = df.with_columns([
        pl.col("turnover").rolling_mean(5).over("ticker").alias("turnover_ma5"),
        pl.col("turnover").rolling_mean(10).over("ticker").alias("turnover_ma10"),
        pl.col("turnover").rolling_mean(20).over("ticker").alias("turnover_ma20"),
    ])

    # 3단계: 표준편차 (변동성)
    df = df.with_columns([
        pl.col("turnover").rolling_std(5).over("ticker").alias("turnover_std5"),
        pl.col("turnover").rolling_std(10).over("ticker").alias("turnover_std10"),
        pl.col("turnover").rolling_std(20).over("ticker").alias("turnover_std20"),
    ])

    # 4단계: 추가 확장 피처들
    df = df.with_columns([
        # 백분위수 (turnover_current_rank와 동일하지만 명시적으로 생성)
        pl.col("turnover").rank(method="dense", descending=True).over(["date"]).alias("turnover_percentile"),

        # 모멘텀 계열
        (pl.col("turnover") / pl.col("turnover").shift(5).over("ticker") - 1).alias("turnover_momentum5"),
        (pl.col("turnover") / pl.col("turnover").shift(10).over("ticker") - 1).alias("turnover_momentum10"),
    ])

    # 5단계: 파생 계산들
    df = df.with_columns([
        # Z-score (표준화)
        ((pl.col("turnover") - pl.col("turnover_ma20")) /
         (pl.col("turnover_std20") + 1e-9)).alias("turnover_zscore"),

        # 변동성 비율
        (pl.col("turnover_std5") / (pl.col("turnover_ma5") + 1e-9)).alias("turnover_volatility_ratio"),

        # 순위 변화
        (pl.col("turnover_current_rank") - pl.col("turnover_rank_prev1")).alias("turnover_rank_change"),
    ])

    # 임시 컬럼 제거
    df = df.drop(["turnover_current_rank"])

    # 모든 turnover 피처들의 결측치 처리
    all_turnover_cols = [
        # 기본 파생 피처들 (TURNOVER_DERIVED_FEATURES)
        "turnover_ratio_1d", "turnover_rank_prev1",

        # Direction Classifier용 추가 피처들
        "is_top100_prev1", "is_top300_prev1",

        # 확장 피처들
        "turnover_ma5", "turnover_ma10", "turnover_ma20",
        "turnover_std5", "turnover_std10", "turnover_std20",
        "turnover_zscore", "turnover_percentile",
        "turnover_momentum5", "turnover_momentum10",
        "turnover_volatility_ratio", "turnover_rank_change"
    ]

    # 결측치를 0.0으로 채움
    for col in all_turnover_cols:
        if col in df.columns:
            df = df.with_columns([
                pl.col(col).fill_null(0.0).alias(col)
            ])

    print(f"통합 피처 생성 완료: {len(all_turnover_cols)}개 피처 추가")
    return df

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
    top_k: int = 5,  # Precision@5의 높은 정확도를 고려하여 5로 설정
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

        # 모델 메타데이터 표시
        dir_metadata = load_model_metadata(dir_model_path)
        display_model_info(dir_metadata, "Direction Classifier")
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

        # 모델 메타데이터 표시
        ev_metadata = load_model_metadata(f"{ev_model_stub}.pth")
        display_model_info(ev_metadata, "Event Detector")
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

    # 모든 turnover 기반 피처 생성 (통합)
    test_df = create_extended_turnover_features(test_df)

    # 사용 가능 피처 출력 (업데이트된 피처 세트 사용)
    ev_base_avail = [c for c in VOLATILITY_FEATURES if c in test_df.columns]
    ev_derived_avail = [c for c in TURNOVER_DERIVED_FEATURES if c in test_df.columns]

    # 확장 turnover 피처들 (Event Detector용)
    extended_turnover_features = [
        "turnover_ma5", "turnover_ma10", "turnover_ma20",
        "turnover_std5", "turnover_std10", "turnover_std20",
        "turnover_zscore", "turnover_percentile",
        "turnover_momentum5", "turnover_momentum10",
        "turnover_volatility_ratio", "turnover_rank_change"
    ]
    extended_avail = [c for c in extended_turnover_features if c in test_df.columns]

    # Direction Classifier 피처들 (이제 통합 함수에서 생성됨)
    dir_turnover_features = [
        "turnover_rank_prev1", "turnover_ratio_1d",
        "is_top100_prev1", "is_top300_prev1"
    ]
    dir_turnover_avail = [c for c in dir_turnover_features if c in test_df.columns]

    dir_avail = [c for c in SELECTED_FEATURES if c in test_df.columns]

    print("  피처 현황:")
    print(f"    Event 기본 피처: {len(ev_base_avail)}/{len(VOLATILITY_FEATURES)} 사용")
    print(f"    Turnover 파생 피처: {len(ev_derived_avail)}/{len(TURNOVER_DERIVED_FEATURES)} 사용")
    print(f"    Turnover 확장 피처: {len(extended_avail)}/{len(extended_turnover_features)} 사용")
    print(f"    Direction 기본 피처: {len(dir_avail)}/{len(SELECTED_FEATURES)} 사용")
    print(f"    Direction Turnover 피처: {len(dir_turnover_avail)}/{len(dir_turnover_features)} 사용")

    miss_ev = [c for c in VOLATILITY_FEATURES if c not in test_df.columns]
    miss_derived = [c for c in TURNOVER_DERIVED_FEATURES if c not in test_df.columns]
    miss_extended = [c for c in extended_turnover_features if c not in test_df.columns]
    miss_dir_turnover = [c for c in dir_turnover_features if c not in test_df.columns]
    miss_dir = [c for c in SELECTED_FEATURES if c not in test_df.columns]

    if miss_ev:
        print(f"  Event 기본 피처 누락: {miss_ev}")
    if miss_derived:
        print(f"  Event Turnover 파생 피처 누락: {miss_derived}")
    if miss_extended:
        print(f"  Event Turnover 확장 피처 누락: {miss_extended}")
    if miss_dir_turnover:
        print(f"  Direction Turnover 피처 누락: {miss_dir_turnover}")
    if miss_dir:
        print(f"  Direction 기본 피처 누락: {miss_dir}")

    # Turnover 파생 피처가 없는 경우 경고
    if not ev_derived_avail:
        print("  Turnover 파생 피처가 없습니다. 모델 성능에 영향이 있을 수 있습니다.")

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
    print("Event Top-K -> Direction 필터 백테스트")
    MARKET = "KR"
    TRAIN_YEARS = [2018, 2019, 2020]
    TEST_YEARS = [2021]
    MAX_TICKERS = 3000
    TOP_K = 10  # Precision@5의 높은 정확도를 고려하여 5로 설정
    DIR_PTH = 0.5  # 상승확률 임계
    MOVE_EXIT = 0.1

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
            print("\n완료: reports/backtest_trigger_size/ 폴더를 확인하세요.")
        else:
            print("\n실패")
    except KeyboardInterrupt:
        print("\n중단됨")


if __name__ == "__main__":
    main()

    # 추가: 빠른 테스트를 위한 함수
    def quick_test():
        """빠른 테스트 실행 (적은 데이터로)"""
        print("\n빠른 테스트 모드")
        result = run_event_topk_direction_backtest(
            market="KR",
            years_train=[2018, 2019],
            years_test=[2020],
            max_tickers=20,  # 적은 종목으로 빠른 테스트
            top_k=5,
            dir_prob_thresh=0.5,
            move_exit_pct=0.05,
        )
        return result

    # 사용법:
    # python backtester/backtest_m001.py  # 전체 테스트
    # 또는 코드에서 quick_test() 호출 # 빠른 테스트
