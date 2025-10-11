#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M002 MultiTask 모델 백테스트

- M002 모델로 rebound 확률, 수익률, drawdown 예측
- 정책 점수 기반 신호 생성
- 기본적인 백테스트 실행
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

# 프로젝트 모듈 경로
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.M002_MultiTask import M002MultiTaskModel, M002TrainingConfig
from data.dataset_builder import build_dataset
from backtester.backtester import (
    BacktestConfig, UniverseRule, SignalRule, ExecutionRule, PortfolioRule,
    backtest, plot_equity, plot_drawdown, plot_monthly_heatmap,
    plot_rolling_sharpe, plot_contrib_by_ticker, plot_signals_per_ticker
)


def load_m002_model(model_path: str) -> M002MultiTaskModel:
    """M002 모델 로드"""
    try:
        import joblib
        model = joblib.load(model_path)
        print(f"M002 모델 로드 성공: {model_path}")
        return model
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return None


def load_model_metadata(model_path: str) -> dict:
    """모델 메타데이터 로드"""
    json_path = model_path.replace('.pkl', '.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"메타데이터 로드 실패: {e}")
    return {}


def display_model_info(metadata: dict):
    """모델 정보 출력"""
    if not metadata:
        print("M002 모델 메타데이터를 찾을 수 없습니다.")
        return

    print("\nM002 모델 정보:")
    print(f"  모델 타입: {metadata.get('model_type', 'N/A')}")
    print(f"  시장: {metadata.get('market', 'N/A')}")
    print(f"  학습 연도: {metadata.get('years', [])}")
    print(f"  예측 기간: {metadata.get('horizon', 'N/A')}일")
    print(f"  Rebound 임계값: {metadata.get('rebound_thresh', 'N/A')}%")
    print(f"  Drawdown 바닥: {metadata.get('drawdown_floor', 'N/A')}%")
    print(f"  리스크 회피도(λ): {metadata.get('risk_aversion', 'N/A')}")
    print(f"  특징 수: {len(metadata.get('feature_columns', []))}")

    if 'validation_metrics' in metadata:
        vm = metadata['validation_metrics']
        print(f"  검증 평균 정밀도: {vm.get('valid_avg_precision', 'N/A'):.4f}")
        print(f"  수익률 MAE: {vm.get('valid_mae_ret', 'N/A'):.4f}")
        print(f"  정책 점수 평균: {vm.get('valid_policy_score_mean', 'N/A'):.4f}")


def create_m002_signals(model: M002MultiTaskModel, df: pl.DataFrame,
                       policy_threshold: float = 0.0) -> pl.DataFrame:
    """
    M002 모델로 신호 생성

    Args:
        model: 학습된 M002 모델
        df: 특성이 포함된 데이터프레임
        policy_threshold: 정책 점수 임계값 (기본 0.0)

    Returns:
        신호가 추가된 데이터프레임
    """
    print("[M002 신호 생성]")

    # 필요한 특성 추출
    feature_cols = model.feature_columns
    available_features = [col for col in feature_cols if col in df.columns]

    if len(available_features) < len(feature_cols) * 0.8:  # 80% 이상 특성 필요
        missing = set(feature_cols) - set(available_features)
        print(f"경고: {len(missing)}개 특성 누락: {list(missing)[:5]}...")
        if len(missing) > 10:
            print(f"  ... 외 {len(missing) - 5}개")

    print(f"사용 특성: {len(available_features)}/{len(feature_cols)}")

    # pandas로 변환하여 예측
    features_df = df.select(available_features).to_pandas()

    # 누락된 특성은 0으로 채움
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0

        # 예측 수행
    try:
        prob, ret_pred, policy = model.predict(features_df)
        print(f"DEBUG: predict returned - prob[0]: {prob[0]:.4f}, ret_pred[0]: {ret_pred[0]:.4f}, policy[0]: {policy[0]:.4f}")

        # 결과 추가
        result_df = df.with_columns([
            pl.Series("m002_rebound_prob", prob, dtype=pl.Float64),
            pl.Series("m002_ret_pred", ret_pred, dtype=pl.Float64),
            pl.Series("m002_policy_score", policy, dtype=pl.Float64),
        ])

        # 정책 점수 기반 신호 생성
        result_df = result_df.with_columns([
            (pl.col("m002_policy_score") > policy_threshold).cast(pl.Int8).alias("final_signal")
        ])

        # 신호 통계 출력
        signal_count = result_df.filter(pl.col("final_signal") == 1).height
        total_count = result_df.height
        signal_ratio = signal_count / total_count if total_count > 0 else 0

        # 정책 점수 분포 분석
        policy_scores = result_df.select("m002_policy_score").to_pandas()["m002_policy_score"]
        print(f"  총 데이터: {total_count:,}")
        print(f"  정책 점수 분포 - 최소: {policy_scores.min():.4f}, 최대: {policy_scores.max():.4f}")
        print(f"  정책 점수 분포 - 평균: {policy_scores.mean():.4f}, 중앙값: {policy_scores.median():.4f}")
        print(f"  정책 점수 임계값: {policy_threshold:.3f}")
        print(f"  임계값 이상 개수: {(policy_scores >= policy_threshold).sum():,}")
        print(f"  신호 수: {signal_count:,} ({signal_ratio:.2%})")

        # 백분위수 정보 추가
        for pct in [10, 25, 50, 75, 90, 95, 99]:
            threshold_pct = np.percentile(policy_scores, pct)
            count_pct = (policy_scores >= threshold_pct).sum()
            print(f"  {pct}th 백분위수: {threshold_pct:.4f} (이상: {count_pct:,}개, {count_pct/total_count:.1%})")

        return result_df

    except Exception as e:
        print(f"M002 예측 실패: {e}")
        import traceback
        traceback.print_exc()

        # 실패 시 기본 신호 (모두 0) 반환
        return df.with_columns([
            pl.lit(0.0).alias("m002_rebound_prob"),
            pl.lit(0.0).alias("m002_ret_pred"),
            pl.lit(0.0).alias("m002_policy_score"),
            pl.lit(0).cast(pl.Int8).alias("final_signal"),
        ])


def run_m002_backtest(
    market: str = "US",
    years_train: list[int] = list(range(2010, 2019)),
    years_test: list[int] = [2019, 2020, 2021],
    max_tickers: int = 50,
    policy_threshold: float = 0.0,
    move_exit_pct: float = 0.05,
    use_existing_model: bool = True,
) -> dict | None:
    """
    M002 모델 백테스트 실행
    """

    print("[M002 MultiTask 백테스트]")
    print("=" * 60)
    print(f"시장         : {market}")
    print(f"학습 연도    : {years_train}")
    print(f"테스트 연도  : {years_test}")
    print(f"최대 종목    : {max_tickers}")
    print(f"정책 임계값  : {policy_threshold:.3f}")
    print(f"Exit on ±n%  : n={move_exit_pct:.2%}")
    print(f"기존 모델 사용: {use_existing_model}")
    print("=" * 60)

    t0 = time.time()

    # 1) M002 모델 준비
    print("\n[M002 모델 준비]")
    model_stub = f"models/saved/m002_multitask_{market}_{'_'.join(map(str, years_train))}"
    model_path = f"{model_stub}.pkl"

    if use_existing_model and os.path.exists(model_path):
        print(f"기존 모델 로드: {model_path}")
        model = load_m002_model(model_path)
        if model is None:
            print("모델 로드 실패, 신규 학습 실행")
            use_existing_model = False
        else:
            # 메타데이터 표시
            metadata = load_model_metadata(model_path)
            display_model_info(metadata)
    else:
        print("신규 M002 모델 학습 실행...")
        config = M002TrainingConfig(
            market=market,
            years=years_train,
        )
        model = M002MultiTaskModel(config=config)
        metrics = model.train()

        print("\n학습 결과:")
        print(f"  검증 평균 정밀도: {metrics['valid_avg_precision']:.4f}")
        print(f"  수익률 MAE: {metrics['valid_mae_ret']:.4f}")
        print(f"  정책 점수 평균: {metrics['valid_policy_score_mean']:.4f}")

    if model is None:
        print("모델 준비 실패")
        return None

    # 2) 테스트 데이터 로드
    print("\n[테스트 데이터 로드]")
    test_df = build_dataset(
        years=years_test,
        market=market,
        max_tickers=max_tickers,
        feature_set="m002",
        label_horizon=5,
        label_task="regression",
        normalize_features=True,
        verbose=True,
    )
    print(f"테스트 데이터: {test_df.height:,} 행, {len(test_df.columns)} 열")

    # 3) 신호 생성
    print("\n[M002 신호 생성]")
    with_signals = create_m002_signals(
        model=model,
        df=test_df,
        policy_threshold=policy_threshold,
    )

    # 4) 백테스트 설정
    print("\n[백테스트 설정]")
    config = BacktestConfig(
        label_col="futret_5",  # 5일 수익률
        signal_col="final_signal",
        universe=UniverseRule(
            top_k_per_day=max_tickers,
            min_turnover=1e3,
            min_price=5_000,
        ),
        signal=SignalRule(
            select_top_n=0,  # 정책 점수 기반 필터링이므로 상위 N개 선택하지 않음
            min_threshold=0.0,
            long_only=True,
        ),
        execution=ExecutionRule(
            mode="next_open_to_close_nmove",
            move_exit_pct=move_exit_pct
        ),
        portfolio=PortfolioRule(
            weighting="equal",
            fee_bps=8.0,
            slippage_bps=5.0,
            capital_per_position=1_000_000,
        ),
        outdir=Path("reports/m002_baseline"),
    )

    print(f"신호 컬럼: {config.signal_col}")
    print(f"레이블 컬럼: {config.label_col}")
    print(f"유니버스 top_k_per_day: {config.universe.top_k_per_day}")

    # 5) 백테스트 실행
    print("\n[백테스트 실행]")
    try:
        result = backtest(with_signals, config)
    except Exception as e:
        print(f"백테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 6) 결과 요약
    print("\n[결과 요약]")
    if "summary" in result:
        s = result["summary"]
        print(f"거래일수   : {s.get('trading_days')}")
        print(f"연간 수익률: {s.get('ret_annual', 0.0):.2%}")
        print(f"변동성     : {s.get('vol_annual', 0.0):.2%}")
        print(f"샤프       : {s.get('sharpe', 0.0):.2f}")
        print(f"MDD        : {s.get('max_drawdown', 0.0):.2%}")
        print(f"승률       : {s.get('win_rate', 0.0):.1%}")
        print(f"총 거래    : {s.get('total_trades', 0)}")

    # 7) 차트 생성
    print("\n[차트 생성]")
    try:
        plot_equity(result, show=False)
        plot_drawdown(result, show=False)
        plot_monthly_heatmap(result, show=False)

        n_days = len(result.get("daily", pl.DataFrame()))
        window = min(30, max(10, n_days // 3)) if n_days else 10
        print(f"Rolling Sharpe window={window}, days={n_days}")
        plot_rolling_sharpe(result, window=window, show=False)
        plot_contrib_by_ticker(result, show=False)
        try:
            plot_signals_per_ticker(result, show=False)
        except Exception as chart_e:
            print(f"plot_signals_per_ticker 차트 생성 실패: {chart_e}")
    except Exception as e:
        print(f"차트 생성 실패: {e}")

    print(f"\n[완료] 총 {time.time() - t0:.2f}s")
    return result


def main():
    print("M002 MultiTask 모델 백테스트")

    # 설정
    MARKET = "US"
    TRAIN_YEARS = list(range(2010, 2019))
    TEST_YEARS = [2019, 2020, 2021]
    MAX_TICKERS = 3000
    POLICY_THRESHOLD = -0.11  # 정책 점수 임계값 (90th 백분위수)
    MOVE_EXIT = 0.1
    USE_EXISTING = True

    try:
        result = run_m002_backtest(
            market=MARKET,
            years_train=TRAIN_YEARS,
            years_test=TEST_YEARS,
            max_tickers=MAX_TICKERS,
            policy_threshold=POLICY_THRESHOLD,
            move_exit_pct=MOVE_EXIT,
            use_existing_model=USE_EXISTING,
        )
        if result:
            print("\n완료: reports/m002_baseline/ 폴더를 확인하세요.")
        else:
            print("\n실패")
    except KeyboardInterrupt:
        print("\n중단됨")


if __name__ == "__main__":
    main()

    # 빠른 테스트를 위한 함수
    def quick_test():
        """빠른 테스트 실행 (적은 데이터로)"""
        print("\n빠른 테스트 모드")
        result = run_m002_backtest(
            market="KR",
            years_train=[2019, 2020],
            years_test=[2021],
            max_tickers=50,
            policy_threshold=0.02,  # 99th 백분위수 근처
            move_exit_pct=0.05,
            use_existing_model=False,  # 새 모델 학습
        )
        return result

    # 사용법:
    # python backtester/backtest_m002_baseline.py  # 전체 테스트
    # 또는 코드에서 quick_test() 호출  # 빠른 테스트
