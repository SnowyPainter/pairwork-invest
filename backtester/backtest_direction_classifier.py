#!/usr/bin/env python3
"""
Direction Classifier 백테스팅 모듈 (개선된 버전)

주요 개선사항:
- Z-score 정규화 문제 해결
- 데이터 로딩 최적화
- 신호 생성 안정화
- nan 결과 방지
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

# 프로젝트 모듈 임포트
sys.path.append(str(Path(__file__).parent.parent))

from models.M001_DirectionClassifier import (
    DirectionClassifierLGBM,
    SELECTED_FEATURES,
    create_direction_classifier_model
)
from data.dataset_builder import build_dataset
from backtester.backtester import (
    BacktestConfig, UniverseRule, SignalRule, ExecutionRule, PortfolioRule,
    backtest, plot_equity, plot_drawdown, plot_monthly_heatmap,
    plot_rolling_sharpe, plot_contrib_by_ticker, quick_run
)


def load_raw_data_for_backtest(years: list, market: str = "KR", max_tickers: int = 100) -> pl.DataFrame:
    """
    백테스팅용 원본 데이터 로드 (Z-score 정규화 없이)
    
    Args:
        years: 연도 리스트
        market: 시장 코드
        max_tickers: 최대 티커 수
    
    Returns:
        원본 값이 보존된 데이터프레임
    """
    print(f"📊 Loading raw data for backtest: {years}, market: {market}")
    
    # 1. 기본 데이터 로드 (feature 생성까지, Z-score 정규화 없이)
    df = build_dataset(
        years=years,
        market=market,
        exchanges=None,
        tickers=None,
        max_tickers=max_tickers,
        start=None,
        end=None,
        feature_set="v2",
        label_horizon=1,
        label_task="classification",
        label_thresh=0.05,
        select_cols=None,
        drop_na_rows=True,
        verbose=False,
        use_cache=True,
        normalize_features=False,  # Z-score 정규화 비활성화 (원본 값 보존)
    )
    
    print(f"✅ Raw data loaded: {len(df):,} rows × {len(df.columns)} columns")
    
    # 2. 필요한 컬럼 확인
    required_cols = ["date", "ticker", "close", "turnover", "futret_1", "label_1d_cls"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Missing required columns: {missing_cols}")
    
    # 3. 선택된 피처들 확인
    available_features = [f for f in SELECTED_FEATURES if f in df.columns]
    missing_features = [f for f in SELECTED_FEATURES if f not in df.columns]
    
    print(f"✅ Available features: {len(available_features)}/{len(SELECTED_FEATURES)}")
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
    
    # 4. 데이터 통계 출력
    print("\n📊 Data Statistics:")
    print(f"  Date range: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  Unique tickers: {df['ticker'].n_unique()}")
    print(f"  Total events (label_1d_cls != 0): {df.filter(pl.col('label_1d_cls') != 0).height:,}")
    
    # 5. 기본 값 범위 확인 (정규화되지 않았는지)
    if 'close' in df.columns:
        close_stats = df.select(pl.col('close')).describe()
        print(f"  Close price range: {df['close'].min():.0f} ~ {df['close'].max():.0f}")
    
    if 'turnover' in df.columns:
        print(f"  Turnover range: {df['turnover'].min():.0f} ~ {df['turnover'].max():.0f}")
    
    return df


def create_simple_signals(model: DirectionClassifierLGBM, df: pl.DataFrame) -> pl.DataFrame:
    """
    간단하고 안정적인 신호 생성
    
    Args:
        model: 학습된 모델
        df: 데이터프레임
    
    Returns:
        신호가 추가된 데이터프레임
    """
    print("🎯 Creating simple direction signals...")
    start_time = time.time()
    
    # 1. 이벤트 데이터만 필터링
    event_df = df.filter(pl.col("label_1d_cls") != 0)
    event_count = len(event_df)
    
    print(f"📊 Event data: {event_count:,} rows")
    
    if event_count == 0:
        print("⚠️ No event data found!")
        return df.with_columns(signal_direction=pl.lit(0.0))
    
    # 2. 사용 가능한 피처 확인
    available_features = [f for f in SELECTED_FEATURES if f in event_df.columns]
    print(f"✅ Using {len(available_features)} features")
    
    if len(available_features) == 0:
        print("❌ No available features!")
        return df.with_columns(signal_direction=pl.lit(0.0))
    
    # 3. 피처 데이터 준비
    try:
        feature_data = event_df.select(available_features).to_pandas()
        
        # 결측치 처리 (간단하게 0으로 채움)
        feature_data = feature_data.fillna(0.0)
        
        print(f"📊 Feature data shape: {feature_data.shape}")
        print(f"📊 Feature data range check:")
        for col in feature_data.columns[:5]:  # 처음 5개 컬럼만 확인
            col_min, col_max = feature_data[col].min(), feature_data[col].max()
            print(f"  {col}: {col_min:.3f} ~ {col_max:.3f}")
        
        # 4. 예측 수행
        print("🔮 Making predictions...")
        y_pred, y_pred_proba = model.predict(feature_data)
        
        # 5. 신호 변환 (양수 클래스 확률)
        if len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] > 1:
            signals = y_pred_proba[:, 1]  # 양수 클래스 확률
        else:
            signals = y_pred_proba.flatten()
        
        # 6. 신호 통계
        print(f"\n📊 Signal Statistics:")
        print(f"  Count: {len(signals):,}")
        print(f"  Mean: {np.mean(signals):.4f}")
        print(f"  Std: {np.std(signals):.4f}")
        print(f"  Min: {np.min(signals):.4f}")
        print(f"  Max: {np.max(signals):.4f}")
        print(f"  > 0.5: {np.sum(signals > 0.5):,} ({np.mean(signals > 0.5)*100:.1f}%)")
        print(f"  > 0.6: {np.sum(signals > 0.6):,} ({np.mean(signals > 0.6)*100:.1f}%)")
        print(f"  > 0.7: {np.sum(signals > 0.7):,} ({np.mean(signals > 0.7)*100:.1f}%)")
        
        # 7. 이벤트 데이터에 신호 추가
        event_df_with_signal = event_df.with_columns(
            signal_direction=pl.Series("signal_direction", signals, dtype=pl.Float64)
        )
        
        # 8. 전체 데이터에 병합
        result_df = df.join(
            event_df_with_signal.select(["date", "ticker", "signal_direction"]),
            on=["date", "ticker"],
            how="left"
        ).with_columns(
            signal_direction=pl.col("signal_direction").fill_null(0.0)
        )
        
        signal_time = time.time() - start_time
        print(f"✅ Signals created in {signal_time:.2f} seconds")
        
        return result_df
        
    except Exception as e:
        print(f"❌ Error creating signals: {e}")
        import traceback
        traceback.print_exc()
        return df.with_columns(signal_direction=pl.lit(0.0))


def run_simple_backtest(market: str = "KR",
                       years_train: list = [2018, 2019, 2020],
                       years_test: list = [2021],
                       max_tickers: int = 50,  # 더 작게
                       top_positions: int = 10,  # 더 작게
                       min_threshold: float = 0.5) -> dict:
    """
    간단한 백테스트 실행
    
    Args:
        market: 시장 코드
        years_train: 학습 연도
        years_test: 테스트 연도
        max_tickers: 최대 티커 수
        top_positions: 상위 포지션 수
        min_threshold: 최소 신호 임계값
    
    Returns:
        백테스트 결과
    """
    print("🚀 Starting Simple Direction Classifier Backtest")
    print("=" * 60)
    print(f"📊 Market: {market}")
    print(f"🏗️ Train Years: {years_train}")
    print(f"🧪 Test Years: {years_test}")
    print(f"📈 Max Tickers: {max_tickers}")
    print(f"🎯 Top Positions: {top_positions}")
    print(f"🎚️ Min Threshold: {min_threshold}")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. 모델 준비
    print("\n🏗️ Preparing model...")
    model_path = f"models/saved/direction_classifier_{market}_{'_'.join(map(str, years_train))}.txt"
    
    if os.path.exists(model_path):
        print(f"📂 Loading existing model: {model_path}")
        model = DirectionClassifierLGBM()
        model.load_model(model_path)
    else:
        print("🏗️ Training new model...")
        model = create_direction_classifier_model(
            market=market,
            years=years_train,
            save_model=True
        )
    
    # 2. 테스트 데이터 로드 (원본 값 유지)
    print("\n📊 Loading test data (preserving original values)...")
    test_df = load_raw_data_for_backtest(
        years=years_test,
        market=market,
        max_tickers=max_tickers
    )
    
    # 3. 신호 생성
    print("\n🎯 Generating signals...")
    test_df_with_signals = create_simple_signals(model, test_df)
    
    # 4. 백테스트 설정 (매우 관대하게)
    print("\n⚙️ Setting up backtest configuration...")
    config = BacktestConfig(
        label_col="futret_1",
        signal_col="signal_direction",
        universe=UniverseRule(
            top_k_per_day=max_tickers,  # 모든 티커 허용
            min_turnover=1e3,   # 매우 낮은 거래대금 (1천)
            min_price=50        # 매우 낮은 가격 (50원)
        ),
        signal=SignalRule(
            select_top_n=top_positions,
            min_threshold=0.1,  # 매우 낮은 임계값 (10%)
            long_only=True
        ),
        execution=ExecutionRule(mode="next_open_to_close"),
        portfolio=PortfolioRule(
            weighting="equal",
            max_gross_leverage=1.0,
            fee_bps=5.0,  # 낮은 수수료
            slippage_bps=3.0,  # 낮은 슬리피지
            capital_per_position=1_000_000  # 종목당 100만원 할당
        ),
        outdir=Path("reports/backtest_direction")
    )
    
    # 5. 사전 검증
    print("\n🔍 Pre-backtest validation:")
    total_rows = len(test_df_with_signals)
    event_rows = test_df_with_signals.filter(pl.col("label_1d_cls") != 0).height
    signal_rows = test_df_with_signals.filter(pl.col("signal_direction") > 0).height
    strong_signals = test_df_with_signals.filter(pl.col("signal_direction") > min_threshold).height
    
    print(f"  Total rows: {total_rows:,}")
    print(f"  Event rows: {event_rows:,}")
    print(f"  Positive signals: {signal_rows:,}")
    print(f"  Strong signals (>{min_threshold}): {strong_signals:,}")
    
    # 6. 백테스트 실행
    print("\n🚀 Running backtest...")
    try:
        result = backtest(test_df_with_signals, config)
        
        # 7. 결과 출력
        print("\n📊 Backtest Results:")
        print("=" * 50)
        summary = result["summary"]
        
        print("🎯 Performance Metrics:")
        print(f"  Trading Days: {summary['trading_days']}")
        print(f"  Annual Return: {summary['ret_annual']:.2%}")
        print(f"  Volatility: {summary['vol_annual']:.2%}")
        print(f"  Sharpe Ratio: {summary['sharpe']:.2f}")
        print(f"  Max Drawdown: {summary['max_drawdown']:.2%}")
        print(f"  Win Rate: {summary['win_rate']:.1%}")
        print(f"  Total Trades: {summary['total_trades']:,}")
        print(f"  Avg Positions: {summary['avg_n_positions']:.1f}")
        
        # 8. 차트 생성
        print("\n📈 Generating charts...")
        plot_equity(result, show=False)
        plot_drawdown(result, show=False)
        plot_monthly_heatmap(result, show=False)
        plot_rolling_sharpe(result, show=False)
        plot_contrib_by_ticker(result, show=False)
        
        total_time = time.time() - start_time
        print(f"\n🏁 Total execution time: {total_time:.2f} seconds")
        print("✅ Backtest completed successfully!")
        
        return result
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """메인 함수"""
    print("🎯 Simple Direction Classifier Backtest")
    print("=" * 50)
    
    # 간단한 설정
    MARKET = "KR"
    TRAIN_YEARS = [2018, 2019, 2020]
    TEST_YEARS = [2021]
    MAX_TICKERS = 50      # 더 작게
    TOP_POSITIONS = 10    # 더 작게
    MIN_THRESHOLD = 0.5   # 적당한 임계값
    
    print(f"📊 Configuration:")
    print(f"  Market: {MARKET}")
    print(f"  Train: {TRAIN_YEARS}")
    print(f"  Test: {TEST_YEARS}")
    print(f"  Max Tickers: {MAX_TICKERS}")
    print(f"  Top Positions: {TOP_POSITIONS}")
    print(f"  Min Threshold: {MIN_THRESHOLD}")
    print("=" * 50)
    
    try:
        result = run_simple_backtest(
            market=MARKET,
            years_train=TRAIN_YEARS,
            years_test=TEST_YEARS,
            max_tickers=MAX_TICKERS,
            top_positions=TOP_POSITIONS,
            min_threshold=MIN_THRESHOLD
        )
        
        if result:
            print("\n🎉 Success! Check reports/backtest_direction/ for results")
        else:
            print("\n❌ Backtest failed")
            
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()