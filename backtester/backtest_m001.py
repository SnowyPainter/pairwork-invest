#!/usr/bin/env python3
"""
Combined Models 백테스팅

Direction Classifier + Event Detector를 함께 사용하는 통합 전략
- Direction Classifier: 상승/하락 방향 예측
- Event Detector: 큰 변동 이벤트 감지
- 두 모델의 신호를 결합하여 더 정확한 예측
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

from models.M001_DirectionClassifier import DirectionClassifierLGBM, create_direction_classifier_model, SELECTED_FEATURES
from models.M001_EventDetector import EventDetectorLGBM, create_event_detector_model
from data.dataset_builder import build_dataset
from backtester.backtester import (
    BacktestConfig, UniverseRule, SignalRule, ExecutionRule, PortfolioRule,
    backtest, plot_equity, plot_drawdown, plot_monthly_heatmap,
    plot_rolling_sharpe, plot_contrib_by_ticker
)


def create_combined_signals(direction_model: DirectionClassifierLGBM, 
                          event_model: EventDetectorLGBM, 
                          df: pl.DataFrame,
                          direction_weight: float = 0.6,
                          event_weight: float = 0.4,
                          min_event_prob: float = 0.3) -> pl.DataFrame:
    """
    Direction Classifier와 Event Detector의 신호를 결합
    
    Args:
        direction_model: 방향 예측 모델
        event_model: 이벤트 감지 모델  
        df: 데이터프레임
        direction_weight: 방향 예측 가중치
        event_weight: 이벤트 감지 가중치
        min_event_prob: 최소 이벤트 확률 (이 이상일 때만 거래)
        
    Returns:
        통합 신호가 추가된 데이터프레임
    """
    print("[통합 신호 생성]")
    start_time = time.time()
    
    # 이벤트 데이터만 필터링
    event_df = df.filter(pl.col("label_1d_cls") != 0)
    event_count = len(event_df)
    
    print(f"  이벤트 데이터: {event_count:,} 행")
    
    if event_count == 0:
        print("  [경고] 이벤트 데이터가 없습니다!")
        return df.with_columns(
            signal_direction=pl.lit(0.0),
            signal_event=pl.lit(0.0),
            signal_combined=pl.lit(0.0)
        )
    
    # 1. Direction Classifier 신호 생성
    print("  방향 예측 신호 생성 중...")
    direction_features = [f for f in SELECTED_FEATURES if f in event_df.columns]
    if direction_features:
        direction_X = event_df.select(direction_features).to_pandas().fillna(0.0)
        _, direction_proba = direction_model.predict(direction_X)
        
        # 이진 분류인 경우 양수 클래스 확률 추출
        if len(direction_proba.shape) == 2:
            direction_signals = direction_proba[:, 1]
        else:
            direction_signals = direction_proba
    else:
        print("    [경고] 방향 예측용 피처가 없습니다!")
        direction_signals = np.zeros(event_count)
    
    # 2. Event Detector 신호 생성
    print("  이벤트 감지 신호 생성 중...")
    print(f"    모델이 가진 피처 수: {len(event_model.features)}")

    # 사용 가능한 피처 확인
    available_features = [f for f in event_model.features if f in event_df.columns]
    missing_features = [f for f in event_model.features if f not in event_df.columns]

    print(f"    데이터에서 사용 가능한 피처: {len(available_features)}개")
    if missing_features:
        print(f"    누락된 피처: {len(missing_features)}개")
        print(f"    누락 피처 샘플: {missing_features[:5]}")

    if available_features:
        print(f"    예측에 사용할 피처: {len(available_features)}개")
        event_X = event_df.select(available_features).to_pandas().fillna(0.0)
        _, event_proba = event_model.predict(event_X)
        event_signals = event_proba
    else:
        print("    [경고] 이벤트 감지용 피처가 없습니다!")
        event_signals = np.zeros(event_count)
    
    # 3. 신호 통합
    print("  신호 통합 중...")
    
    # 가중 평균으로 통합
    combined_signals = (direction_weight * direction_signals + 
                       event_weight * event_signals)

    # 이벤트 확률이 낮으면 신호 강도 감소
    event_filter = event_signals >= min_event_prob
    combined_signals = np.where(event_filter, combined_signals, combined_signals * 0.5)
    
    # 신호 통계
    print(f"  신호 통계:")
    print(f"    방향 신호 평균: {np.mean(direction_signals):.3f}")
    print(f"    이벤트 신호 평균: {np.mean(event_signals):.3f}")
    print(f"    통합 신호 평균: {np.mean(combined_signals):.3f}")
    print(f"    강한 신호 (>0.6): {np.sum(combined_signals > 0.6):,}개")
    print(f"    이벤트 필터 통과: {np.sum(event_filter):,}개")
    
    # 이벤트 데이터에 신호 추가
    event_df_with_signals = event_df.with_columns([
        pl.Series("signal_direction", direction_signals, dtype=pl.Float64),
        pl.Series("signal_event", event_signals, dtype=pl.Float64),
        pl.Series("signal_combined", combined_signals, dtype=pl.Float64)
    ])
    
    # 전체 데이터에 병합
    result_df = df.join(
        event_df_with_signals.select(["date", "ticker", "signal_direction", "signal_event", "signal_combined"]),
        on=["date", "ticker"],
        how="left"
    ).with_columns([
        pl.col("signal_direction").fill_null(0.0),
        pl.col("signal_event").fill_null(0.0),
        pl.col("signal_combined").fill_null(0.0)
    ])
    
    signal_time = time.time() - start_time
    print(f"  소요시간: {signal_time:.2f}초")
    
    return result_df


def run_combined_backtest(market: str = "KR",
                         years_train: list = [2018, 2019, 2020],
                         years_test: list = [2021],
                         max_tickers: int = 50,
                         top_positions: int = 10,
                         direction_weight: float = 0.6,
                         event_weight: float = 0.4,
                         min_signal_threshold: float = 0.5,
                         min_event_prob: float = 0.3) -> dict:
    """
    통합 모델 백테스트 실행
    
    Args:
        market: 시장 코드
        years_train: 학습 연도
        years_test: 테스트 연도
        max_tickers: 최대 종목 수
        top_positions: 상위 포지션 수
        direction_weight: 방향 예측 가중치
        event_weight: 이벤트 감지 가중치
        min_signal_threshold: 최소 신호 임계값
        min_event_prob: 최소 이벤트 확률
        
    Returns:
        백테스트 결과
    """
    print("[통합 모델 백테스트]")
    print("=" * 60)
    print(f"  시장: {market}")
    print(f"  학습 연도: {years_train}")
    print(f"  테스트 연도: {years_test}")
    print(f"  최대 종목: {max_tickers}개")
    print(f"  상위 포지션: {top_positions}개")
    print(f"  방향/이벤트 가중치: {direction_weight:.1f}/{event_weight:.1f}")
    print(f"  신호 임계값: {min_signal_threshold}")
    print(f"  이벤트 확률 임계값: {min_event_prob}")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. Direction Classifier 준비
    print("\n[Direction Classifier 준비]")
    direction_model_path = f"models/saved/direction_classifier_{market}_{'_'.join(map(str, years_train))}.txt"
    
    if os.path.exists(direction_model_path):
        print(f"  기존 모델 로드: {direction_model_path}")
        direction_model = DirectionClassifierLGBM()
        direction_model.load_model(direction_model_path)
    else:
        print("  새로운 Direction Classifier 학습...")
        direction_model = create_direction_classifier_model(
            market=market,
            years=years_train,
            save_model=True
        )
    
        # 2. Event Detector 준비 (향상된 피처 사용)
    print("\n[Event Detector 준비]")
    event_model_path = f"models/saved/event_detector_{market}_{'_'.join(map(str, years_train))}_5pct.txt"

    if os.path.exists(event_model_path):
        print(f"  기존 향상된 모델 로드: {event_model_path}")
        event_model = EventDetectorLGBM(threshold=0.05, use_enhanced_features=True)
        event_model.load_model(event_model_path)
    else:
        print("  새로운 Event Detector 학습 (향상된 피처)...")
        print("  [참고] 기존 모델과 호환되지 않으므로 새로운 모델을 학습합니다")
        event_model = create_event_detector_model(
            market=market,
            years=years_train,
            threshold=0.05,
            target="big_move_event",
            max_tickers=max_tickers,
            save_model=True,
            use_enhanced_features=True  # 향상된 피처 사용
        )
    
    # 3. 테스트 데이터 로드 및 Event Detector 피처 생성
    print("\n[테스트 데이터 로드]")
    test_df = build_dataset(
        years=years_test,
        market=market,
        max_tickers=max_tickers,
        feature_set="v2",  # 기본 피처 세트
        label_horizon=1,
        label_task="classification",
        normalize_features=True,
        verbose=True
    )

    print(f"  테스트 데이터: {len(test_df):,} 행")
    print(f"  기본 피처 수: {len(test_df.columns)}개")

    # Event Detector의 향상된 피처들 생성
    print("\n[Event Detector 향상된 피처 생성]")
    test_df = event_model._generate_regime_features(test_df)
    print(f"  향상된 피처 적용 후 컬럼 수: {len(test_df.columns)}개")
    
    # 4. 통합 신호 생성
    print("\n[통합 신호 생성]")
    test_df_with_signals = create_combined_signals(
        direction_model=direction_model,
        event_model=event_model,
        df=test_df,
        direction_weight=direction_weight,
        event_weight=event_weight,
        min_event_prob=min_event_prob
    )
    
    # 5. 백테스트 설정 (더 관대하게)
    print("\n[백테스트 설정]")
    config = BacktestConfig(
        label_col="futret_1",
        signal_col="signal_combined",
        universe=UniverseRule(
            top_k_per_day=max_tickers,
            min_turnover=1e3,
            min_price=50
        ),
        signal=SignalRule(
            select_top_n=top_positions,
            min_threshold=min_signal_threshold,  # 임계값을 매우 낮게 설정
            long_only=True
        ),
        execution=ExecutionRule(mode="next_open_to_close"),
        portfolio=PortfolioRule(
            weighting="equal",
            fee_bps=8.0,
            slippage_bps=5.0,
            capital_per_position=1_000_000
        ),
        outdir=Path("reports/backtest_combined")
    )
    
    print(f"  설정된 임계값: {config.signal.min_threshold} (원래 요청: {min_signal_threshold})")
    print(f"  유니버스 크기: {config.universe.top_k_per_day} (원래: {max_tickers})")
    print(f"  최소 거래대금: {config.universe.min_turnover:,}원")
    print(f"  최소 가격: {config.universe.min_price}원")
    
    # 6. 백테스트 실행
    print("\n[백테스트 실행]")
    try:
        result = backtest(test_df_with_signals, config)
        
        # 7. 결과 출력
        print("\n[백테스트 결과]")
        summary = result["summary"]
        
        print("  성과 지표:")
        print(f"    거래일수: {summary['trading_days']}")
        print(f"    연간 수익률: {summary['ret_annual']:.2%}")
        print(f"    변동성: {summary['vol_annual']:.2%}")
        print(f"    샤프 비율: {summary['sharpe']:.2f}")
        print(f"    최대 낙폭: {summary['max_drawdown']:.2%}")
        print(f"    승률: {summary['win_rate']:.1%}")
        print(f"    총 거래: {summary['total_trades']:,}")
        
        # 8. 신호별 성과 분석 및 디버깅
        print("\n  신호 필터링 과정 분석:")
        
        # 각 단계별 필터링 현황
        total_data = len(test_df_with_signals)
        has_signal = test_df_with_signals.filter(pl.col("signal_combined") > 0).height
        strong_signal = test_df_with_signals.filter(pl.col("signal_combined") > min_signal_threshold).height
        
        print(f"    전체 데이터: {total_data:,} 행")
        print(f"    신호 있음 (>0): {has_signal:,} 행")
        print(f"    강한 신호 (>{min_signal_threshold}): {strong_signal:,} 행")
        
        # 백테스터 결과에서 선택된 데이터 분석
        if "daily" in result and len(result["daily"]) > 0:
            daily_data = result["daily"].to_pandas()
            print(f"    백테스터 거래일수: {len(daily_data)} 일")
            print(f"    총 거래수: {daily_data['n_positions'].sum()}")
            print(f"    평균 일별 포지션: {daily_data['n_positions'].mean():.1f}")
            
            # 실제 거래 수익률 분석
            if len(daily_data) > 0:
                print(f"    평균 일별 수익률: {daily_data['daily_return'].mean():.4f} ({daily_data['daily_return'].mean()*100:.2f}%)")
                print(f"    일별 수익률 범위: {daily_data['daily_return'].min():.4f} ~ {daily_data['daily_return'].max():.4f}")
                
                # 양수/음수 일 분석
                positive_days = (daily_data['daily_return'] > 0).sum()
                total_days = len(daily_data)
                print(f"    수익 일수: {positive_days}/{total_days} ({positive_days/total_days:.1%})")
        
        # 백테스터 처리된 데이터 분석
        if "processed_data" in result and len(result["processed_data"]) > 0:
            processed_data = result["processed_data"]
            selected_data = processed_data.filter(pl.col("selected") == True)

            print(f"\n  백테스터 처리 결과:")
            print(f"    처리된 총 데이터: {len(processed_data):,} 행")
            print(f"    선택된 포지션: {len(selected_data):,} 행")

            if len(selected_data) > 0:
                selected_pd = selected_data.to_pandas()

                # 1. 개별 거래의 순수익 총합
                total_net_pnl = selected_pd['net_pnl'].sum()
                # 2. 개별 거래의 총 투자금 총합
                total_invested = selected_pd['position_capital'].sum()
                print(f"    총 투자금: {total_invested:,.0f}원")

                # 3. 개별 거래별 수익률 분석
                if len(selected_pd) > 0:
                    individual_returns = selected_pd['net_pnl'] / selected_pd['position_capital']
                    avg_individual_return = individual_returns.mean()
                    print(f"    개별 거래 평균 수익률: {avg_individual_return:.4f} ({avg_individual_return*100:.2f}%)")
                    print(f"    개별 거래 수익률 범위: {individual_returns.min():.4f} ~ {individual_returns.max():.4f}")
                # 4. 백테스터의 최종 수익률
                if "daily" in result and len(result["daily"]) > 0:
                    final_equity = result["daily"].select("equity").to_pandas().iloc[-1, 0]
                    backtest_return = final_equity - 1
                    print(f"    백테스터 최종 수익률: {backtest_return:.4f} ({backtest_return*100:.2f}%)")
                # 실제 선택된 거래의 신호 분석
                signal_col = config.signal_col
                if signal_col in selected_pd.columns and "futret_1" in selected_pd.columns:
                    signal_corr = selected_pd[signal_col].corr(selected_pd["futret_1"])
                    print(f"    선택된 거래의 신호-수익률 상관관계: {signal_corr:.3f}")
                    
                    print(f"    선택된 거래 신호 통계:")
                    print(f"      평균: {selected_pd[signal_col].mean():.3f}")
                    print(f"      범위: {selected_pd[signal_col].min():.3f} ~ {selected_pd[signal_col].max():.3f}")
                    
                    print(f"    선택된 거래 수익률 통계:")
                    print(f"      평균: {selected_pd['futret_1'].mean():.4f} ({selected_pd['futret_1'].mean()*100:.2f}%)")
                    print(f"      범위: {selected_pd['futret_1'].min():.4f} ~ {selected_pd['futret_1'].max():.4f}")
                    
                    # 일별 거래 분포
                    daily_trades = selected_pd.groupby('date').size()
                    print(f"    일별 거래 분포:")
                    print(f"      평균: {daily_trades.mean():.1f} 거래/일")
                    print(f"      최대: {daily_trades.max()} 거래/일")
                    print(f"      총 거래일: {len(daily_trades)} 일")
        else:
            print("\n  백테스터 처리된 데이터를 찾을 수 없습니다.")
        
        # 입력 데이터 디버깅
        print(f"\n  입력 데이터 체크:")
        print(f"    컬럼: {list(test_df_with_signals.columns)}")
        
        # 강한 신호 분석
        strong_signal_data = test_df_with_signals.filter(
            pl.col("signal_combined") > min_signal_threshold
        )
        
        if len(strong_signal_data) > 0:
            print(f"    강한 신호 데이터 ({len(strong_signal_data)}개):")
            
            # 상위 몇 개 신호 출력
            top_signals = strong_signal_data.sort("signal_combined", descending=True).head(5)
            print("    상위 5개 신호:")
            for row in top_signals.to_dicts():
                print(f"      {row['date']} {row['ticker']}: signal={row['signal_combined']:.3f}, ret={row['futret_1']:.4f}")
        else:
            print("    강한 신호 데이터가 없습니다.")
        
        # 9. 차트 생성
        print("\n[차트 생성]")
        plot_equity(result, show=False)
        plot_drawdown(result, show=False)
        plot_monthly_heatmap(result, show=False)
        # 데이터 길이에 맞는 window size로 조정
        data_length = len(result['daily']) if 'daily' in result and len(result['daily']) > 0 else 0
        window_size = min(30, max(10, data_length // 3))  # 데이터 길이의 1/3, 최소 10, 최대 30
        print(f"[Rolling Sharpe] 데이터 길이: {data_length}일, window size: {window_size}일")
        plot_rolling_sharpe(result, window=window_size, show=False)
        plot_contrib_by_ticker(result, show=False)
        
        total_time = time.time() - start_time
        print(f"\n[완료] 총 {total_time:.2f}초 소요")
        
        return result
        
    except Exception as e:
        print(f"  [오류] 백테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """메인 함수"""
    print("🎯 Combined Models Backtest")
    print("Direction Classifier + Event Detector (향상된 피처)")
    print("=" * 60)

    # 설정
    MARKET = "KR"
    TRAIN_YEARS = [2018, 2019, 2020]
    TEST_YEARS = [2021]
    MAX_TICKERS = 50
    TOP_POSITIONS = 10
    DIRECTION_WEIGHT = 0.5
    EVENT_WEIGHT = 0.5
    MIN_SIGNAL_THRESHOLD = 0.5
    MIN_EVENT_PROB = 0.3

    print(f"🔧 Event Detector: 향상된 피처 세트 사용 (상관관계 + 단기 V-score 패턴)")
    print(f"📊 예상 피처 수: 95개 (기존 67개 + 28개 향상)")
    
    try:
        result = run_combined_backtest(
            market=MARKET,
            years_train=TRAIN_YEARS,
            years_test=TEST_YEARS,
            max_tickers=MAX_TICKERS,
            top_positions=TOP_POSITIONS,
            direction_weight=DIRECTION_WEIGHT,
            event_weight=EVENT_WEIGHT,
            min_signal_threshold=MIN_SIGNAL_THRESHOLD,
            min_event_prob=MIN_EVENT_PROB
        )
        
        if result:
            print("\n🎉 통합 모델 백테스트 완료!")
            print("📁 결과는 reports/backtest_combined/ 에서 확인하세요")
        else:
            print("\n❌ 백테스트 실패")
            
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
