#!/usr/bin/env python3
"""
M001 모델 백테스트 (vectorbt 기반)

M001은 두 개의 모델로 구성:
1. EventDetector (TCN): 급등락 이벤트 발생 예측 (확률)
2. DirectionClassifier (LGBM): 이벤트 발생 시 방향(상승/하락) 예측

매매 로직:
- EventDetector가 높은 확률로 이벤트를 예측하고
- DirectionClassifier가 상승(+1)을 예측하면 Long
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore')

import polars as pl
import pandas as pd
import numpy as np
import vectorbt as vbt
from tqdm import tqdm

from data.dataset_builder import build_dataset
from models.M001_EventDetector import EventDetectorManager
from models.M001_DirectionClassifier import DirectionClassifierLGBM


class M001Backtester:
    """M001 모델 백테스트 (vectorbt 기반)"""
    
    def __init__(
        self,
        event_model_path: str,
        direction_model_path: str,
        event_threshold: float = 0.5,
        top_n_events: int = 20,
        commission: float = 0.001,  # 0.1% 수수료
        initial_cash: float = 10_000,
    ):
        """
        Args:
            event_model_path: EventDetector 모델 경로
            direction_model_path: DirectionClassifier 모델 경로
            event_threshold: 이벤트 확률 임계값
            top_n_events: 일별 상위 N개 이벤트만 선택
            commission: 수수료율 (0.001 = 0.1%)
            initial_cash: 초기 자본
        """
        self.event_threshold = event_threshold
        self.top_n_events = top_n_events
        self.commission = commission
        self.initial_cash = initial_cash
        
        # 모델 로드
        print("[M001 모델 로드]")
        self.event_detector = EventDetectorManager()
        self.event_detector.load_model(event_model_path)
        print(f"  ✅ EventDetector 로드: {event_model_path}")
        
        self.direction_classifier = DirectionClassifierLGBM()
        self.direction_classifier.load_model(direction_model_path)
        print(f"  ✅ DirectionClassifier 로드: {direction_model_path}")
    
    def prepare_data(
        self,
        market: str = "KR",
        years: list = [2021, 2022],
        max_tickers: int = 100
    ) -> pl.DataFrame:
        """백테스트용 데이터 준비"""
        print(f"[데이터 준비] {market} 시장, {years} 연도")
        
        # 데이터셋 빌드 (v2 feature set)
        df = build_dataset(
            years=years,
            market=market,
            max_tickers=max_tickers,
            feature_set="v2",
            label_horizon=1,
            label_task="classification",
            label_thresh=0.05,
            verbose=False,
            normalize_features=False
        )
        
        print(f"  로드된 데이터: {len(df):,} 행 × {len(df.columns)} 열")
        print(f"  날짜 범위: {df['date'].min()} ~ {df['date'].max()}")
        print(f"  종목 수: {df['ticker'].n_unique()}개")
        
        return df
    
    def generate_signals(self, df: pl.DataFrame) -> pd.DataFrame:
        """
        M001 모델로 시그널 생성
        
        Returns:
            시그널 데이터프레임 (ticker × date pivot)
        """
        print("\n[M001 시그널 생성]")
        
        # 1. EventDetector로 이벤트 확률 예측
        print("  1. EventDetector 예측 중...")
        event_preds, event_probs = self.event_detector.predict(df)
        
        # 2. 이벤트가 발생할 것으로 예측된 행만 필터링
        df_with_event = df.with_columns([
            pl.Series("event_pred", event_preds),
            pl.Series("event_prob", event_probs)
        ])
        
        # NaN 및 -1 값 필터링 (예측 실패한 행)
        df_events = df_with_event.filter(
            (pl.col("event_pred") == 1) &
            (pl.col("event_prob") >= self.event_threshold) &
            (~pl.col("event_prob").is_nan())
        )
        
        print(f"    이벤트 예측: {len(df_events):,} / {len(df):,} ({len(df_events)/len(df):.1%})")
        
        if len(df_events) == 0:
            print("    ⚠️ 예측된 이벤트가 없습니다!")
            return self._empty_signals(df)
        
        # 3. 일별 상위 N개 이벤트만 선택
        df_top_events = (
            df_events
            .with_columns(
                pl.col("event_prob").rank("dense", descending=True).over("date").alias("event_rank")
            )
            .filter(pl.col("event_rank") <= self.top_n_events)
        )
        
        print(f"    상위 {self.top_n_events}개 이벤트 선택: {len(df_top_events):,}개")
        
        # 4. DirectionClassifier로 방향 예측
        print("  2. DirectionClassifier 예측 중...")
        
        # DirectionClassifier 입력 준비
        direction_features = self.direction_classifier.feature_list
        available_features = [f for f in direction_features if f in df_top_events.columns]
        
        if not available_features:
            print("    ⚠️ DirectionClassifier 피처가 없습니다!")
            return self._empty_signals(df)
        
        X_direction = df_top_events.select(available_features).to_pandas().fillna(0.0)
        
        # 방향 예측 (0: 하락, 1: 상승)
        dir_preds, dir_probs = self.direction_classifier.predict(X_direction)
        
        # 5. Long 시그널 생성 (상승 예측)
        df_signals = df_top_events.with_columns([
            pl.Series("direction_pred", dir_preds),
            pl.Series("direction_prob", dir_probs)
        ])
        
        # 상승 예측(1)만 Long
        df_long_signals = df_signals.filter(pl.col("direction_pred") == 1)
        
        print(f"    Long 시그널: {len(df_long_signals):,}개")
        
        # 6. Pivot to ticker × date
        signal_df = self._create_signal_matrix(df, df_long_signals)
        
        return signal_df
    
    def _empty_signals(self, df: pl.DataFrame) -> pd.DataFrame:
        """빈 시그널 매트릭스 생성"""
        dates = sorted(df['date'].unique().to_list())
        tickers = sorted(df['ticker'].unique().to_list())
        return pd.DataFrame(0, index=dates, columns=tickers)
    
    def _create_signal_matrix(self, df_all: pl.DataFrame, df_signals: pl.DataFrame) -> pd.DataFrame:
        """
        시그널을 ticker × date 매트릭스로 변환
        
        Args:
            df_all: 전체 데이터
            df_signals: 시그널 데이터 (Long할 종목)
        
        Returns:
            시그널 매트릭스 (1 = Long, 0 = 관망)
        """
        # 전체 날짜와 종목 리스트
        dates = sorted(df_all['date'].unique().to_list())
        tickers = sorted(df_all['ticker'].unique().to_list())
        
        # 빈 매트릭스 생성
        signal_matrix = pd.DataFrame(0, index=dates, columns=tickers)
        
        # 시그널이 있는 (date, ticker) 위치에 1 설정
        if len(df_signals) > 0:
            signals_pd = df_signals.select(['date', 'ticker']).to_pandas()
            
            for _, row in signals_pd.iterrows():
                date = row['date']
                ticker = row['ticker']
                if date in signal_matrix.index and ticker in signal_matrix.columns:
                    signal_matrix.loc[date, ticker] = 1
        
        return signal_matrix
    
    def run(
        self,
        market: str = "KR",
        years: list = [2021, 2022],
        max_tickers: int = 100,
        save_dir: str = "reports/m001_backtest"
    ):
        """백테스트 실행"""
        print("\n" + "=" * 60)
        print("[M001 백테스트 시작]")
        print("=" * 60)
        
        # 1. 데이터 준비
        df = self.prepare_data(market=market, years=years, max_tickers=max_tickers)
        
        # 2. 시그널 생성
        signals = self.generate_signals(df)
        
        if signals.sum().sum() == 0:
            print("\n⚠️ 시그널이 없어 백테스트를 수행할 수 없습니다!")
            return None
        
        # 3. 가격 데이터 준비 (pivot)
        print("\n[가격 데이터 준비]")
        price_pivot = self._prepare_price_data(df)
        
        # 4. vectorbt 백테스트
        print("\n[vectorbt 백테스트 실행]")
        portfolio = vbt.Portfolio.from_signals(
            close=price_pivot,
            entries=signals == 1,  # Long entry
            exits=signals == 0,    # Exit when signal disappears
            direction='longonly',
            init_cash=self.initial_cash,
            fees=self.commission,
            freq='1D'
        )
        
        # 5. 결과 출력
        print("\n" + "=" * 60)
        print("[백테스트 결과]")
        print("=" * 60)
        stats = portfolio.stats()
        print(stats)
        
        # 6. 결과 저장
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # 통계 저장
        stats.to_csv(f"{save_dir}/stats.csv")
        print(f"\n✅ 통계 저장: {save_dir}/stats.csv")
        
        # 차트 저장
        fig = portfolio.plot()
        fig.write_html(f"{save_dir}/equity_curve.html")
        print(f"✅ 차트 저장: {save_dir}/equity_curve.html")
        
        return portfolio
    
    def _prepare_price_data(self, df: pl.DataFrame) -> pd.DataFrame:
        """가격 데이터를 ticker × date 매트릭스로 변환"""
        price_df = df.select(['date', 'ticker', 'close']).to_pandas()
        price_pivot = price_df.pivot(index='date', columns='ticker', values='close')
        
        # 결측치 forward fill
        price_pivot = price_pivot.fillna(method='ffill')
        
        print(f"  가격 데이터: {price_pivot.shape} (날짜 × 종목)")
        print(f"  결측치: {price_pivot.isna().sum().sum()}개")
        
        return price_pivot


def main():
    """M001 백테스트 실행"""

    print("[M001 모델들 로드]")

    # 모델 경로 (저장된 모델 사용)
    event_model_path = "models/saved/tcn_event_detector_KR_2018_2019_2020_100pct_L60"
    direction_model_path = "models/saved/direction_classifier_KR_2018_2019_2020.txt"

    # 모델 파일 존재 확인
    import os
    event_exists = os.path.exists(f"{event_model_path}.pth")
    direction_exists = os.path.exists(direction_model_path)

    if not event_exists:
        print(f"  ❌ EventDetector 모델 파일을 찾을 수 없습니다: {event_model_path}.pth")
        return None

    if not direction_exists:
        print(f"  ❌ DirectionClassifier 모델 파일을 찾을 수 없습니다: {direction_model_path}")
        return None

    print(f"  ✅ EventDetector 모델: {event_model_path}.pth")
    print(f"  ✅ DirectionClassifier 모델: {direction_model_path}")

    # 백테스터 생성
    backtester = M001Backtester(
        event_model_path=event_model_path,
        direction_model_path=direction_model_path,
        event_threshold=0.5,      # 이벤트 확률 50% 이상
        top_n_events=20,          # 일별 상위 20개 이벤트
        commission=0.001,         # 0.1% 수수료
        initial_cash=10_000
    )

    # 백테스트 실행
    portfolio = backtester.run(
        market="KR",
        years=[2021, 2022],
        max_tickers=100,
        save_dir="reports/m001_backtest"
    )

    return portfolio


if __name__ == "__main__":
    try:
        portfolio = main()
        print("\n✅ M001 백테스트 완료!")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

