#!/usr/bin/env python3
"""
M002 Baseline 모델 백테스트 (vectorbt 기반)

M002 Baseline = MultiTask 모델
- Trigger classifier: 리바운드/브레이크다운 이벤트 발생 예측
- Regressor: 수익률 및 최대 낙폭 예측
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
import torch

from data.dataset_builder import build_dataset
from models.M002_MultiTask import M002MultiTaskModel, M002TrainingConfig


class M002BaselineBacktester:
    """M002 Baseline 백테스트 (vectorbt 기반)"""
    
    def __init__(
        self,
        model: M002MultiTaskModel,
        trigger_threshold: float = 0.5,
        top_n_signals: int = 20,
        min_expected_return: float = 1.0,  # 최소 기대 수익률 (%)
        max_drawdown: float = -3.0,        # 최대 허용 낙폭 (%)
        commission: float = 0.001,
        initial_cash: float = 10_000,
    ):
        """
        Args:
            model: 학습된 M002MultiTask 모델
            trigger_threshold: 트리거 확률 임계값
            top_n_signals: 일별 상위 N개 시그널만 선택
            min_expected_return: 최소 기대 수익률 (%)
            max_drawdown: 최대 허용 낙폭 (%)
            commission: 수수료율
            initial_cash: 초기 자본
        """
        self.model = model
        self.trigger_threshold = trigger_threshold
        self.top_n_signals = top_n_signals
        self.min_expected_return = min_expected_return
        self.max_drawdown = max_drawdown
        self.commission = commission
        self.initial_cash = initial_cash
    
    def prepare_data(
        self,
        market: str = "US",
        years: list = [2019, 2020],
        max_tickers: int = 100
    ) -> pl.DataFrame:
        """백테스트용 데이터 준비"""
        print(f"[데이터 준비] {market} 시장, {years} 연도")
        
        config = self.model.config
        
        df = build_dataset(
            years=years,
            market=market,
            max_tickers=max_tickers,
            feature_set=config.feature_set,
            label_horizon=config.horizon,
            label_task="regression",
            verbose=False,
            normalize_features=config.normalize_features
        )
        
        print(f"  로드된 데이터: {len(df):,} 행 × {len(df.columns)} 열")
        print(f"  날짜 범위: {df['date'].min()} ~ {df['date'].max()}")
        print(f"  종목 수: {df['ticker'].n_unique()}개")
        
        return df
    
    def generate_signals(self, df: pl.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        M002 Baseline 모델로 시그널 생성 (LONG/SHORT 지원)

        Returns:
            (entries, exits) 튜플
            - entries: Long/Short entry 시그널 (양수=Long, 음수=Short)
            - exits: Exit 시그널
        """
        print("\n[M002 Baseline 시그널 생성]")

        # 1. 예측
        print("  예측 중...")
        trigger_prob, expected_return, policy_score = self.model.predict(df)

        # 2. 시그널 필터링 및 의사결정
        df_pred = df.with_columns([
            pl.Series("trigger_prob", trigger_prob),
            pl.Series("expected_return", expected_return),
            pl.Series("policy_score", policy_score)
        ])

        # 기본 필터링 (기존 조건)
        df_filtered = df_pred.filter(
            (pl.col("trigger_prob") >= self.trigger_threshold) &
            (pl.col("expected_return") >= self.min_expected_return) &
            (pl.col("expected_drawdown") >= self.max_drawdown)
        )

        print(f"    필터링된 시그널: {len(df_filtered):,} / {len(df):,} ({len(df_filtered)/len(df):.1%})")

        if len(df_filtered) == 0:
            print("    ⚠️ 조건을 만족하는 시그널이 없습니다!")
            return self._empty_signals(df), self._empty_signals(df)

        # 3. 일별 상위 N개 선택 (policy_score 기준)
        df_top_signals = (
            df_filtered
            .with_columns(
                pl.col("policy_score").rank("dense", descending=True).over("date").alias("signal_rank")
            )
            .filter(pl.col("signal_rank") <= self.top_n_signals)
        )

        print(f"    상위 {self.top_n_signals}개 시그널: {len(df_top_signals):,}개")

        # 4. LONG/SHORT/FLAT 의사결정 (policy_score 기반)
        df_decisions = df_top_signals.with_columns([
            # 양수 policy_score: LONG (+1)
            # 음수 policy_score: SHORT (-1)
            # policy_score = 0 근처: FLAT (0)
            pl.when(pl.col("policy_score") > 0.01)
            .then(1)  # LONG
            .when(pl.col("policy_score") < -0.01)
            .then(-1)  # SHORT
            .otherwise(0)  # FLAT
            .alias("decision")
        ])

        # Entry 시그널: Long/Short만
        df_entries = df_decisions.filter(pl.col("decision") != 0)
        entries = self._create_signal_matrix(df, df_entries, include_direction=True)

        # Exit 시그널: 현재 포지션과 반대되는 신호
        # (실제로는 더 복잡한 로직이 필요하지만, 간단히 구현)
        df_exits = df_decisions.filter(pl.col("decision") == 0)
        exits = self._create_signal_matrix(df, df_exits)

        return entries, exits
    
    def _empty_signals(self, df: pl.DataFrame) -> pd.DataFrame:
        """빈 시그널 매트릭스 생성"""
        dates = sorted(df['date'].unique().to_list())
        tickers = sorted(df['ticker'].unique().to_list())
        return pd.DataFrame(0, index=dates, columns=tickers)
    
    def _create_signal_matrix(self, df_all: pl.DataFrame, df_signals: pl.DataFrame, include_direction: bool = False) -> pd.DataFrame:
        """시그널을 ticker × date 매트릭스로 변환"""
        dates = sorted(df_all['date'].unique().to_list())
        tickers = sorted(df_all['ticker'].unique().to_list())

        signal_matrix = pd.DataFrame(0, index=dates, columns=tickers)

        if len(df_signals) > 0:
            if include_direction:
                # decision 컬럼 포함 (LONG=1, SHORT=-1)
                signals_pd = df_signals.select(['date', 'ticker', 'decision']).to_pandas()

                for _, row in signals_pd.iterrows():
                    date = row['date']
                    ticker = row['ticker']
                    decision = row['decision']
                    if date in signal_matrix.index and ticker in signal_matrix.columns:
                        signal_matrix.loc[date, ticker] = decision
            else:
                # 일반 시그널 (1 또는 0)
                signals_pd = df_signals.select(['date', 'ticker']).to_pandas()

                for _, row in signals_pd.iterrows():
                    date = row['date']
                    ticker = row['ticker']
                    if date in signal_matrix.index and ticker in signal_matrix.columns:
                        signal_matrix.loc[date, ticker] = 1

        return signal_matrix
    
    def run(
        self,
        market: str = "US",
        years: list = [2019, 2020],
        max_tickers: int = 100,
        save_dir: str = "reports/m002_baseline_backtest"
    ):
        """백테스트 실행 (LONG/SHORT 지원)"""
        print("\n" + "=" * 60)
        print("[M002 Baseline 백테스트 시작]")
        print("=" * 60)

        # 1. 데이터 준비
        df = self.prepare_data(market=market, years=years, max_tickers=max_tickers)

        # 2. 시그널 생성
        entries, exits = self.generate_signals(df)

        if entries.abs().sum().sum() == 0:
            print("\n⚠️ Entry 시그널이 없어 백테스트를 수행할 수 없습니다!")
            return None

        # 3. 가격 데이터 준비
        print("\n[가격 데이터 준비]")
        price_pivot = self._prepare_price_data(df)

        # 4. vectorbt 백테스트 (LONG/SHORT 지원)
        print("\n[vectorbt 백테스트 실행]")
        portfolio = vbt.Portfolio.from_signals(
            close=price_pivot,
            entries=entries != 0,  # Long/Short entry (양수 또는 음수)
            exits=exits == 1,      # Exit signal
            direction='all',       # LONG/SHORT 모두 지원
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

        # 6. 시그널 차트 생성 (랜덤 종목 샘플링)
        self._plot_sample_signals(df, entries, exits, save_dir)

        # 7. 결과 저장
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        stats.to_csv(f"{save_dir}/stats.csv")
        print(f"\n✅ 통계 저장: {save_dir}/stats.csv")

        fig = portfolio.plot()
        fig.write_html(f"{save_dir}/equity_curve.html")
        print(f"✅ 차트 저장: {save_dir}/equity_curve.html")

        return portfolio
    
    def _prepare_price_data(self, df: pl.DataFrame) -> pd.DataFrame:
        """가격 데이터를 ticker × date 매트릭스로 변환"""
        price_df = df.select(['date', 'ticker', 'close']).to_pandas()
        price_pivot = price_df.pivot(index='date', columns='ticker', values='close')
        price_pivot = price_pivot.fillna(method='ffill')
        
        print(f"  가격 데이터: {price_pivot.shape} (날짜 × 종목)")
        print(f"  결측치: {price_pivot.isna().sum().sum()}개")
        
        return price_pivot

    def _plot_sample_signals(self, df: pl.DataFrame, entries: pd.DataFrame, exits: pd.DataFrame, save_dir: str, n_samples: int = 5):
        """
        랜덤하게 선택된 종목들의 시그널 차트를 생성 (M002 Baseline용)

        Args:
            df: 원본 데이터
            entries: Entry 시그널 매트릭스
            exits: Exit 시그널 매트릭스
            save_dir: 저장 디렉토리
            n_samples: 샘플링할 종목 수
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        print(f"\n[시그널 차트 생성] 랜덤 {n_samples}개 종목 샘플링")

        # 실제 포지션을 취한 종목들 추출 (entries에서 0이 아닌 값)
        traded_tickers = []
        for col in entries.columns:
            if (entries[col] != 0).any():
                traded_tickers.append(col)

        if len(traded_tickers) == 0:
            print("  ⚠️ 거래된 종목이 없어 시그널 차트를 생성할 수 없습니다.")
            return

        # 랜덤하게 종목 선택
        if len(traded_tickers) <= n_samples:
            sample_tickers = traded_tickers
        else:
            sample_tickers = np.random.choice(traded_tickers, n_samples, replace=False)

        print(f"  샘플링된 종목들: {sample_tickers}")

        # 각 종목별 차트 생성
        for ticker in sample_tickers:
            self._plot_ticker_signals_baseline(df, ticker, entries, exits, save_dir)

    def _plot_ticker_signals_baseline(self, df: pl.DataFrame, ticker: str, entries: pd.DataFrame, exits: pd.DataFrame, save_dir: str):
        """
        특정 종목의 시그널 차트를 생성 (M002 Baseline용)

        Args:
            df: 원본 데이터 (예측 결과가 포함됨)
            ticker: 대상 종목
            entries: Entry 시그널 매트릭스
            exits: Exit 시그널 매트릭스
            save_dir: 저장 디렉토리
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 해당 종목 데이터 추출
        ticker_df = df.filter(pl.col("ticker") == ticker).sort("date")

        if len(ticker_df) == 0:
            return

        # pandas로 변환
        ticker_pd = ticker_df.to_pandas()

        # Entry/Exit 시그널 추출
        ticker_entries = entries[ticker][entries[ticker] != 0]
        ticker_exits = exits[ticker][exits[ticker] != 0]

        # 차트 생성
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])

        # 1. 가격 차트
        ax1.plot(ticker_pd['date'], ticker_pd['close'], linewidth=2, color='black', label='Close Price')

        # Entry 시그널 표시 (양수=LONG, 음수=SHORT)
        if not ticker_entries.empty:
            for date, signal in ticker_entries.items():
                if date in ticker_pd['date'].values:
                    price = ticker_pd[ticker_pd['date'] == date]['close'].iloc[0]
                    color = 'red' if signal > 0 else 'blue'
                    marker = '^' if signal > 0 else 'v'
                    label = 'LONG Entry' if signal > 0 else 'SHORT Entry'

                    ax1.scatter(date, price, color=color, s=100, marker=marker,
                               label=label, edgecolors='black', linewidth=2, zorder=5)

        # Exit 시그널 표시
        if not ticker_exits.empty:
            for date, _ in ticker_exits.items():
                if date in ticker_pd['date'].values:
                    price = ticker_pd[ticker_pd['date'] == date]['close'].iloc[0]
                    ax1.scatter(date, price, color='green', s=80, marker='x',
                               label='Exit', edgecolors='black', linewidth=2, zorder=5)

        ax1.set_title(f'{ticker} - Price & Signals (M002 Baseline)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Policy Score 차트
        if 'policy_score' in ticker_pd.columns:
            ax2.plot(ticker_pd['date'], ticker_pd['policy_score'], linewidth=2, color='green', label='Policy Score')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Neutral')

            # 양수/음수 영역 표시
            ax2.fill_between(ticker_pd['date'], ticker_pd['policy_score'], 0,
                           where=(ticker_pd['policy_score'] > 0), color='red', alpha=0.2, label='Positive Score')
            ax2.fill_between(ticker_pd['date'], ticker_pd['policy_score'], 0,
                           where=(ticker_pd['policy_score'] < 0), color='blue', alpha=0.2, label='Negative Score')

            ax2.set_title('Policy Score', fontsize=12, fontweight='bold')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Policy Score 데이터 없음', transform=ax2.transAxes,
                    ha='center', va='center', fontsize=12)

        ax2.set_xlabel('Date')
        ax2.set_ylabel('Policy Score')
        ax2.grid(True, alpha=0.3)

        # 시그널 정보 텍스트 추가
        signal_info = []
        if not ticker_entries.empty:
            for date, signal in ticker_entries.items():
                if date in ticker_pd['date'].values:
                    row = ticker_pd[ticker_pd['date'] == date].iloc[0]
                    action = 'LONG' if signal > 0 else 'SHORT'
                    info = f"{action}: {date.date()}\n"
                    if 'policy_score' in row:
                        info += f"Policy: {row['policy_score']:.3f}\n"
                    if 'trigger_prob' in row:
                        info += f"Trigger Prob: {row['trigger_prob']:.3f}\n"
                    if 'expected_return' in row:
                        info += f"Exp Ret: {row['expected_return']:.2f}%"
                    signal_info.append(info)

        # 정보 텍스트를 차트에 추가 (우상단)
        if signal_info:
            info_text = "\n\n".join(signal_info[:3])  # 최대 3개만 표시
            ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
                    fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        # 저장
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        chart_path = f"{save_dir}/signal_chart_{ticker}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ {ticker} 시그널 차트 저장: {chart_path}")

        plt.close()


def main():
    """M002 Baseline 백테스트 실행"""

    # 1. 저장된 모델 로드
    print("[M002 Baseline 모델 로드]")

    import joblib

    # 저장된 모델 파일 경로
    model_path = "models/saved/m002_multitask_US_2000-2018.pkl"

    try:
        # 모델 로드
        model = joblib.load(model_path)
        print(f"  ✅ 모델 로드 성공: {model_path}")

        # 설정 정보 출력
        print(f"  📊 시장: {model.config.market}")
        print(f"  📊 학습 연도: {model.config.years}")
        print(f"  📊 예측 기간: {model.config.horizon}일")
        print(f"  📊 리스크 회피도(λ): {model.config.risk_aversion}")

    except FileNotFoundError:
        print(f"  ❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        print("  🔄 새로운 모델을 학습합니다...")

        # 모델이 없으면 새로 학습
        config = M002TrainingConfig(
            market="US",
            years=list(range(2000, 2019)),
            horizon=5,
            rebound_thresh=1.0,
            drawdown_floor=-3.0
        )

        model = M002MultiTaskModel(config=config)

        # 학습 데이터 로드
        train_df = build_dataset(
            years=config.years,
            market=config.market,
            max_tickers=100,
            feature_set=config.feature_set,
            label_horizon=config.horizon,
            label_task="regression",
            verbose=False,
            normalize_features=config.normalize_features
        )

        # 모델 학습
        print("\n[모델 학습]")
        model.train(train_df)

        # 모델 저장
        joblib.dump(model, model_path)
        print(f"  💾 모델 저장됨: {model_path}")

    # 2. 백테스터 생성
    backtester = M002BaselineBacktester(
        model=model,
        trigger_threshold=0.5,
        top_n_signals=20,
        min_expected_return=1.0,
        max_drawdown=-3.0,
        commission=0.001,
        initial_cash=10_000
    )

    # 3. 백테스트 실행
    portfolio = backtester.run(
        market="US",
        years=[2019, 2020],
        max_tickers=100,
        save_dir="reports/m002_baseline_backtest"
    )

    return portfolio


if __name__ == "__main__":
    try:
        portfolio = main()
        print("\n✅ M002 Baseline 백테스트 완료!")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
