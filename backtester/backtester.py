#!/usr/bin/env python3
"""
개선된 백테스팅 프레임워크

주요 개선사항:
- nan 결과 문제 해결
- 데이터 검증 강화
- 메모리 효율성 개선
- 명확한 에러 처리
"""

import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale = 1.5,font ='NanumGothic' )


@dataclass
class UniverseRule:
    """유니버스 선택 규칙"""
    top_k_per_day: int = 50
    min_turnover: float = 1e4  # 최소 거래대금 (더 낮게)
    min_price: float = 100     # 최소 가격 (더 낮게)


@dataclass
class SignalRule:
    """신호 선택 규칙"""
    select_top_n: int = 20
    min_threshold: float = 0.1  # 매우 낮은 임계값
    long_only: bool = True


@dataclass
class ExecutionRule:
    """실행 규칙"""
    mode: str = "next_open_to_close"


@dataclass
class PortfolioRule:
    """포트폴리오 규칙"""
    weighting: str = "equal"
    max_gross_leverage: float = 1.0
    fee_bps: float = 8.0
    slippage_bps: float = 5.0
    capital_per_position: float = 1_000_000  # 종목당 할당 자산 (기본 100만원)


@dataclass
class BacktestConfig:
    """백테스트 설정"""
    label_col: str
    signal_col: str
    universe: UniverseRule
    signal: SignalRule
    execution: ExecutionRule
    portfolio: PortfolioRule
    outdir: Path


def _safe_mean(arr):
    """안전한 평균 계산"""
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr))


def _safe_std(arr):
    """안전한 표준편차 계산"""
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def _select_universe(df: pl.DataFrame, config: BacktestConfig) -> pl.DataFrame:
    """
    유니버스 선택 (개선된 버전)
    """
    print("[유니버스 선택 시작]")
    start_time = time.time()

    initial_count = len(df)
    print(f"  초기 데이터: {initial_count:,} 행")

    if initial_count == 0:
        print("  [경고] 처리할 데이터가 없습니다!")
        return df
    
    # 1. 기본 필터링
    universe = df.filter(
        (pl.col("turnover") >= config.universe.min_turnover) &
        (pl.col("close") >= config.universe.min_price) &
        (~pl.col("close").is_null()) &
        (~pl.col("turnover").is_null())
    )
    
    after_basic_filter = len(universe)
    print(f"  기본 필터링 후: {after_basic_filter:,} 행")

    if after_basic_filter == 0:
        print("  [경고] 기본 필터링 후 데이터가 없습니다!")
        return universe

    # 2. 일별 상위 종목 선택 (개선된 로직)
    if config.universe.top_k_per_day > 0:
        # turnover 기준으로 일별 상위 선택
        universe = (
            universe
            .with_columns(
                rank=pl.col("turnover").rank("dense", descending=True).over("date")
            )
            .filter(pl.col("rank") <= config.universe.top_k_per_day)
            .drop("rank")
        )

        after_top_k = len(universe)
        print(f"  상위 {config.universe.top_k_per_day}개 선택 후: {after_top_k:,} 행")

    universe_time = time.time() - start_time
    print(f"  소요시간: {universe_time:.2f}초")
    
    return universe


def _select_positions(df: pl.DataFrame, config: BacktestConfig) -> pl.DataFrame:
    """
    포지션 선택 (개선된 버전)
    """
    print("[포지션 선택 시작]")
    start_time = time.time()

    # 신호 컬럼 존재 확인
    if config.signal_col not in df.columns:
        print(f"  [오류] 신호 컬럼 '{config.signal_col}'을 찾을 수 없습니다!")
        return df.with_columns(selected=pl.lit(False))

    # 신호 값 검증
    signal_stats = df.select(pl.col(config.signal_col)).describe()
    print("  신호 통계:")
    print(signal_stats)

    # 1. 신호 임계값 필터링
    positions = df.filter(
        (pl.col(config.signal_col) >= config.signal.min_threshold) &
        (~pl.col(config.signal_col).is_null())
    )

    after_threshold = len(positions)
    print(f"  임계값 필터링 후 (>={config.signal.min_threshold}): {after_threshold:,} 행")

    if after_threshold == 0:
        print("  [경고] 임계값 필터링 후 포지션이 없습니다!")
        return df.with_columns(selected=pl.lit(False))
    
    # 2. 일별 상위 신호 선택
    if config.signal.select_top_n > 0:
        positions = (
            positions
            .with_columns(
                signal_rank=pl.col(config.signal_col).rank("dense", descending=True).over("date")
            )
            .filter(pl.col("signal_rank") <= config.signal.select_top_n)
        )

        after_top_n = len(positions)
        print(f"  상위 {config.signal.select_top_n}개 선택 후: {after_top_n:,} 행")

    # 3. Long only 필터링
    if config.signal.long_only:
        positions = positions.filter(pl.col(config.signal_col) > 0)
        after_long_only = len(positions)
        print(f"  롱온리 필터링 후: {after_long_only:,} 행")

    # 선택된 포지션 표시
    selected_positions = positions.select(["date", "ticker"]).with_columns(selected=pl.lit(True))

    # 전체 데이터에 병합
    result = df.join(selected_positions, on=["date", "ticker"], how="left").with_columns(
        selected=pl.col("selected").fill_null(False)
    )

    position_time = time.time() - start_time
    print(f"  소요시간: {position_time:.2f}초")
    
    return result


def _assign_weights(df: pl.DataFrame, config: BacktestConfig) -> pl.DataFrame:
    """
    가중치 할당 (종목당 고정 자산 방식)
    """
    print("[가중치 및 자본 할당 시작]")
    start_time = time.time()
    
    capital_per_position = config.portfolio.capital_per_position
    
    # 선택된 포지션에만 고정 자산 할당
    result = df.with_columns([
        # 종목당 고정 자산 할당
        pl.when(pl.col("selected"))
        .then(pl.lit(capital_per_position))
        .otherwise(0.0)
        .alias("position_capital"),
        
        # 가중치는 1.0 (각 종목이 독립적으로 운용)
        pl.when(pl.col("selected"))
        .then(pl.lit(1.0))
        .otherwise(0.0)
        .alias("weight")
    ])
    
    # 일별 총 투입 자본 계산
    daily_capital = (
        result.filter(pl.col("selected"))
        .group_by("date")
        .agg([
            pl.col("position_capital").sum().alias("total_capital"),
            pl.len().alias("n_positions")
        ])
    )
    
    # 총 투입 자본 정보 추가
    result = result.join(daily_capital, on="date", how="left").with_columns([
        pl.col("total_capital").fill_null(0.0),
        pl.col("n_positions").fill_null(0)
    ])
    
    weight_time = time.time() - start_time
    print(f"  소요시간: {weight_time:.2f}초")

    # 자본 배분 통계 출력
    if len(result.filter(pl.col("selected"))) > 0:
        capital_stats = result.filter(pl.col("selected")).select([
            pl.col("total_capital").mean().alias("avg_daily_capital"),
            pl.col("n_positions").mean().alias("avg_positions"),
            pl.col("total_capital").max().alias("max_daily_capital")
        ]).row(0)

        print("  자본 배분 현황:")
        print(f"    종목당 자본: {capital_per_position:,.0f}원")
        print(f"    일평균 포지션: {capital_stats[1]:.1f}개")
        print(f"    일평균 자본: {capital_stats[0]:,.0f}원")
        print(f"    최대 일자본: {capital_stats[2]:,.0f}원")
    
    return result


def _apply_fees(df: pl.DataFrame, config: BacktestConfig) -> pl.DataFrame:
    """
    수수료 적용 (종목당 고정 자산 방식)
    """
    print("[수수료 적용 및 수익률 계산 시작]")
    start_time = time.time()
    
    # 수수료율 계산
    fee_rate = (config.portfolio.fee_bps + config.portfolio.slippage_bps) / 10000.0
    
    result = df.with_columns([
        # 종목별 총 수익/손실 (자산 * 수익률)
        (pl.col("position_capital") * pl.col(config.label_col)).alias("gross_pnl"),
        
        # 종목별 수수료 (자산 * 수수료율)
        (pl.col("position_capital") * fee_rate).alias("fee_cost"),
        
        # 종목별 순 수익/손실 (총수익 - 수수료)
        (pl.col("position_capital") * pl.col(config.label_col) - pl.col("position_capital") * fee_rate).alias("net_pnl"),
        
        # 종목별 수익률 (순수익 / 투입자본)
        ((pl.col("position_capital") * pl.col(config.label_col) - pl.col("position_capital") * fee_rate) / pl.col("position_capital")).alias("net_return")
    ])
    
    fee_time = time.time() - start_time
    print(f"  소요시간: {fee_time:.2f}초")
    
    return result


def backtest(df: pl.DataFrame, config: BacktestConfig) -> Dict[str, Any]:
    """
    백테스트 실행 (완전히 재구현된 버전)
    """
    print("\n[백테스트 실행 시작]")
    print("=" * 60)

    start_time = time.time()
    performance = {}

    # 데이터 검증
    print("[데이터 검증]")
    print(f"  입력 데이터: {len(df):,} 행 × {len(df.columns)} 열")
    print(f"  날짜 범위: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  고유 종목수: {df['ticker'].n_unique()}개")

    required_cols = [config.label_col, config.signal_col, "date", "ticker", "close", "turnover"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"필요한 컬럼이 누락되었습니다: {missing_cols}")

    # 1. 유니버스 선택
    universe_start = time.time()
    universe_df = _select_universe(df, config)
    performance['universe_time'] = time.time() - universe_start

    if len(universe_df) == 0:
        print("  [오류] 유니버스 데이터가 없습니다!")
        return _empty_backtest_result()
    
    # 2. 포지션 선택
    position_start = time.time()
    position_df = _select_positions(universe_df, config)
    performance['position_time'] = time.time() - position_start
    
    selected_count = position_df.filter(pl.col("selected")).height
    print(f"  총 선택 포지션: {selected_count:,}개")

    if selected_count == 0:
        print("  [오류] 선택된 포지션이 없습니다!")
        return _empty_backtest_result()

    # 3. 가중치 할당
    weight_start = time.time()
    weighted_df = _assign_weights(position_df, config)
    performance['weight_time'] = time.time() - weight_start

    # 4. 수수료 적용
    fee_start = time.time()
    final_df = _apply_fees(weighted_df, config)
    performance['fee_time'] = time.time() - fee_start

        # 5. 일별 집계 (포트폴리오 레벨)
    print("[일별 결과 집계]")
    agg_start = time.time()
    
    daily_results = (
        final_df
        .filter(pl.col("selected"))
        .group_by("date")
           .agg([
            pl.col("gross_pnl").sum().alias("gross_pnl"),
            pl.col("fee_cost").sum().alias("fee_cost"),
            pl.col("net_pnl").sum().alias("net_pnl"),
            pl.col("position_capital").sum().alias("total_capital"),
            pl.len().alias("n_positions")
        ])
           .sort("date")
    )
    
    # 종목별 성과 집계
    ticker_results = (
        final_df
        .filter(pl.col("selected"))
        .group_by("ticker")
        .agg([
            pl.col("net_pnl").sum().alias("total_pnl"),
            pl.col("gross_pnl").sum().alias("total_gross_pnl"),
            pl.col("fee_cost").sum().alias("total_fees"),
            pl.col("position_capital").first().alias("capital_per_trade"),
            pl.len().alias("n_trades"),
            (pl.col("net_pnl") > 0).sum().alias("winning_trades")
        ])
        .with_columns([
            (pl.col("total_pnl") / (pl.col("capital_per_trade") * pl.col("n_trades"))).alias("avg_return_per_trade"),
            (pl.col("winning_trades") / pl.col("n_trades")).alias("win_rate")
        ])
        .sort("total_pnl", descending=True)
    )
    
    performance['agg_time'] = time.time() - agg_start
    
    if len(daily_results) == 0:
        print("  [오류] 생성된 일별 결과가 없습니다!")
        return _empty_backtest_result()

    print(f"  거래일수: {len(daily_results)}일")

    # 6. 누적 성과 계산 (포트폴리오 레벨)
    print("[자본 곡선 계산]")
    equity_start = time.time()
    
    daily_df = daily_results.to_pandas()
    
    # 일별 수익률 계산 (순수익 / 총투입자본)
    daily_df['daily_return'] = daily_df['net_pnl'] / daily_df['total_capital']
    daily_df['daily_return'] = daily_df['daily_return'].fillna(0)
    
    # 누적 수익률 계산
    daily_df['equity'] = (1 + daily_df['daily_return']).cumprod()
    daily_df['drawdown'] = (daily_df['equity'] / daily_df['equity'].expanding().max() - 1)
    
    performance['equity_time'] = time.time() - equity_start
    
    # 7. 성과 지표 계산
    print("[성과 지표 계산]")
    stats_start = time.time()
    
    returns = daily_df['daily_return'].values
    equity_values = daily_df['equity'].values
    
    # 기본 통계
    total_days = len(returns)
    trading_days = np.sum(returns != 0)
    total_return = equity_values[-1] - 1 if len(equity_values) > 0 else 0
    
    # 연간화 지표
    if total_days > 0:
        annual_factor = 252 / total_days
        ret_annual = (1 + total_return) ** annual_factor - 1
        vol_annual = _safe_std(returns) * np.sqrt(252)
        sharpe = ret_annual / vol_annual if vol_annual > 0 else 0
    else:
        ret_annual = vol_annual = sharpe = 0
    
    # 위험 지표
    max_drawdown = daily_df['drawdown'].min() if len(daily_df) > 0 else 0
    win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
    
    # 포지션 통계
    avg_positions = daily_df['n_positions'].mean() if len(daily_df) > 0 else 0
    total_trades = daily_df['n_positions'].sum() if len(daily_df) > 0 else 0
    avg_capital = daily_df['total_capital'].mean() if len(daily_df) > 0 else 0
    max_capital = daily_df['total_capital'].max() if len(daily_df) > 0 else 0
    
    performance['stats_time'] = time.time() - stats_start
    
        # 8. 종목별 성과 분석
    print("[종목별 성과 분석]")
    contrib_start = time.time()
    
    ticker_df = ticker_results.to_pandas()
    
    performance['contrib_time'] = time.time() - contrib_start
    
    # 결과 정리
    summary = {
        'days': total_days,
        'trading_days': trading_days,
        'ret_total': total_return,
        'ret_annual': ret_annual,
        'vol_annual': vol_annual,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_n_positions': avg_positions,
        'total_trades': int(total_trades),
        'avg_capital': avg_capital,
        'max_capital': max_capital,
        'capital_per_position': config.portfolio.capital_per_position
    }
    
    total_time = time.time() - start_time
    print(f"\n[백테스트 완료] 총 {total_time:.2f}초 소요")
    print("=" * 60)
    print(f"[최종 성과]")
    print(f"  최종 자본: {equity_values[-1]:.2f}" if len(equity_values) > 0 else "  최종 자본: 1.00")
    print(f"  연간 수익률: {ret_annual:.2%}")
    print(f"  샤프 비율: {sharpe:.2f}")
    print(f"  최대 낙폭: {max_drawdown:.2%}")
    print(f"  승률: {win_rate:.1%}")

    # 자본 배분 요약
    print(f"\n[자본 배분 요약]")
    print(f"  종목당 자본: {config.portfolio.capital_per_position:,.0f}원")
    print(f"  일평균 자본: {avg_capital:,.0f}원")
    print(f"  최대 일자본: {max_capital:,.0f}원")
    print(f"  총 고유 종목: {len(ticker_df) if len(ticker_df) > 0 else 0}개")

    # 상위 종목별 성과 출력
    if len(ticker_df) > 0:
        print(f"\n[상위 10개 종목 성과]")
        for i, row in ticker_df.head(10).iterrows():
            total_return_pct = (row['total_pnl'] / (row['capital_per_trade'] * row['n_trades'])) * 100
            print(f"  {i+1}. {row['ticker']}: {row['total_pnl']:,.0f}원 ({total_return_pct:+.1f}%) [{row['n_trades']}거래, {row['win_rate']:.1%}승률]")
    
    # 출력 디렉토리 생성
    config.outdir.mkdir(parents=True, exist_ok=True)
    
    return {
        'summary': summary,
        'daily': pl.from_pandas(daily_df),
        'ticker_performance': ticker_df,  # 종목별 성과로 변경
        'performance': performance,
        'config': config
    }


def _empty_backtest_result() -> Dict[str, Any]:
    """빈 백테스트 결과 반환"""
    return {
        'summary': {
            'days': 0,
            'trading_days': 0,
            'ret_total': 0.0,
            'ret_annual': 0.0,
            'vol_annual': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_n_positions': 0.0,
            'total_trades': 0,
            'avg_capital': 0.0,
            'max_capital': 0.0,
            'capital_per_position': 0.0
        },
        'daily': pl.DataFrame(),
        'ticker_performance': pd.DataFrame(),
        'performance': {},
        'config': None
    }


def plot_equity(backtest_result: Dict[str, Any], show: bool = True):
    """자본 곡선 플롯"""
    daily = backtest_result['daily']
    if len(daily) == 0:
        print("[경고] 자본 곡선 플롯을 위한 데이터가 없습니다")
        return

    daily_pd = daily.to_pandas()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_pd['date'], daily_pd['equity'], linewidth=2, color='blue')
    ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.grid(True, alpha=0.3)

    # 출력 디렉토리에 저장
    config = backtest_result.get('config')
    if config and config.outdir:
        plt.savefig(config.outdir / 'equity_curve.png', dpi=300, bbox_inches='tight')
        print(f"[저장] 자본 곡선: {config.outdir / 'equity_curve.png'}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_drawdown(backtest_result: Dict[str, Any], show: bool = True):
    """낙폭 플롯"""
    daily = backtest_result['daily']
    if len(daily) == 0:
        print("[경고] 낙폭 플롯을 위한 데이터가 없습니다")
        return

    daily_pd = daily.to_pandas()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(daily_pd['date'], daily_pd['drawdown'], 0, alpha=0.5, color='red')
    ax.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)

    config = backtest_result.get('config')
    if config and config.outdir:
        plt.savefig(config.outdir / 'drawdown.png', dpi=300, bbox_inches='tight')
        print(f"[저장] 낙폭 차트: {config.outdir / 'drawdown.png'}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_monthly_heatmap(backtest_result: Dict[str, Any], show: bool = True):
    """월별 수익률 히트맵"""
    daily = backtest_result['daily']
    if len(daily) == 0:
        print("[경고] 월별 히트맵을 위한 데이터가 없습니다")
        return

    daily_pd = daily.to_pandas()
    daily_pd['year'] = daily_pd['date'].dt.year
    daily_pd['month'] = daily_pd['date'].dt.month

    monthly = daily_pd.groupby(['year', 'month'])['net_pnl'].apply(lambda x: (1 + x).prod() - 1)
    monthly_pivot = monthly.unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(monthly_pivot * 100, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax)
    ax.set_title('Monthly Returns (%)', fontsize=14, fontweight='bold')

    config = backtest_result.get('config')
    if config and config.outdir:
        plt.savefig(config.outdir / 'monthly_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"[저장] 월별 히트맵: {config.outdir / 'monthly_heatmap.png'}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_rolling_sharpe(backtest_result: Dict[str, Any], window: int = 60, show: bool = True):
    """롤링 샤프 비율 플롯"""
    daily = backtest_result['daily']
    if len(daily) == 0:
        print("[경고] 롤링 샤프 비율을 위한 데이터가 없습니다")
        return

    daily_pd = daily.to_pandas()
    rolling_sharpe = daily_pd['net_pnl'].rolling(window).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_pd['date'], rolling_sharpe, linewidth=2, color='green')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title(f'Rolling Sharpe Ratio ({window} days)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio')
    ax.grid(True, alpha=0.3)

    config = backtest_result.get('config')
    if config and config.outdir:
        plt.savefig(config.outdir / 'rolling_sharpe.png', dpi=300, bbox_inches='tight')
        print(f"[저장] 롤링 샤프 비율: {config.outdir / 'rolling_sharpe.png'}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_contrib_by_ticker(backtest_result: Dict[str, Any], top: int = 20, show: bool = True):
    """종목별 성과 플롯"""
    ticker_perf = backtest_result['ticker_performance']
    if len(ticker_perf) == 0:
        print("[경고] 종목별 성과 플롯을 위한 데이터가 없습니다")
        return

    top_tickers = ticker_perf.head(top)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 1. 총 수익/손실
    bars1 = ax1.barh(range(len(top_tickers)), top_tickers['total_pnl'] / 1000)  # 천원 단위
    ax1.set_yticks(range(len(top_tickers)))
    ax1.set_yticklabels(top_tickers['ticker'])
    ax1.set_title(f'Top {top} Tickers - Total PnL (천원)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Total PnL (천원)')

    # 색상 구분 (양수/음수)
    for i, bar in enumerate(bars1):
        if top_tickers.iloc[i]['total_pnl'] >= 0:
            bar.set_color('green')
        else:
            bar.set_color('red')

    # 2. 평균 거래당 수익률
    bars2 = ax2.barh(range(len(top_tickers)), top_tickers['avg_return_per_trade'] * 100)  # 퍼센트
    ax2.set_yticks(range(len(top_tickers)))
    ax2.set_yticklabels(top_tickers['ticker'])
    ax2.set_title(f'Top {top} Tickers - Avg Return per Trade (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Avg Return per Trade (%)')

    # 색상 구분 (양수/음수)
    for i, bar in enumerate(bars2):
        if top_tickers.iloc[i]['avg_return_per_trade'] >= 0:
            bar.set_color('blue')
        else:
            bar.set_color('orange')

    plt.tight_layout()

    config = backtest_result.get('config')
    if config and config.outdir:
        plt.savefig(config.outdir / 'ticker_performance.png', dpi=300, bbox_inches='tight')
        print(f"[저장] 종목별 성과: {config.outdir / 'ticker_performance.png'}")
    
    if show:
        plt.show()
    else:
        plt.close()


def quick_run(df: pl.DataFrame, signal_col: str, label_col: str = "futret_1", 
              top_k: int = 20, min_threshold: float = 0.1, capital_per_position: float = 1_000_000) -> Dict[str, Any]:
    """빠른 백테스트 실행"""
    config = BacktestConfig(
        label_col=label_col,
        signal_col=signal_col,
        universe=UniverseRule(top_k_per_day=100, min_turnover=1e4, min_price=100),
        signal=SignalRule(select_top_n=top_k, min_threshold=min_threshold, long_only=True),
        execution=ExecutionRule(mode="next_open_to_close"),
        portfolio=PortfolioRule(weighting="equal", fee_bps=8.0, slippage_bps=5.0, capital_per_position=capital_per_position),
        outdir=Path("reports/quick_backtest")
    )
    
    return backtest(df, config)
