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
    # next_open_to_close_tp: 다음날 시가 매수 후 intraday 목표수익 달성 시 청산
    take_profit_pct: float = 0.0  # 예: 0.03 → +3% 도달 시 당일 청산
    # next_open_to_close_nmove: 다음날 시가 대비 절대 변동률이 n% 이상이면 청산(상/하락 모두)
    move_exit_pct: float = 0.0


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
        (pl.col(config.signal_col) > config.signal.min_threshold) &
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
    가중치 할당 (시그널당 고정 자산 방식)
    - 각 시그널(종목)마다 동일한 금액 투자
    - 동시 포지션 수와 무관하게 독립적 투자
    """
    print("[가중치 및 자본 할당 시작]")
    start_time = time.time()

    # 시그널당 투자 금액 (종목 불문하고 동일 금액)
    capital_per_signal = config.portfolio.capital_per_position

    # 선택된 포지션에 시그널당 고정 자산 할당
    result = df.with_columns([
        # 각 시그널마다 동일 금액 투자
        pl.when(pl.col("selected"))
        .then(pl.lit(capital_per_signal))
        .otherwise(0.0)
        .alias("position_capital"),

        # 가중치는 1.0 (각 시그널이 독립적으로 운용)
        pl.when(pl.col("selected"))
        .then(pl.lit(1.0))
        .otherwise(0.0)
        .alias("weight")
    ])

    # 일별 총 투입 자본 계산 (동시 포지션 수 × 시그널당 금액)
    daily_capital = (
        result.filter(pl.col("selected"))
        .group_by("date")
        .agg([
            (pl.col("position_capital").sum()).alias("total_capital"),
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
        print(f"    시그널당 투자금: {capital_per_signal:,.0f}원 (고정)")
        print(f"    일평균 포지션: {capital_stats[1]:.1f}개")
        print(f"    일평균 총 투자금: {capital_stats[0]:,.0f}원")
        print(f"    최대 일 투자금: {capital_stats[2]:,.0f}원")

    return result


def _apply_fees(df: pl.DataFrame, config: BacktestConfig) -> pl.DataFrame:
    """
    수수료 적용 (시그널당 고정 자산 방식)
    """
    print("[수수료 적용 및 수익률 계산 시작]")
    start_time = time.time()
    
    # 수수료율 계산
    fee_rate = (config.portfolio.fee_bps + config.portfolio.slippage_bps) / 10000.0

    # 수익률 산출 모드 결정
    use_tp_mode = (
        hasattr(config, "execution")
        and config.execution is not None
        and config.execution.mode == "next_open_to_close_tp"
        and config.execution.take_profit_pct is not None
        and config.execution.take_profit_pct > 0
    )
    use_nmove_mode = (
        hasattr(config, "execution")
        and config.execution is not None
        and config.execution.mode in ("next_open_to_close_nmove", "next_open_nmove")
        and ((config.execution.move_exit_pct is not None and config.execution.move_exit_pct > 0) or (config.execution.take_profit_pct is not None and config.execution.take_profit_pct > 0))
    )

    if use_tp_mode or use_nmove_mode:
        # 임계값: TP 모드면 take_profit_pct, nmove 모드면 move_exit_pct 우선 (fallback to take_profit_pct)
        th = None
        if use_tp_mode:
            th = float(config.execution.take_profit_pct)
        else:
            th = float(config.execution.move_exit_pct or config.execution.take_profit_pct)

        # 다음날 OHLC 준비 (티커 단위 시프트)
        df = df.sort(["ticker", "date"]).with_columns([
            pl.col("open").shift(-1).over("ticker").alias("open_next"),
            pl.col("high").shift(-1).over("ticker").alias("high_next"),
            pl.col("low").shift(-1).over("ticker").alias("low_next"),
            pl.col("close").shift(-1).over("ticker").alias("close_next"),
        ])

        if use_tp_mode:
            # TP 충족 여부 및 실현 수익률 계산 (롱 기준, 상승만)
            df = df.with_columns([
                (pl.col("open_next") * (1 + th)).alias("tp_price"),
                (pl.col("high_next") >= pl.col("tp_price")).alias("hit_up"),
                pl.lit(False).alias("hit_down"),
                pl.lit(False).alias("both_hit"),
                pl.when(pl.col("selected") & pl.col("open_next").is_not_null())
                .then(
                    pl.when(pl.col("hit_up")).then(pl.lit(th))
                    .otherwise((pl.col("close_next") / pl.col("open_next") - 1.0))
                )
                .otherwise(0.0)
                .alias("realized_return")
            ])
        else:
            # 절대 변동 n% 이상 시 청산 (상/하락 모두)
            df = df.with_columns([
                (pl.col("open_next") * (1 + th)).alias("tp_price"),
                (pl.col("open_next") * (1 - th)).alias("sl_price"),
            ]).with_columns([
                (pl.col("high_next") >= pl.col("tp_price")).alias("hit_up"),
                (pl.col("low_next") <= pl.col("sl_price")).alias("hit_down"),
            ]).with_columns([
                (pl.col("hit_up") & pl.col("hit_down")).alias("both_hit")
            ]).with_columns([
                pl.when(pl.col("selected") & pl.col("open_next").is_not_null())
                .then(
                    pl.when(pl.col("hit_up") & ~pl.col("hit_down")).then(pl.lit(th))
                    .when(pl.col("hit_down") & ~pl.col("hit_up")).then(pl.lit(-th))
                    # 동시 터치 시 보수적으로 손절 우선으로 처리하려면 아래 줄을 바꾸세요
                    .when(pl.col("both_hit")).then(pl.lit(-th))
                    .otherwise((pl.col("close_next") / pl.col("open_next") - 1.0))
                )
                .otherwise(0.0)
                .alias("realized_return")
            ])

        result = df.with_columns([
            # 종목별 총 수익/손실 (자산 * 실현 수익률)
            (pl.col("position_capital") * pl.col("realized_return")).alias("gross_pnl"),

            # 종목별 수수료 (자산 * 수수료율)
            (pl.col("position_capital") * fee_rate).alias("fee_cost"),

            # 종목별 순 수익/손실 (총수익 - 수수료)
            (pl.col("position_capital") * pl.col("realized_return") - pl.col("position_capital") * fee_rate).alias("net_pnl"),

            # 종목별 수익률 (순수익 / 투입자본)
            ((pl.col("position_capital") * pl.col("realized_return") - pl.col("position_capital") * fee_rate) / pl.col("position_capital")).alias("net_return")
        ])
    else:
        # 라벨 기반 (예: futret_1 = next_open_to_close)
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
    try:
        position_df = _select_positions(universe_df, config)
        performance['position_time'] = time.time() - position_start
        
        # selected 컬럼 존재 확인
        if "selected" not in position_df.columns:
            print("  [오류] 'selected' 컬럼이 생성되지 않았습니다!")
            return _empty_backtest_result()
        
        selected_count = position_df.filter(pl.col("selected")).height
        print(f"  총 선택 포지션: {selected_count:,}개")

        if selected_count == 0:
            print("  [오류] 선택된 포지션이 없습니다!")
            return _empty_backtest_result()
            
    except Exception as e:
        print(f"  [오류] 포지션 선택 중 에러: {e}")
        import traceback
        traceback.print_exc()
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

        # 6. 누적 성과 계산 (간단하고 정확한 방식)
    print("[자본 곡선 계산]")
    equity_start = time.time()

    daily_df = daily_results.to_pandas()

    # 누적 성과 계산 (실제 투자 방식: 각 날짜별 독립 수익률)
    if len(daily_df) > 0:
        # 각 날짜별 수익률 = 그날의 net_pnl / 그날의 total_capital
        daily_df['daily_return'] = daily_df['net_pnl'] / daily_df['total_capital']
        daily_df['daily_return'] = daily_df['daily_return'].fillna(0)

        # 자본곡선 = 일별 수익률의 누적 곱 (실제 투자 방식)
        daily_df['equity'] = (1 + daily_df['daily_return']).cumprod()

        # 드로우다운 계산
        daily_df['drawdown'] = (daily_df['equity'] / daily_df['equity'].expanding().max() - 1)
    else:
        daily_df['daily_return'] = 0.0
        daily_df['equity'] = 1.0
        daily_df['drawdown'] = 0.0
    
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
    
    # 연간화 지표 (더 안전한 계산)
    if total_days > 0 and trading_days > 0:
        # 실제 거래일수 기준으로 연간화
        annual_factor = 252 / trading_days
        
        # 연간화 팩터가 너무 크면 제한 (최대 10배)
        annual_factor = min(annual_factor, 10.0)
        
        # 총 수익률이 너무 크거나 작으면 제한
        total_return = max(min(total_return, 10.0), -0.99)
        
        ret_annual = (1 + total_return) ** annual_factor - 1
        vol_annual = _safe_std(returns) * np.sqrt(252)
        sharpe = ret_annual / vol_annual if vol_annual > 0 else 0
        
        # 비정상적인 값 체크
        if abs(ret_annual) > 100:  # 10000% 이상이면 제한
            ret_annual = np.sign(ret_annual) * 1.0  # 100%로 제한
            
        print(f"  연간화 계산: total_days={total_days}, trading_days={trading_days}, factor={annual_factor:.2f}")
        print(f"  총 수익률: {total_return:.4f}, 연간 수익률: {ret_annual:.4f}")
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
        'config': config,
        'processed_data': final_df.select([
            # 기본 키/상태
            "date", "ticker", "selected", "weight", "position_capital",
            # 가격/거래 관련
            "open", "high", "low", "close", "turnover",
            # 다음날 참고 (TP/변동률 모드일 때 생성됨) - 안전하게 처리
            *([col for col in ["open_next", "high_next", "low_next", "close_next", "tp_price", "hit_up", "hit_down", "both_hit", "realized_return"] if col in final_df.columns]),
            # 신호 관련 확률값들 (안전하게 처리)
            *([col for col in ["signal_trigger_prob", "signal_event_prob"] if col in final_df.columns]),
            # 성과/신호
            "gross_pnl", "net_pnl", config.signal_col, config.label_col
        ])  # 처리된 데이터 추가
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
        'config': None,
        'processed_data': pl.DataFrame()  # 빈 processed_data 추가
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

    # 월별 수익률 계산: 일별 PnL / 일별 투자금을 사용
    monthly = daily_pd.groupby(['year', 'month']).apply(
        lambda g: g['net_pnl'].sum() / g['total_capital'].sum() if g['total_capital'].sum() > 0 else 0
    )
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

    # Rolling Sharpe 계산 (디버깅 강화)
    series = daily_pd['daily_return'] if 'daily_return' in daily_pd.columns else (daily_pd['net_pnl'] / daily_pd['total_capital']).fillna(0)

    print(f"[Rolling Sharpe 디버깅] window={window}")
    print(f"  series 통계: mean={series.mean():.8f}, std={series.std():.8f}")
    print(f"  series 범위: {series.min():.8f} ~ {series.max():.8f}")
    print(f"  데이터 길이: {len(series)}")

    # 문제 진단
    if series.std() == 0 or series.std() < 1e-10:
        print("  ⚠️  WARNING: series 표준편차가 0이거나 매우 작습니다")
        if (abs(series) < 1e-10).all():
            print("  ⚠️  WARNING: 모든 값이 0에 가깝습니다")

    if len(series) < window:
        print(f"  ⚠️  WARNING: 데이터 길이({len(series)})가 window({window})보다 작습니다")

    # Sharpe 계산 (개선된 로직)
    rolling_sharpe = series.rolling(window, min_periods=1).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if len(x) > 1 and x.std() > 1e-10 else np.nan
    )

    print(f"  rolling_sharpe 통계: mean={rolling_sharpe.mean():.6f}, std={rolling_sharpe.std():.6f}")
    print(f"  NaN 개수: {rolling_sharpe.isna().sum()}, 유효 값 개수: {rolling_sharpe.notna().sum()}")

    # NaN을 0으로 채우기 (차트 표시를 위해)
    rolling_sharpe = rolling_sharpe.fillna(0)

    print(f"  최종 rolling_sharpe 범위: {rolling_sharpe.min():.6f} ~ {rolling_sharpe.max():.6f}")

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


def plot_signals_per_ticker(backtest_result: Dict[str, Any], show: bool = False, max_tickers: int | None = None):
    """종목별 차트 + 시그널 지점 표시 (디버깅용)
    - close 라인 + 시그널 발생일 점 표시
    - TP 모드인 경우, 다음날 TP 가격 라인 보조 표시
    """
    processed = backtest_result.get('processed_data', pl.DataFrame())
    if len(processed) == 0:
        print("[경고] 시그널 차트를 위한 processed_data가 없습니다")
        return

    df = processed.to_pandas().sort_values(['ticker', 'date'])

    # 출력 디렉토리
    config = backtest_result.get('config')
    outdir = None
    if config and config.outdir:
        outdir = config.outdir / 'charts_by_ticker'
        outdir.mkdir(parents=True, exist_ok=True)

    # 티커 순회 (선택된 신호가 존재하는 티커만)
    tickers = df.loc[df['selected'] == True, 'ticker'].unique().tolist()
    if max_tickers is not None:
        tickers = tickers[:max_tickers]

    is_tp_mode = False
    tp = 0.0
    if config and config.execution and config.execution.mode == 'next_open_to_close_tp' and config.execution.take_profit_pct and config.execution.take_profit_pct > 0:
        is_tp_mode = True
        tp = float(config.execution.take_profit_pct)

    for ticker in tickers:
        sub = df[df['ticker'] == ticker].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(sub['date'], sub['close'], linewidth=1.5, color='black', label='Close')

        sig = sub[sub['selected'] == True]
        if not sig.empty:
            # 수익 방향에 따라 색상 구분
            if 'realized_return' in sig.columns and not sig['realized_return'].isna().all():
                pos = sig[sig['realized_return'] >= 0]
                neg = sig[sig['realized_return'] < 0]
                if not pos.empty:
                    ax.scatter(pos['date'], pos['close'], color='green', s=28, label='Signal (↑ or TP)')
                    # direction_proba와 event_proba 값 표시
                    for _, row in pos.iterrows():
                        try:
                            dir_prob = float(row.get('signal_trigger_prob', 0))
                            ev_prob = float(row.get('signal_event_prob', 0))
                            ax.annotate(f"D:{dir_prob:.2f}\nE:{ev_prob:.2f}",
                                      (row['date'], row['close']),
                                      xytext=(5, 5), textcoords='offset points',
                                      fontsize=7, color='green')
                        except (ValueError, TypeError):
                            pass  # 값이 유효하지 않으면 표시하지 않음
                if not neg.empty:
                    ax.scatter(neg['date'], neg['close'], color='orange', s=28, label='Signal (↓)')
                    # direction_proba와 event_proba 값 표시
                    for _, row in neg.iterrows():
                        try:
                            dir_prob = float(row.get('signal_trigger_prob', 0))
                            ev_prob = float(row.get('signal_event_prob', 0))
                            ax.annotate(f"D:{dir_prob:.2f}\nE:{ev_prob:.2f}",
                                      (row['date'], row['close']),
                                      xytext=(5, -15), textcoords='offset points',
                                      fontsize=7, color='orange')
                        except (ValueError, TypeError):
                            pass  # 값이 유효하지 않으면 표시하지 않음
            else:
                ax.scatter(sig['date'], sig['close'], color='red', s=25, label='Signal Day')
                # direction_proba와 event_proba 값 표시
                for _, row in sig.iterrows():
                    try:
                        dir_prob = float(row.get('signal_trigger_prob', 0))
                        ev_prob = float(row.get('signal_event_prob', 0))
                        ax.annotate(f"D:{dir_prob:.2f}\nE:{ev_prob:.2f}",
                                  (row['date'], row['close']),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=7, color='red')
                    except (ValueError, TypeError):
                        pass  # 값이 유효하지 않으면 표시하지 않음

        if is_tp_mode and 'open_next' in sub.columns and 'tp_price' in sub.columns:
            # 다음날 TP 레벨 보조 표기 (선택적인 참고용)
            next_day = sub.dropna(subset=['open_next', 'tp_price']).copy()
            if not next_day.empty:
                # 신호 발생일의 다음날에 TP 라인 표시를 위해 x축을 다음날로 쉬프트한 보조 시리즈 생성
                # 단순화를 위해 다음날 date를 계산하지 않고, 참고용 라인만 표시
                pass  # 세부 표시는 생략 (복잡도 대비 효용 낮음)

        ax.set_title(f"{ticker} - Price with Signals", fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if outdir is not None:
            fp = outdir / f"{ticker}.png"
            plt.savefig(fp, dpi=200, bbox_inches='tight')
            print(f"[저장] 시그널 차트: {fp}")

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
