from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_TICKER = "AAPL"
DEFAULT_LOOKBACK_DAYS = 252  # ~1 trading year
DEFAULT_OUTPUT = Path("reports/paper_plots/figures/backtest_equity.png")


def _download_prices(
    ticker: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Please install `yfinance` to fetch Yahoo Finance data.") from exc

    end = datetime.utcnow()
    start = end - timedelta(days=int(lookback_days * 1.8))
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker}.")

    # Handle MultiIndex columns from newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten MultiIndex columns
        df.columns = df.columns.get_level_values(0)

    df = (
        df.reset_index()
        .rename(columns=str.lower)
        .sort_values("date")
        .tail(lookback_days)
        .reset_index(drop=True)
    )
    return df


def _run_m002_inspired_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Run backtest using M002-inspired strategy (simplified version)."""
    df = df.copy()
    df["ret"] = df["close"].pct_change()

    # Calculate technical indicators inspired by M002
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Price momentum (slope)
    df['price_slope'] = df['close'].pct_change(3)

    # Curvature (second derivative approximation)
    df['curvature'] = df['price_slope'].diff()

    # Volume analysis
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    # ATR for volatility
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()

    # M002-inspired trading logic
    # Long: RSI oversold + positive curvature + high volume
    long_condition = (
        (df['rsi'] < 35) &  # Oversold
        (df['curvature'] > df['curvature'].quantile(0.6)) &  # Positive curvature
        (df['volume_ratio'] > 1.2)  # High volume
    )

    # Short: RSI overbought + negative curvature + high volume
    short_condition = (
        (df['rsi'] > 65) &  # Overbought
        (df['curvature'] < df['curvature'].quantile(0.4)) &  # Negative curvature
        (df['volume_ratio'] > 1.2)  # High volume
    )

    # Position sizing based on ATR (volatility-adjusted)
    df['volatility'] = df['atr'] / df['close']
    base_position = 0.5  # Base position size
    df['position_size'] = base_position / (1 + df['volatility'])  # Reduce size in high vol

    # Generate signals
    df['signal'] = 0.0
    df.loc[long_condition, 'signal'] = 1.0
    df.loc[short_condition, 'signal'] = -1.0

    # Fill missing values
    df = df.fillna(0.0)

    # Calculate strategy returns with position sizing
    df["strategy_ret"] = df["signal"].shift(1).fillna(0.0) * df["position_size"].shift(1).fillna(0.0) * df["ret"]

    # Calculate equity curve
    df["equity"] = (1.0 + df["strategy_ret"]).cumprod()
    df["equity"] = df["equity"].ffill().fillna(1.0)
    df["equity_peak"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] / df["equity_peak"] - 1.0

    return df


def _run_simple_backtest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    fast = df["close"].rolling(20).mean()
    slow = df["close"].rolling(60).mean()
    df["signal"] = np.where(fast > slow, 1.0, -1.0)
    df["signal"] = df["signal"].fillna(0.0)
    df["strategy_ret"] = df["signal"].shift(1).fillna(0.0) * df["ret"]
    df["equity"] = (1.0 + df["strategy_ret"]).cumprod()
    df["equity"] = df["equity"].ffill().fillna(1.0)
    df["equity_peak"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] / df["equity_peak"] - 1.0
    return df


def _compute_metrics(df: pd.DataFrame) -> dict:
    n_days = df["strategy_ret"].dropna().shape[0]
    if n_days == 0:
        raise ValueError("Not enough data to compute metrics.")

    total_return = df["equity"].iloc[-1] - 1.0
    ann_factor = 252 / n_days
    ann_return = (1.0 + total_return) ** ann_factor - 1.0
    ann_vol = df["strategy_ret"].std(ddof=0) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    max_dd = df["drawdown"].min()

    trades = df["signal"].diff().abs() >= 2.0
    num_trades = int(trades.sum())

    trade_returns = df.loc[df["strategy_ret"] != 0, "strategy_ret"]
    win_rate = (trade_returns > 0).mean() if not trade_returns.empty else np.nan

    return {
        "Total Return": f"{total_return * 100:+.1f}%",
        "CAGR": f"{ann_return * 100:+.1f}%",
        "Sharpe": f"{sharpe:.2f}" if not np.isnan(sharpe) else "n/a",
        "Max DD": f"{max_dd * 100:.1f}%",
        "# Trades": str(num_trades),
        "Win Rate": f"{win_rate * 100:.1f}%" if not np.isnan(win_rate) else "n/a",
    }


def plot_backtest_from_yahoo(
    ticker: str = DEFAULT_TICKER,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    output: Path = DEFAULT_OUTPUT,
    show: bool = False,
    use_m002_model: bool = False,
) -> Path:
    prices = _download_prices(ticker, lookback_days)

    if use_m002_model:
        # Run M002-inspired strategy (simplified version)
        bt = _run_m002_inspired_backtest(prices)
        strategy_name = "M002-Inspired Strategy"
    else:
        # Run simple SMA crossover backtest
        bt = _run_simple_backtest(prices)
        strategy_name = "SMA(20/60) Crossover"

    metrics = _compute_metrics(bt)

    dates = bt["date"]
    normalized = (bt["equity"] - 1.0) * 100.0
    drawdowns = bt["drawdown"] * 100.0

    fig, (ax_eq, ax_dd) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax_eq.plot(dates, normalized, color="#005f73", linewidth=2)
    ax_eq.set_ylabel("Return since start (%)", fontsize=11)
    ax_eq.set_title(
        f"{ticker} · {strategy_name}",
        fontsize=15,
        fontweight="bold",
    )
    ax_eq.grid(alpha=0.25)

    text = "\n".join(f"{k}: {v}" for k, v in metrics.items())
    ax_eq.text(
        0.99,
        0.02,
        text,
        transform=ax_eq.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.4"),
    )

    ax_dd.fill_between(dates, drawdowns, 0, color="#ee6c4d", alpha=0.5)
    ax_dd.set_ylabel("Drawdown (%)", fontsize=11)
    ax_dd.set_xlabel("Date", fontsize=11)
    ax_dd.grid(alpha=0.25)

    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax_eq.xaxis.set_major_locator(locator)
    ax_eq.xaxis.set_major_formatter(formatter)
    ax_dd.xaxis.set_major_locator(locator)
    ax_dd.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest visualization from Yahoo Finance data.")
    parser.add_argument("--ticker", type=str, default=DEFAULT_TICKER, help="Ticker symbol (default: AAPL).")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="Number of recent trading days to use (default ~1Y).",
    )
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output PNG path.")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively.")
    parser.add_argument("--use-m002", action="store_true", help="Use M002-inspired strategy instead of simple SMA strategy.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plot_backtest_from_yahoo(
        ticker=args.ticker,
        lookback_days=args.lookback_days,
        output=Path(args.output),
        show=args.show,
        use_m002_model=args.use_m002,
    )


if __name__ == "__main__":
    main()
