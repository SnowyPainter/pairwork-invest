#!/usr/bin/env python3
"""
Pipeline helper for scoring fresh Yahoo Finance data with the M002 full architecture.

Usage (example):

    python -m apps.m002_yfinance_predict \\
        --tickers AAPL MSFT \\
        --start 2020-01-01 \\
        --end 2024-01-01 \\
        --model-path models/saved/m002_full_architecture_US_2010_2011_2012_2013_2014_2015_2016_2017_2018.pkl \\
        --save-csv reports/m002_yf_scores.csv

The script will:
  1. Download OHLCV bars via yfinance
  2. Convert them to the silver schema expected by the feature factory
  3. Build the M002 feature set
  4. Apply saved normalization statistics when available
  5. Feed the features into the pre-trained M002 full architecture
  6. Emit policy signals and optional artifacts
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import glob

import matplotlib
# Try different backends for GUI support
backends = ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'Agg']
for backend in backends:
    try:
        matplotlib.use(backend)
        break
    except ImportError:
        continue
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

import pandas as pd
import polars as pl

try:  # yfinance is listed in requirements, keep a clear error if missing
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    raise SystemExit("yfinance is required. Install it with `pip install yfinance`.") from exc

try:
    import joblib
except ImportError:  # pragma: no cover - instruct the caller to add dependency
    joblib = None

from features.feature_sets import add_feature_set
from models.M002_MultiTask import M002TrainingConfig
from models.M002_FullArchitecture import (
    RegimeConfig,
    FullArchitectureConfig,
    PolicyConfig,
    STATE_PROB_COLS,
    M002FullArchitecture
)
from models.M002_RegimeClassifier import DEFAULT_REGIME_FEATURES


def find_latest_m002_model(saved_dir: Path = Path("models/saved")) -> Optional[Path]:
    """Find the latest M002 full architecture model in the saved directory."""
    if not saved_dir.exists():
        return None

    pattern = str(saved_dir / "m002_full_architecture*.pkl")
    model_files = glob.glob(pattern)

    if not model_files:
        return None

    # Return the most recently modified file
    return Path(max(model_files, key=lambda f: Path(f).stat().st_mtime))

# Base schema columns that should never be normalized
BASE_COLUMNS = {
    "date",
    "ticker",
    "market",
    "exchange",
    "currency",
    "year",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "turnover",
}


@dataclass(frozen=True)
class DownloadConfig:
    tickers: Sequence[str]
    start: str
    end: Optional[str]
    interval: str = "1d"
    auto_adjust: bool = True
    progress: bool = False
    market: str = "US"
    exchange: Optional[str] = None
    currency: str = "USD"


def _normalize_tickers(raw: Iterable[str]) -> List[str]:
    seen: List[str] = []
    for ticker in raw:
        if not ticker:
            continue
        normalized = ticker.strip().upper()
        if normalized and normalized not in seen:
            seen.append(normalized)
    return seen


def download_prices(cfg: DownloadConfig) -> pl.DataFrame:
    frames: List[pl.DataFrame] = []
    for ticker in cfg.tickers:
        logging.info("Downloading %s from Yahoo Finance ...", ticker)
        data = yf.download(
            ticker,
            start=cfg.start,
            end=cfg.end,
            interval=cfg.interval,
            auto_adjust=cfg.auto_adjust,
            progress=cfg.progress,
            threads=False,
        )
        if data.empty:
            logging.warning("No data returned for %s.", ticker)
            continue

        frame = data.reset_index()

        # Handle MultiIndex columns from yfinance
        if isinstance(frame.columns, pd.MultiIndex):
            # Flatten MultiIndex columns
            frame.columns = [col[0].lower().replace(" ", "_") if isinstance(col, tuple) else str(col).lower().replace(" ", "_")
                           for col in frame.columns]
        else:
            frame.columns = [str(col).lower().replace(" ", "_") for col in frame.columns]

        if "adj_close" not in frame.columns:
            frame["adj_close"] = frame.get("close")

        pl_frame = pl.from_pandas(frame)
        pl_frame = pl_frame.with_columns([
            pl.col("date").cast(pl.Date),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("adj_close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64).fill_null(0.0),
        ])
        pl_frame = pl_frame.with_columns([
            (pl.col("close") * pl.col("volume")).alias("turnover"),
            pl.lit(ticker).alias("ticker"),
            pl.lit(cfg.market).alias("market"),
            pl.lit(cfg.exchange.upper() if cfg.exchange else None).alias("exchange"),
            pl.lit(cfg.currency).alias("currency"),
            pl.col("date").dt.year().alias("year"),
        ])
        frames.append(pl_frame.select(list(BASE_COLUMNS)))

    if not frames:
        raise RuntimeError("No price data could be downloaded for the requested tickers.")

    ohlcv = pl.concat(frames, how="vertical")
    return (
        ohlcv
        .sort(["ticker", "date"])
        .unique(subset=["ticker", "date"], keep="last")
    )


def build_feature_frame(ohlcv: pl.DataFrame, feature_set: str = "m002") -> pl.DataFrame:
    lf = (
        ohlcv
        .lazy()
        .sort(["ticker", "date"])
    )
    lf = add_feature_set(lf, feature_set=feature_set)
    return lf.collect()


def load_m002_model(path: Path) -> M002FullArchitecture:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    if joblib is None:
        raise RuntimeError(
            "joblib is required to load the saved model. Install it with `pip install joblib`."
        )

    model = joblib.load(path)
    if not isinstance(model, M002FullArchitecture):
        raise TypeError(f"Unexpected object loaded from {path}: {type(model)!r}")

    # Ensure policy config exposes risk_aversion even if older artifacts omit it
    fallback_lambda = getattr(model.config.multitask, "risk_aversion", 0.5)
    if not hasattr(model.config.policy, "risk_aversion"):  # older pickles lacked the field
        setattr(model.config.policy, "risk_aversion", fallback_lambda)
    if not hasattr(model.policy_cfg, "risk_aversion"):
        setattr(model.policy_cfg, "risk_aversion", fallback_lambda)

    return model


def load_normalization_stats(path: Optional[Path]) -> Optional[Dict[str, Dict[str, float]]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Normalization stats JSON must contain a top-level object.")
    return {
        str(col): {
            "mean": float(stats.get("mean", 0.0)),
            "std": float(stats.get("std", 1.0)) if abs(stats.get("std", 0.0)) > 1e-12 else 1.0,
        }
        for col, stats in payload.items()
        if isinstance(stats, dict)
    }


def maybe_extract_model_stats(model: M002FullArchitecture) -> Optional[Dict[str, Dict[str, float]]]:
    for attr in ("normalization_stats", "normalization_stats_used"):
        stats = getattr(model, attr, None)
        if isinstance(stats, dict) and stats:
            return stats

    regime_stats = getattr(model.regime, "normalization_stats", None)
    if isinstance(regime_stats, dict) and regime_stats:
        return regime_stats
    return None


def apply_normalization(df: pl.DataFrame, stats: Dict[str, Dict[str, float]]) -> pl.DataFrame:
    exprs = []
    for col, spec in stats.items():
        if col in BASE_COLUMNS or col not in df.columns:
            continue
        mean = spec.get("mean", 0.0)
        std = spec.get("std", 1.0) or 1.0
        exprs.append(((pl.col(col) - mean) / std).alias(col))
    if exprs:
        df = df.with_columns(exprs)
    return df


def filter_required_rows(df: pl.DataFrame, needed: Sequence[str]) -> pl.DataFrame:
    missing_cols = [col for col in needed if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Feature frame missing required columns: {missing_cols}")
    if not needed:
        return df
    return df.drop_nulls(needed)


def predict(model: M002FullArchitecture, feature_df: pl.DataFrame) -> pl.DataFrame:
    preds_pd = model.predict(feature_df)
    preds = pl.from_pandas(preds_pd)
    preds = preds.with_columns([
        pl.col("date").cast(pl.Date),
        pl.col("ticker").cast(pl.Utf8),
    ])
    return preds


def merge_predictions(
    base: pl.DataFrame,
    features: pl.DataFrame,
    predictions: pl.DataFrame,
) -> pl.DataFrame:
    join_cols = ["ticker", "date"]
    combined = (
        base.join(features, on=join_cols, how="left")
        .join(predictions, on=join_cols, how="left")
    )
    return combined


def create_trading_chart(merged_df: pl.DataFrame, ticker: str, save_path: Optional[Path] = None) -> None:
    """Create a trading chart with price, positions, and performance metrics."""
    # Convert to pandas for easier plotting
    df = merged_df.to_pandas()
    df['date'] = pd.to_datetime(df['date'])

    # Create figure with subplots - professional styling
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[3, 1])
    fig.suptitle(f'📈 {ticker} Trading Analysis - M002 Model', fontsize=18, fontweight='bold', y=0.95)

    # Set background colors
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('#f8f9fa')
    ax2.set_facecolor('#f8f9fa')

    # Main price chart
    ax1.plot(df['date'], df['close'], linewidth=2, label='Close Price', color='navy')

    # Add moving averages for reference
    if len(df) > 20:
        df['MA20'] = df['close'].rolling(window=20).mean()
        ax1.plot(df['date'], df['MA20'], linewidth=1, label='20-day MA', color='orange', linestyle='--')

    if len(df) > 50:
        df['MA50'] = df['close'].rolling(window=50).mean()
        ax1.plot(df['date'], df['MA50'], linewidth=1, label='50-day MA', color='purple', linestyle='--')

    # Add position signals with background highlighting
    if 'action' in df.columns:
        # Create background colors for different positions
        colors = {'LONG': 'lightgreen', 'SHORT': 'lightcoral', 'FLAT': 'lightgray'}

        # Group consecutive signals for background highlighting
        df['action_group'] = (df['action'] != df['action'].shift()).cumsum()

        for action in ['LONG', 'SHORT']:  # Only highlight LONG and SHORT periods
            action_data = df[df['action'] == action]
            if action_data.empty:
                continue

            # Background highlighting for position periods
            for _, group in action_data.groupby('action_group'):
                if len(group) > 1:  # Only highlight if more than 1 consecutive day
                    start_date = group['date'].min()
                    end_date = group['date'].max()
                    ax1.axvspan(start_date, end_date, alpha=0.2, color=colors[action],
                               label=f'{action} Period' if f'{action} Period' not in [l.get_label() for l in ax1.patches] else "")

        # Signal markers on price chart
        long_signals = df[df['action'] == 'LONG']
        if not long_signals.empty:
            ax1.scatter(long_signals['date'], long_signals['close'],
                       marker='^', color='green', s=120, label='LONG Signal',
                       edgecolors='black', linewidth=1, zorder=6)

        short_signals = df[df['action'] == 'SHORT']
        if not short_signals.empty:
            ax1.scatter(short_signals['date'], short_signals['close'],
                       marker='v', color='red', s=120, label='SHORT Signal',
                       edgecolors='black', linewidth=1, zorder=6)


    ax1.set_ylabel('Price ($)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, fontsize=10)

    # Remove x-axis labels from top chart
    ax1.set_xlabel('')

    # Performance metrics subplot
    if 'policy_score' in df.columns:
        ax2.plot(df['date'], df['policy_score'], linewidth=2, label='Policy Score', color='purple')

        # Add threshold lines
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Neutral Threshold')
        ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Strong Signal')

    ax2.set_ylabel('Policy Score', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, fontsize=10)

    # Calculate and display performance metrics
    if len(df) > 1:
        initial_price = df['close'].iloc[0]
        final_price = df['close'].iloc[-1]
        total_return = ((final_price - initial_price) / initial_price) * 100

        # Enhanced signal analysis
        if 'action' in df.columns:
            long_count = (df['action'] == 'LONG').sum()
            short_count = (df['action'] == 'SHORT').sum()
            flat_count = (df['action'] == 'FLAT').sum()

            # Calculate position changes for trade analysis
            df['position_change'] = df['action'] != df['action'].shift()
            trade_signals = df[df['position_change'] & (df['action'] != 'FLAT')]
            total_trades = len(trade_signals)

            # Calculate win rate (simplified: assuming LONG beats buy&hold in bull periods)
            if total_trades > 0:
                # Simple win rate based on position direction vs price movement
                wins = 0
                for i in range(1, len(df)):
                    if df['action'].iloc[i-1] == 'LONG' and df['close'].iloc[i] > df['close'].iloc[i-1]:
                        wins += 1
                    elif df['action'].iloc[i-1] == 'SHORT' and df['close'].iloc[i] < df['close'].iloc[i-1]:
                        wins += 1
                win_rate = (wins / len(df)) * 100 if len(df) > 0 else 0
            else:
                win_rate = 0.0
        else:
            long_count = short_count = flat_count = total_trades = 0
            win_rate = 0.0

        # Add metrics text
        metrics_text = f"""
        📊 Performance Metrics:
        Initial Price: ${initial_price:.2f}
        Final Price: ${final_price:.2f}
        Total Return: {total_return:+.2f}%

        📈 Trading Signals:
        LONG: {long_count}  SHORT: {short_count}  FLAT: {flat_count}
        Total Trades: {total_trades}
        Signal Accuracy: {win_rate:.1f}%
        Data Points: {len(df)}
        """

        fig.text(0.02, 0.98, metrics_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Chart saved to {save_path}")
    else:
        try:
            plt.show(block=True)
            logging.info("Chart displayed. Close the window to continue.")
        except Exception as e:
            logging.warning(f"Could not display chart interactively: {e}")
            logging.info("Try using --save-chart option instead to save chart as image file.")

    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the M002 full architecture on fresh yfinance data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols (space separated).")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD). If not provided, defaults to 1 year ago from today.")
    parser.add_argument("--end", help="End date (YYYY-MM-DD). If not provided, defaults to today.")
    parser.add_argument("--interval", default="1d", help="Download interval supported by yfinance.")
    parser.add_argument("--market", default="US", help="Market code stored in the silver schema.")
    parser.add_argument("--exchange", help="Optional exchange hint stored alongside the bars.")
    parser.add_argument("--currency", default="USD", help="Currency code for turnover calculation.")
    parser.add_argument("--no-auto-adjust", action="store_true", help="Disable yfinance price adjustments.")
    parser.add_argument(
        "--feature-set",
        default="m002",
        choices=["v1", "v2", "v3", "m002"],
        help="Feature factory preset to use.",
    )
    parser.add_argument(
        "--model-path",
        help="Path to a pickled M002FullArchitecture artifact. If not provided, automatically finds the latest M002 model.",
    )
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        help="Optional JSON file with per-feature mean/std used during training.",
    )
    parser.add_argument("--save-csv", type=Path, help="Optional CSV output path for merged predictions.")
    parser.add_argument("--save-parquet", type=Path, help="Optional Parquet output path for merged predictions.")
    parser.add_argument("--save-chart", type=Path, help="Optional chart output path (PNG format).")
    parser.add_argument("--show-chart", action="store_true", help="Show interactive chart in window.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level (e.g. INFO, DEBUG).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(message)s")

    # Set default dates if not provided
    today = datetime.now().date()
    if args.start is None:
        args.start = (today - timedelta(days=365)).strftime("%Y-%m-%d")
        logging.info("Using default start date: %s (1 year ago)", args.start)
    if args.end is None:
        args.end = today.strftime("%Y-%m-%d")
        logging.info("Using default end date: %s (today)", args.end)

    tickers = _normalize_tickers(args.tickers)
    if not tickers:
        raise SystemExit("No valid tickers provided.")

    # Auto-find model path if not provided
    if args.model_path is None:
        model_path = find_latest_m002_model()
        if model_path is None:
            raise SystemExit("No M002 model found in models/saved/ directory. Please specify --model-path explicitly.")
        logging.info("Auto-selected model: %s", model_path)
    else:
        model_path = Path(args.model_path)
    logging.info("Loading model from %s", model_path)
    model = load_m002_model(model_path)
    logging.info("Model ready with %d head features.", len(model.head_features))

    download_cfg = DownloadConfig(
        tickers=tickers,
        start=args.start,
        end=args.end,
        interval=args.interval,
        auto_adjust=not args.no_auto_adjust,
        market=args.market,
        exchange=args.exchange,
        currency=args.currency,
    )
    ohlcv = download_prices(download_cfg)
    logging.info("Downloaded %d rows across %d tickers.", ohlcv.height, len(tickers))

    feature_df = build_feature_frame(ohlcv, feature_set=args.feature_set)
    logging.info("Feature frame built with %d columns.", len(feature_df.columns))

    # Add state probabilities from regime classifier
    state_probs = model.regime.predict_probabilities(
        feature_df.select(["ticker", "date", *DEFAULT_REGIME_FEATURES])
    )
    feature_df = feature_df.join(state_probs, on=["ticker", "date"], how="left")

    # Fill null values in state probability columns
    for col in STATE_PROB_COLS:
        if col in feature_df.columns:
            feature_df = feature_df.with_columns(pl.col(col).fill_null(0.0))

    stats = load_normalization_stats(args.normalization_stats)
    if stats is None and getattr(model.config, "normalize_features", True):
        stats = maybe_extract_model_stats(model)
        if stats:
            logging.info("Using normalization stats embedded in the trained model.")
        else:
            logging.warning(
                "No normalization statistics provided; proceeding without Z-score scaling. "
                "Predictions may be skewed if the model was trained on normalized features."
            )

    if stats:
        logging.info("Applying Z-score normalization (mean=0, std=1) using provided stats for %d features", len(stats))
        feature_df = apply_normalization(feature_df, stats)
    else:
        logging.info("No normalization stats provided - using raw feature values")

    required = sorted(set(model.head_features) | set(DEFAULT_REGIME_FEATURES))
    feature_df = filter_required_rows(feature_df, required)
    logging.info("Feature frame filtered down to %d rows after dropping nulls.", feature_df.height)

    predictions = predict(model, feature_df)
    logging.info("Generated %d prediction rows.", predictions.height)

    merged = merge_predictions(ohlcv, feature_df.select(["ticker", "date"]), predictions)

    # Sort by ticker and date for consistent ordering
    merged = merged.sort(["ticker", "date"])

    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.write_csv(args.save_csv)
        logging.info("Saved CSV predictions to %s", args.save_csv)

    if args.save_parquet:
        args.save_parquet.parent.mkdir(parents=True, exist_ok=True)
        merged.write_parquet(args.save_parquet)
        logging.info("Saved Parquet predictions to %s", args.save_parquet)

    # Display full results
    logging.info("Full Results:")
    logging.info("=" * 100)
    result_cols = ["ticker", "date", "close", "policy_score", "action", "pred_rebound_prob"]
    available_cols = [col for col in result_cols if col in merged.columns]
    if available_cols:
        # Convert to pandas for better display
        result_df = merged.select(available_cols).to_pandas()
        logging.info("\n%s", result_df.to_string(index=False))

    # Create charts for each ticker
    if args.save_chart or args.show_chart:
        unique_tickers = merged.select("ticker").unique().to_series().to_list()
        for ticker in unique_tickers:
            ticker_data = merged.filter(pl.col("ticker") == ticker)
            if args.save_chart:
                chart_path = args.save_chart.parent / f"{args.save_chart.stem}_{ticker}{args.save_chart.suffix}"
                create_trading_chart(ticker_data, ticker, chart_path)
            elif args.show_chart:
                create_trading_chart(ticker_data, ticker, None)


if __name__ == "__main__":
    main()
