#!/usr/bin/env python3
"""
Pipeline helper for scoring fresh Yahoo Finance data with the M002 full architecture.

Usage (example):

    python -m apps.m002_yfinance_predict \\
        --tickers AAPL MSFT \\
        --start 2020-01-01 \\
        --end 2024-01-01 \\
        --model-path models/saved/m002_full_architecture_US_2000-2018.pkl \\
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

# matplotlib is imported by backtesting.py when needed

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

try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover

    class M002BacktestStrategy(Strategy):
        """
        M002 model signals based backtesting strategy using backtesting.py

        This strategy follows the action signals from the M002 model:
        - LONG: Enter long position
        - SHORT: Enter short position (close long if exists, then short)
        - FLAT: Close any open position
        """

        def init(self):
            """Initialize strategy - called once before backtest starts"""
            pass  # Signals are accessed directly from self.data in next()

        def next(self):
            """
            Main strategy logic - called for each candle/bar

            backtesting.py provides self.data as a pandas Series-like object
            with OHLCV + custom columns (including our 'action' column)
            """
            # Get the latest action signal
            current_signal = self.data['action'][-1]

            if current_signal == 'LONG':
                # Enter long position if not already in one
                if not self.position:
                    self.buy()

            elif current_signal == 'SHORT':
                # Enter short position
                if not self.position:
                    #self.sell()
                    pass
                elif self.position.is_long:
                    # Close long position first, then enter short
                    self.position.close()
                    #self.sell()

            elif current_signal == 'FLAT':
                # Close any open position
                if self.position:
                    #self.position.close()
                    pass

except ImportError:
    Backtest = None
    Strategy = None
    M002BacktestStrategy = None


def run_backtest(df: pl.DataFrame) -> tuple[Dict[str, float], any]:
    """
    Run backtest using backtesting.py and return metrics and bt object.
    Falls back to basic calculations if backtesting.py fails.
    """
    df_pd = df.to_pandas()
    df_pd['date'] = pd.to_datetime(df_pd['date'])

    # Basic metrics (always available)
    metrics = {}
    if len(df_pd) > 1:
        initial_price = df_pd['close'].iloc[0]
        final_price = df_pd['close'].iloc[-1]
        total_return = ((final_price - initial_price) / initial_price) * 100

        metrics.update({
            'initial_price': float(initial_price),
            'final_price': float(final_price),
            'total_return_pct': float(total_return),
            'data_points': len(df_pd),
            'start_date': df_pd['date'].min().strftime('%Y-%m-%d'),
            'end_date': df_pd['date'].max().strftime('%Y-%m-%d'),
        })

    # Signal analysis
    if 'action' in df_pd.columns:
        long_count = (df_pd['action'] == 'LONG').sum()
        short_count = (df_pd['action'] == 'SHORT').sum()
        flat_count = (df_pd['action'] == 'FLAT').sum()
        metrics.update({
            'long_signals': int(long_count),
            'short_signals': int(short_count),
            'flat_signals': int(flat_count),
        })

    bt = None
    # Use backtesting.py for comprehensive metrics
    if Backtest is not None and Strategy is not None and M002BacktestStrategy is not None:
        try:
            bt_data = df_pd.set_index('date')[['open', 'high', 'low', 'close', 'volume']].copy()
            # Rename columns to match backtesting.py requirements
            bt_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            # Add action column for strategy access
            bt_data['action'] = df_pd.set_index('date')['action']
            bt = Backtest(bt_data, M002BacktestStrategy, cash=10000, commission=.002)
            result = bt.run()

            # Extract key metrics
            backtest_metrics = {
                'return_pct': result.get('Return [%]', 0),
                'buy_and_hold_return_pct': result.get('Buy & Hold Return [%]', 0),
                'sharpe_ratio': result.get('Sharpe Ratio', 0),
                'max_drawdown_pct': result.get('Max. Drawdown [%]', 0),
                'win_rate_pct': result.get('Win Rate [%]', 0),
                'total_trades': result.get('# Trades', 0),
                'equity_final': result.get('Equity Final [$]', 10000),
                'equity_peak': result.get('Equity Peak [$]', 10000),
            }

            # Convert types
            for key, value in backtest_metrics.items():
                if hasattr(value, 'item'):
                    backtest_metrics[key] = value.item()

            metrics.update(backtest_metrics)
            logging.info("Successfully calculated backtest metrics using backtesting.py")

        except Exception as e:
            logging.warning(f"backtesting.py failed: {e}")
            logging.info("Falling back to basic metrics")

    return metrics, bt


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




def generate_backtest_filename(ticker: str, start_date: str, end_date: str, reports_dir: Path = Path("reports")) -> Path:
    """Generate a filename for backtest results including ticker and date range."""
    start_year = start_date.split('-')[0]
    end_year = end_date.split('-')[0]
    filename = f"{ticker}_{start_year}_{end_year}_backtest_results.json"
    return reports_dir / filename


def save_backtest_results(ticker: str, metrics: Dict[str, float], start_date: str, end_date: str, model_path: Path) -> None:
    """Save backtest performance metrics to a JSON file in reports/ directory."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Convert numpy/pandas types to Python native types for JSON serialization
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {key: convert_to_python_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_python_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj

    backtest_data = {
        "metadata": {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "model_path": str(model_path),
            "generated_at": datetime.now().isoformat(),
            "model_name": "M002_Full_Architecture"
        },
        "performance_metrics": convert_to_python_types(metrics)
    }

    filename = generate_backtest_filename(ticker, start_date, end_date)
    with filename.open("w", encoding="utf-8") as f:
        json.dump(backtest_data, f, indent=2, ensure_ascii=False)

    logging.info(f"Backtest results saved to {filename}")


def create_chart(df: pl.DataFrame, ticker: str, bt: any, save_path: Optional[Path] = None) -> None:
    """Create a trading chart using backtesting.py's built-in plotting."""
    if bt is None:
        logging.warning("No backtest object available for charting")
        return

    try:
        if save_path:
            bt.plot(filename=str(save_path), open_browser=False)
            logging.info(f"Chart saved to {save_path}")
        else:
            bt.plot(open_browser=False)
            logging.info("Chart displayed")
    except Exception as e:
        logging.warning(f"Could not create chart: {e}")




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
    parser.add_argument("--show-chart", action="store_true", help="Show chart using backtesting.py.")
    parser.add_argument("--run-backtest", action="store_true", help="Run backtest and save performance metrics to reports/ folder.")
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

    # Create charts and run backtests for each ticker
    if args.save_chart or args.show_chart or args.run_backtest:
        unique_tickers = merged.select("ticker").unique().to_series().to_list()
        for ticker in unique_tickers:
            ticker_data = merged.filter(pl.col("ticker") == ticker)

            # Run backtest and get metrics + bt object
            metrics, bt = run_backtest(ticker_data)

            # Create chart if requested
            if args.save_chart:
                chart_path = args.save_chart.parent / f"{args.save_chart.stem}_{ticker}{args.save_chart.suffix}"
                create_chart(ticker_data, ticker, bt, chart_path)
            elif args.show_chart:
                create_chart(ticker_data, ticker, bt, None)

            # Save backtest results if requested
            if args.run_backtest:
                save_backtest_results(ticker, metrics, args.start, args.end, model_path)
                # Also save chart when running backtest
                chart_filename = f"{ticker}_{args.start.split('-')[0]}_{args.end.split('-')[0]}_backtest_chart.html"
                chart_path = Path("reports") / chart_filename
                create_chart(ticker_data, ticker, bt, chart_path)

if __name__ == "__main__":
    main()
