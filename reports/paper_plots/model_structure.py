from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
DEFAULT_ARCH_PUML_PATH = DEFAULT_OUTPUT_DIR / "method_architecture.puml"
DEFAULT_EVENTS_PATH = DEFAULT_OUTPUT_DIR / "method_events.png"
DEFAULT_PRED_PATH = DEFAULT_OUTPUT_DIR / "method_predictions.png"
LOOKBACK_DAYS = 180
ARCHITECTURE_PUML = """@startuml method_architecture
skinparam backgroundColor #ffffff
skinparam roundcorner 15
skinparam linetype ortho
skinparam shadowing false

left to right direction

rectangle LayerA [
  Layer A
  Regime Classifier
  --
  * Morphological labeler -> interpretable regimes
  * Inputs: curvature, slope, event flags
  * Outputs: state probabilities
]

rectangle LayerB [
  Layer B
  Prediction Head
  --
  * LightGBM rebound classifier
  * Multi-output regression (return / drawdown)
  * Feature fusion: raw + regime probs
]

rectangle LayerC [
  Layer C
  Policy & Execution
  --
  * Soft utility: prob * ret - lambda * drawdown
  * Discrete policy (flat / long / short)
  * Volatility-aware sizing & safeguards
]

LayerA --> LayerB : state probabilities
LayerB --> LayerC : policy scores

legend right
Three-tier M002 methodology
Layer A: Interpret Regimes
Layer B: Predict Returns & DD
Layer C: Execute Policy
end legend

@enduml
"""


def _maybe_save(fig: plt.Figure, save_path: Optional[str | os.PathLike]) -> Optional[Path]:
    target = None
    if save_path:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=300, bbox_inches="tight")
    return target


def write_architecture_puml(save_path: Optional[str | os.PathLike] = None) -> Path:
    target = Path(save_path or DEFAULT_ARCH_PUML_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(ARCHITECTURE_PUML.strip() + "\n", encoding="utf-8")
    return target


def _download_aapl_history(
    start: Optional[str] = None,
    end: Optional[str] = None,
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install `yfinance` to pull AAPL history.") from exc

    end_dt = datetime.utcnow() if end is None else datetime.fromisoformat(end)
    extra_days = max(lookback_days + 10, lookback_days * 2)
    if start is None:
        start_dt = end_dt - timedelta(days=extra_days)
        start = start_dt.strftime("%Y-%m-%d")

    data = yf.download("AAPL", start=start, end=end_dt.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError("No AAPL data returned from Yahoo Finance.")

    # Handle MultiIndex columns from newer yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        # Flatten MultiIndex columns
        data.columns = data.columns.get_level_values(0)

    df = data.reset_index().rename(columns=str.lower).sort_values("date")
    df = df.tail(lookback_days).reset_index(drop=True)
    return df


def _draw_candles(ax: plt.Axes, df: pd.DataFrame, width: float = 0.6) -> None:
    required_cols = {"open", "high", "low", "close"}
    df = df.copy()
    df.columns = [str(col).lower() for col in df.columns]
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Candlestick data missing columns: {missing}")

    dates = mdates.date2num(df["date"])
    price_rows = df[["open", "high", "low", "close"]].itertuples(index=False, name=None)

    for x, (open_, high, low, close) in zip(dates, price_rows):
        color = "#2a9d8f" if close >= open_ else "#e63946"
        ax.plot([x, x], [low, high], color=color, linewidth=1.2, solid_capstyle="round")
        body_height = max(abs(close - open_), 1e-3)
        lower = min(open_, close)
        rect = Rectangle(
            (x - width / 2, lower),
            width,
            body_height,
            facecolor=color,
            edgecolor=color,
            alpha=0.85,
        )
        ax.add_patch(rect)
    ax.set_xlim(dates[0] - 0.6, dates[-1] + 0.6)


def _mark_event_points(mask: pd.Series) -> pd.Series:
    mask = mask.fillna(False)
    shifted = mask.shift(1, fill_value=False)
    return mask & ~shifted


def plot_aapl_event_visualization(
    start: Optional[str] = None,
    end: Optional[str] = None,
    save_path: Optional[str | os.PathLike] = None,
    show: bool = False,
    data: Optional[pd.DataFrame] = None,
) -> Optional[Path]:
    df = data.copy() if data is not None else _download_aapl_history(start=start, end=end)
    df["slope"] = df["close"].pct_change(3)
    df["curvature"] = df["slope"].diff()
    df["volatility"] = df["close"].pct_change().rolling(10).std()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df.dropna(inplace=True)

    events = {
        "Momentum Burst": df["slope"] > df["slope"].quantile(0.85),
        "Mean-Revert Setup": df["curvature"] < df["curvature"].quantile(0.15),
        "Volume Surge": df["volume_ratio"] > 1.5,
    }
    colors = {
        "Momentum Burst": "#ff6b6b",
        "Mean-Revert Setup": "#4d908e",
        "Volume Surge": "#f9c74f",
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    price_ax, signal_ax = axes

    _draw_candles(price_ax, df)
    price_ax.set_ylabel("Close", fontsize=11)
    price_ax.set_title("AAPL · Morphological Event Overlay", fontsize=15, fontweight="bold")

    for name, mask in events.items():
        markers = _mark_event_points(mask)
        price_ax.scatter(
            df.loc[markers, "date"],
            df.loc[markers, "close"],
            color=colors[name],
            label=name,
            s=50,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )

    price_ax.legend(loc="upper left", frameon=True)
    price_ax.grid(alpha=0.2)

    signal_ax.plot(df["date"], df["slope"], label="Slope (ΔP)", color="#577590", linewidth=1.6)
    signal_ax.plot(
        df["date"],
        df["curvature"],
        label="Curvature (Δ²P)",
        color="#d62828",
        linewidth=1.4,
        alpha=0.8,
    )
    signal_ax.plot(
        df["date"],
        df["volatility"],
        label="Rolling Volatility",
        color="#90be6d",
        linewidth=1.4,
        alpha=0.9,
    )
    signal_ax.set_ylabel("Signal", fontsize=11)
    signal_ax.set_xlabel("Date", fontsize=11)
    signal_ax.legend(loc="upper right")
    signal_ax.grid(alpha=0.2)

    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    price_ax.xaxis.set_major_locator(locator)
    price_ax.xaxis.set_major_formatter(formatter)
    signal_ax.xaxis.set_major_locator(locator)
    signal_ax.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    saved_path = _maybe_save(fig, save_path or DEFAULT_EVENTS_PATH)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved_path


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def plot_prediction_results(
    start: Optional[str] = None,
    end: Optional[str] = None,
    save_path: Optional[str | os.PathLike] = None,
    show: bool = False,
    data: Optional[pd.DataFrame] = None,
) -> Optional[Path]:
    df = data.copy() if data is not None else _download_aapl_history(start=start, end=end)

    df["ret_1d"] = df["close"].pct_change()
    df["slope_3d"] = df["close"].pct_change(3)
    df["curvature"] = df["slope_3d"].diff()
    df["volatility_10d"] = df["ret_1d"].rolling(10).std()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["ret_5d_ahead"] = df["close"].shift(-5) / df["close"] - 1.0
    df.dropna(inplace=True)

    features = (
        2.1 * df["slope_3d"]
        - 1.3 * df["curvature"]
        - 0.4 * df["volatility_10d"]
        + 0.6 * (df["volume_ratio"] - df["volume_ratio"].median())
    )
    df["prob_rebound"] = _sigmoid(features * 4.5)

    long_thr, short_thr = 0.62, 0.4
    df["action"] = np.where(
        df["prob_rebound"] >= long_thr,
        "Long",
        np.where(df["prob_rebound"] <= short_thr, "Short", "Flat"),
    )
    df["pred_dir"] = df["action"].map({"Long": 1, "Flat": 0, "Short": -1})
    df["future_dir"] = np.where(df["ret_5d_ahead"] > 0, 1, np.where(df["ret_5d_ahead"] < 0, -1, 0))
    df["hit"] = (df["pred_dir"] == df["future_dir"]).astype(int)

    def _segment_boundaries(series: pd.Series):
        start_idx = 0
        for idx in range(1, len(series)):
            if series.iloc[idx] != series.iloc[idx - 1]:
                yield start_idx, idx - 1, series.iloc[idx - 1]
                start_idx = idx
        yield start_idx, len(series) - 1, series.iloc[-1]

    action_colors = {"Long": "#2a9d8f", "Flat": "#bdbdbd", "Short": "#ef476f"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    price_ax, prob_ax = axes

    _draw_candles(price_ax, df)
    for start_idx, end_idx, action in _segment_boundaries(df["action"]):
        if action == "Flat":
            continue
        price_ax.axvspan(
            df["date"].iloc[start_idx],
            df["date"].iloc[end_idx],
            color=action_colors[action],
            alpha=0.15,
        )
    change_points = df["action"].ne(df["action"].shift()).fillna(True) & (df["action"] != "Flat")
    price_ax.scatter(
        df.loc[change_points, "date"],
        df.loc[change_points, "close"],
        color=df.loc[change_points, "action"].map(action_colors),
        s=32,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        label="Trade switch",
    )
    price_ax.set_ylabel("Price", fontsize=11)
    price_ax.set_title("Demo: Model Predictions vs AAPL (5-day horizon)", fontsize=15, fontweight="bold")
    price_ax.grid(alpha=0.2)

    prob_ax.plot(df["date"], df["prob_rebound"], color="#264653", linewidth=2, label="Prob(Rebound)")
    prob_ax.fill_between(
        df["date"],
        0,
        df["ret_5d_ahead"],
        where=df["ret_5d_ahead"] >= 0,
        color="#90be6d",
        alpha=0.25,
        label="+5d Return",
    )
    prob_ax.fill_between(
        df["date"],
        0,
        df["ret_5d_ahead"],
        where=df["ret_5d_ahead"] < 0,
        color="#f94144",
        alpha=0.25,
    )
    prob_ax.axhline(long_thr, color="#2a9d8f", linestyle="--", linewidth=1, label="Long threshold")
    prob_ax.axhline(short_thr, color="#ef476f", linestyle="--", linewidth=1, label="Short threshold")
    prob_ax.set_ylabel("Prob / Return", fontsize=11)
    prob_ax.set_xlabel("Date", fontsize=11)

    trade_mask = df["action"] != "Flat"
    hit_rate = df.loc[trade_mask, "hit"].mean()
    long_mask = df["action"] == "Long"
    short_mask = df["action"] == "Short"
    long_hit = df.loc[long_mask, "hit"].mean() if long_mask.any() else float("nan")
    short_hit = df.loc[short_mask, "hit"].mean() if short_mask.any() else float("nan")
    avg_long_ret = df.loc[long_mask, "ret_5d_ahead"].mean() if long_mask.any() else float("nan")
    hit_text = f"{hit_rate:.2%}" if not np.isnan(hit_rate) else "N/A"
    long_hit_text = f"{long_hit:.2%}" if not np.isnan(long_hit) else "N/A"
    short_hit_text = f"{short_hit:.2%}" if not np.isnan(short_hit) else "N/A"
    avg_long_text = f"{avg_long_ret:+.2%}" if not np.isnan(avg_long_ret) else "N/A"
    summary = (
        f"Trade hit-rate: {hit_text} (n={int(trade_mask.sum())})\n"
        f"Long acc: {long_hit_text} | Short acc: {short_hit_text}\n"
        f"Avg +5d return when Long: {avg_long_text}"
    )
    prob_ax.text(
        0.01,
        0.95,
        summary,
        transform=prob_ax.transAxes,
        fontsize=10,
        color="#333",
        va="top",
        bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.4"),
    )
    prob_ax.legend(loc="upper right")
    prob_ax.grid(alpha=0.2)

    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    price_ax.xaxis.set_major_locator(locator)
    price_ax.xaxis.set_major_formatter(formatter)
    prob_ax.xaxis.set_major_locator(locator)
    prob_ax.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    saved_path = _maybe_save(fig, save_path or DEFAULT_PRED_PATH)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Methodology visuals for M002.")
    parser.add_argument(
        "--mode",
        choices=["all", "puml", "events", "predictions"],
        default="all",
        help="Which artifact(s) to generate.",
    )
    parser.add_argument("--output-dir", type=str, help="Directory to drop default outputs.")
    parser.add_argument("--output-arch-puml", type=str, help="Custom path for the architecture PlantUML.")
    parser.add_argument("--output-events", type=str, help="Custom path for the event overlay PNG.")
    parser.add_argument("--output-pred", type=str, help="Custom path for the prediction PNG.")
    parser.add_argument("--start", type=str, default=None, help="Start date for Yahoo Finance pulls (default: auto).")
    parser.add_argument("--end", type=str, default=None, help="End date for Yahoo Finance pulls (default: today).")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=LOOKBACK_DAYS,
        help=f"Number of most recent trading days to display (default {LOOKBACK_DAYS}).",
    )
    parser.add_argument("--show", action="store_true", help="Display Matplotlib figures instead of closing.")
    return parser.parse_args()


def _resolve_outputs(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    base_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    arch_path = Path(args.output_arch_puml) if args.output_arch_puml else base_dir / DEFAULT_ARCH_PUML_PATH.name
    events_path = Path(args.output_events) if args.output_events else base_dir / DEFAULT_EVENTS_PATH.name
    pred_path = Path(args.output_pred) if args.output_pred else base_dir / DEFAULT_PRED_PATH.name
    return arch_path, events_path, pred_path


def main() -> None:
    args = _parse_args()
    arch_path, events_path, pred_path = _resolve_outputs(args)
    cached_df = None

    if args.mode in {"all", "puml"}:
        write_architecture_puml(save_path=arch_path)

    if args.mode in {"all", "events", "predictions"}:
        cached_df = _download_aapl_history(start=args.start, end=args.end, lookback_days=args.lookback_days)

    if args.mode in {"all", "events"}:
        plot_aapl_event_visualization(
            start=args.start,
            end=args.end,
            save_path=events_path,
            show=args.show,
            data=cached_df,
        )

    if args.mode in {"all", "predictions"}:
        plot_prediction_results(
            start=args.start,
            end=args.end,
            save_path=pred_path,
            show=args.show,
            data=cached_df,
        )


if __name__ == "__main__":
    main()


__all__ = [
    "write_architecture_puml",
    "plot_aapl_event_visualization",
    "plot_prediction_results",
]
