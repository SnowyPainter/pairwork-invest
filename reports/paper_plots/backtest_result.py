#!/usr/bin/env python3
"""
Compare the 3계층 분석 모델 (M002 full architecture) versus an HMM-based UP/DOWN/FLAT regime model.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib
import seaborn as sns
sns.set(font_scale=1.0, font="NanumGothic")
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from apps.m002_yfinance_predict import (  # noqa: E402
    DownloadConfig,
    download_prices,
    build_feature_frame,
    find_latest_m002_model,
    load_m002_model,
    maybe_extract_model_stats,
    apply_normalization,
    filter_required_rows,
    predict as run_m002_predict,
    merge_predictions,
)
from models.M002_RegimeClassifier import DEFAULT_REGIME_FEATURES  # noqa: E402
from models.M002_FullArchitecture import M002FullArchitecture, STATE_PROB_COLS  # noqa: E402
from models.HMM import MarketRegimeHMM  # noqa: E402

LSTM_STATE_ORDER = ["Accumulation", "EarlyUp", "Peak", "Distribution", "LateDown"]
LSTM_STATE_COLORS = {
    "Accumulation": "#7f8c8d",
    "EarlyUp": "#1abc9c",
    "Peak": "#f1c40f",
    "Distribution": "#d35400",
    "LateDown": "#c0392b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize differences between 3계층 분석 모델 신호와 HMM 레짐 신호.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", default="AAPL", help="Yahoo Finance ticker.")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD). Defaults to 2y ago.")
    parser.add_argument("--end", help="End date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--interval", default="1d", help="yfinance interval.")
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to a pickled 3계층 분석 모델(M002) artifact. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        help="Optional JSON file with normalization stats.",
    )
    parser.add_argument(
        "--lookback-train",
        type=int,
        default=252,
        help="Minimum rows required to train the HMM.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/paper_plots/hmm_vs_m002.png"),
        help="Figure output path.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def ensure_dates(args: argparse.Namespace) -> Tuple[datetime.date, datetime.date]:
    today = datetime.utcnow().date()
    if args.end is None:
        args.end = today.strftime("%Y-%m-%d")
    if args.start is None:
        args.start = (today - timedelta(days=2 * 365)).strftime("%Y-%m-%d")
    start_dt = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").date()
    if end_dt < start_dt:
        raise SystemExit("end date must be on or after start date.")
    return start_dt, end_dt


def load_stats(path: Optional[Path]) -> Optional[Dict[str, Dict[str, float]]]:
    if path is None:
        return None
    if not path.exists():
        logging.warning("Normalization stats file %s not found.", path)
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    stats: Dict[str, Dict[str, float]] = {}
    for col, spec in payload.items():
        stats[str(col)] = {
            "mean": float(spec.get("mean", 0.0)),
            "std": float(spec.get("std", 1.0)) or 1.0,
        }
    return stats


def prepare_m002_predictions(
    ticker: str,
    model: M002FullArchitecture,
    download_cfg: DownloadConfig,
    stats_path: Optional[Path],
) -> pl.DataFrame:
    ohlcv = download_prices(download_cfg)
    feature_df = build_feature_frame(ohlcv, feature_set="m002")

    regime_probs = model.regime.predict_probabilities(
        feature_df.select(["ticker", "date", *DEFAULT_REGIME_FEATURES])
    )
    feature_df = feature_df.join(regime_probs, on=["ticker", "date"], how="left")

    for col in STATE_PROB_COLS:
        if col in feature_df.columns:
            feature_df = feature_df.with_columns(pl.col(col).fill_null(0.0))

    stats = load_stats(stats_path)
    if stats is None and getattr(model.config, "normalize_features", True):
        stats = maybe_extract_model_stats(model)
    if stats:
        feature_df = apply_normalization(feature_df, stats)

    required = sorted(set(model.head_features) | set(DEFAULT_REGIME_FEATURES))
    feature_df = filter_required_rows(feature_df, required)

    preds = run_m002_predict(model, feature_df)
    merged = merge_predictions(
        ohlcv,
        feature_df.select(["ticker", "date"]),
        preds,
    )
    merged = merged.filter(pl.col("ticker") == ticker).sort("date")
    return merged


def train_hmm(prices: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    if len(prices) < min_rows:
        raise RuntimeError(
            f"Need at least {min_rows} rows to train the HMM. Increase lookback window."
        )
    hmm = MarketRegimeHMM()
    hmm.fit(prices, n_iter=75)
    states = hmm.predict(prices)
    return states


def _signal_direction(signal: str) -> int:
    mapping = {"LONG": 1, "SHORT": -1}
    return mapping.get(signal, 0)


def compute_accuracy(df: pd.DataFrame, min_return: float = 0.0, eval_horizon: int = 20) -> pd.DataFrame:
    df = df.copy()
    horizon = max(1, int(eval_horizon))
    df["future_ret"] = df["close"].shift(-horizon) / df["close"] - 1.0
    df["hmm_signal"] = df["hmm_state"].map({"UP": "LONG", "DOWN": "SHORT", "FLAT": "FLAT"})
    df["model_signal"] = df["action"].fillna("FLAT")

    prob_cols = [col for col in df.columns if col.startswith("state_prob_")]
    if prob_cols:
        prob_values = np.nan_to_num(df[prob_cols].to_numpy(), nan=-1.0)
        best_idx = np.argmax(prob_values, axis=1)
        df["lstm_state"] = [prob_cols[i].replace("state_prob_", "") for i in best_idx]
    else:
        df["lstm_state"] = "Unknown"

    def _is_correct(signal: str, ret: float) -> float:
        if pd.isna(ret):
            return np.nan
        direction = _signal_direction(signal)
        if direction == 0:
            return 1.0  # FLAT은 무위험으로 간주하여 적중 처리
        if abs(ret) <= min_return:
            return np.nan
        return 1.0 if direction * ret > 0 else 0.0

    df["model_correct"] = [
        _is_correct(sig, ret) for sig, ret in zip(df["model_signal"], df["future_ret"])
    ]
    df["hmm_correct"] = [
        _is_correct(sig, ret) for sig, ret in zip(df["hmm_signal"], df["future_ret"])
    ]
    return df


def _state_segments(dates: Sequence[pd.Timestamp], states: Sequence[str]) -> Iterable[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    if len(dates) == 0:
        return []
    prev_state = states[0]
    start_idx = 0
    for idx in range(1, len(dates)):
        if states[idx] != prev_state:
            yield prev_state, dates[start_idx], dates[idx - 1]
            prev_state = states[idx]
            start_idx = idx
    yield prev_state, dates[start_idx], dates[-1]


def plot_comparison(df: pd.DataFrame, ticker: str, output: Path, eval_horizon: int = 20) -> None:
    if df.empty:
        raise ValueError("No rows available for plotting.")
    date_series = pd.to_datetime(df["date"])
    df = df.copy()
    df["position_size"] = df.get("position_size", 0.0).fillna(0.0)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 9), constrained_layout=True)

    # Price + regime boundaries
    ax_price = axes[0]
    ax_price.plot(date_series, df["close"], color="#34495e", linewidth=1.5, label="Close")
    seen_labels = set()
    for state, start, end in _state_segments(date_series.to_numpy(), df["lstm_state"]):
        color = LSTM_STATE_COLORS.get(state, "#bdc3c7")
        label = f"LSTM {state}" if state not in seen_labels else None
        ax_price.axvspan(start, end, color=color, alpha=0.18, label=label)
        seen_labels.add(state)
    hmm_transitions = [
        date_series[i]
        for i in range(1, len(df))
        if df["hmm_state"].iloc[i] != df["hmm_state"].iloc[i - 1]
    ]
    for x in hmm_transitions:
        ax_price.axvline(x, color="#2c3e50", linestyle="--", linewidth=0.8, alpha=0.4)
    ax_price.set_ylabel("Price")
    ax_price.set_title(f"{ticker} 가격 및 국면 전이")
    handles, labels = ax_price.get_legend_handles_labels()
    ax_price.legend(handles, labels, loc="upper left", ncol=3)
    ax_price.grid(True, linewidth=0.3, alpha=0.4)

    # LSTM probability landscape vs HMM boundaries
    ax_prob = axes[1]
    prob_arrays = [df.get(f"state_prob_{state}", pd.Series([0.0] * len(df))).to_numpy() for state in LSTM_STATE_ORDER]
    ax_prob.stackplot(
        date_series,
        *prob_arrays,
        labels=LSTM_STATE_ORDER,
        colors=[LSTM_STATE_COLORS.get(state, "#cccccc") for state in LSTM_STATE_ORDER],
        alpha=0.8,
    )
    for x in hmm_transitions:
        ax_prob.axvline(x, color="#2c3e50", linestyle="--", linewidth=0.6, alpha=0.4)
    ax_prob.set_ylim(0, 1)
    ax_prob.set_ylabel("LSTM Regime Prob.")
    ax_prob.set_title("LSTM 기반 확률 모델: 부드러운 전이")
    ax_prob.legend(loc="upper left", ncol=5)
    ax_prob.grid(True, linewidth=0.3, alpha=0.4)

    # Rolling accuracy
    ax_roll = axes[2]
    window = max(1, int(eval_horizon))
    model_roll = df["model_correct"].rolling(window, min_periods=max(5, window // 2)).mean().ffill().bfill()
    hmm_roll = df["hmm_correct"].rolling(window, min_periods=max(5, window // 2)).mean().ffill().bfill()
    ax_roll.plot(date_series, model_roll, color="#1f77b4", linewidth=1.4, label="3계층 분석 모델")
    ax_roll.plot(date_series, hmm_roll, color="#2ecc71", linewidth=1.4, label="HMM")
    ax_roll.set_ylabel(f"Rolling Accuracy ({window}d)")
    ax_roll.set_ylim(0, 1)
    ax_roll.set_title("누적 적중률 추이")
    ax_roll.grid(True, linewidth=0.3, alpha=0.4)
    ax_roll.legend(loc="upper left")

    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

    fig.suptitle(f"{ticker} — HMM vs. 3계층 분석 모델", fontsize=16, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    logging.info("Saved comparison chart to %s", output)


def summarize(df: pd.DataFrame) -> None:
    total = len(df)
    model_hits = df["model_correct"].dropna().mean() * 100 if total else 0.0
    hmm_hits = df["hmm_correct"].dropna().mean() * 100 if total else 0.0
    logging.info(
        "Rows: %d | 3계층 분석 모델 정확도: %.1f%% | HMM 정확도: %.1f%%",
        total,
        model_hits,
        hmm_hits,
    )
    sample_cols = ["date", "close", "hmm_state", "model_signal", "hmm_signal", "future_ret"]
    diff = df[
        (df["model_correct"] != df["hmm_correct"])
        & df["model_correct"].notna()
        & df["hmm_correct"].notna()
    ]
    if not diff.empty:
        logging.info("서로 다른 판정을 낸 사례:\n%s", diff[sample_cols].head(10).to_string(index=False))


def main() -> None:
    args = parse_args()
    view_start, view_end = ensure_dates(args)
    buffer_days = timedelta(days=730)
    train_start = view_start - buffer_days
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(message)s")
    if train_start < view_start:
        logging.info("훈련용 HMM 데이터 범위를 %s까지 확장합니다 (시각화 범위: %s~%s).",
                     train_start.strftime("%Y-%m-%d"),
                     view_start.strftime("%Y-%m-%d"),
                     view_end.strftime("%Y-%m-%d"))

    if args.model_path is None:
        model_path = find_latest_m002_model()
        if model_path is None:
            raise SystemExit("No trained 3계층 분석 모델 아티팩트를 models/saved/에서 찾을 수 없습니다.")
        logging.info("Auto-selected model: %s", model_path)
    else:
        model_path = args.model_path

    model = load_m002_model(model_path)
    logging.info("3계층 분석 모델 로드 완료 (head features: %d)", len(model.head_features))

    logging.info("시각화 범위: %s ~ %s", view_start.strftime("%Y-%m-%d"), view_end.strftime("%Y-%m-%d"))

    download_cfg = DownloadConfig(
        tickers=[args.ticker],
        start=train_start.strftime("%Y-%m-%d"),
        end=view_end.strftime("%Y-%m-%d"),
        interval=args.interval,
        auto_adjust=True,
        progress=False,
    )
    merged = prepare_m002_predictions(args.ticker, model, download_cfg, args.normalization_stats)
    merged_pd = merged.to_pandas()
    merged_pd["date"] = pd.to_datetime(merged_pd["date"])

    price_cols = ["date", "open", "high", "low", "close", "volume"]
    prices_pd = merged_pd[price_cols].dropna().sort_values("date").reset_index(drop=True)
    train_mask = (prices_pd["date"].dt.date >= train_start) & (prices_pd["date"].dt.date <= view_end)
    prices_train = prices_pd.loc[train_mask].reset_index(drop=True)
    hmm_df = train_hmm(prices_train, args.lookback_train)

    view_mask = (merged_pd["date"].dt.date >= view_start) & (merged_pd["date"].dt.date <= view_end)
    merged_view = merged_pd.loc[view_mask].reset_index(drop=True)
    hmm_view_mask = (hmm_df["date"].dt.date >= view_start) & (hmm_df["date"].dt.date <= view_end)
    hmm_view = hmm_df.loc[hmm_view_mask].reset_index(drop=True)

    combined = merged_view.merge(hmm_view, on="date", how="inner")
    eval_horizon = 20
    combined = compute_accuracy(combined, eval_horizon=eval_horizon)

    summarize(combined)
    plot_comparison(combined, args.ticker, args.output, eval_horizon=eval_horizon)


if __name__ == "__main__":
    main()
