#!/usr/bin/env python3
"""
형태학적 5국면 오버레이 플롯.

M002 형태학 라벨(Accumulation~LateDown)을 가격 위에 밴드로 겹쳐서,
곡률/이벤트가 국면에 따라 어떻게 분포하는지 직관적으로 보여준다.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib
import seaborn as sns
sns.set(font_scale=1.0, font="NanumGothic")
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from apps.m002_yfinance_predict import (  # noqa: E402
    DownloadConfig,
    download_prices,
    build_feature_frame,
)
from models.M002_RegimeClassifier import _assign_regime_labels  # noqa: E402

STATE_COLORS = {
    "Accumulation": "#7f8c8d",
    "EarlyUp": "#1abc9c",
    "Peak": "#f1c40f",
    "Distribution": "#d35400",
    "LateDown": "#c0392b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="형태학적 5국면 라벨을 가격 위에 오버레이하는 플롯 생성.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", default="AAPL", help="Yahoo Finance 티커 심볼")
    parser.add_argument("--start", default="2020-01-01", help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default="2022-12-31", help="종료일 (YYYY-MM-DD)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/paper_plots/morphology_overlay.png"),
        help="플롯 저장 경로",
    )
    parser.add_argument("--log-level", default="INFO", help="로그 레벨")
    return parser.parse_args()


def load_morphology_frame(
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    # Download OHLCV data using yfinance
    download_cfg = DownloadConfig(
        tickers=[ticker],
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        market="US",
    )
    ohlcv = download_prices(download_cfg)

    # Build features
    feature_df = build_feature_frame(ohlcv, feature_set="m002")

    # Assign regime labels
    feature_df = _assign_regime_labels(feature_df)

    # Convert to pandas and prepare for plotting
    pdf = feature_df.to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf = pdf.sort_values("date").reset_index(drop=True)
    return pdf


def _segments(dates: Iterable[pd.Timestamp], states: Iterable[str]) -> Iterable[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    dates = list(dates)
    states = list(states)
    if not dates:
        return []
    start_idx = 0
    prev_state = states[0]
    for idx in range(1, len(dates)):
        if states[idx] != prev_state:
            yield prev_state, dates[start_idx], dates[idx - 1]
            prev_state = states[idx]
            start_idx = idx
    yield prev_state, dates[start_idx], dates[-1]


def plot_morphology(df: pd.DataFrame, ticker: str, output: Path) -> None:
    dates = df["date"]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8), constrained_layout=True)

    # Price overlay with regime states and events
    ax.plot(dates, df["close"], color="#2c3e50", linewidth=1.5, label="Close")

    for state, start, end in _segments(dates, df["regime_state"]):
        ax.axvspan(
            start,
            end,
            color=STATE_COLORS.get(state, "#bdc3c7"),
            alpha=0.18,
            label=state if state not in ax.get_legend_handles_labels()[1] else None,
        )

    # Add events on price chart
    event_configs = {
        "event_rebound_candidate": {"color": "#1abc9c", "marker": "^", "label": "Rebound"},
        "event_breakdown_risk": {"color": "#e74c3c", "marker": "v", "label": "Breakdown"},
        "event_local_vol_spike": {"color": "#f39c12", "marker": "*", "label": "Vol Spike"},
        "event_volume_regain": {"color": "#9b59b6", "marker": "s", "label": "Vol Regain"},
        "event_exhaustion_candidate": {"color": "#e67e22", "marker": "D", "label": "Exhaustion"},
    }

    for event_col, config in event_configs.items():
        if event_col in df.columns:
            mask = df[event_col] == 1
            if mask.any():
                # 이벤트 점들을 더 크게 표시하고 약간 투명하게 해서 겹침 시각화
                ax.scatter(
                    dates[mask],
                    df.loc[mask, "close"],
                    color=config["color"],
                    marker=config["marker"],
                    s=80,  # 2배 크기로 증가
                    alpha=0.8,
                    label=config["label"],
                    zorder=5,
                    edgecolors='white',
                    linewidth=1,
                )

                # 겹치는 이벤트들을 화살표로 표시 (같은 날짜에 여러 이벤트가 있는 경우)
                event_dates = dates[mask]
                event_prices = df.loc[mask, "close"]

                # 같은 날짜에 여러 이벤트가 있는지 확인
                date_counts = event_dates.value_counts()
                overlapping_dates = date_counts[date_counts > 1].index

                for overlap_date in overlapping_dates:
                    overlap_mask = event_dates == overlap_date
                    if overlap_mask.sum() > 1:
                        # 겹치는 이벤트들을 화살표로 연결해서 표시
                        overlap_prices = event_prices[overlap_mask]
                        min_price = overlap_prices.min()
                        max_price = overlap_prices.max()

                        # 화살표로 범위 표시
                        ax.annotate('',
                                  xy=(overlap_date, max_price),
                                  xytext=(overlap_date, min_price),
                                  arrowprops=dict(arrowstyle='<->', color='red', alpha=0.7, linewidth=2),
                                  zorder=6)

    ax.set_title(f"{ticker} – Morphology-Based 5 Regime Overlay")
    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", ncol=4, fontsize=12, framealpha=0.9)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"✅ Overlay saved to {output}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(message)s")

    df = load_morphology_frame(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
    )
    if df.empty:
        raise SystemExit("데이터가 비어 있습니다. 기간 또는 티커를 확인하세요.")

    # Log regime distribution
    import polars as pl
    regime_counts = pl.from_pandas(df).group_by("regime_state").len().sort("len", descending=True)
    logging.info("✅ Regime Label Distribution:")
    logging.info(regime_counts)

    plot_morphology(df, args.ticker, args.output)


if __name__ == "__main__":
    main()
