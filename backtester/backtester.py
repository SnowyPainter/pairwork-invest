"""
Backtesting skeleton focused on *good visualization* and fast Polars pipelines.
- Input: Polars DataFrame from `build_dataset(...)` with BASE_COLS and optional feature/signal columns.
- Supports: open→close or next-open→close execution, top-N or threshold selection, equal-weight or score-weight, fees/slippage.
- Outputs: summary stats + artifacts in `reports/backtest` and ready-made matplotlib plots.

Usage (example):

from data.dataset_builder import build_dataset
from backtester_skeleton import (
    BacktestConfig, UniverseRule, SignalRule, ExecutionRule, PortfolioRule,
    backtest, plot_equity, plot_drawdown, plot_monthly_heatmap,
    plot_rolling_sharpe, plot_contrib_by_ticker, plot_confusion_if_classification
)

CFG = BacktestConfig(
    label_col="futret_1",                 # or your computed daily return column
    signal_col="score_long",              # positive = long intent
    # signal_col can also be class probs/logits; see SignalRule
    universe=UniverseRule(top_k_per_day=100),
    signal=SignalRule(select_top_n=20, min_threshold=None, long_only=True),
    execution=ExecutionRule(mode="next_open_to_close"),
    portfolio=PortfolioRule(weighting="equal", max_gross_leverage=1.0,
                            fee_bps=8, slippage_bps=5),
)

df = build_dataset(years=[2020], market="KR", feature_set="v3",
                   label_horizon=1, label_task="regression",
                   verbose=True)

res = backtest(df, CFG)
print(res["summary"])  # dict

# Plots (saved to outdir and returned as figure for interactive use)
plot_equity(res, show=False)
plot_drawdown(res, show=False)
plot_monthly_heatmap(res, show=False)
plot_rolling_sharpe(res, window=60, show=False)
plot_contrib_by_ticker(res, top=20, show=False)

# If you have classification labels like label_up in df
# plot_confusion_if_classification(res, label_true_col="label_up", show=False)

"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Tuple
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------
# Constants / helpers
# ---------------------------------
BASE_COLS = {
    "date","ticker","market","exchange","currency","year",
    "open","high","low","close","adj_close","volume","turnover",
}

OUTDIR = Path("reports/backtest")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Safe conversion for plotting
_to_pd = lambda df: df.to_pandas(use_pyarrow_extension_array=True) if isinstance(df, pl.DataFrame) else df

# ---------------------------------
# Config dataclasses
# ---------------------------------
@dataclass
class UniverseRule:
    top_k_per_day: Optional[int] = None  # choose by absolute |signal| desc
    min_turnover: Optional[float] = None
    min_price: Optional[float] = None

@dataclass
class SignalRule:
    # Selection policy: choose positions from a *score* column
    select_top_n: Optional[int] = 20
    min_threshold: Optional[float] = None  # require score >= threshold
    long_only: bool = True                 # if False → long/short by sign

@dataclass
class ExecutionRule:
    # Trade when the signal is known relative to price path
    mode: Literal["open_to_close", "next_open_to_close"] = "next_open_to_close"

@dataclass
class PortfolioRule:
    weighting: Literal["equal", "score"] = "equal"
    max_gross_leverage: float = 1.0
    fee_bps: float = 8.0
    slippage_bps: float = 5.0

@dataclass
class BacktestConfig:
    label_col: str                     # realized return per trade horizon (e.g., futret_1)
    signal_col: str                    # score for ranking/selecting longs/shorts
    universe: UniverseRule
    signal: SignalRule
    execution: ExecutionRule
    portfolio: PortfolioRule
    outdir: Path = OUTDIR

# ---------------------------------
# Return constructors
# ---------------------------------
def realized_return(df: pl.DataFrame, exec_rule: ExecutionRule) -> pl.Series:
    """Construct realized 1D return if label_col not provided. (Optional helper)
    For simplicity we rely on a label_col already in df. This is a placeholder
    in case users want to switch to open/close derived returns.
    """
    raise NotImplementedError

# ---------------------------------
# Core Backtest
# ---------------------------------

def _select_universe(lf: pl.LazyFrame, uni: UniverseRule, signal_col: str) -> pl.LazyFrame:
    # Optionally filter by liquidity/price
    conds = []
    if uni.min_turnover is not None:
        conds.append(pl.col("turnover") >= uni.min_turnover)
    if uni.min_price is not None:
        conds.append(pl.col("close") >= uni.min_price)
    if conds:
        lf = lf.filter(pl.all_horizontal(conds))

    if uni.top_k_per_day is None:
        return lf

    # Top-|signal| per date
    return (
        lf.with_columns(abs_score=pl.col(signal_col).abs())
          .sort(["date", "abs_score"], descending=[False, True])
          .group_by("date")
          .head(uni.top_k_per_day)
          .drop("abs_score")
    )


def _select_positions(lf: pl.LazyFrame, sig: SignalRule, signal_col: str) -> pl.LazyFrame:
    # apply threshold
    if sig.min_threshold is not None:
        lf = lf.filter(pl.col(signal_col) >= sig.min_threshold)

    if sig.select_top_n is not None:
        lf = (lf.sort(["date", signal_col], descending=[False, True])
                .group_by("date").head(sig.select_top_n))

    # side and raw position intent
    if sig.long_only:
        side_expr = pl.lit(1.0)
    else:
        side_expr = pl.when(pl.col(signal_col) >= 0).then(1.0).otherwise(-1.0)

    return lf.with_columns(
        side=side_expr,
        sel_rank=pl.int_range(0, pl.len()).over("date")  # for debugging
    )


def _assign_weights(lf: pl.LazyFrame, prt: PortfolioRule, signal_col: str) -> pl.LazyFrame:
    if prt.weighting == "equal":
        w = (pl.lit(1.0)
             .group_by("date").count()
             .over("date")
        )
        # 1 / count  * side
        return lf.with_columns(weight = (pl.col("side") / w))

    # score-proportional weights (positive-only scores assumed; if long/short, use |score|)
    denom = (pl.col(signal_col).abs() if True else pl.col(signal_col))
    return (
        lf.with_columns(score_pos = denom)
          .with_columns(score_sum = pl.col("score_pos").sum().over("date"))
          .with_columns(weight = pl.col("side") * (pl.col("score_pos") / pl.col("score_sum").replace(0, np.nan)))
          .drop(["score_pos", "score_sum"]) 
    )


def _apply_fees(lf: pl.LazyFrame, prt: PortfolioRule) -> pl.LazyFrame:
    # Fees & slippage per turnover day: |weight change| * bps. Skeleton: use gross exposure as proxy
    bps = (prt.fee_bps + prt.slippage_bps) / 1e4
    # Approximate daily cost = bps * sum(|weight|) over positions
    fees = (pl.col("weight").abs().sum().over("date")) * bps
    return lf.with_columns(daily_fee = fees)


def backtest(df: pl.DataFrame, cfg: BacktestConfig) -> Dict[str, object]:
    """Main backtest entry.
    Returns dict with `daily`, `positions`, `summary`, etc.
    Required columns in df: BASE_COLS ∪ { cfg.label_col, cfg.signal_col }
    """
    for c in ["date", "ticker", cfg.label_col, cfg.signal_col, "close"]:
        if c not in df.columns:
            raise ValueError(f"Required column missing: {c}")

    lf0 = df.lazy()

    # Universe selection
    lfU = _select_universe(lf0, cfg.universe, cfg.signal_col)

    # Position selection
    lfS = _select_positions(lfU, cfg.signal, cfg.signal_col)

    # Weights
    lfW = _assign_weights(lfS, cfg.portfolio, cfg.signal_col)

    # Realized return from label_col
    lfR = lfW.with_columns(
        pnl = pl.col(cfg.label_col) * pl.col("weight")
    )

    # Fees
    lfF = _apply_fees(lfR, cfg.portfolio)

    # Aggregate daily PnL
    daily = (
        lfF.group_by("date")
           .agg([
               pl.col("pnl").sum().alias("gross_pnl"),
               pl.col("daily_fee").max().alias("fee_proxy"),  # already per date
               pl.len().alias("n_positions"),
           ])
           .with_columns(net_pnl = pl.col("gross_pnl") - pl.col("fee_proxy"))
           .sort("date")
           .collect(streaming=True)
    )

    daily = daily.with_columns(
        equity = (1.0 + pl.col("net_pnl")).cum_prod(),
        dd = (pl.col("equity").cum_max() - pl.col("equity")) / pl.col("equity").cum_max()
    )

    # Per-ticker contribution
    contrib = (
        lfF.group_by(["date", "ticker"]).agg(pl.col("pnl").sum().alias("pnl"))
           .group_by("ticker").agg(pl.col("pnl").sum().alias("total_pnl"))
           .sort("total_pnl", descending=True)
           .collect(streaming=True)
    )

    # Summary stats
    equ = daily.select(["equity"]).to_series().to_numpy()
    net = daily.select(["net_pnl"]).to_series().to_numpy()
    ann = 252
    def _safe_mean(x):
        return float(np.nanmean(x)) if len(x) else float("nan")
    def _safe_std(x):
        return float(np.nanstd(x, ddof=1)) if len(x) > 1 else float("nan")

    ret = _safe_mean(net) * ann
    vol = _safe_std(net) * np.sqrt(ann)
    sharpe = ret / vol if vol and vol == vol else float("nan")
    max_dd = float(daily["dd"].max()) if daily.height else float("nan")

    summary = {
        "days": int(daily.height),
        "ret_annual": ret,
        "vol_annual": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_equity": float(daily["equity"][-1]) if daily.height else 1.0,
        "avg_n_positions": float(daily["n_positions"].mean()) if daily.height else 0.0,
        "fee_bps": cfg.portfolio.fee_bps,
        "slippage_bps": cfg.portfolio.slippage_bps,
    }

    res = {
        "config": cfg,
        "daily": daily,
        "positions": lfW.collect(streaming=True),
        "contrib": contrib,
        "summary": summary,
    }

    return res

# ---------------------------------
# Visualization helpers
# ---------------------------------

def _save(fig: plt.Figure, name: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(fp, dpi=160)
    return fp


def plot_equity(res: Dict[str, object], *, show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    daily: pl.DataFrame = res["daily"]
    pdf = _to_pd(daily)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(pdf["date"]), pdf["equity"], lw=1.5)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (× initial)")
    _save(fig, "equity_curve", res["config"].outdir)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_drawdown(res: Dict[str, object], *, show: bool = True):
    daily: pl.DataFrame = res["daily"]
    pdf = _to_pd(daily)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.fill_between(pd.to_datetime(pdf["date"]), -pdf["dd"], 0.0, step=None)
    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("DD")
    _save(fig, "drawdown", res["config"].outdir)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_rolling_sharpe(res: Dict[str, object], window: int = 60, *, show: bool = True):
    daily: pl.DataFrame = res["daily"]
    pdf = _to_pd(daily)
    rtn = pdf["net_pnl"].to_numpy()
    roll = pd.Series(rtn).rolling(window)
    sharpe = (roll.mean() * 252) / (roll.std(ddof=1) * np.sqrt(252))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(pd.to_datetime(pdf["date"]), sharpe)
    ax.axhline(0, lw=1)
    ax.set_title(f"Rolling Sharpe ({window}d)")
    ax.set_xlabel("Date")
    _save(fig, "rolling_sharpe", res["config"].outdir)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_monthly_heatmap(res: Dict[str, object], *, show: bool = True):
    daily: pl.DataFrame = res["daily"]
    pdf = _to_pd(daily)
    idx = pd.to_datetime(pdf["date"])  # ensures datetime index
    mret = pd.Series(pdf["net_pnl"].to_numpy(), index=idx).resample("M").apply(lambda x: (1+x).prod()-1)
    tbl = mret.to_frame("ret").assign(year=mret.index.year, month=mret.index.strftime("%b"))
    pt = tbl.pivot(index="year", columns="month", values="ret").fillna(0.0)
    # order months
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pt = pt.reindex(columns=months)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pt.values, aspect="auto")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months)
    ax.set_yticks(range(len(pt.index)))
    ax.set_yticklabels(pt.index)
    ax.set_title("Monthly Return Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.025)
    _save(fig, "monthly_heatmap", res["config"].outdir)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_contrib_by_ticker(res: Dict[str, object], top: int = 20, *, show: bool = True):
    contrib: pl.DataFrame = res["contrib"]
    pdf = _to_pd(contrib.head(top))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(pdf["ticker"], pdf["total_pnl"])
    ax.set_title(f"Contribution by Ticker (Top {top})")
    ax.set_xticklabels(pdf["ticker"], rotation=60, ha="right")
    _save(fig, "contrib_by_ticker", res["config"].outdir)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_confusion_if_classification(res: Dict[str, object], label_true_col: str = "label_up", *, show: bool = True):
    """If you ran a classification-based strategy, compute confusion on selected positions.
    Expects `positions` to have `pred` or `signal_col` thresholded as positive class,
    and df to include `label_true_col` ∈ {0,1}.
    """
    pos: pl.DataFrame = res["positions"]
    cfg: BacktestConfig = res["config"]
    if label_true_col not in pos.columns:
        # Try to join from original daily? Skipping to keep skeleton light.
        print(f"[confusion] '{label_true_col}' not found in positions. Skipped.")
        return None, None

    pdf = _to_pd(pos.select(["date","ticker", cfg.signal_col, label_true_col]))
    # predict positive if signal >= 0 (or threshold)
    thr = 0.0
    pred = (pdf[cfg.signal_col] >= thr).astype(int)
    y = pdf[label_true_col].astype(int)
    tp = int(((pred==1) & (y==1)).sum())
    fp = int(((pred==1) & (y==0)).sum())
    tn = int(((pred==0) & (y==0)).sum())
    fn = int(((pred==0) & (y==1)).sum())

    fig, ax = plt.subplots(figsize=(4, 4))
    mat = np.array([[tp, fp],[fn, tn]])
    ax.imshow(mat, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred 1","Pred 0"]) ; ax.set_yticklabels(["True 1","True 0"])
    for (i,j), val in np.ndenumerate(mat):
        ax.text(j, i, str(val), ha='center', va='center')
    _save(fig, "confusion_matrix", res["config"].outdir)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax

# ---------------------------------
# Convenience: quick runner
# ---------------------------------

def quick_run(df: pl.DataFrame, *, label_col: str, signal_col: str, outdir: str | Path = OUTDIR) -> Dict[str, object]:
    cfg = BacktestConfig(
        label_col=label_col,
        signal_col=signal_col,
        universe=UniverseRule(top_k_per_day=100),
        signal=SignalRule(select_top_n=20, min_threshold=None, long_only=True),
        execution=ExecutionRule(mode="next_open_to_close"),
        portfolio=PortfolioRule(weighting="equal", max_gross_leverage=1.0,
                                fee_bps=8, slippage_bps=5),
        outdir=Path(outdir)
    )
    return backtest(df, cfg)
