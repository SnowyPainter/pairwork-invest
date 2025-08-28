from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple

import polars as pl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from data.dataset_builder import build_dataset

BASE_COLS = {
    "date","ticker","market","exchange","currency","year",
    "open","high","low","close","adj_close","volume","turnover",
}

def _ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def _pick_feature_cols(df: pl.DataFrame, target_col: str) -> List[str]:
    cols = []
    for c, dt in df.schema.items():
        if c in BASE_COLS:
            continue
        if c == target_col or c.startswith("label_") or c.startswith("futret_"):
            continue
        if dt in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64):
            cols.append(c)
    return cols

def _z(col: str, eps: float = 1e-9) -> pl.Expr:
    x = pl.col(col)
    return (x - x.mean()) / (x.std() + eps)

def compute_cs_ic(df: pl.DataFrame, feature_cols: List[str], target_col: str = "futret_1") -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    날짜별 교차섹션 IC(피어슨)를 안전하게 계산:
    IC(date, f) = mean( zscore(f) * zscore(target) ) within date
    반환:
      - ic_long: columns=[date, feature, ic, ic_ma60, ic_std60]
      - ic_summary: per-feature summary(mean, std, IR, n_days)
    """
    # 날짜 단위 그룹에서 피처별 IC 계산
    agg_exprs = [((_z(f) * _z(target_col)).mean()).alias(f) for f in feature_cols]
    ic_wide = (
        df.lazy()
          .group_by("date")
          .agg(agg_exprs)
          .sort("date")
          .collect(streaming=True)
    )
    # Long으로 변환
    ic_long = (
        ic_wide.melt(id_vars=["date"], variable_name="feature", value_name="ic")
              .drop_nulls(["ic"])
    )
    # 피처별 날짜 정렬 후 60-일 롤링 평균/표준편차
    ic_long = (
        ic_long.sort(["feature","date"])
               .with_columns([
                   pl.col("ic").rolling_mean(60).over("feature").alias("ic_ma60"),
                   pl.col("ic").rolling_std(60).over("feature").alias("ic_std60"),
               ])
    )
    # 요약 통계
    ic_summary = (
        ic_long.group_by("feature")
               .agg([
                   pl.len().alias("n_days"),
                   pl.col("ic").mean().alias("ic_mean"),
                   pl.col("ic").std().alias("ic_std"),
               ])
               .with_columns([
                   (pl.col("ic_mean") / (pl.col("ic_std") + 1e-9)).alias("ic_ir"),
               ])
               .sort("ic_ir", descending=True)
    )
    return ic_long, ic_summary

def event_filter(df: pl.DataFrame, target_col: str = "futret_1", thresh: float = 0.05) -> pl.DataFrame:
    return df.filter(pl.col(target_col).abs() >= thresh)

def plot_scatter_event(df: pl.DataFrame, feature: str, target_col: str, out_path: Path, title_suffix: str = ""):
    pdf = df.select(["date","ticker", feature, target_col]).to_pandas()
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=pdf, x=feature, y=target_col, alpha=0.35, s=12, edgecolor=None)
    sns.regplot(data=pdf, x=feature, y=target_col, scatter=False, color="red", lowess=True, line_kws={"lw": 1.5})
    plt.title(f"{feature} vs {target_col} {title_suffix}".strip())
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_scatter_all(df: pl.DataFrame, feature: str, target_col: str, out_path: Path, title_suffix: str = ""):
    pdf = df.select(["date","ticker", feature, target_col]).to_pandas()
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=pdf, x=feature, y=target_col, alpha=0.15, s=10, edgecolor=None)
    sns.regplot(data=pdf, x=feature, y=target_col, scatter=False, color="red", lowess=True, line_kws={"lw": 1.2})
    plt.title(f"{feature} vs {target_col} {title_suffix}".strip())
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def build_train_frame(
    market: str = "KR",
    years: Iterable[int] = (2018, 2019, 2020),
    max_tickers: int = 30,
    feature_set: str = "v2",
    label_horizon: int = 1,
    label_task: str = "classification",
    label_thresh: float = 0.05,
    verbose: bool = True,
) -> pl.DataFrame:
    df = build_dataset(
        years=years,
        market=market,
        exchanges=None,
        tickers=None,
        max_tickers=max_tickers,
        start=None,
        end=None,
        feature_set=feature_set,
        label_horizon=label_horizon,
        label_task=label_task,
        label_thresh=label_thresh,
        select_cols=None,
        drop_na_rows=True,
        verbose=verbose,
    )
    # 분류 라벨과 회귀 타깃이 함께 필요하니 horizon=1일의 futret_1도 존재해야 함
    # build_dataset는 always futret_{horizon}를 생성
    return df

def run_analytics(
    market: str = "KR",
    years_train: Iterable[int] = (2018, 2019, 2020),
    max_tickers: int = 30,
    feature_set: str = "v2",
    target_col: str = "futret_1",
    event_thresh: float = 0.05,
    topk_plots: int = 8,
):
    root = Path(__file__).resolve().parent
    out_dir = _ensure_outdir(root / "outputs" / "M001")
    plots_dir = _ensure_outdir(out_dir / "plots")
    tables_dir = _ensure_outdir(out_dir / "tables")

    # 1) 데이터셋 로드
    df = build_train_frame(
        market=market,
        years=years_train,
        max_tickers=max_tickers,
        feature_set=feature_set,
        label_horizon=1,
        label_task="classification",
        label_thresh=event_thresh,
        verbose=False,
    )

    # 2) 피처 컬럼 선택
    feature_cols = _pick_feature_cols(df, target_col=target_col)
    if len(feature_cols) == 0:
        raise RuntimeError("No feature columns detected.")

    # 3) 교차섹션 IC 계산(+ 롤링)
    ic_long, ic_summary = compute_cs_ic(df, feature_cols, target_col=target_col)

    # 4) 결과 저장
    ic_long.write_csv(str(tables_dir / "ic_by_date_long.csv"))
    ic_summary.write_csv(str(tables_dir / "ic_summary.csv"))

    # 5) 이벤트 집중 산점도
    ev_df = event_filter(df, target_col=target_col, thresh=event_thresh)
    # 안정적/유의성이 높은 피처 선정: IR 상위 우선
    top_features = (
        ic_summary.filter(pl.col("n_days") >= 30)
                  .sort("ic_ir", descending=True)
                  .head(topk_plots)
                  .get_column("feature")
                  .to_list()
    )

    # 산점도 저장
    for f in top_features:
        plot_scatter_event(ev_df, f, target_col, plots_dir / f"scatter_event_{f}.png", title_suffix="(event |Δ|≥5%)")
        plot_scatter_all(df, f, target_col, plots_dir / f"scatter_all_{f}.png", title_suffix="(all)")

    # 6) 이벤트 비율/요약 저장
    event_rate_by_date = (
        df.select([
            "date",
            (pl.col(target_col).abs() >= event_thresh).cast(pl.Float64).alias("is_event")
        ])
        .group_by("date")
        .agg(pl.col("is_event").mean().alias("event_rate"))
        .sort("date")
    )
    event_rate_by_date.write_csv(str(tables_dir / "event_rate_by_date.csv"))

    # 간단 로그
    print(f"[M001] features analyzed: {len(feature_cols)}")
    print(f"[M001] top features (by IC IR): {top_features}")
    print(f"[M001] outputs saved under: {out_dir}")

if __name__ == "__main__":
    # 기본 실행: KR, 2018–2020, v2, 최대 30티커, 이벤트 임계 5%
    sns.set_context("talk")
    sns.set_style("whitegrid")
    run_analytics(
        market="KR",
        years_train=(2018, 2019, 2020),
        max_tickers=30,
        feature_set="v2",
        target_col="futret_1",
        event_thresh=0.05,
        topk_plots=8,
    )
