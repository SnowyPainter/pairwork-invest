from __future__ import annotations

"""

종합 결론

feature별 이벤트 맞출 성공률
rel_range: 변동폭, 7%
vol/obv/parkinson/gk: 변동성/거래량 6%
atr: 방향 정확도는 50% 정도지만 단일 성공률은 매우 낮음

보통의 오르내림 등 미약한 변동성/거래량 증가만으로는 좋은 시그널이 아니지만 extreme zone(>2)에서는 이벤트와 근접한 수익률
* 극단값만 필터링, 다중 필터링(단순히 한 feature 값에 대한 기준이 아님)
* 따라서 일단 event detectection -> 방향 분류기 추가
* 이후 해당 피쳐들로 예측 성능 높히기

정리하자면, rel_range, vol, obv, atr은 "변동성 탐지"에 유의미, 그러나 정말 +인지 -인지 탐지에는 문제가 있음.
따라서 방향 정확도를 높이는 feature들을 선발해야함

경험상 One-Shot으로 +, -, else 3class 분류기를 만들고 성공도 해봤으나 클래스불균형+연속실패로 "좆된 적이 있음"

따라서, Event Detector + Direction Classifier, No-trade zone 확보 필요

"""

import os
from pathlib import Path
from typing import Iterable, List, Tuple

import polars as pl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

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

def compute_event_prediction_performance(df: pl.DataFrame, feature_cols: List[str], label_col: str = "label_1d_cls") -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    이벤트 예측 성능을 날짜별로 계산:
    각 날짜에서 피처 값이 높은/낮은 주식들이 실제 이벤트를 얼마나 잘 예측하는지 측정
    
    성능 지표:
    - Event Prediction Score: 피처 상위 20% vs 하위 20%의 이벤트 발생률 차이
    - Directional Accuracy: +이벤트 vs -이벤트 방향 예측 정확도
    
    반환:
      - perf_long: columns=[date, feature, event_pred_score, direction_accuracy]
      - perf_summary: per-feature summary(mean, std, success_rate)
    """
    performance_results = []
    
    # 각 날짜별로 이벤트 예측 성능 계산
    for date in df["date"].unique().sort():
        date_df = df.filter(pl.col("date") == date)
        
        if len(date_df) < 10:  # 최소 10개 주식 필요
            continue
            
        date_results = {"date": date}
        
        for feature in feature_cols:
            feature_data = date_df.select([feature, label_col]).to_pandas()
            feature_data = feature_data.dropna()
            
            if len(feature_data) < 10:
                continue
                
            # 피처 값 기준 상위/하위 20% 분할
            top_20_threshold = feature_data[feature].quantile(0.8)
            bottom_20_threshold = feature_data[feature].quantile(0.2)
            
            top_20_mask = feature_data[feature] >= top_20_threshold
            bottom_20_mask = feature_data[feature] <= bottom_20_threshold
            
            # 이벤트 발생률 계산 (0이 아닌 라벨 = 이벤트)
            top_20_event_rate = (feature_data.loc[top_20_mask, label_col] != 0).mean()
            bottom_20_event_rate = (feature_data.loc[bottom_20_mask, label_col] != 0).mean()
            
            # Event Prediction Score: 상위 20% vs 하위 20%의 이벤트 발생률 차이
            event_pred_score = top_20_event_rate - bottom_20_event_rate
            
            # Directional Accuracy: +이벤트 vs -이벤트 방향 예측
            pos_events = feature_data[feature_data[label_col] == 1]  # +5% 이상
            neg_events = feature_data[feature_data[label_col] == -1]  # -5% 이하
            
            direction_accuracy = 0.5  # 기본값
            if len(pos_events) > 0 and len(neg_events) > 0:
                pos_feature_mean = pos_events[feature].mean()
                neg_feature_mean = neg_events[feature].mean()
                
                # 양의 이벤트에서 피처가 더 높으면 방향 예측 성공
                if pos_feature_mean > neg_feature_mean:
                    direction_accuracy = 1.0
                else:
                    direction_accuracy = 0.0
            
            date_results[f"{feature}_event_pred_score"] = event_pred_score
            date_results[f"{feature}_direction_accuracy"] = direction_accuracy
        
        performance_results.append(date_results)
    
    # DataFrame으로 변환
    perf_df = pl.DataFrame(performance_results)
    
    # Long format으로 변환
    event_score_cols = [c for c in perf_df.columns if c.endswith("_event_pred_score")]
    direction_cols = [c for c in perf_df.columns if c.endswith("_direction_accuracy")]
    
    # Event prediction scores를 long format으로
    perf_long_list = []
    for score_col, dir_col in zip(event_score_cols, direction_cols):
        feature_name = score_col.replace("_event_pred_score", "")
        
        temp_df = perf_df.select([
            "date", 
            score_col, 
            dir_col
        ]).rename({
            score_col: "event_pred_score",
            dir_col: "direction_accuracy"
        }).with_columns([
            pl.lit(feature_name).alias("feature")
        ]).select(["date", "feature", "event_pred_score", "direction_accuracy"])
        
        perf_long_list.append(temp_df)
    
    if not perf_long_list:
        # 빈 DataFrame 반환
        perf_long = pl.DataFrame({
            "date": [], "feature": [], "event_pred_score": [], "direction_accuracy": []
        })
        perf_summary = pl.DataFrame({
            "feature": [], "n_days": [], "event_pred_mean": [], "direction_acc_mean": [], "success_rate": []
        })
        return perf_long, perf_summary
    
    perf_long = pl.concat(perf_long_list).drop_nulls()
    
    # 롤링 통계 추가
    perf_long = (
        perf_long.sort(["feature", "date"])
               .with_columns([
                    pl.col("event_pred_score").rolling_mean(30).over("feature").alias("event_pred_ma30"),
                    pl.col("direction_accuracy").rolling_mean(30).over("feature").alias("direction_acc_ma30"),
               ])
    )
    
    # 요약 통계
    perf_summary = (
        perf_long.group_by("feature")
               .agg([
                   pl.len().alias("n_days"),
                    pl.col("event_pred_score").mean().alias("event_pred_mean"),
                    pl.col("direction_accuracy").mean().alias("direction_acc_mean"),
                    (pl.col("event_pred_score") > 0).mean().alias("positive_pred_rate"),
               ])
               .with_columns([
                    # 성공률 = 이벤트 예측력 + 방향 정확도의 조합
                    (pl.col("event_pred_mean").abs() * pl.col("direction_acc_mean")).alias("success_rate")
               ])
                .sort("success_rate", descending=True)
    )
    
    return perf_long, perf_summary

def event_filter(df: pl.DataFrame, target_col: str = "futret_1", thresh: float = 0.05) -> pl.DataFrame:
    return df.filter(pl.col(target_col).abs() >= thresh)

def plot_scatter_overlay(df_all: pl.DataFrame, df_event: pl.DataFrame, feature: str, label_col: str, out_path: Path, success_rate: float = None):
    """전체 데이터와 이벤트 데이터를 겹쳐서 보여주는 산점도"""
    # futret_1을 Y축으로 사용 (시각화용)
    futret_col = "futret_1"
    pdf_all = df_all.select(["date","ticker", feature, futret_col]).to_pandas()
    pdf_event = df_event.select(["date","ticker", feature, futret_col]).to_pandas()
    
    plt.figure(figsize=(10, 6))
    
    # 전체 데이터 (연한 배경)
    sns.scatterplot(data=pdf_all, x=feature, y=futret_col, alpha=0.1, s=8, color='lightblue', label='All data')
    
    # 이벤트 데이터 (강조)
    sns.scatterplot(data=pdf_event, x=feature, y=futret_col, alpha=0.6, s=15, color='red', label='Event (|Δ|≥5%)')
    
    # 회귀선들
    sns.regplot(data=pdf_all, x=feature, y=futret_col, scatter=False, color="blue", lowess=True, 
                line_kws={"lw": 2, "alpha": 0.7}, label='All trend')
    sns.regplot(data=pdf_event, x=feature, y=futret_col, scatter=False, color="red", lowess=True, 
                line_kws={"lw": 2.5}, label='Event trend')
    
    # 이벤트 경계선 표시
    plt.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, linewidth=1)
    plt.axhline(y=-0.05, color='orange', linestyle='--', alpha=0.7, linewidth=1)
    plt.fill_between(plt.xlim(), -0.05, 0.05, alpha=0.1, color='gray', label='Non-event zone')
    
    # 제목과 레이블
    title = f"{feature} vs {futret_col}"
    if success_rate is not None:
        title += f" (Success Rate: {success_rate:.3f})"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(feature, fontsize=12)
    plt.ylabel(futret_col, fontsize=12)
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()

def plot_scatter_comparison_matrix(df_all: pl.DataFrame, df_event: pl.DataFrame, features: List[str], 
                                 target_col: str, out_path: Path, perf_summary: pl.DataFrame):
    """상위 지표들을 한 번에 비교할 수 있는 subplot 매트릭스"""
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # 성능 값들을 딕셔너리로 변환
    perf_dict = {row['feature']: row['success_rate'] for row in perf_summary.to_dicts()}
    
    for idx, feature in enumerate(features):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # futret_1을 Y축으로 사용
        futret_col = "futret_1"
        pdf_all = df_all.select([feature, futret_col]).to_pandas()
        pdf_event = df_event.select([feature, futret_col]).to_pandas()
        
        # 전체 데이터
        ax.scatter(pdf_all[feature], pdf_all[futret_col], alpha=0.1, s=4, color='lightblue')
        # 이벤트 데이터
        ax.scatter(pdf_event[feature], pdf_event[futret_col], alpha=0.6, s=8, color='red')
        
        # 회귀선
        sns.regplot(data=pdf_event, x=feature, y=futret_col, scatter=False, color="red", 
                   lowess=True, line_kws={"lw": 2}, ax=ax)
        
        # 이벤트 경계선
        ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5)
        ax.axhline(y=-0.05, color='orange', linestyle='--', alpha=0.5)
        
        # 제목
        perf_val = perf_dict.get(feature, 0)
        ax.set_title(f"{feature}\n(Success: {perf_val:.3f})", fontsize=10, fontweight='bold')
        ax.set_xlabel(feature, fontsize=9)
        ax.set_ylabel(futret_col, fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 빈 subplot 제거
    for idx in range(n_features, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])
    
    plt.suptitle(f"Top {n_features} Features Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()

def plot_performance_timeseries(perf_long: pl.DataFrame, top_features: List[str], out_path: Path):
    """상위 지표들의 이벤트 예측 성능 시계열 차트"""
    perf_data = perf_long.filter(pl.col("feature").is_in(top_features)).to_pandas()
    
    plt.figure(figsize=(15, 10))
    
    # 각 지표별로 서브플롯 생성
    n_features = len(top_features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    for idx, feature in enumerate(top_features):
        plt.subplot(n_rows, n_cols, idx + 1)
        
        feature_data = perf_data[perf_data['feature'] == feature]
        feature_data['date'] = pd.to_datetime(feature_data['date'])
        feature_data = feature_data.sort_values('date')
        
        # 이벤트 예측 성능 시계열
        plt.plot(feature_data['date'], feature_data['event_pred_score'], alpha=0.6, color='blue', linewidth=1, label='Event Pred Score')
        # 방향 정확도 시계열
        plt.plot(feature_data['date'], feature_data['direction_accuracy'], alpha=0.6, color='green', linewidth=1, label='Direction Accuracy')
        
        # 30일 이동평균
        if 'event_pred_ma30' in feature_data.columns:
            plt.plot(feature_data['date'], feature_data['event_pred_ma30'], color='red', linewidth=2, label='30d MA')
        
        # 기준선들
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Random (0.5)')
        
        plt.title(f"{feature} Event Prediction Performance", fontweight='bold')
        plt.ylabel('Performance Score')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.ylim(-0.1, 1.1)
        
        if idx == 0:
            plt.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()

def plot_event_rate_timeseries(event_rate_data: pl.DataFrame, out_path: Path):
    """이벤트 발생률 시계열 차트"""
    pdf = event_rate_data.to_pandas()
    pdf['date'] = pd.to_datetime(pdf['date'])
    pdf = pdf.sort_values('date')
    
    plt.figure(figsize=(15, 6))
    
    # 이벤트율 시계열
    plt.plot(pdf['date'], pdf['event_rate'], color='red', linewidth=1.5, alpha=0.8)
    plt.fill_between(pdf['date'], pdf['event_rate'], alpha=0.3, color='red')
    
    # 평균선
    mean_rate = pdf['event_rate'].mean()
    plt.axhline(y=mean_rate, color='blue', linestyle='--', linewidth=2, 
                label=f'Average: {mean_rate:.3f}')
    
    # 고변동성 구간 표시 (상위 10%)
    high_vol_threshold = pdf['event_rate'].quantile(0.9)
    plt.axhline(y=high_vol_threshold, color='orange', linestyle='--', linewidth=2,
                label=f'90th percentile: {high_vol_threshold:.3f}')
    
    plt.title('Event Rate Time Series (Daily % of stocks with |return| ≥ 5%)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Event Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()

def plot_feature_correlation_heatmap(df: pl.DataFrame, top_features: List[str], out_path: Path):
    """상위 지표들 간의 상관관계 히트맵"""
    corr_data = df.select(top_features).to_pandas().corr()
    
    plt.figure(figsize=(10, 8))
    
    # 마스크 생성 (상삼각형 숨기기)
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    
    # 히트맵
    sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                fmt='.3f', annot_kws={'size': 9})
    
    plt.title('Feature Correlation Heatmap (Top Performers)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()

def plot_performance_summary_bar(perf_summary: pl.DataFrame, out_path: Path):
    """지표별 이벤트 예측 성능 요약 바 차트"""
    # 상위 15개 지표만 표시
    top_data = perf_summary.head(15).to_pandas()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Success Rate 바 차트
    bars1 = ax1.barh(range(len(top_data)), top_data['success_rate'], 
                     color=['green' if x >= 0.6 else 'orange' if x >= 0.4 else 'red' 
                           for x in top_data['success_rate']])
    ax1.set_yticks(range(len(top_data)))
    ax1.set_yticklabels(top_data['feature'], fontsize=10)
    ax1.set_xlabel('Success Rate (Event Pred × Direction Acc)', fontsize=12)
    ax1.set_title('Event Prediction Success Rate', fontsize=14, fontweight='bold')
    ax1.axvline(x=0.6, color='green', linestyle='--', alpha=0.7, label='Good (≥0.6)')
    ax1.axvline(x=0.4, color='orange', linestyle='--', alpha=0.7, label='Fair (≥0.4)')
    ax1.axvline(x=0.25, color='red', linestyle='--', alpha=0.7, label='Random (0.25)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 값 레이블 추가
    for i, v in enumerate(top_data['success_rate']):
        ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    # Direction Accuracy 바 차트
    bars2 = ax2.barh(range(len(top_data)), top_data['direction_acc_mean'], 
                     color=['darkgreen' if x >= 0.7 else 'darkblue' if x >= 0.6 else 'gray' 
                           for x in top_data['direction_acc_mean']])
    ax2.set_yticks(range(len(top_data)))
    ax2.set_yticklabels(top_data['feature'], fontsize=10)
    ax2.set_xlabel('Direction Accuracy', fontsize=12)
    ax2.set_title('Event Direction Prediction Accuracy', fontsize=14, fontweight='bold')
    ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Random (0.5)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 값 레이블 추가
    for i, v in enumerate(top_data['direction_acc_mean']):
        ax2.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()

def build_train_frame(
    market: str = "KR",
    years: Iterable[int] = (2018, 2019, 2020),
    max_tickers: int = 100,
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
    label_col: str = "label_1d_cls",
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
        verbose=True,
    )

    # 2) 피처 컬럼 선택
    feature_cols = _pick_feature_cols(df, target_col=label_col)
    if len(feature_cols) == 0:
        raise RuntimeError("No feature columns detected.")

    # 3) 이벤트 예측 성능 분석
    print(f"[M001] Computing event prediction performance...")
    perf_long, perf_summary = compute_event_prediction_performance(df, feature_cols, label_col=label_col)
    
    # 결과 저장
    perf_long.write_csv(str(tables_dir / "event_prediction_performance.csv"))
    perf_summary.write_csv(str(tables_dir / "event_prediction_summary.csv"))
    
    # 성능 딕셔너리 생성
    perf_dict = {row['feature']: row['success_rate'] for row in perf_summary.to_dicts()}

    # 4) 이벤트 데이터 필터링 (시각화용)
    ev_df = df.filter(pl.col(label_col) != 0)  # 0이 아닌 라벨 = 이벤트

    # 5) 최고 성능 피처들 선정
    top_features = (
        perf_summary.filter(pl.col("n_days") >= 30)
                   .sort("success_rate", descending=True)
                  .head(topk_plots)
                  .get_column("feature")
                  .to_list()
    )

    # 6) 이벤트 비율/요약 저장
    event_rate_by_date = (
        df.select([
            "date",
            (pl.col(label_col) != 0).cast(pl.Float64).alias("is_event")
        ])
        .group_by("date")
        .agg(pl.col("is_event").mean().alias("event_rate"))
        .sort("date")
    )
    event_rate_by_date.write_csv(str(tables_dir / "event_rate_by_date.csv"))

    # 7) 추가 시각화들
    print(f"[M001] Generating time series and summary plots...")
    
    # 성능 시계열 차트 (IC 대신 이벤트 예측 성능)
    plot_performance_timeseries(perf_long, top_features, plots_dir / "performance_timeseries_top_features.png")
    
    # 이벤트율 시계열 차트
    plot_event_rate_timeseries(event_rate_by_date, plots_dir / "event_rate_timeseries.png")
    
    # 지표 간 상관관계 히트맵
    plot_feature_correlation_heatmap(df, top_features, plots_dir / "feature_correlation_heatmap.png")
    
    # 성능 요약 바 차트
    plot_performance_summary_bar(perf_summary, plots_dir / "performance_summary_bars.png")

    # 8) 추가 통계 테이블 생성
    print(f"[M001] Generating additional analysis tables...")
    
    # 개별 overlay 산점도 (전체 + 이벤트 겹쳐서)
    for f in top_features:
        plot_scatter_overlay(df, ev_df, f, label_col, 
                           plots_dir / f"scatter_overlay_{f}.png", 
                           success_rate=perf_dict.get(f))

    # 상위 지표들 비교 매트릭스
    plot_scatter_comparison_matrix(df, ev_df, top_features, label_col, 
                                 plots_dir / "comparison_matrix_top_features.png", 
                                 perf_summary)
    
    # 월별 성능 분석은 제거 (이미 위에서 처리됨)
    # 월별 성능 분석
    monthly_performance = (
        perf_long.with_columns([
            pl.col("date").dt.strftime("%Y-%m").alias("year_month")
        ])
        .group_by(["year_month", "feature"])
        .agg([
            pl.col("event_pred_score").mean().alias("monthly_event_pred_mean"),
            pl.col("direction_accuracy").mean().alias("monthly_direction_acc_mean"),
            pl.len().alias("days_in_month")
        ])
        .with_columns([
            (pl.col("monthly_event_pred_mean").abs() * pl.col("monthly_direction_acc_mean")).alias("monthly_success_rate")
        ])
        .sort(["year_month", "monthly_success_rate"], descending=[False, True])
    )
    monthly_performance.write_csv(str(tables_dir / "monthly_performance.csv"))

    # 간단 로그
    print(f"[M001] ✅ Analysis complete!")
    print(f"[M001] Features analyzed: {len(feature_cols)}")
    print(f"[M001] Top features (by Success Rate): {top_features}")
    print(f"[M001] Generated files:")
    print(f"  📊 {len(top_features)} overlay scatter plots")
    print(f"  📈 1 comparison matrix")
    print(f"  📉 4 summary/timeseries charts")
    print(f"  📋 4 analysis tables")
    print(f"[M001] All outputs saved under: {out_dir}")

if __name__ == "__main__":
    # 기본 실행: KR, 2018–2020, v2, 최대 30티커, 이벤트 임계 5%
    sns.set_context("talk")
    sns.set_style("whitegrid")
    run_analytics(
        market="KR",
        years_train=(2018, 2019, 2020),
        max_tickers=30,
        feature_set="v2",
        label_col="label_1d_cls",
        event_thresh=0.05,
        topk_plots=8,
    )
