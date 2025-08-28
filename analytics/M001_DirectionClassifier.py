# M001_DirectionClassifier.py
"""
방향 분류기 Feature 분석 모듈

이 모듈은 올라갈것인가? 내려갈 것인가?를 예측하는 방향 분류 모델을 위한
feature 선별 및 분석을 수행합니다.

주요 기능:
- 방향 예측 성능 측정 (Positive vs Negative event prediction)
- Feature별 방향 예측 정확도 분석
- 방향 예측에 특화된 시각화
- 상위 성능 Feature 선별 및 상관관계 분석
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import polars as pl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from data.dataset_builder import build_dataset

BASE_COLS = {
    "date","ticker","market","exchange","currency","year",
    "open","high","low","close","adj_close","volume","turnover",
}

def _ensure_outdir(path: Path) -> Path:
    """출력 디렉토리 생성"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def _pick_feature_cols(df: pl.DataFrame, target_col: str) -> List[str]:
    """Feature 컬럼 선택 (기본 컬럼과 타깃 컬럼 제외)"""
    cols = []
    for c, dt in df.schema.items():
        if c in BASE_COLS:
            continue
        if c == target_col or c.startswith("label_") or c.startswith("futret_"):
            continue
        if dt in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64):
            cols.append(c)
    return cols

def compute_direction_prediction_performance(df: pl.DataFrame, feature_cols: List[str], label_col: str = "label_1d_cls") -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    방향 예측 성능을 날짜별로 계산

    각 날짜에서 피처 값이 +이벤트와 -이벤트를 얼마나 잘 구분하는지 측정

    성능 지표:
    - Direction Prediction Score: +이벤트와 -이벤트 구분력
    - Direction Accuracy: 방향 예측 정확도
    - Class Separation: 클래스별 피처 값 분포 차이

    반환:
      - perf_long: columns=[date, feature, direction_pred_score, direction_accuracy, class_separation]
      - perf_summary: per-feature summary(mean, std, success_rate)
    """
    performance_results = []

        # 각 날짜별로 방향 예측 성능 계산
    for date in df["date"].unique().sort():
        try:
            date_df = df.filter(pl.col("date") == date)

            if len(date_df) < 20:  # 최소 20개 주식 필요
                continue

            # 이벤트만 필터링 (+1 또는 -1)
            event_df = date_df.filter(pl.col(label_col) != 0)
            if len(event_df) < 10:
                continue

            date_results = {"date": date}

            for feature in feature_cols:
                try:
                    feature_data = event_df.select([feature, label_col]).to_pandas()
                    feature_data = feature_data.dropna()

                    if len(feature_data) < 10:
                        continue

                    # +이벤트와 -이벤트 분리
                    pos_events = feature_data[feature_data[label_col] == 1]
                    neg_events = feature_data[feature_data[label_col] == -1]

                    if len(pos_events) < 3 or len(neg_events) < 3:
                        continue

                    # 방향 예측 점수 계산
                    pos_mean = pos_events[feature].mean()
                    neg_mean = neg_events[feature].mean()
                    pos_std = pos_events[feature].std()
                    neg_std = neg_events[feature].std()

                    # Direction Prediction Score: 클래스 평균 차이 (표준화)
                    mean_diff = abs(pos_mean - neg_mean)
                    avg_std = (pos_std + neg_std) / 2
                    direction_pred_score = mean_diff / (avg_std + 1e-9)

                    # Direction Accuracy: 평균 기반 방향 예측 정확도
                    correct_predictions = 0
                    total_predictions = len(feature_data)

                    for _, row in feature_data.iterrows():
                        pred_direction = 1 if row[feature] > (pos_mean + neg_mean) / 2 else -1
                        if pred_direction == row[label_col]:
                            correct_predictions += 1

                    direction_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.5

                    # Class Separation: 클래스 간 분포 분리도
                    class_separation = abs(pos_mean - neg_mean) / (pos_std + neg_std + 1e-9)

                    date_results[f"{feature}_direction_pred_score"] = direction_pred_score
                    date_results[f"{feature}_direction_accuracy"] = direction_accuracy
                    date_results[f"{feature}_class_separation"] = class_separation

                except Exception as e:
                    print(f"⚠️ Error processing {feature} on {date}: {e}")
                    continue

            performance_results.append(date_results)

        except Exception as e:
            print(f"⚠️ Error processing date {date}: {e}")
            continue

    # DataFrame으로 변환
    if not performance_results:
        # 빈 DataFrame 반환
        perf_long = pl.DataFrame({
            "date": [], "feature": [], "direction_pred_score": [], "direction_accuracy": [], "class_separation": []
        })
        perf_summary = pl.DataFrame({
            "feature": [], "n_days": [], "direction_pred_mean": [], "direction_acc_mean": [], "class_sep_mean": [], "success_rate": []
        })
        return perf_long, perf_summary

    perf_df = pl.DataFrame(performance_results)

    # Long format으로 변환
    direction_score_cols = [c for c in perf_df.columns if c.endswith("_direction_pred_score")]
    direction_acc_cols = [c for c in perf_df.columns if c.endswith("_direction_accuracy")]
    class_sep_cols = [c for c in perf_df.columns if c.endswith("_class_separation")]

    perf_long_list = []
    for score_col, acc_col, sep_col in zip(direction_score_cols, direction_acc_cols, class_sep_cols):
        feature_name = score_col.replace("_direction_pred_score", "")

        temp_df = perf_df.select([
            "date",
            score_col,
            acc_col,
            sep_col
        ]).rename({
            score_col: "direction_pred_score",
            acc_col: "direction_accuracy",
            sep_col: "class_separation"
        }).with_columns([
            pl.lit(feature_name).alias("feature")
        ]).select(["date", "feature", "direction_pred_score", "direction_accuracy", "class_separation"])

        perf_long_list.append(temp_df)

    if not perf_long_list:
        perf_long = pl.DataFrame({
            "date": [], "feature": [], "direction_pred_score": [], "direction_accuracy": [], "class_separation": []
        })
        perf_summary = pl.DataFrame({
            "feature": [], "n_days": [], "direction_pred_mean": [], "direction_acc_mean": [], "class_sep_mean": [], "success_rate": []
        })
        return perf_long, perf_summary

    perf_long = pl.concat(perf_long_list).drop_nulls()

    # 롤링 통계 추가
    perf_long = (
        perf_long.sort(["feature", "date"])
               .with_columns([
                    pl.col("direction_pred_score").rolling_mean(30).over("feature").alias("direction_pred_ma30"),
                    pl.col("direction_accuracy").rolling_mean(30).over("feature").alias("direction_acc_ma30"),
                    pl.col("class_separation").rolling_mean(30).over("feature").alias("class_sep_ma30"),
               ])
    )

    # 요약 통계
    perf_summary = (
        perf_long.group_by("feature")
               .agg([
                   pl.len().alias("n_days"),
                   pl.col("direction_pred_score").mean().alias("direction_pred_mean"),
                   pl.col("direction_accuracy").mean().alias("direction_acc_mean"),
                   pl.col("class_separation").mean().alias("class_sep_mean"),
               ])
               .with_columns([
                   # 종합 성공률: 방향 예측 정확도 × 클래스 분리도
                   (pl.col("direction_acc_mean") * pl.col("class_sep_mean")).alias("success_rate")
               ])
               .sort("success_rate", descending=True)
    )

    return perf_long, perf_summary

def plot_direction_separation_analysis(df: pl.DataFrame, feature: str, label_col: str, out_path: Path, success_rate: float = None):
    """+이벤트와 -이벤트의 피처 분포를 비교하는 분석 차트"""
    # 이벤트만 필터링
    event_df = df.filter(pl.col(label_col) != 0)
    pdf = event_df.select([feature, label_col]).to_pandas()
    pdf = pdf.dropna()

    if len(pdf) < 10:
        print(f"⚠️ Skipping {feature}: insufficient data ({len(pdf)} samples)")
        return

    # 각 클래스별 데이터 확인
    pos_data = pdf[pdf[label_col] == 1][feature]
    neg_data = pdf[pdf[label_col] == -1][feature]

    if len(pos_data) == 0 or len(neg_data) == 0:
        print(f"⚠️ Skipping {feature}: missing positive or negative data")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 박스플롯: 클래스별 분포 비교
    bp_data = [pos_data, neg_data]
    ax1.boxplot(bp_data, labels=['+Event', '-Event'], patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax1.set_title(f'{feature} Distribution by Direction', fontweight='bold')
    ax1.set_ylabel(feature)
    ax1.grid(True, alpha=0.3)

    # 2. KDE 플롯: 밀도 분포 비교
    if len(pos_data) > 5 and len(neg_data) > 5:
        try:
            sns.kdeplot(data=pos_data, ax=ax2, label='+Event', color='green', fill=True, alpha=0.3)
            sns.kdeplot(data=neg_data, ax=ax2, label='-Event', color='red', fill=True, alpha=0.3)

            # 평균선 추가 (데이터가 있는 경우에만)
            if len(pos_data) > 0:
                ax2.axvline(pos_data.mean(), color='green', linestyle='--', alpha=0.7, label='+Mean')
            if len(neg_data) > 0:
                ax2.axvline(neg_data.mean(), color='red', linestyle='--', alpha=0.7, label='-Mean')
        except Exception as e:
            print(f"⚠️ KDE plot failed for {feature}: {e}")
            ax2.text(0.5, 0.5, 'KDE plot failed', transform=ax2.transAxes, ha='center')

    ax2.set_title(f'{feature} Density by Direction', fontweight='bold')
    ax2.set_xlabel(feature)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 스캐터 플롯: 피처 vs 수익률
    futret_col = "futret_1"
    if futret_col in df.columns:
        scatter_df = event_df.select([feature, futret_col, label_col]).to_pandas()
        colors = ['green' if l == 1 else 'red' for l in scatter_df[label_col]]
        ax3.scatter(scatter_df[feature], scatter_df[futret_col], c=colors, alpha=0.6, s=20)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel(feature)
        ax3.set_ylabel(futret_col)
        ax3.set_title('Feature vs Return (by Direction)', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 범례 추가
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='+Event'),
            Patch(facecolor='red', label='-Event')
        ]
        ax3.legend(handles=legend_elements)

    # 4. Confusion Matrix 기반 방향 예측 성능
    if len(pos_data) > 0 and len(neg_data) > 0 and len(pos_data) > 3 and len(neg_data) > 3:
        try:
            # 간단한 threshold 기반 예측
            threshold = (pos_data.mean() + neg_data.mean()) / 2

            y_true = pdf[label_col]
            y_pred = [1 if x > threshold else -1 for x in pdf[feature]]

            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                       xticklabels=['-Event', '+Event'], yticklabels=['-Event', '+Event'])
            ax4.set_title(f'Direction Prediction Confusion Matrix\nThreshold: {threshold:.3f}', fontweight='bold')
            ax4.set_ylabel('True Direction')
            ax4.set_xlabel('Predicted Direction')
        except Exception as e:
            print(f"⚠️ Confusion matrix failed for {feature}: {e}")
            ax4.text(0.5, 0.5, 'Confusion matrix failed', transform=ax4.transAxes, ha='center')
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for\nconfusion matrix', transform=ax4.transAxes, ha='center')

    # 전체 제목
    title = f"Direction Classification Analysis: {feature}"
    if success_rate is not None:
        title += f" (Success Rate: {success_rate:.3f})"
    fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()

def plot_direction_performance_timeseries(perf_long: pl.DataFrame, top_features: List[str], out_path: Path):
    """상위 지표들의 방향 예측 성능 시계열 차트"""
    perf_data = perf_long.filter(pl.col("feature").is_in(top_features)).to_pandas()

    plt.figure(figsize=(15, 12))

    # 각 지표별로 서브플롯 생성
    n_features = len(top_features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols

    for idx, feature in enumerate(top_features):
        plt.subplot(n_rows, n_cols, idx + 1)

        feature_data = perf_data[perf_data['feature'] == feature]
        feature_data['date'] = pd.to_datetime(feature_data['date'])
        feature_data = feature_data.sort_values('date')

        # 방향 예측 성능 시계열
        plt.plot(feature_data['date'], feature_data['direction_accuracy'], alpha=0.8, color='blue', linewidth=2, label='Direction Accuracy')
        plt.plot(feature_data['date'], feature_data['direction_pred_score'], alpha=0.6, color='green', linewidth=1, label='Direction Pred Score')
        plt.plot(feature_data['date'], feature_data['class_separation'], alpha=0.6, color='orange', linewidth=1, label='Class Separation')

        # 30일 이동평균
        if 'direction_acc_ma30' in feature_data.columns:
            plt.plot(feature_data['date'], feature_data['direction_acc_ma30'], color='red', linewidth=2, label='30d MA (Acc)')

        # 기준선들
        plt.axhline(y=0.5, color='black', linestyle='-', alpha=0.5, label='Random (0.5)')
        plt.axhline(y=0.6, color='green', linestyle='--', alpha=0.7, label='Good (0.6)')

        plt.title(f"{feature} Direction Prediction", fontweight='bold')
        plt.ylabel('Performance Score')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)

        if idx == 0:
            plt.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()

def plot_direction_feature_correlation_heatmap(df: pl.DataFrame, top_features: List[str], out_path: Path):
    """방향 예측 상위 지표들 간의 상관관계 히트맵"""
    # 이벤트 데이터만 사용
    event_df = df.filter(pl.col("label_1d_cls") != 0)
    corr_data = event_df.select(top_features).to_pandas().corr()

    plt.figure(figsize=(12, 10))

    # 마스크 생성 (상삼각형 숨기기)
    mask = np.triu(np.ones_like(corr_data, dtype=bool))

    # 히트맵
    sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                fmt='.3f', annot_kws={'size': 9})

    plt.title('Direction Classification Feature Correlation Heatmap\n(Events Only)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()

def plot_direction_performance_summary(perf_summary: pl.DataFrame, out_path: Path):
    """방향 예측 성능 요약 바 차트"""
    # 상위 15개 지표만 표시
    top_data = perf_summary.head(15).to_pandas()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

    # 1. Success Rate 바 차트
    bars1 = ax1.barh(range(len(top_data)), top_data['success_rate'],
                     color=['darkgreen' if x >= 0.4 else 'orange' if x >= 0.25 else 'red'
                           for x in top_data['success_rate']])
    ax1.set_yticks(range(len(top_data)))
    ax1.set_yticklabels(top_data['feature'], fontsize=9)
    ax1.set_xlabel('Success Rate (Acc × Separation)', fontsize=12)
    ax1.set_title('Direction Classification Success Rate', fontsize=14, fontweight='bold')
    ax1.axvline(x=0.4, color='darkgreen', linestyle='--', alpha=0.7, label='Excellent (≥0.4)')
    ax1.axvline(x=0.25, color='orange', linestyle='--', alpha=0.7, label='Good (≥0.25)')
    ax1.axvline(x=0.15, color='red', linestyle='--', alpha=0.7, label='Random (0.15)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')

    # 값 레이블 추가
    for i, v in enumerate(top_data['success_rate']):
        ax1.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)

    # 2. Direction Accuracy 바 차트
    bars2 = ax2.barh(range(len(top_data)), top_data['direction_acc_mean'],
                     color=['darkblue' if x >= 0.65 else 'blue' if x >= 0.55 else 'lightblue'
                           for x in top_data['direction_acc_mean']])
    ax2.set_yticks(range(len(top_data)))
    ax2.set_yticklabels(top_data['feature'], fontsize=9)
    ax2.set_xlabel('Direction Accuracy', fontsize=12)
    ax2.set_title('Direction Prediction Accuracy', fontsize=14, fontweight='bold')
    ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Random (0.5)')
    ax2.axvline(x=0.6, color='orange', linestyle='--', alpha=0.7, label='Good (0.6)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')

    # 값 레이블 추가
    for i, v in enumerate(top_data['direction_acc_mean']):
        ax2.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)

    # 3. Class Separation 바 차트
    bars3 = ax3.barh(range(len(top_data)), top_data['class_sep_mean'],
                     color=['purple' if x >= 1.0 else 'mediumpurple' if x >= 0.5 else 'thistle'
                           for x in top_data['class_sep_mean']])
    ax3.set_yticks(range(len(top_data)))
    ax3.set_yticklabels(top_data['feature'], fontsize=9)
    ax3.set_xlabel('Class Separation', fontsize=12)
    ax3.set_title('Class Separation Strength', fontsize=14, fontweight='bold')
    ax3.axvline(x=0.5, color='purple', linestyle='--', alpha=0.7, label='Strong (≥0.5)')
    ax3.axvline(x=0.25, color='mediumpurple', linestyle='--', alpha=0.7, label='Moderate (≥0.25)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')

    # 값 레이블 추가
    for i, v in enumerate(top_data['class_sep_mean']):
        ax3.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=8)

    plt.suptitle('Direction Classification Feature Performance Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()

def plot_direction_comparison_matrix(df: pl.DataFrame, features: List[str], label_col: str, out_path: Path, perf_summary: pl.DataFrame):
    """상위 지표들을 한 번에 비교할 수 있는 방향 예측 매트릭스"""
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # 이벤트 데이터만 사용
    event_df = df.filter(pl.col(label_col) != 0)

    # 성능 값들을 딕셔너리로 변환
    perf_dict = {row['feature']: row['success_rate'] for row in perf_summary.to_dicts()}

    for idx, feature in enumerate(features):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        try:
            # +이벤트와 -이벤트 분포 비교
            pos_data = event_df.filter(pl.col(label_col) == 1).select(feature).to_pandas()[feature].dropna()
            neg_data = event_df.filter(pl.col(label_col) == -1).select(feature).to_pandas()[feature].dropna()

            if len(pos_data) > 5 and len(neg_data) > 5:
                # 박스플롯
                bp_data = [pos_data, neg_data]
                ax.boxplot(bp_data, labels=['+Event', '-Event'], patch_artist=True,
                          boxprops=dict(facecolor=['lightgreen', 'lightcoral'], alpha=0.7),
                          medianprops=dict(color='red', linewidth=1.5))

                # 평균선 추가 (데이터가 있는 경우에만)
                if len(pos_data) > 0:
                    ax.axhline(pos_data.mean(), color='green', linestyle='--', alpha=0.7, linewidth=1)
                if len(neg_data) > 0:
                    ax.axhline(neg_data.mean(), color='red', linestyle='--', alpha=0.7, linewidth=1)

            elif len(pos_data) == 0 or len(neg_data) == 0:
                ax.text(0.5, 0.5, 'Missing data\nfor one class', transform=ax.transAxes, ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'Insufficient\ndata', transform=ax.transAxes, ha='center', va='center')

        except Exception as e:
            print(f"⚠️ Error plotting {feature}: {e}")
            ax.text(0.5, 0.5, f'Error:\n{feature}', transform=ax.transAxes, ha='center', va='center')

        # 제목
        perf_val = perf_dict.get(feature, 0)
        ax.set_title(f"{feature}\n(Success: {perf_val:.3f})", fontsize=11, fontweight='bold')
        ax.set_ylabel(feature, fontsize=9)
        ax.grid(True, alpha=0.3)

    # 빈 subplot 제거
    for idx in range(n_features, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    plt.suptitle('Direction Classification Feature Comparison Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()

def analyze_feature_importance_for_direction(df: pl.DataFrame, feature_cols: List[str], label_col: str, out_path: Path):
    """방향 예측을 위한 피처 중요도 분석"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    try:
        # 이벤트 데이터만 사용
        event_df = df.filter(pl.col(label_col) != 0)
        if event_df.is_empty():
            print("⚠️ No event data found for feature importance analysis")
            return None

        feature_data = event_df.select(feature_cols + [label_col]).to_pandas().dropna()

        if len(feature_data) < 50:
            print(f"⚠️ Not enough data for feature importance analysis ({len(feature_data)} samples)")
            return None

        # 클래스 균형 확인
        class_counts = feature_data[label_col].value_counts()
        if len(class_counts) < 2:
            print("⚠️ Need both positive and negative classes for feature importance analysis")
            return None

        min_class_samples = class_counts.min()
        if min_class_samples < 10:
            print(f"⚠️ Insufficient samples in minority class ({min_class_samples} samples)")
            return None

        X = feature_data[feature_cols]
        y = feature_data[label_col]

        # 데이터 스케일링
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Random Forest 모델 학습
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_scaled, y)

        # 피처 중요도 추출
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        # 시각화
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)

        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Direction Classification Feature Importance\n(Random Forest)', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        # 값 레이블 추가
        for i, v in enumerate(top_features['importance']):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(out_path, dpi=160, bbox_inches='tight')
        plt.close()

        # 중요도 데이터 저장
        feature_importance.to_csv(str(out_path.parent / "feature_importance_direction.csv"), index=False)

        return feature_importance

    except Exception as e:
        print(f"⚠️ Feature importance analysis failed: {e}")
        return None

def build_direction_classification_frame(
    market: str = "KR",
    years: Iterable[int] = (2018, 2019, 2020),
    max_tickers: int = 100,
    feature_set: str = "v2",
    label_horizon: int = 1,
    label_task: str = "classification",
    label_thresh: float = 0.05,
    verbose: bool = False,
) -> pl.DataFrame:
    """방향 분류를 위한 데이터셋 빌드"""
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
    return df

def run_direction_analysis(
    market: str = "KR",
    years_train: Iterable[int] = (2018, 2019, 2020),
    max_tickers: int = 50,
    feature_set: str = "v2",
    label_col: str = "label_1d_cls",
    direction_thresh: float = 0.05,
    topk_plots: int = 10,
):
    """방향 분류를 위한 종합 feature 분석 실행"""
    root = Path(__file__).resolve().parent
    out_dir = _ensure_outdir(root / "outputs" / "M001_Direction")
    plots_dir = _ensure_outdir(out_dir / "plots")
    tables_dir = _ensure_outdir(out_dir / "tables")

    print("🔍 Starting Direction Classification Feature Analysis...")

    # 1) 데이터셋 로드
    print(f"[Direction] Loading dataset for {market} market...")
    df = build_direction_classification_frame(
        market=market,
        years=years_train,
        max_tickers=max_tickers,
        feature_set=feature_set,
        label_horizon=1,
        label_task="classification",
        label_thresh=direction_thresh,
    )

    # 2) 피처 컬럼 선택
    feature_cols = _pick_feature_cols(df, target_col=label_col)
    if len(feature_cols) == 0:
        raise RuntimeError("No feature columns detected.")

    print(f"[Direction] Analyzing {len(feature_cols)} features...")

    # 3) 방향 예측 성능 분석
    print(f"[Direction] Computing direction prediction performance...")
    perf_long, perf_summary = compute_direction_prediction_performance(df, feature_cols, label_col=label_col)

    # 결과 저장
    perf_long.write_csv(str(tables_dir / "direction_prediction_performance.csv"))
    perf_summary.write_csv(str(tables_dir / "direction_prediction_summary.csv"))

    # 성능 딕셔너리 생성
    perf_dict = {row['feature']: row['success_rate'] for row in perf_summary.to_dicts()}

    # 4) 이벤트 데이터 필터링 (방향 분석용)
    direction_df = df.filter(pl.col(label_col) != 0)  # 0이 아닌 라벨 = 방향 이벤트

    # 5) 최고 성능 피처들 선정
    if perf_summary.is_empty():
        print("⚠️ No performance data available. Skipping detailed analysis.")
        return None, []

    top_features = (
        perf_summary.filter(pl.col("n_days") >= 30)
                   .sort("success_rate", descending=True)
                   .head(topk_plots)
                   .get_column("feature")
                   .to_list()
    )

    if len(top_features) == 0:
        print("⚠️ No features meet the criteria (n_days >= 30). Using all available features.")
        top_features = (
            perf_summary.sort("success_rate", descending=True)
                       .head(topk_plots)
                       .get_column("feature")
                       .to_list()
        )

    print(f"[Direction] Top {len(top_features)} features selected for detailed analysis")

    # 6) 이벤트 비율/요약 저장
    direction_rate_by_date = (
        df.select([
            "date",
            (pl.col(label_col) != 0).cast(pl.Float64).alias("is_direction_event"),
            (pl.col(label_col) == 1).cast(pl.Float64).alias("is_positive"),
            (pl.col(label_col) == -1).cast(pl.Float64).alias("is_negative")
        ])
        .group_by("date")
        .agg([
            pl.col("is_direction_event").mean().alias("direction_event_rate"),
            pl.col("is_positive").mean().alias("positive_rate"),
            pl.col("is_negative").mean().alias("negative_rate")
        ])
        .sort("date")
    )
    direction_rate_by_date.write_csv(str(tables_dir / "direction_event_rate_by_date.csv"))

    # 7) 시각화 생성
    print(f"[Direction] Generating visualization plots...")

    # 방향 예측 성능 시계열 차트
    plot_direction_performance_timeseries(perf_long, top_features, plots_dir / "direction_performance_timeseries.png")

    # 방향 예측 피처 상관관계 히트맵
    plot_direction_feature_correlation_heatmap(df, top_features, plots_dir / "direction_feature_correlation_heatmap.png")

    # 방향 예측 성능 요약 바 차트
    plot_direction_performance_summary(perf_summary, plots_dir / "direction_performance_summary.png")

    # 8) 개별 방향 분리 분석 차트
    print(f"[Direction] Generating individual direction separation plots...")
    for f in top_features[:8]:  # 상위 8개만
        plot_direction_separation_analysis(df, f, label_col,
                                         plots_dir / f"direction_separation_{f}.png",
                                         success_rate=perf_dict.get(f))

    # 9) 방향 예측 비교 매트릭스
    plot_direction_comparison_matrix(df, top_features, label_col,
                                   plots_dir / "direction_comparison_matrix.png",
                                   perf_summary)

    # 10) 피처 중요도 분석
    print(f"[Direction] Analyzing feature importance...")
    analyze_feature_importance_for_direction(df, feature_cols, label_col, plots_dir / "feature_importance_analysis.png")

    # 11) 월별 방향 예측 성능 분석
    monthly_direction_performance = (
        perf_long.with_columns([
            pl.col("date").dt.strftime("%Y-%m").alias("year_month")
        ])
        .group_by(["year_month", "feature"])
        .agg([
            pl.col("direction_accuracy").mean().alias("monthly_direction_acc_mean"),
            pl.col("direction_pred_score").mean().alias("monthly_direction_pred_mean"),
            pl.col("class_separation").mean().alias("monthly_class_sep_mean"),
            pl.len().alias("days_in_month")
        ])
        .with_columns([
            (pl.col("monthly_direction_acc_mean") * pl.col("monthly_class_sep_mean")).alias("monthly_success_rate")
        ])
        .sort(["year_month", "monthly_success_rate"], descending=[False, True])
    )
    monthly_direction_performance.write_csv(str(tables_dir / "monthly_direction_performance.csv"))

    # 12) 최종 결과 출력
    print("✅ Direction Classification Feature Analysis Complete!")
    print(f"[Direction] Features analyzed: {len(feature_cols)}")
    print(f"[Direction] Top features (by Success Rate): {top_features[:5]}")  # 상위 5개만 출력
    print(f"[Direction] Generated files:")
    print(f"  📊 {min(8, len(top_features))} direction separation analysis plots")
    print(f"  📈 1 direction comparison matrix")
    print(f"  📉 4 summary/timeseries charts")
    print(f"  📋 4 analysis tables")
    print(f"  🤖 1 feature importance analysis")
    print(f"[Direction] All outputs saved under: {out_dir}")

    # 13) 방향 예측을 위한 추천 피처 출력
    print("Recommended Features for Direction Classification:")
    recommended_features = (
        perf_summary.filter(
            (pl.col("n_days") >= 50) &
            (pl.col("success_rate") >= 0.25) &
            (pl.col("direction_acc_mean") >= 0.55)
        )
        .sort("success_rate", descending=True)
        .head(10)
    )

    if len(recommended_features) > 0:
        print("\nTop Recommended Features:")
        for row in recommended_features.to_dicts():
            print("2d")

    return perf_summary, top_features

if __name__ == "__main__":
    # 기본 실행: KR, 2018–2020, v2, 최대 50티커, 방향 임계 5%
    sns.set_context("talk")
    sns.set_style("whitegrid")

    run_direction_analysis(
        market="KR",
        years_train=(2018, 2019, 2020),
        max_tickers=50,
        feature_set="v2",
        label_col="label_1d_cls",
        direction_thresh=0.05,
        topk_plots=10
    )