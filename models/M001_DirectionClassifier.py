# models/M001_DirectionClassifier.py
"""
방향 분류 모델 (LightGBM 기반 이진 분류기)

다중공선성 문제를 고려하여 선별된 feature들을 사용하여
상승/하락 방향을 예측하는 모델입니다.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from data.dataset_builder import build_dataset

# === Feature Selection (다중공선성 고려) ===
SELECTED_FEATURES = [
    # RSI 그룹 (rsi14 선택 - 가장 안정적)
    'rsi14',

    # 이동평균 그룹 (ema20, sma50 선택 - 장단기 균형)
    'ema20',
    'sma50',

    # 수익률 그룹 (roc10 선택 - 적절한 기간)
    'roc10',

    # 스토캐스틱 그룹 (stochd14 선택 - smoothed 버전)
    'stochd14',

    # 거래량 그룹 (vol_z20 선택 - 표준화된 변동성)
    'vol_z20',

    # 변동성 그룹 (parkinson20 선택 - Parkinson 변동성)
    'parkinson20',

    # MACD 그룹 (macd_hist 선택 - 히스토그램)
    'macd_hist',

    # ATR 그룹 (atr14 선택 - 안정적)
    'atr14',

    # VWAP 그룹 (vwap20 선택 - 더 긴 기간)
    'vwap20',

    # 가격 구조 (pos_in_don20 선택 - Donchian position)
    'pos_in_don20',

    # OBV 그룹 (obv 선택 - On Balance Volume)
    'obv',

    # MFI 그룹 (mfi14 선택)
    'mfi14',

    # CCI 그룹 (cci20 선택)
    'cci20',

    # Williams %R (willr14 선택)
    'willr14'
]

class DirectionClassifierLGBM:
    """
    LightGBM 기반 방향 분류 모델

    Features: 다중공선성 고려하여 선별된 16개 feature
    Target: +1 (상승), -1 (하락) 이진 분류
    """

    def __init__(self,
                 model_params: Optional[Dict] = None,
                 feature_list: Optional[List[str]] = None):
        """
        Args:
            model_params: LightGBM 모델 파라미터
            feature_list: 사용할 feature 리스트 (기본값: SELECTED_FEATURES)
        """
        self.feature_list = feature_list or SELECTED_FEATURES
        self.model_params = model_params or self._get_default_params()
        self.model = None
        self.feature_importance = None
        self.training_metrics = {}

    def _get_default_params(self) -> Dict:
        """기본 LightGBM 파라미터"""
        return {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 100,
            'max_depth': 6,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }

    def load_data(self,
                  market: str = "KR",
                  years: List[int] = [2018, 2019, 2020],
                  max_tickers: int = 100,
                  feature_set: str = "v2",
                  label_horizon: int = 1,
                  label_thresh: float = 0.05) -> Tuple[pd.DataFrame, pd.Series]:
        """
        학습 데이터를 로드하고 전처리

        Args:
            market: 시장 코드
            years: 학습 연도
            max_tickers: 최대 티커 수
            feature_set: feature set
            label_horizon: 라벨 horizon
            label_thresh: 라벨 threshold

        Returns:
            X: feature 데이터프레임
            y: target 시리즈 (0: 상승, 1: 하락)
        """
        print(f"📊 Loading data for {market} market, years {years}...")

        # 데이터 로드
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
            label_task="classification",
            label_thresh=label_thresh,
            select_cols=None,
            drop_na_rows=True,
            verbose=False,
        )

        print(f"✅ Loaded {len(df)} samples, {len(df.columns)} columns")

        # 이벤트 데이터만 필터링 (label_1d_cls가 0이 아닌 경우)
        event_df = df.filter(pl.col("label_1d_cls") != 0)
        print(f"✅ Filtered to {len(event_df)} directional events")

        if len(event_df) < 1000:
            print(f"⚠️ Warning: Only {len(event_df)} samples available. Consider using more data.")

        # Feature와 Target 분리
        available_features = [f for f in self.feature_list if f in event_df.columns]
        missing_features = [f for f in self.feature_list if f not in event_df.columns]

        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            print(f"📊 Using {len(available_features)} available features: {available_features}")

        # pandas로 변환
        feature_df = event_df.select(available_features).to_pandas()
        target_series = event_df.select("label_1d_cls").to_pandas()["label_1d_cls"]

        # Target 변환: -1, 1 -> 0, 1
        y = ((target_series + 1) // 2).astype(int)

        print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
        print(f"📊 Features shape: {feature_df.shape}")

        return feature_df, y

    def train(self,
              X: pd.DataFrame,
              y: pd.Series,
              test_size: float = 0.2,
              use_cv: bool = True,
              cv_folds: int = 5) -> Dict:
        """
        모델 학습

        Args:
            X: feature 데이터
            y: target 데이터
            test_size: 테스트 세트 비율
            use_cv: 교차검증 사용 여부
            cv_folds: 교차검증 fold 수

        Returns:
            학습 결과 메트릭
        """
        print("🚀 Training Direction Classifier...")

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"📊 Train set: {X_train.shape}, Test set: {X_test.shape}")

        # LightGBM 데이터셋 생성
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # 모델 학습
        self.model = lgb.train(
            self.model_params,
            train_data,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(period=10)
            ]
        )

        # 예측
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # 메트릭 계산
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        # 교차검증 (옵션)
        if use_cv:
            cv_scores = cross_val_score(
                lgb.LGBMClassifier(**self.model_params),
                X, y, cv=cv_folds, scoring='accuracy'
            )
            metrics['cv_accuracy_mean'] = cv_scores.mean()
            metrics['cv_accuracy_std'] = cv_scores.std()

        # Feature Importance 저장
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        self.training_metrics = metrics

        print("✅ Training completed!")
        print(f"🎯 Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"🎯 Test ROC-AUC: {metrics['roc_auc']:.4f}")

        return metrics

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        예측 수행

        Args:
            X: feature 데이터

        Returns:
            y_pred: 예측 클래스 (0, 1)
            y_pred_proba: 예측 확률
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        y_pred_proba = self.model.predict(X)
        y_pred = (y_pred_proba > 0.5).astype(int)

        return y_pred, y_pred_proba

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        모델 평가

        Args:
            X: feature 데이터
            y: 실제 target

        Returns:
            평가 메트릭
        """
        y_pred, y_pred_proba = self.predict(X)

        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred, zero_division=0)
        }

    def plot_feature_importance(self, save_path: Optional[str] = None, top_n: int = 20):
        """Feature Importance 시각화"""
        if self.feature_importance is None:
            raise ValueError("Model not trained yet or no feature importance available.")

        plt.figure(figsize=(12, 8))

        top_features = self.feature_importance.head(top_n)
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance (Gain)')
        plt.title(f'Direction Classifier - Top {top_n} Feature Importance', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        # 값 레이블 추가
        for i, v in enumerate(top_features['importance']):
            plt.text(v + max(top_features['importance']) * 0.01, i,
                    '.3f', va='center', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=160, bbox_inches='tight')
            print(f"💾 Feature importance plot saved to {save_path}")

        plt.show()

    def plot_confusion_matrix(self, X: pd.DataFrame, y: pd.Series, save_path: Optional[str] = None):
        """Confusion Matrix 시각화"""
        y_pred, _ = self.predict(X)
        cm = confusion_matrix(y, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Down (-1)', 'Up (+1)'],
                   yticklabels=['Down (-1)', 'Up (+1)'])
        plt.title('Direction Classifier - Confusion Matrix', fontweight='bold')
        plt.ylabel('True Direction')
        plt.xlabel('Predicted Direction')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=160, bbox_inches='tight')
            print(f"💾 Confusion matrix saved to {save_path}")

        plt.show()

    def save_model(self, filepath: str):
        """모델 저장"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        # 모델 디렉토리 생성
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # LightGBM 모델 저장
        self.model.save_model(filepath)

        # 메타데이터 저장
        metadata = {
            'feature_list': self.feature_list,
            'model_params': self.model_params,
            'feature_importance': self.feature_importance.to_dict() if self.feature_importance is not None else None,
            'training_metrics': self.training_metrics
        }

        metadata_path = filepath.replace('.txt', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"💾 Model saved to {filepath}")
        print(f"💾 Metadata saved to {metadata_path}")

    def load_model(self, filepath: str):
        """모델 로드"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # LightGBM 모델 로드
        self.model = lgb.Booster(model_file=filepath)

        # 메타데이터 로드
        metadata_path = filepath.replace('.txt', '_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            self.feature_list = metadata.get('feature_list', [])
            self.model_params = metadata.get('model_params', {})
            self.feature_importance = pd.DataFrame(metadata.get('feature_importance', {}))
            self.training_metrics = metadata.get('training_metrics', {})

        print(f"📂 Model loaded from {filepath}")

    def get_feature_correlation(self, X: pd.DataFrame, save_path: Optional[str] = None):
        """Feature 간 상관관계 분석"""
        corr_matrix = X.corr()

        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   fmt='.2f', annot_kws={'size': 8})
        plt.title('Direction Classifier Features - Correlation Matrix', fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=160, bbox_inches='tight')
            print(f"💾 Correlation matrix saved to {save_path}")

        plt.show()

        return corr_matrix

def create_direction_classifier_model(market: str = "KR",
                                    years: List[int] = [2018, 2019, 2020],
                                    save_model: bool = True,
                                    model_dir: str = "models/saved") -> DirectionClassifierLGBM:
    """
    방향 분류 모델 생성 및 학습

    Args:
        market: 시장 코드
        years: 학습 연도
        save_model: 모델 저장 여부
        model_dir: 모델 저장 디렉토리

    Returns:
        학습된 모델 인스턴스
    """
    print("🎯 Creating Direction Classifier Model...")
    print(f"📊 Selected Features ({len(SELECTED_FEATURES)}):")
    for i, feature in enumerate(SELECTED_FEATURES, 1):
        print(f"  {i}. {feature}")
    print()

    # 모델 생성
    model = DirectionClassifierLGBM()

    # 데이터 로드
    X, y = model.load_data(market=market, years=years)

    # Feature 상관관계 확인
    print("🔍 Checking feature correlations...")
    corr_matrix = model.get_feature_correlation(X)

    # 상관관계가 0.8 이상인 feature 쌍 출력
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= 0.8:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))

    if high_corr_pairs:
        print("⚠️ High correlation pairs (|corr| >= 0.8):")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"  {feat1} - {feat2}: {corr:.3f}")
        print()
    else:
        print("✅ No high correlation pairs found!")
        print()

    # 모델 학습
    metrics = model.train(X, y)

    # 결과 출력
    print("📊 Model Performance:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    print()

    # Feature Importance 출력
    if model.feature_importance is not None:
        print("🎯 Top 10 Feature Importance:")
        top_10 = model.feature_importance.head(10)
        for i, row in top_10.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        print()

    # 모델 저장
    if save_model:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"direction_classifier_{market}_{'_'.join(map(str, years))}.txt")
        model.save_model(model_path)

    return model

if __name__ == "__main__":
    # 예시 실행
    model = create_direction_classifier_model(
        market="KR",
        years=[2018, 2019, 2020],
        save_model=True
    )

    print("✅ Direction Classifier Model created successfully!")
    print(f"🎯 Features used: {len(SELECTED_FEATURES)}")
    print(f"🏆 Best accuracy: {model.training_metrics.get('accuracy', 'N/A')}")
    print(f"🏆 Best ROC-AUC: {model.training_metrics.get('roc_auc', 'N/A')}")
