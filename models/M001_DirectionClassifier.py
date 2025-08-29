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
    # 수익률 그룹 (roc10 선택 - 가장 높은 중요도)
    'roc10',

    # CCI 그룹 (cci20 선택 - 두 번째로 높은 중요도)
    'cci20',

    # Williams %R (willr14 선택 - 세 번째로 높은 중요도)
    'willr14',

    # OBV 그룹 (obv 선택 - On Balance Volume, 네 번째 중요도)
    'obv',

    # ATR 그룹 (atr14 선택 - 다섯 번째 중요도)
    'atr14',

    # MACD 그룹 (macd_hist 선택 - 히스토그램, 여섯 번째 중요도)
    'macd_hist',

    # RSI 그룹 (rsi14 선택 - 일곱 번째 중요도)
    'rsi14',

    # MFI 그룹 (mfi14 선택 - 여덟 번째 중요도)
    'mfi14',

    # Donchian 채널 (pos_in_don20 선택 - 아홉 번째 중요도)
    'pos_in_don20',

    # 변동성 그룹 (parkinson20 선택 - Parkinson 변동성, 열 번째 중요도)
    'parkinson20',

    # 스토캐스틱 그룹 (stochd14 선택 - smoothed 버전)
    'stochd14',

    # 거래량 그룹 (vol_z20 선택 - 표준화된 변동성)
    'vol_z20',

    # VWAP 그룹 (vwap20 선택 - 더 긴 기간)
    'vwap20'
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
              cv_folds: int = 5,
              validation_years: Optional[List[int]] = [2021]) -> Dict:
        """
        모델 학습 (교차검증 + Validation 포함)

        Args:
            X: feature 데이터
            y: target 데이터
            test_size: 테스트 세트 비율
            use_cv: 교차검증 사용 여부
            cv_folds: 교차검증 fold 수
            validation_years: validation용 데이터 연도 (기본값: [2021])

        Returns:
            학습 결과 메트릭
        """
        print("🚀 Training Direction Classifier...")

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"📊 Train set: {X_train.shape}, Test set: {X_test.shape}")

        # Validation 데이터 로드 (2021년)
        validation_metrics = None
        if validation_years:
            print(f"📊 Loading validation data for years: {validation_years}")
            try:
                X_val, y_val = self.load_data(
                    market="KR",
                    years=validation_years,
                    max_tickers=100,
                    feature_set="v2",
                    label_horizon=1,
                    label_thresh=0.05
                )
                print(f"📊 Validation set: {X_val.shape}")

                # Validation 데이터에서 사용할 수 있는 feature만 선택
                common_features = [f for f in self.feature_list if f in X_val.columns]
                X_val = X_val[common_features]
                X_train = X_train[common_features]
                X_test = X_test[common_features]

                print(f"📊 Using {len(common_features)} common features for validation")

            except Exception as e:
                print(f"⚠️ Warning: Could not load validation data: {e}")
                X_val, y_val = None, None
        else:
            X_val, y_val = None, None

        # LightGBM 데이터셋 생성
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        valid_sets = [train_data, test_data]
        valid_names = ['train', 'valid']

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('validation')

        # 모델 학습
        self.model = lgb.train(
            self.model_params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(period=10)
            ]
        )

        # Train/Test 예측 및 메트릭
        y_pred_proba_test = self.model.predict(X_test)
        y_pred_test = (y_pred_proba_test > 0.5).astype(int)

        # 메트릭 계산
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, zero_division=0),
            'recall': recall_score(y_test, y_pred_test, zero_division=0),
            'f1_score': f1_score(y_test, y_pred_test, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba_test),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'classification_report': classification_report(y_test, y_pred_test, zero_division=0)
        }

        # Validation 평가 (2021년 데이터)
        if X_val is not None and y_val is not None:
            y_pred_proba_val = self.model.predict(X_val)
            y_pred_val = (y_pred_proba_val > 0.5).astype(int)

            validation_metrics = {
                'val_accuracy': accuracy_score(y_val, y_pred_val),
                'val_precision': precision_score(y_val, y_pred_val, zero_division=0),
                'val_recall': recall_score(y_val, y_pred_val, zero_division=0),
                'val_f1_score': f1_score(y_val, y_pred_val, zero_division=0),
                'val_roc_auc': roc_auc_score(y_val, y_pred_proba_val),
                'val_confusion_matrix': confusion_matrix(y_val, y_pred_val),
                'val_classification_report': classification_report(y_val, y_pred_val, zero_division=0)
            }

            metrics.update(validation_metrics)

        # 교차검증 (옵션)
        if use_cv:
            cv_scores = cross_val_score(
                lgb.LGBMClassifier(**self.model_params),
                X_train, y_train, cv=cv_folds, scoring='accuracy'
            )
            metrics['cv_accuracy_mean'] = cv_scores.mean()
            metrics['cv_accuracy_std'] = cv_scores.std()

        # Feature Importance 저장
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        self.training_metrics = metrics

        print("✅ Training completed!")
        print("\n📊 Test Performance:")
        print(f"  🎯 Accuracy: {metrics['accuracy']:.4f}")
        print(f"  🎯 ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  🎯 Precision: {metrics['precision']:.4f}")
        print(f"  🎯 Recall: {metrics['recall']:.4f}")
        print(f"  🎯 F1-Score: {metrics['f1_score']:.4f}")

        if validation_metrics:
            print("\n📊 Validation Performance (2021):")
            print(f"  🎯 Accuracy: {validation_metrics['val_accuracy']:.4f}")
            print(f"  🎯 ROC-AUC: {validation_metrics['val_roc_auc']:.4f}")
            print(f"  🎯 Precision: {validation_metrics['val_precision']:.4f}")
            print(f"  🎯 Recall: {validation_metrics['val_recall']:.4f}")
            print(f"  🎯 F1-Score: {validation_metrics['val_f1_score']:.4f}")

        if use_cv:
            print("\n📊 Cross-Validation:")
            print(f"  🎯 CV Accuracy: {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")

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

        # 메타데이터 저장 (JSON으로 안전하게 저장)
        import json
        metadata = {
            'feature_list': self.feature_list,
            'model_params': self.model_params,
            'feature_importance': self.feature_importance.to_dict() if self.feature_importance is not None else None,
            'training_metrics': self.training_metrics
        }

        # JSON으로 메타데이터 저장 (pickle 문제 회피)
        json_metadata_path = filepath.replace('.txt', '_metadata.json')
        with open(json_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # 기존 pickle 방식도 유지 (호환성)
        try:
            pickle_metadata_path = filepath.replace('.txt', '_metadata.pkl')
            with open(pickle_metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
        except Exception as e:
            print(f"  [경고] Pickle 저장 실패 (JSON으로 저장됨): {e}")

        print(f"💾 Model saved to {filepath}")
        print(f"💾 Metadata saved to {json_metadata_path}")

    def load_model(self, filepath: str):
        """모델 로드"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # LightGBM 모델 로드
        self.model = lgb.Booster(model_file=filepath)

        # 메타데이터 로드 (JSON 우선, pickle fallback)
        import json
        metadata = None

        # 1. JSON 파일 시도 (안전한 방식)
        json_metadata_path = filepath.replace('.txt', '_metadata.json')
        if os.path.exists(json_metadata_path):
            try:
                with open(json_metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"  ✅ JSON 메타데이터 로드 성공")
            except Exception as e:
                print(f"  [경고] JSON 메타데이터 로드 실패: {e}")

        # 2. Pickle 파일 시도 (기존 호환성)
        if metadata is None:
            pickle_metadata_path = filepath.replace('.txt', '_metadata.pkl')
            if os.path.exists(pickle_metadata_path):
                try:
                    with open(pickle_metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                    print(f"  ✅ Pickle 메타데이터 로드 성공")
                except Exception as e:
                    print(f"  [경고] Pickle 메타데이터 로드 실패: {e}")

        # 3. 메타데이터가 있으면 적용
        if metadata is not None:
            self.feature_list = metadata.get('feature_list', [])
            self.model_params = metadata.get('model_params', {})
            self.feature_importance = pd.DataFrame(metadata.get('feature_importance', {}))
            self.training_metrics = metadata.get('training_metrics', {})
        else:
            print(f"  [경고] 메타데이터 파일이 없어 기본값 사용")

        print(f"📂 Model loaded from {filepath}")

def create_direction_classifier_model(market: str = "KR",
                                    years: List[int] = [2018, 2019, 2020],
                                    validation_years: Optional[List[int]] = [2021],
                                    save_model: bool = True,
                                    model_dir: str = "models/saved") -> DirectionClassifierLGBM:
    """
    방향 분류 모델 생성 및 학습 (교차검증 + Validation 포함)

    Args:
        market: 시장 코드
        years: 학습 연도
        validation_years: validation용 데이터 연도 (기본값: [2021])
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

    # 모델 학습 (교차검증 + Validation 포함)
    metrics = model.train(X, y, validation_years=validation_years)

    # Classification Report 출력
    print("\n📊 Detailed Classification Report (Test Set):")
    print(metrics['classification_report'])

    if 'val_classification_report' in metrics:
        print("📊 Detailed Classification Report (Validation Set - 2021):")
        print(metrics['val_classification_report'])

    # Confusion Matrix 출력
    print("📊 Confusion Matrix (Test Set):")
    print(metrics['confusion_matrix'])

    if 'val_confusion_matrix' in metrics:
        print("📊 Confusion Matrix (Validation Set - 2021):")
        print(metrics['val_confusion_matrix'])
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
    # 예시 실행 (교차검증 + 2021년 Validation)
    model = create_direction_classifier_model(
        market="KR",
        years=[2018, 2019, 2020],
        validation_years=[2021],
        save_model=True
    )

    print("✅ Direction Classifier Model created successfully!")
    print(f"🎯 Features used: {len(SELECTED_FEATURES)}")
    print(f"🏆 Test Accuracy: {model.training_metrics.get('accuracy', 'N/A')}")
    print(f"🏆 Test ROC-AUC: {model.training_metrics.get('roc_auc', 'N/A')}")

    if 'val_accuracy' in model.training_metrics:
        print(f"🏆 Validation Accuracy (2021): {model.training_metrics.get('val_accuracy', 'N/A')}")
        print(f"🏆 Validation ROC-AUC (2021): {model.training_metrics.get('val_roc_auc', 'N/A')}")

    if 'cv_accuracy_mean' in model.training_metrics:
        print(f"🏆 CV Accuracy: {model.training_metrics.get('cv_accuracy_mean', 'N/A'):.4f} ± {model.training_metrics.get('cv_accuracy_std', 'N/A'):.4f}")

