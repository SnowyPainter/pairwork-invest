#!/usr/bin/env python3
"""
Event Detector - M001 시리즈 (TCN 기반)

실제 변동폭을 측정하여 5% 이상 등락 이벤트를 감지하는 모델
TCN(Temporal Convolutional Network) 아키텍처 사용
Direction Classifier와 함께 사용하여 더 정확한 예측 가능

특징:
- TCN 아키텍처 (L=60, dilation pyramid: 1,2,4,8,16)
- 채널 수: 160~256
- Dropout: 0.2
- FocalLoss 적용
- Class-balanced sampler
- Temperature scaling
- 레짐 변경 감지용 고급 피처 세트 무조건 사용

특이사항
2021년도 데이터로 가면 "박살이 나버림"

"""

import os
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# PyTorch 관련 임포트
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# 프로젝트 모듈 임포트
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset_builder import build_dataset


# ============================================================================
# TCN 아키텍처 구현
# ============================================================================

class Chomp1d(nn.Module):
    """
    Causal convolution을 위한 헬퍼 클래스
    미래 정보를 보지 않도록 하기 위해 출력의 끝부분을 제거
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    TCN의 기본 Residual Block
    Dilated causal convolution + residual connection
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # 첫 번째 dilated convolution
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 두 번째 dilated convolution
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection을 위한 1x1 convolution (차원 맞추기용)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """가중치 초기화"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network 메인 아키텍처
    여러 dilated convolution 레이어를 쌓아 시계열 패턴 학습
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs: 입력 피처 수
            num_channels: 각 레이어의 채널 수 리스트 [160, 192, 224, 256, 256]
            kernel_size: convolution 커널 크기
            dropout: dropout 비율
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i  # 1, 2, 4, 8, 16...
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                    stride=1, dilation=dilation_size,
                                    padding=(kernel_size-1) * dilation_size,
                                    dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FocalLoss(nn.Module):
    """
    Focal Loss 구현
    클래스 불균형 문제를 해결하기 위한 손실 함수
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 변동폭 감지용 피처 세트
VOLATILITY_FEATURES = [
    # 가격 변동성 지표
    'atr14', 'atr5',           # Average True Range
    'parkinson20',         # Parkinson Volatility
    'vol_z20',              # Volume Z-score

    # 가격 위치 지표
    'rsi5',                    # RSI
    'stoch_spread',             # Stochastic %K-%D spread
    'willr14',                          # Williams %R

    # 이동평균 관련
    'ema5',           # Exponential Moving Average

    # 모멘텀 지표
    'roc5',           # Rate of Change
    'macd_hist', # MACD

    # 거래량 지표
    'vwap20',                  # VWAP
    'obv',
    'cmf20',
    'cci20',            # Volume indicators
]





# 통합 피처 세트 (변동성 피처만 사용)
ENHANCED_FEATURES = VOLATILITY_FEATURES


class TimeSeriesDataset(Dataset):
    """
    시계열 데이터를 위한 PyTorch Dataset
    TCN 모델 입력을 위한 시퀀스 데이터 생성
    """
    def __init__(self, df: pl.DataFrame, features: List[str], target_col: str,
                 sequence_length: int = 60, stride: int = 1):
        """
        Args:
            df: 입력 데이터프레임 (이벤트 라벨이 이미 계산된 상태)
            features: 사용할 피처 리스트
            target_col: 타겟 컬럼명
            sequence_length: 시퀀스 길이 (L=60)
            stride: 시퀀스 생성 간격
        """
        self.sequence_length = sequence_length
        self.stride = stride

        # 타겟 컬럼 존재 확인 (예측 시에는 없어도 됨)
        if target_col and target_col not in df.columns:
            # 예측 시에는 타겟 컬럼이 없어도 진행
            target_col = None

        if target_col is not None:
            # 결측치가 없는 데이터만 사용
            df_clean = df.filter(
                (~pl.col(target_col).is_null()) &
                (~pl.any_horizontal(pl.col(features).is_null()))
            )

            # 데이터를 pandas로 변환
            self.data = df_clean.select(features).to_pandas().values.astype(np.float32)
            targets_series = df_clean.select(target_col).to_pandas().iloc[:, 0]

            # None/NaN 값 처리
            targets_series = targets_series.fillna(0).astype(int)
            self.targets = targets_series.values.astype(np.int64)
        else:
            # 예측 시에는 타겟 없이 피처만 사용
            df_clean = df.filter(
                ~pl.any_horizontal(pl.col(features).is_null())
            )
            self.data = df_clean.select(features).to_pandas().values.astype(np.float32)
            self.targets = np.zeros(len(self.data), dtype=np.int64)  # 더미 타겟

        # 시퀀스 생성을 위한 인덱스 계산
        self.valid_indices = []
        for i in range(0, len(self.data) - sequence_length + 1, stride):
            self.valid_indices.append(i)

        print(f"  시퀀스 데이터셋 생성: {len(self.valid_indices)}개 시퀀스")
        print(f"  데이터 shape: {self.data.shape}, 타겟 분포: {np.bincount(self.targets)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length

        # 시퀀스 데이터: [sequence_length, num_features]
        x = self.data[start_idx:end_idx].T  # Conv1d 입력 형식: [channels, length]
        y = self.targets[end_idx - 1]  # 마지막 타임스텝의 타겟

        return torch.FloatTensor(x), torch.LongTensor([y]).squeeze()


class EventDetectorTCN(nn.Module):
    """
    TCN 기반 이벤트 감지기
    레짐 변경 감지용 고급 피처 세트를 무조건 사용
    """
    def __init__(self, num_features: int, sequence_length: int = 60,
                 num_channels: List[int] = None, dropout: float = 0.2):
        """
        Args:
            num_features: 입력 피처 수
            sequence_length: 시퀀스 길이
            num_channels: TCN 채널 수 리스트
            dropout: dropout 비율
        """
        super(EventDetectorTCN, self).__init__()

        # 기본 채널 설정 (160~256)
        if num_channels is None:
            num_channels = [160, 192, 224, 256, 256]

        # TCN 네트워크
        self.tcn = TemporalConvNet(
            num_inputs=num_features,
            num_channels=num_channels,
            kernel_size=2,
            dropout=dropout
        )

        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Binary classification
        )

        # 가중치 초기화
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_features, sequence_length]
        Returns:
            logits: [batch_size, 2]
        """
        # TCN 적용
        tcn_out = self.tcn(x)  # [batch_size, num_channels[-1], output_length]

        # Global average pooling (시간 차원 축소)
        pooled = torch.mean(tcn_out, dim=2)  # [batch_size, num_channels[-1]]

        # 분류
        logits = self.classifier(pooled)
        return logits


class EventDetectorManager:
    """
    TCN 기반 이벤트 감지기 관리 클래스
    레짐 변경 감지용 고급 피처 세트를 무조건 사용

    특징:
    - TCN 아키텍처 (L=60, dilation pyramid: 1,2,4,8,16)
    - 채널 수: 160~256
    - Dropout: 0.2
    - FocalLoss 적용
    - Class-balanced sampler
    - Temperature scaling
    """

    def __init__(self, threshold: float = 1.0, sequence_length: int = 60,
                 num_channels: List[int] = None, dropout: float = 0.2,
                 device: str = 'auto'):
        """
        ATR 기반 정규화 이벤트 감지기 초기화

        Args:
            threshold: ATR 정규화 이벤트 감지 임계값 (기본 1.0 = ATR의 100%)
            sequence_length: TCN 시퀀스 길이 (L=60)
            num_channels: TCN 채널 수 리스트 [160, 192, 224, 256, 256]
            dropout: dropout 비율 (0.2)
            device: 연산 장치 ('auto', 'cpu', 'cuda')
        """
        self.threshold = threshold
        self.sequence_length = sequence_length
        self.dropout = dropout

        # 레짐 변경 감지용 고급 피처 세트 무조건 사용
        self.features = ENHANCED_FEATURES

        # TCN 채널 설정
        if num_channels is None:
            self.num_channels = [160, 192, 224, 256, 256]

        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 모델 및 학습 관련 변수들
        self.model = None
        self.scaler = StandardScaler()
        self.training_history = {}

        # FocalLoss와 관련 파라미터들
        self.focal_loss_alpha = 1.0
        self.focal_loss_gamma = 2.0

        # Calibration 관련 변수들
        self.probability_calibrator = None
        self.precision_threshold = None
        self.target_precision = 0.4  # 목표 precision (40%)
        self.daily_top_k = 5  # 기본 daily top-k (하루에 최대 5개 이벤트)

        print(f"[EventDetectorManager] 초기화 완료")
        print(f"  디바이스: {self.device}")
        print(f"  시퀀스 길이: {self.sequence_length}")
        print(f"  피처 수: {len(self.features)}")
        print(f"  TCN 채널: {self.num_channels}")



    def _create_class_balanced_sampler(self, targets: torch.Tensor) -> WeightedRandomSampler:
        """
        Class-balanced sampler 생성
        클래스 불균형 문제를 해결하기 위한 가중치 기반 샘플러

        Args:
            targets: 타겟 라벨 텐서

        Returns:
            WeightedRandomSampler
        """
        class_counts = torch.bincount(targets)
        class_weights = 1.0 / class_counts.float()

        # 각 샘플의 가중치 계산
        sample_weights = class_weights[targets]

        # WeightedRandomSampler 생성
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return sampler

    def _get_default_params(self, scale_pos_weight: float = 1.0) -> Dict[str, Any]:
        """LightGBM 기본 파라미터"""
        return {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'scale_pos_weight': scale_pos_weight,  # 클래스 불균형 처리
            'random_state': 42,
            'verbose': -1
        }
    
    def _calculate_event_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Intraday max/min 기반 정규화 이벤트 라벨 계산

        ATR을 수익률 단위로 변환하여 이벤트 감지:
        event = max(high/open-1, open/low-1) >= k * ATR%
        
        Args:
            df: 입력 데이터프레임
            
        Returns:
            이벤트 라벨이 추가된 데이터프레임
        """
        print(f"[Intraday range 기반 이벤트 라벨 계산] 임계값: {self.threshold:.2f}")

        # 다음날 intraday max/min 기반 수익률 계산 (None 값 안전하게 처리)
        # max(high_{t+1}/open_{t+1}-1, open_{t+1}/low_{t+1}-1)
        df_with_events = df.with_columns([
            pl.when(
                pl.col("open").shift(-1).is_null() |
                pl.col("high").shift(-1).is_null() |
                pl.col("low").shift(-1).is_null()
            )
              .then(None)
              .otherwise(
                  pl.max_horizontal([
                      (pl.col("high").shift(-1) / pl.col("open").shift(-1) - 1).abs(),
                      (pl.col("open").shift(-1) / pl.col("low").shift(-1) - 1).abs()
                  ])
              )
              .alias("next_day_intraday_range")
        ])

        # ATR14 결측 처리 및 수익률 단위 변환
        df_with_events = df_with_events.with_columns([
            # ATR14 결측 처리: 결측시 기본 변동성 값 사용 (2%)
            pl.when(
                pl.col("atr14").is_null() |
                (pl.col("atr14") == 0) |
                (pl.col("atr14") < 0) |
                pl.col("close").is_null() |
                (pl.col("close") == 0)
            )
              .then(0.05)
              .otherwise(pl.col("atr14") / pl.col("close"))
              .alias("atr_percent")
        ])

        # 이벤트 라벨 계산: intraday range >= k * ATR% (None 값 처리)
        df_with_events = df_with_events.with_columns([
            pl.when(pl.col("next_day_intraday_range").is_null())
              .then(False)
              .otherwise(pl.col("next_day_intraday_range") >= self.threshold * pl.col("atr_percent"))
              .alias("big_move_event"),

            pl.when(pl.col("next_day_intraday_range").is_null())
              .then(False)
              .otherwise(pl.col("next_day_intraday_range") >= self.threshold * pl.col("atr_percent"))
              .alias("big_up_event"),

            pl.when(pl.col("next_day_intraday_range").is_null())
              .then(False)
              .otherwise(pl.col("next_day_intraday_range") >= self.threshold * pl.col("atr_percent"))
              .alias("big_down_event"),

            pl.when(pl.col("next_day_intraday_range").is_null())
              .then(False)
              .otherwise(pl.col("next_day_intraday_range") >= self.threshold * pl.col("atr_percent"))
              .alias("big_directional_event")
        ])

        # 디버깅: 라벨 분포 확인
        label_stats = df_with_events.select([
            pl.col("big_move_event").sum().alias("events"),
            pl.len().alias("total")
        ]).row(0)

        print(f"  이벤트 라벨 분포: {label_stats[0]}/{label_stats[1]} ({label_stats[0]/label_stats[1]:.1%})")

        return df_with_events

    def _tune_threshold_with_cv(self, df: pl.DataFrame, k_values: List[float] = None,
                               target_col: str = "big_move_event") -> Dict[str, Any]:
        """
        연도별-티커독립 시계열 CV로 k(ATR 배수) 튜닝
        
        Args:
            df: 입력 데이터프레임
            k_values: 튜닝할 k 값들 (기본: [0.8, 1.0, 1.2, 1.5])
            target_col: 타겟 컬럼명
            
        Returns:
            튜닝 결과 딕셔너리
        """
        if k_values is None:
            k_values = [0.8, 1.0, 1.2, 1.5]

        print(f"[k(ATR 배수) 튜닝] k ∈ {k_values}")

        # 시간 기반 CV를 위한 데이터 준비 (시점 leakage 방지)
        df_sorted = df.sort("date")
        dates = df_sorted.select("date").to_pandas()['date'].values
        unique_dates = sorted(np.unique(dates))

        if len(unique_dates) < 2:
            print("  [경고] CV를 위한 충분한 데이터가 없습니다. 기본값 k=1.0 사용")
            return {'optimal_k': 1.0, 'results': {}}

        # 시간 기반 CV 수행 (Expanding window 방식)
        cv_results = {}
        best_k = 1.0
        best_pr_auc = 0.0

        # CV fold 수 결정 (최대 3-fold)
        n_folds = min(3, len(unique_dates) // 2)

        for k in k_values:
            pr_aucs = []
            print(f"  k = {k:.1f} 평가 중...")

            for fold in range(n_folds):
                # Train: 이전 데이터들
                train_end_idx = int(len(unique_dates) * (fold + 1) / n_folds)
                train_dates = unique_dates[:train_end_idx]

                # Test: 다음 데이터들 (시간 순서 유지)
                test_start_idx = train_end_idx
                test_end_idx = int(len(unique_dates) * (fold + 2) / n_folds) if fold < n_folds - 1 else len(unique_dates)
                test_dates = unique_dates[test_start_idx:test_end_idx]

                if len(test_dates) == 0:
                    continue

                # 날짜를 datetime.date로 변환하여 필터링 (다양한 포맷 지원)
                from datetime import date, datetime
                import pandas as pd

                def parse_date(d):
                    if isinstance(d, date):
                        return d
                    try:
                        # ISO 포맷 문자열 처리
                        date_str = str(d).split('T')[0] if 'T' in str(d) else str(d).split()[0]
                        return date.fromisoformat(date_str)
                    except (ValueError, AttributeError):
                        # pandas를 사용한 유연한 파싱
                        try:
                            return pd.to_datetime(d).date()
                        except:
                            raise ValueError(f"날짜 파싱 실패: {d}")

                train_dates_dt = [parse_date(d) for d in train_dates]
                test_dates_dt = [parse_date(d) for d in test_dates]

                train_df = df_sorted.filter(pl.col("date").is_in(train_dates_dt))
                test_df = df_sorted.filter(pl.col("date").is_in(test_dates_dt))

                if len(train_df) == 0 or len(test_df) == 0:
                    continue

                # 임시 threshold로 이벤트 라벨 계산
                temp_detector = EventDetectorManager(threshold=k, sequence_length=self.sequence_length)
                train_with_labels = temp_detector._calculate_event_labels(train_df)
                test_with_labels = temp_detector._calculate_event_labels(test_df)

                # 간단한 baseline 모델로 평가 (여기서는 이벤트 발생률 기반)
                train_event_rate = train_with_labels.select(target_col).mean().to_numpy()[0]
                test_events = test_with_labels.select(target_col).to_numpy().flatten()
                test_intraday_ranges = test_with_labels.select("next_day_intraday_range").to_numpy().flatten()

                if len(test_events) == 0 or test_events.sum() == 0:
                    pr_aucs.append(0.0)
                    continue

                # PR-AUC 계산 (간단한 baseline)
                # 이벤트가 발생한 케이스들의 평균 intraday range vs non-event 케이스들
                event_ranges = test_intraday_ranges[test_events == 1]
                non_event_ranges = test_intraday_ranges[test_events == 0]

                if len(event_ranges) == 0 or len(non_event_ranges) == 0:
                    pr_aucs.append(0.0)
                    continue

                # 간단한 precision/recall 계산
                threshold = k * train_df.select("atr14").mean().to_numpy()[0] / train_df.select("close").mean().to_numpy()[0]
                pred_events = test_intraday_ranges >= threshold

                if pred_events.sum() == 0:
                    pr_aucs.append(0.0)
            else:
                    precision = (pred_events & test_events).sum() / pred_events.sum()
                    recall = (pred_events & test_events).sum() / test_events.sum()
                    pr_auc = precision * recall  # 간단한 근사
                    pr_aucs.append(pr_auc)

            avg_pr_auc = np.mean(pr_aucs) if pr_aucs else 0.0
            cv_results[k] = avg_pr_auc

            print(f"    k = {k:.1f}: PR-AUC = {avg_pr_auc:.4f}")

            if avg_pr_auc > best_pr_auc:
                best_pr_auc = avg_pr_auc
                best_k = k

        print(f"  최적 k: {best_k:.1f} (PR-AUC: {best_pr_auc:.4f})")

        return {
            'optimal_k': best_k,
            'best_pr_auc': best_pr_auc,
            'results': cv_results
        }

    def _tune_threshold_with_precision(self, df: pl.DataFrame, target_precision: float = 0.4,
                                     target_col: str = "big_move_event") -> Dict[str, Any]:
        """
        Precision@k 기반 임계값 튜닝

        Args:
            df: 입력 데이터프레임
            target_precision: 목표 precision (예: 0.4 = 40%)
            target_col: 타겟 컬럼명

        Returns:
            튜닝 결과 딕셔너리
        """
        print(f"[Precision@k 기반 임계값 튜닝] 목표 precision: {target_precision:.1%}")

        # 데이터 준비
        pred_df = self._calculate_event_labels(df)
        available_features = [f for f in self.features if f in pred_df.columns]

        if not available_features:
            print("  [경고] 사용 가능한 피처가 없습니다.")
            return {'precision_threshold': 0.5, 'results': {}}

        # 결측치 처리
        pred_df = pred_df.filter(~pl.any_horizontal(pl.col(available_features).is_null()))

        if len(pred_df) == 0:
            print("  [경고] 예측 가능한 데이터가 없습니다.")
            return {'precision_threshold': 0.5, 'results': {}}

        # 시퀀스 데이터셋 생성
        pred_dataset = TimeSeriesDataset(
            df=pred_df,
            features=available_features,
            target_col=target_col,
            sequence_length=self.sequence_length,
            stride=1
        )

        pred_loader = DataLoader(pred_dataset, batch_size=64, shuffle=False, drop_last=False)

        # 모델 예측
        self.model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in pred_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                probs = torch.softmax(outputs, dim=1)[:, 1]

                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        y_true = np.array(all_targets)
        y_proba = np.array(all_probs)

        if len(y_true) == 0 or y_true.sum() == 0:
            print("  [경고] 이벤트 데이터가 없습니다.")
            return {'precision_threshold': 0.5, 'results': {}}

        # Precision@k 곡선 계산
        thresholds = np.linspace(0.01, 0.99, 99)
        precision_at_k = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            if y_pred.sum() > 0:
                precision = (y_true & y_pred).sum() / y_pred.sum()
                precision_at_k.append((threshold, precision))
            else:
                precision_at_k.append((threshold, 0.0))

        # 목표 precision에 가장 가까운 threshold 찾기
        precision_at_k = np.array(precision_at_k)
        precision_values = precision_at_k[:, 1]

        # 목표 precision 이상인 것들 중 가장 낮은 threshold 선택
        valid_indices = precision_values >= target_precision
        if valid_indices.any():
            # 목표 precision 이상인 것들 중 threshold가 가장 낮은 것 선택
            valid_thresholds = precision_at_k[valid_indices]
            optimal_idx = np.argmin(valid_thresholds[:, 0])
            optimal_threshold = valid_thresholds[optimal_idx, 0]
            optimal_precision = valid_thresholds[optimal_idx, 1]
        else:
            # 목표 precision을 달성하지 못하면 가장 높은 precision의 threshold 선택
            optimal_idx = np.argmax(precision_values)
            optimal_threshold = precision_at_k[optimal_idx, 0]
            optimal_precision = precision_at_k[optimal_idx, 1]

        print(f"  최적 임계값: {optimal_threshold:.3f} (달성 precision: {optimal_precision:.1%})")

        # 결과 저장
        self.precision_threshold = optimal_threshold

        return {
            'precision_threshold': optimal_threshold,
            'achieved_precision': optimal_precision,
            'target_precision': target_precision,
            'precision_curve': precision_at_k.tolist()
        }

    def _tune_daily_top_k(self, df: pl.DataFrame, k: int = 5,
                         target_col: str = "big_move_event") -> Dict[str, Any]:
        """
        하루마다 확률 상위 K개만 이벤트로 간주하는 방식의 튜닝

        Args:
            df: 입력 데이터프레임
            k: 상위 K개
            target_col: 타겟 컬럼명

        Returns:
            튜닝 결과 딕셔너리
        """
        print(f"[일별 상위 {k}개 이벤트 선택]")

        # 데이터 준비
        pred_df = self._calculate_event_labels(df)
        available_features = [f for f in self.features if f in pred_df.columns]

        if not available_features:
            print("  [경고] 사용 가능한 피처가 없습니다.")
            return {'daily_top_k': k, 'results': {}}

        # 결측치 처리
        pred_df = pred_df.filter(~pl.any_horizontal(pl.col(available_features).is_null()))

        if len(pred_df) == 0:
            print("  [경고] 예측 가능한 데이터가 없습니다.")
            return {'daily_top_k': k, 'results': {}}

        # 날짜별로 그룹화하여 예측
        dates = pred_df.select("date").unique().sort("date")

        total_events = 0
        total_predictions = 0
        daily_precisions = []

        for date_row in dates.rows():
            date = date_row[0]
            daily_df = pred_df.filter(pl.col("date") == date)

            if len(daily_df) == 0:
                continue

            # 해당 날짜 데이터로 예측
            daily_dataset = TimeSeriesDataset(
                df=daily_df,
                features=available_features,
                target_col=target_col,
                sequence_length=self.sequence_length,
                stride=1
            )

            daily_loader = DataLoader(daily_dataset, batch_size=64, shuffle=False, drop_last=False)

            daily_probs = []
            daily_targets = []

            with torch.no_grad():
                for batch_x, batch_y in daily_loader:
                    batch_x = batch_x.to(self.device)
                    outputs = self.model(batch_x)
                    probs = torch.softmax(outputs, dim=1)[:, 1]

                    daily_probs.extend(probs.cpu().numpy())
                    daily_targets.extend(batch_y.cpu().numpy())

            if len(daily_probs) == 0:
                continue

            # 상위 k개 선택
            top_k_indices = np.argsort(daily_probs)[-k:]
            y_pred_daily = np.zeros(len(daily_probs))
            y_pred_daily[top_k_indices] = 1

            y_true_daily = np.array(daily_targets)

            # 해당 날짜의 precision 계산
            if y_pred_daily.sum() > 0:
                daily_precision = (y_true_daily & y_pred_daily).sum() / y_pred_daily.sum()
                daily_precisions.append(daily_precision)
                total_events += y_true_daily.sum()
                total_predictions += y_pred_daily.sum()

        # 전체 평균 precision
        if daily_precisions:
            avg_precision = np.mean(daily_precisions)
            print(f"  일별 평균 precision: {avg_precision:.1%}")
            print(f"  총 이벤트 수: {total_events}, 총 예측 수: {total_predictions}")
        else:
            avg_precision = 0.0
        # 결과 저장
        self.daily_top_k = k

        return {
            'daily_top_k': k,
            'avg_precision': avg_precision,
            'total_events': total_events,
            'total_predictions': total_predictions,
            'daily_precisions': daily_precisions
        }

    def _calibrate_probabilities(self, df: pl.DataFrame, method: str = "isotonic",
                               target_col: str = "big_move_event") -> Dict[str, Any]:
        """
        확률 교정 (Calibration) 수행

        Args:
            df: 입력 데이터프레임
            method: 교정 방법 ("isotonic" 또는 "platt")
            target_col: 타겟 컬럼명

        Returns:
            교정 결과 딕셔너리
        """
        print(f"[확률 교정] 방법: {method}")

        # 데이터 준비
        pred_df = self._calculate_event_labels(df)
        available_features = [f for f in self.features if f in pred_df.columns]

        if not available_features:
            print("  [경고] 사용 가능한 피처가 없습니다.")
            return {'calibrator': None, 'results': {}}

        # 결측치 처리
        pred_df = pred_df.filter(~pl.any_horizontal(pl.col(available_features).is_null()))

        if len(pred_df) == 0:
            print("  [경고] 예측 가능한 데이터가 없습니다.")
            return {'calibrator': None, 'results': {}}

        # 시퀀스 데이터셋 생성
        pred_dataset = TimeSeriesDataset(
            df=pred_df,
            features=available_features,
            target_col=target_col,
            sequence_length=self.sequence_length,
            stride=1
        )

        pred_loader = DataLoader(pred_dataset, batch_size=64, shuffle=False, drop_last=False)

        # 모델 예측
        self.model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in pred_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                probs = torch.softmax(outputs, dim=1)[:, 1]

                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        y_true = np.array(all_targets)
        y_proba = np.array(all_probs)

        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            print("  [경고] 교정에 충분한 데이터가 없습니다.")
            return {'calibrator': None, 'results': {}}

        # 교정 수행
        if method == "isotonic":
            # Isotonic Regression
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(y_proba, y_true)

        elif method == "platt":
            # Platt scaling (Logistic Regression)
            calibrator = LogisticRegression()
            # 확률을 logit 변환
            eps = 1e-15
            y_proba_clipped = np.clip(y_proba, eps, 1 - eps)
            logits = np.log(y_proba_clipped / (1 - y_proba_clipped))
            calibrator.fit(logits.reshape(-1, 1), y_true)

        else:
            raise ValueError(f"지원하지 않는 교정 방법: {method}")

        # 교정된 확률 계산
        if method == "isotonic":
            y_proba_calibrated = calibrator.predict(y_proba)
        else:  # platt
            eps = 1e-15
            y_proba_clipped = np.clip(y_proba, eps, 1 - eps)
            logits = np.log(y_proba_clipped / (1 - y_proba_clipped))
            platt_probs = calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
            y_proba_calibrated = platt_probs

        # 교정 성능 평가
        from sklearn.metrics import brier_score_loss

        # 교정 전/후 Brier score 비교
        brier_before = brier_score_loss(y_true, y_proba)
        brier_after = brier_score_loss(y_true, y_proba_calibrated)

        print(f"  Brier Score - 교정 전: {brier_before:.4f}, 교정 후: {brier_after:.4f}")
        print(f"  개선량: {brier_before - brier_after:.4f}")

        # 교정 결과 저장
        self.probability_calibrator = {
            'method': method,
            'calibrator': calibrator
        }

        return {
            'method': method,
            'calibrator': calibrator,
            'brier_before': brier_before,
            'brier_after': brier_after,
            'improvement': brier_before - brier_after,
            'original_probs': y_proba.tolist(),
            'calibrated_probs': y_proba_calibrated.tolist(),
            'true_labels': y_true.tolist()
        }

    def _apply_probability_calibration(self, probabilities: np.ndarray) -> np.ndarray:
        """
        학습된 확률 교정 적용

        Args:
            probabilities: 원본 확률 배열

        Returns:
            교정된 확률 배열
        """
        if self.probability_calibrator is None:
            print("  [경고] 확률 교정기가 학습되지 않았습니다. 원본 확률을 반환합니다.")
            return probabilities

        method = self.probability_calibrator['method']
        calibrator = self.probability_calibrator['calibrator']

        if method == "isotonic":
            calibrated_probs = calibrator.predict(probabilities)

        elif method == "platt":
            eps = 1e-15
            probs_clipped = np.clip(probabilities, eps, 1 - eps)
            logits = np.log(probs_clipped / (1 - probs_clipped))
            platt_probs = calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
            calibrated_probs = platt_probs

        else:
            calibrated_probs = probabilities

        return calibrated_probs
    
    def load_data(self, market: str = "KR", years: List[int] = [2018, 2019, 2020], 
                  max_tickers: int = 100, normalize_features: bool = False) -> pl.DataFrame:
        """
        학습용 데이터 로드
        
        Args:
            market: 시장 코드
            years: 학습 연도
            max_tickers: 최대 종목 수
            normalize_features: 피처 정규화 여부
            
        Returns:
            학습용 데이터프레임
        """
        print(f"[데이터 로드] {market} 시장, {years} 연도, 최대 {max_tickers}개 종목")
        
        # 데이터셋 빌드
        df = build_dataset(
            years=years,
            market=market,
            max_tickers=max_tickers,
            feature_set="v2",
            label_horizon=1,
            label_task="regression",  # 회귀로 로드 후 직접 라벨 계산
            verbose=False,
            normalize_features=normalize_features
        )
        
        print(f"  로드된 데이터: {len(df):,} 행 × {len(df.columns)} 열")
        
        # 이벤트 라벨 계산 (초기 threshold로)
        df_with_events = self._calculate_event_labels(df)
        
        # k(ATR 배수) 튜닝 수행 (튜닝된 k로 라벨 재계산)
        if len(years) > 1:  # 최소 2개 이상의 연도가 있어야 CV 가능
            tuning_result = self._tune_threshold_with_cv(df)
            optimal_k = tuning_result['optimal_k']
            print(f"  튜닝된 k: {optimal_k:.2f} (기존: {self.threshold:.2f})")

            # 최적 k로 threshold 업데이트
            self.threshold = optimal_k

            # 튜닝된 k로 라벨 재계산 (일관성 유지)
            print(f"  튜닝된 k({optimal_k:.2f})로 라벨 재계산 중...")
            df_with_events = self._calculate_event_labels(df)
        

        
        # 이벤트 통계 출력
        event_stats = df_with_events.select([
            pl.col("big_move_event").sum().alias("events"),
            pl.len().alias("total")
        ]).row(0)

        print(f"  Intraday range 이벤트 통계 (임계값: {self.threshold:.2f}):")
        print(f"    이벤트 수: {event_stats[0]:,}개 ({event_stats[0]/event_stats[1]:.1%})")
        print(f"    이벤트 라벨: max(high/open-1, open/low-1) >= {self.threshold:.2f} × ATR%")
        
        return df_with_events
    
    def train(self, df: pl.DataFrame, target_col: str = "big_move_event",
              test_size: float = 0.2, batch_size: int = 64, epochs: int = 50,
              learning_rate: float = 1e-3, patience: int = 10,
              use_class_balanced_sampler: bool = True) -> Dict[str, Any]:
        """
        TCN 모델 학습

        Args:
            df: 학습 데이터
            target_col: 타겟 컬럼명
            test_size: 테스트 셋 비율
            batch_size: 배치 크기
            epochs: 최대 에포크 수
            learning_rate: 학습률
            patience: early stopping patience
            use_class_balanced_sampler: 클래스 균형 샘플러 사용 여부

        Returns:
            학습 결과
        """
        print(f"[TCN 모델 학습] 타겟: {target_col}")

        # 사용 가능한 피처 확인
        available_features = [f for f in self.features if f in df.columns]
        missing_features = [f for f in self.features if f not in df.columns]

        if missing_features:
            print(f"  누락된 피처: {missing_features}")

        if not available_features:
            raise ValueError("사용 가능한 피처가 없습니다!")

        print(f"  사용 피처: {len(available_features)}개")

        # 결측치가 없는 데이터만 사용
        train_df = df.filter(
            (~pl.col(target_col).is_null()) &
            (~pl.any_horizontal(pl.col(available_features).is_null()))
        )

        print(f"  학습 데이터: {len(train_df):,} 행")

        if len(train_df) == 0:
            raise ValueError("학습 가능한 데이터가 없습니다!")

        # 시간 기반 train/validation split (시점 leakage 방지)
        # 먼저 날짜순으로 정렬
        train_df_sorted = train_df.sort("date")

        # 전체 데이터를 날짜 기준으로 분할 (시점 leakage 방지)
        dates = train_df_sorted.select("date").to_pandas()['date'].values
        unique_dates = sorted(np.unique(dates))
        split_idx = int(len(unique_dates) * (1 - test_size))

        train_dates = unique_dates[:split_idx]
        valid_dates = unique_dates[split_idx:]

        # 날짜 기반 분할 (다양한 포맷 지원)
        from datetime import date, datetime
        import pandas as pd

        def parse_date(d):
            if isinstance(d, date):
                return d
            try:
                # ISO 포맷 문자열 처리
                date_str = str(d).split('T')[0] if 'T' in str(d) else str(d).split()[0]
                return date.fromisoformat(date_str)
            except (ValueError, AttributeError):
                # pandas를 사용한 유연한 파싱
                try:
                    return pd.to_datetime(d).date()
                except:
                    raise ValueError(f"날짜 파싱 실패: {d}")

        train_dates_dt = [parse_date(d) for d in train_dates]
        valid_dates_dt = [parse_date(d) for d in valid_dates]

        train_raw_df = train_df_sorted.filter(pl.col("date").is_in(train_dates_dt))
        valid_raw_df = train_df_sorted.filter(pl.col("date").is_in(valid_dates_dt))

        print(f"  학습 날짜 범위: {train_dates[0]} ~ {train_dates[-1]}")
        print(f"  검증 날짜 범위: {valid_dates[0]} ~ {valid_dates[-1]}")

        # 이벤트 라벨 재계산 (동일한 threshold 사용)
        train_processed_df = self._calculate_event_labels(train_raw_df)
        valid_processed_df = self._calculate_event_labels(valid_raw_df)

        # 재계산된 라벨로 데이터 준비
        X_train = train_processed_df.select(available_features).to_pandas()
        y_train = train_processed_df.select(target_col).to_pandas().iloc[:, 0].astype(int)
        X_valid = valid_processed_df.select(available_features).to_pandas()
        y_valid = valid_processed_df.select(target_col).to_pandas().iloc[:, 0].astype(int)

        # 티커 정보 추출
        train_tickers = train_processed_df.select("ticker").unique().to_pandas()['ticker'].tolist()
        valid_tickers = valid_processed_df.select("ticker").unique().to_pandas()['ticker'].tolist()

        print(f"  학습 이벤트 비율: {y_train.mean():.1%} (총 {len(y_train)}개)")
        print(f"  검증 이벤트 비율: {y_valid.mean():.1%} (총 {len(y_valid)}개)")

        # 클래스 분포 출력 (재계산 후)
        train_class_counts = y_train.value_counts()
        valid_class_counts = y_valid.value_counts()

        # 전체 데이터의 클래스 분포 계산
        all_targets = pd.concat([y_train, y_valid])
        all_class_counts = all_targets.value_counts()

        # None 값 안전하게 처리
        train_neg = int(train_class_counts.get(0, 0) or 0)
        train_pos = int(train_class_counts.get(1, 0) or 0)
        valid_neg = int(valid_class_counts.get(0, 0) or 0)
        valid_pos = int(valid_class_counts.get(1, 0) or 0)

        # 전체 데이터 클래스 분포 (결과 저장용)
        class_counts_all = {
            'total_negative': int(all_class_counts.get(0, 0) or 0),
            'total_positive': int(all_class_counts.get(1, 0) or 0),
            'train_negative': train_neg,
            'train_positive': train_pos,
            'valid_negative': valid_neg,
            'valid_positive': valid_pos
        }

        print("  [재계산된 라벨 분포]")
        print(f"    Train - Negative (0): {train_neg:,}개, Positive (1): {train_pos:,}개 ({train_pos / max(train_neg + train_pos, 1):.1%})")
        print(f"    Valid - Negative (0): {valid_neg:,}개, Positive (1): {valid_pos:,}개 ({valid_pos / max(valid_neg + valid_pos, 1):.1%})")

        print(f"  훈련 티커: {len(train_tickers)}개")
        print(f"  검증 티커: {len(valid_tickers)}개")
        print(f"  훈련 데이터: {len(X_train):,} 행")
        print(f"  검증 데이터: {len(X_valid):,} 행")

        # 피처 스케일링
        print("  🔄 피처 스케일링 적용 중...")
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)

        # 스케일링된 피처로 데이터프레임 업데이트
        print("  📊 스케일링된 피처로 데이터프레임 업데이트")
        train_scaled_df = train_processed_df.with_columns([
            pl.Series(name=f, values=X_train_scaled[:, i], dtype=pl.Float64)
            for i, f in enumerate(available_features)
        ])

        valid_scaled_df = valid_processed_df.with_columns([
            pl.Series(name=f, values=X_valid_scaled[:, i], dtype=pl.Float64)
            for i, f in enumerate(available_features)
        ])

        # PyTorch 데이터셋 생성 (스케일링된 데이터 사용)
        train_dataset = TimeSeriesDataset(
            df=train_scaled_df,
            features=available_features,
            target_col=target_col,
            sequence_length=self.sequence_length
        )

        valid_dataset = TimeSeriesDataset(
            df=valid_scaled_df,
            features=available_features,
            target_col=target_col,
            sequence_length=self.sequence_length
        )

        # DataLoader 생성
        if use_class_balanced_sampler:
            # 클래스 균형 샘플러 사용
            train_targets = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
            train_sampler = self._create_class_balanced_sampler(train_targets)
            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                    sampler=train_sampler, drop_last=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=True, drop_last=True)

        valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                                shuffle=False, drop_last=False)

        # TCN 모델 생성
        num_features = len(available_features)
        self.model = EventDetectorTCN(
            num_features=num_features,
            sequence_length=self.sequence_length,
            num_channels=self.num_channels,
            dropout=self.dropout
        ).to(self.device)

        # Focal Loss와 최적화 설정
        criterion = FocalLoss(alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma)
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        # 학습 루프
        best_f1 = 0.0
        best_model_state = None
        patience_counter = 0

        print("  TCN 학습 시작...")
        for epoch in range(epochs):
            # 학습
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 검증
            self.model.eval()
            valid_loss = 0.0
            all_preds = []
            all_targets = []
            all_probs = []

            with torch.no_grad():
                for batch_x, batch_y in valid_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    valid_loss += loss.item()

                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    preds = torch.argmax(outputs, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            # 리스트를 numpy array로 변환
            all_targets = np.array(all_targets)
            all_preds = np.array(all_preds)
            all_probs = np.array(all_probs)

            valid_loss /= len(valid_loader)

            # 성능 지표 계산
            accuracy = (all_preds == all_targets).mean()
            f1 = f1_score(all_targets, all_preds, zero_division=0)

            # 추가 지표들 계산
            if len(np.unique(all_targets)) > 1:
                roc_auc = roc_auc_score(all_targets, all_probs)
                precision, recall, _ = precision_recall_curve(all_targets, all_probs)
                pr_auc = auc(recall, precision)

                # Precision, Recall 계산
                p_score = precision_score(all_targets, all_preds, zero_division=0)
                r_score = recall_score(all_targets, all_preds, zero_division=0)

                # Balanced Accuracy
                balanced_acc = balanced_accuracy_score(all_targets, all_preds)

                # Precision@k 계산 (학습 중 모니터링용)
                precision_at_k_results = {}
                if len(all_targets) > 0 and all_targets.sum() > 0:
                    k_values = [1, 3, 5, 10]
                    for k in k_values:
                        if k <= len(all_probs):
                            # 상위 k개 예측 선택
                            top_k_indices = np.argsort(all_probs)[-k:]
                            y_pred_top_k = np.zeros(len(all_probs))
                            y_pred_top_k[top_k_indices] = 1

                            # precision@k 계산
                            if y_pred_top_k.sum() > 0:
                                # 타입 변환 후 bitwise AND 연산
                                y_true_int = all_targets.astype(int)
                                y_pred_int = y_pred_top_k.astype(int)
                                precision_at_k = (y_true_int & y_pred_int).sum() / y_pred_top_k.sum()
                            else:
                                precision_at_k = 0.0

                            precision_at_k_results[f'precision@{k}'] = precision_at_k
            else:
                roc_auc = 0.0
                pr_auc = 0.0
                p_score = 0.0
                r_score = 0.0
                balanced_acc = 0.0
                precision_at_k_results = {}

            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
                print(f"      F1: {f1:.4f} | Precision: {p_score:.4f} | Recall: {r_score:.4f}")
                print(f"      ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Balanced Acc: {balanced_acc:.4f}")

                # Precision@k 출력 (5에포크마다)
                if precision_at_k_results:
                    print(f"      📊 Precision@5: {precision_at_k_results.get('precision@5', 0.0):.4f} | "
                          f"@10: {precision_at_k_results.get('precision@10', 0.0):.4f}")

            # Early stopping
            if f1 > best_f1:
                best_f1 = f1
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

            scheduler.step()

        # 최적 모델 복원
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        print("[학습 결과]")
        print(f"  정확도: {accuracy:.3f}")
        print(f"  F1 스코어: {best_f1:.3f}")
        print(f"  Precision: {p_score:.3f}")
        print(f"  Recall: {r_score:.3f}")
        print(f"  ROC-AUC: {roc_auc:.3f}")
        print(f"  PR-AUC: {pr_auc:.3f}")
        print(f"  Balanced Accuracy: {balanced_acc:.3f}")

        # Calibration 적용 (선택사항)
        if hasattr(self, 'apply_calibration_after_training') and self.apply_calibration_after_training:
            print("\n[학습 후 Calibration 적용]")
            try:
                # 검증 데이터로 확률 교정 수행
                calib_result = self._calibrate_probabilities(valid_raw_df, method="isotonic", target_col=target_col)

                # Precision 기반 threshold 튜닝
                precision_result = self._tune_threshold_with_precision(valid_raw_df, target_precision=self.target_precision, target_col=target_col)

                # Daily top-k 튜닝 (선택사항)
                if self.daily_top_k is None:  # daily_top_k가 설정되지 않은 경우에만
                    daily_k_result = self._tune_daily_top_k(valid_raw_df, k=5, target_col=target_col)
                    print(f"  📊 Daily top-k: {daily_k_result.get('avg_precision', 'N/A'):.3f}")

                print(f"  📊 Calibration 완료 - Brier 개선: {calib_result.get('improvement', 0):.4f}")
                print(f"  📊 Precision threshold: {precision_result.get('precision_threshold', 'N/A')}")

            except Exception as e:
                print(f"  [경고] Calibration 실패: {e}")

        # 결과 저장
        results = {
            'model': self.model,
            'features': available_features,
            'target': target_col,
            'accuracy': accuracy,
            'f1_score': best_f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'class_distribution': class_counts_all,
            'threshold': self.threshold,
            'epochs_trained': epoch + 1,
            'best_f1': best_f1,
            'calibration_applied': self.probability_calibrator is not None,
            'precision_threshold': self.precision_threshold,
            'daily_top_k': self.daily_top_k
        }

        self.training_history = results

        return results
    
    def predict(self, df: pl.DataFrame, batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        TCN 기반 예측 수행 (학습 시 사용한 스케일러 적용)

        Args:
            df: 예측 데이터
            batch_size: 배치 크기

        Returns:
            (예측 클래스, 예측 확률)
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다!")

        # 사용 가능한 피처 확인
        available_features = [f for f in self.features if f in df.columns]

        if not available_features:
            print("  [경고] 사용 가능한 피처가 없습니다!")
            return np.zeros(len(df)), np.zeros(len(df))

        # 결측치가 없는 데이터만 사용
        pred_df = df.filter(
            ~pl.any_horizontal(pl.col(available_features).is_null())
        )

        if len(pred_df) == 0:
            print("  [경고] 예측 가능한 데이터가 없습니다!")
            return np.zeros(len(df)), np.zeros(len(df))

        print(f"  예측 데이터: {len(pred_df):,} 행")

        # 예측 시에는 이벤트 라벨 계산 생략 (이미 학습된 모델 사용)
        # pred_df = self._calculate_event_labels(pred_df)

        # 📊 스케일러 적용 (학습 시와 동일한 스케일링)
        if self.scaler is not None:
            print("  🔄 학습 시 사용한 스케일러 적용 중...")
            # 피처 데이터를 pandas로 변환하여 스케일링
            X_pred = pred_df.select(available_features).to_pandas()
            X_pred_scaled = self.scaler.transform(X_pred)

            # 스케일링된 피처로 데이터프레임 업데이트 (동일한 방식으로)
            print("  📊 스케일링된 피처로 예측 데이터프레임 업데이트")
            pred_df = pred_df.with_columns([
                pl.Series(name=f, values=X_pred_scaled[:, i], dtype=pl.Float64)
                for i, f in enumerate(available_features)
            ])

        # 시퀀스 데이터셋 생성 (예측 시에는 타겟 컬럼 없이)
        pred_dataset = TimeSeriesDataset(
            df=pred_df,
            features=available_features,
            target_col=None,  # 예측 시에는 타겟 컬럼 불필요
            sequence_length=self.sequence_length,
            stride=1
        )

        pred_loader = DataLoader(pred_dataset, batch_size=batch_size,
                               shuffle=False, drop_last=False)

        # 모델을 평가 모드로 설정
        self.model.eval()

        all_probs = []
        all_preds = []

        with torch.no_grad():
            for batch_x, _ in pred_loader:
                batch_x = batch_x.to(self.device)

                outputs = self.model(batch_x)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # 전체 데이터 길이에 맞게 결과 확장 (안전한 인덱스 매핑)
        y_pred_proba = np.full(len(df), np.nan)  # NaN으로 초기화하여 예측되지 않은 행 구분
        y_pred = np.full(len(df), -1)  # -1로 초기화하여 예측되지 않은 행 구분

        # null이 없는 행들의 인덱스 계산
        mask = df.select(
            pl.any_horizontal([pl.col(c).is_null() for c in available_features]).alias("row_has_null")
        )["row_has_null"]
        valid_indices = np.where(~mask.to_numpy())[0]

        # 예측 결과와 valid_indices의 길이 확인 및 조정
        num_predictions = min(len(all_probs), len(valid_indices))

        # 안전하게 인덱스 매핑
        for i in range(num_predictions):
            if i < len(valid_indices):
                data_idx = valid_indices[i]
                if data_idx < len(df):
                    y_pred_proba[data_idx] = all_probs[i]
                    y_pred[data_idx] = all_preds[i]
        
        return y_pred.astype(int), y_pred_proba
    
    def evaluate(self, df: pl.DataFrame, target_col: str = "big_move_event",
                 batch_size: int = 64) -> Dict[str, Any]:
        """
        TCN 모델 평가 (동일한 이벤트 라벨 계산 적용)

        Args:
            df: 평가 데이터 (이벤트 라벨이 계산되지 않은 raw 데이터)
            target_col: 타겟 컬럼명
            batch_size: 배치 크기

        Returns:
            평가 결과
        """
        print(f"[TCN 모델 평가] 타겟: {target_col}")

        # 이벤트 라벨 계산 (학습 시와 동일한 로직 적용)
        eval_df = self._calculate_event_labels(df)

        # 사용 가능한 피처 확인
        available_features = [f for f in self.features if f in eval_df.columns]

        if not available_features:
            print("  [경고] 사용 가능한 피처가 없습니다!")
            return {'accuracy': 0.0, 'roc_auc': 0.0}

        # 결측치가 없는 데이터만 사용
        eval_df = eval_df.filter(
            (~pl.col(target_col).is_null()) &
            (~pl.any_horizontal(pl.col(available_features).is_null()))
        )

        if len(eval_df) == 0:
            print("  [경고] 평가 가능한 데이터가 없습니다!")
            return {'accuracy': 0.0, 'roc_auc': 0.0}

        print(f"  평가 데이터: {len(eval_df):,} 행")

        # 📊 스케일러 적용 (학습 시와 동일한 스케일링)
        if self.scaler is not None:
            print("  🔄 학습 시 사용한 스케일러로 평가 데이터 스케일링")
            # 피처 데이터를 pandas로 변환하여 스케일링
            X_eval = eval_df.select(available_features).to_pandas()
            X_eval_scaled = self.scaler.transform(X_eval)

            # 스케일링된 피처로 데이터프레임 업데이트
            print("  📊 스케일링된 피처로 평가 데이터프레임 업데이트")
            eval_df = eval_df.with_columns([
                pl.Series(name=f, values=X_eval_scaled[:, i], dtype=pl.Float64)
                for i, f in enumerate(available_features)
            ])

        # 시퀀스 데이터셋 생성
        eval_dataset = TimeSeriesDataset(
            df=eval_df,
            features=available_features,
            target_col=target_col,
            sequence_length=self.sequence_length,
            stride=1
        )

        eval_loader = DataLoader(eval_dataset, batch_size=batch_size,
                               shuffle=False, drop_last=False)

        # 모델을 평가 모드로 설정
        self.model.eval()

        all_targets = []
        all_probs = []
        all_preds = []

        with torch.no_grad():
            for batch_x, batch_y in eval_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                outputs = self.model(batch_x)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

                all_targets.extend(batch_y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # 평가 지표 계산
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_pred_proba = np.array(all_probs)

        # 클래스 분포 확인
        unique_classes = np.unique(y_true)
        if len(y_true) > 0:
            class_counts = np.bincount(y_true.astype(int), minlength=2)
            print(f"  클래스 분포: {class_counts}")
        else:
            print(f"  클래스 분포: [0 0] (빈 데이터)")

        # 기본 지표
        accuracy = (y_pred == y_true).mean()
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # ROC-AUC와 PR-AUC 계산
        roc_auc = 0.0
        pr_auc = 0.0

        if len(unique_classes) > 1:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = auc(recall_vals, precision_vals)

            # 추가 지표 계산
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            balanced_acc = balanced_accuracy_score(y_true, y_pred)

            print(f"  정확도: {accuracy:.3f}")
            print(f"  F1 스코어: {f1:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  ROC-AUC: {roc_auc:.3f}")
            print(f"  PR-AUC: {pr_auc:.3f}")
            print(f"  Balanced Accuracy: {balanced_acc:.3f}")
        else:
            print(f"  정확도: {accuracy:.3f}")
            print(f"  F1 스코어: {f1:.3f}")
            print(f"  AUC: 계산 불가 (단일 클래스)")

        # 분류 리포트
        try:
            print("분류 리포트:")
            print(classification_report(y_true, y_pred))
        except Exception as e:
            print(f"  분류 리포트: 생성 불가 ({e})")

        # Precision@k 기반 평가 추가
        precision_at_k_results = {}

        # 다양한 k 값에 대한 precision@k 계산
        if len(y_true) > 0 and y_true.sum() > 0:
            k_values = [1, 3, 5, 10, min(20, len(y_true))]
            for k in k_values:
                if k <= len(y_pred_proba):
                    # 상위 k개 예측 선택
                    top_k_indices = np.argsort(y_pred_proba)[-k:]
                    y_pred_top_k = np.zeros(len(y_pred_proba))
                    y_pred_top_k[top_k_indices] = 1

                    # precision@k 계산
                    if y_pred_top_k.sum() > 0:
                        # 두 배열을 같은 타입으로 변환하여 bitwise AND 연산
                        y_true_int = y_true.astype(int)
                        y_pred_int = y_pred_top_k.astype(int)
                        precision_at_k = (y_true_int & y_pred_int).sum() / y_pred_top_k.sum()
                    else:
                        precision_at_k = 0.0

                    precision_at_k_results[f'precision@{k}'] = precision_at_k

                    if k == 5:  # 기본 k=5에 대해 자세히 출력
                        # recall@k 계산 (타입 변환 후)
                        y_true_int = y_true.astype(int)
                        y_pred_int = y_pred_top_k.astype(int)
                        recall_at_k = (y_true_int & y_pred_int).sum() / y_true.sum() if y_true.sum() > 0 else 0.0
                        print(f"  Precision@5: {precision_at_k:.3f}")
                        print(f"  Recall@5: {recall_at_k:.3f}")

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'balanced_accuracy': balanced_acc,
            'precision_at_k': precision_at_k_results,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def plot_feature_importance(self, top_n: int = 20, show: bool = True):
        """피처 중요도 시각화"""
        if self.feature_importance_ is None:
            print("피처 중요도 정보가 없습니다. 먼저 모델을 학습하세요.")
            return
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance_.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance (Event Detector)')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_event_distribution(self, df: pl.DataFrame, show: bool = True):
        """Intraday range 기반 이벤트 분포 시각화"""
        print("[Intraday range 이벤트 분포 분석]")

        # 이벤트 라벨 계산 (항상 재계산하여 일관성 유지)
        df = self._calculate_event_labels(df)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Intraday range 분포
        intraday_ranges = df['next_day_intraday_range'].to_pandas()
        finite_ranges = intraday_ranges[np.isfinite(intraday_ranges)]
        axes[0, 0].hist(finite_ranges * 100, bins=50, alpha=0.7)
        axes[0, 0].set_xlabel('Intraday Range (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Intraday Range Distribution')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. ATR% 분포
        atr_percent = df['atr_percent'].to_pandas()
        finite_atr = atr_percent[np.isfinite(atr_percent)]
        axes[0, 1].hist(finite_atr * 100, bins=50, alpha=0.7)
        axes[0, 1].axvline(self.threshold * np.median(finite_atr) * 100, color='red', linestyle='--',
                          label=f'Threshold × Median ATR%')
        axes[0, 1].set_xlabel('ATR (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('ATR Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 이벤트별 월별 분포
        df_pd = df.to_pandas()
        df_pd['year_month'] = pd.to_datetime(df_pd['date']).dt.to_period('M')
        monthly_events = df_pd.groupby('year_month')['big_move_event'].sum()
        axes[1, 0].plot(monthly_events.index.astype(str), monthly_events.values, marker='o')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Event Count')
        axes[1, 0].set_title('Monthly Event Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 이벤트 크기 분포 (Intraday Range / ATR%)
        event_sizes = (df['next_day_intraday_range'] / df['atr_percent']).to_pandas()
        finite_events = event_sizes[np.isfinite(event_sizes) & (event_sizes >= self.threshold)]
        if len(finite_events) > 0:
            axes[1, 1].hist(finite_events, bins=30, alpha=0.7)
            axes[1, 1].axvline(self.threshold, color='red', linestyle='--', label=f'Threshold ({self.threshold:.2f})')
            axes[1, 1].set_xlabel('Event Size (Intraday Range / ATR%)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Event Size Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'Intraday Range Event Analysis (Threshold: {self.threshold:.2f})', fontsize=14)
        plt.tight_layout()

        if show:
            plt.show()
        else:
            plt.close()
    
    def save_model(self, path: str):
        """TCN 모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다!")

        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # PyTorch 모델 저장
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler
        }, str(model_path.with_suffix('.pth')))

        # 메타데이터 저장 (JSON으로 안전하게 저장)
        import json
        metadata = {
            'threshold': self.threshold,
            'sequence_length': self.sequence_length,
            'dropout': self.dropout,
            'num_channels': self.num_channels,
            'features': self.features,
            'device': str(self.device),
            'focal_loss_alpha': self.focal_loss_alpha,
            'focal_loss_gamma': self.focal_loss_gamma,
            'training_history': self.training_history
        }

        # JSON으로 메타데이터 저장 (pickle 문제 회피)
        with open(model_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # 기존 pickle 방식도 유지 (호환성)
        try:
            with open(model_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(metadata, f)
        except Exception as e:
            print(f"  [경고] Pickle 저장 실패 (JSON으로 저장됨): {e}")

        print(f"[TCN 모델 저장] {model_path}")

    def load_model(self, path: str):
        """TCN 모델 로드"""
        model_path = Path(path)
        import json

        # 메타데이터 로드 (JSON 우선, pickle fallback)
        metadata = None

        # 1. JSON 파일 시도 (안전한 방식)
        json_path = model_path.with_suffix('.json')
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                print(f"  ✅ JSON 메타데이터 로드 성공")
            except Exception as e:
                print(f"  [경고] JSON 메타데이터 로드 실패: {e}")

        # 2. Pickle 파일 시도 (기존 호환성)
        if metadata is None:
            pickle_path = model_path.with_suffix('.pkl')
            if pickle_path.exists():
                try:
                    # 모델 클래스들을 import하여 pickle이 찾을 수 있도록 함
                    from models.M001_EventDetector import EventDetectorTCN, FocalLoss

                    with open(pickle_path, 'rb') as f:
                        metadata = pickle.load(f)
                    print(f"  ✅ Pickle 메타데이터 로드 성공")
                except Exception as e:
                    print(f"  [경고] Pickle 메타데이터 로드 실패: {e}")

        # 3. 메타데이터가 없으면 모델 파일에서 추출
        if metadata is None:
            print("  [경고] 메타데이터 파일이 없어 모델 파일에서 정보 추출 시도...")
            checkpoint = torch.load(str(model_path.with_suffix('.pth')), map_location=self.device)
            if 'scaler' in checkpoint:
                self.scaler = checkpoint['scaler']
                print(f"  ✅ 스케일러 정보 추출 성공")
            else:
                print("  [경고] 스케일러 정보가 모델 파일에 없습니다.")

            # 기본 메타데이터 설정
            metadata = {
                'threshold': getattr(self, 'threshold', 1.0),
                'sequence_length': getattr(self, 'sequence_length', 60),
                'dropout': getattr(self, 'dropout', 0.2),
                'num_channels': getattr(self, 'num_channels', [160, 192, 224, 256, 256]),
                'features': getattr(self, 'features', []),
                'device': str(self.device),
                'focal_loss_alpha': getattr(self, 'focal_loss_alpha', 1.0),
                'focal_loss_gamma': getattr(self, 'focal_loss_gamma', 2.0),
                'training_history': {}
            }

        # 메타데이터로 초기화
        self.threshold = metadata['threshold']
        self.sequence_length = metadata['sequence_length']
        self.dropout = metadata['dropout']
        self.num_channels = metadata['num_channels']
        self.features = metadata['features']
        self.device = torch.device(metadata.get('device', 'cpu'))
        self.focal_loss_alpha = metadata.get('focal_loss_alpha', 1.0)
        self.focal_loss_gamma = metadata.get('focal_loss_gamma', 2.0)
        self.training_history = metadata.get('training_history', {})

        # PyTorch 모델 로드
        checkpoint = torch.load(str(model_path.with_suffix('.pth')), map_location=self.device)

        # TCN 모델 생성 및 로드
        num_features = len(self.features)
        self.model = EventDetectorTCN(
            num_features=num_features,
            sequence_length=self.sequence_length,
            num_channels=self.num_channels,
            dropout=self.dropout
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Scaler 로드
        self.scaler = checkpoint['scaler']

        print(f"[TCN 모델 로드] {model_path}")
        print(f"  디바이스: {self.device}")
        print(f"  시퀀스 길이: {self.sequence_length}")
        print(f"  피처 수: {len(self.features)}")
        print(f"  TCN 채널: {self.num_channels}")


def create_event_detector_model(market: str = "KR", years: List[int] = [2018, 2019, 2020],
                               threshold: float = 1.0, target: str = "big_move_event",
                               max_tickers: int = 100, save_model: bool = True,
                               sequence_length: int = 60, batch_size: int = 64,
                               epochs: int = 50, learning_rate: float = 1e-3,
                               use_class_balanced_sampler: bool = True,
                               apply_calibration: bool = True,
                               daily_top_k: int = 5, target_precision: float = 0.4) -> EventDetectorManager:
    """
    Intraday range 기반 TCN Event Detector 모델 생성 및 학습

    이벤트 라벨: max(high/open-1, open/low-1) >= k × ATR%
    ATR을 수익률 단위로 변환하여 종목별·레짐별 변동성 보정
    연도별 CV를 통해 최적 k 자동 튜닝
    확률 교정 및 Precision@k 기반 임계값 튜닝 적용

    Args:
        market: 시장 코드
        years: 학습 연도
        threshold: 초기 이벤트 감지 임계값 (CV로 자동 튜닝됨)
        target: 예측 타겟 ("big_move_event", "big_directional_event" 등)
        max_tickers: 최대 종목 수
        save_model: 모델 저장 여부
        sequence_length: TCN 시퀀스 길이 (L=60)
        batch_size: 배치 크기
        epochs: 최대 에포크 수
        learning_rate: 학습률
        use_class_balanced_sampler: 클래스 균형 샘플러 사용
        apply_calibration: 학습 후 확률 교정 및 precision 튜닝 적용 여부
        daily_top_k: 일별 최대 이벤트 예측 수 (precision@k 용도)
        target_precision: 목표 precision 값 (0.4 = 40%)

    Returns:
        학습된 EventDetectorManager 모델
    """
    print(f"[Intraday range 기반 TCN Event Detector 생성]")
    print(f"  시장: {market}")
    print(f"  연도: {years}")
    print(f"  임계값: {threshold:.2f} (ATR% 배수)")
    print(f"  타겟: {target}")
    print(f"  시퀀스 길이: {sequence_length}")
    print(f"  배치 크기: {batch_size}")
    print(f"  에포크: {epochs}")
    print(f"  학습률: {learning_rate}")
    print(f"  클래스 균형 샘플러: {use_class_balanced_sampler}")
    print(f"  📊 Precision@k 설정 - Daily top-k: {daily_top_k}, Target precision: {target_precision:.1%}")
    print("  이벤트 라벨: |수익률| >= 임계값 × ATR%")
    print("=" * 50)

    # 모델 생성
    detector = EventDetectorManager(
        threshold=threshold,
        sequence_length=sequence_length
    )

    # Precision@k 설정
    detector.daily_top_k = daily_top_k
    detector.target_precision = target_precision

    # Calibration 적용 설정
    detector.apply_calibration_after_training = apply_calibration

    # 데이터 로드
    df = detector.load_data(
        market=market,
        years=years,
        max_tickers=max_tickers,
        normalize_features=False  # 원본 값 사용
    )

    # 모델 학습
    results = detector.train(
        df=df,
        target_col=target,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        use_class_balanced_sampler=use_class_balanced_sampler
    )

    # 모델 저장
    if save_model:
        model_name = f"tcn_event_detector_{market}_{'_'.join(map(str, years))}_{int(threshold*100)}pct_L{sequence_length}"
        save_path = f"models/saved/{model_name}"
        detector.save_model(save_path)

    print(f"\n[ATR 수익률 기반 TCN Event Detector 완성]")
    print(f"  최종 정확도: {results['accuracy']:.3f}")
    print(f"  최종 F1 스코어: {results['f1_score']:.3f}")
    print(f"  최종 ROC-AUC: {results['roc_auc']:.3f}")
    print(f"  최종 PR-AUC: {results['pr_auc']:.3f}")
    print(f"  학습 에포크: {results['epochs_trained']}")
    print(f"  📊 이벤트 라벨: |수익률| >= {threshold:.2f} × ATR%")

    # Precision@k 결과 출력
    if 'precision_at_k' in results and results['precision_at_k']:
        print("  📊 Precision@k 결과:")
        for k, precision in results['precision_at_k'].items():
            print(f"    {k}: {precision:.3f}")

    # 클래스 분포 정보 출력
    if 'class_distribution' in results:
        cd = results['class_distribution']
        total_samples = cd['total_negative'] + cd['total_positive']
        print(f"  📊 클래스 분포: Negative={cd['total_negative']:,} ({cd['total_negative']/total_samples:.1%}), "
              f"Positive={cd['total_positive']:,} ({cd['total_positive']/total_samples:.1%})")

    return detector


if __name__ == "__main__":
    try:
        print("🚀 Intraday range 기반 TCN Event Detector")
        print("특징:")
        print("  • TCN 아키텍처 (L=60, dilation pyramid: 1,2,4,8,16)")
        print("  • Intraday range 기반 이벤트 라벨: max(high/open-1, open/low-1) >= k × ATR%")
        print("  • 연도별 CV를 통한 자동 k 튜닝")
        print("  • 확률 교정 (Isotonic Regression)")
        print("  • Precision@k 기반 임계값 튜닝")
        print("  • FocalLoss, Class-balanced sampler")
        print("  • 변동성 기반 피처 세트 사용")
        print("=" * 50)

        detector = create_event_detector_model(
            market="KR",
            years=[2018, 2019, 2020],
            threshold=1.0,  # ATR%의 1배 (자동 튜닝됨)
            target="big_move_event",
            max_tickers=500,
            save_model=True,
            sequence_length=60,
            batch_size=64,
            epochs=50,
            learning_rate=1e-3,
            use_class_balanced_sampler=True,
            apply_calibration=True,
            daily_top_k=5,      # 일별 최대 5개 이벤트 예측
            target_precision=0.4  # 목표 precision 40%
        )

        # 테스트 데이터 평가 (학습된 모델과 동일한 threshold 사용)
        print("\n[테스트 데이터 평가]")
        print(f"  학습된 threshold: {detector.threshold:.3f}")

        # 학습된 모델과 동일한 threshold로 테스트 detector 생성
        test_detector = EventDetectorManager(threshold=detector.threshold, sequence_length=60)

        # 테스트 데이터 로드 (학습된 threshold로 라벨 계산)
        test_raw_df = test_detector.load_data(
            market="KR",
            years=[2021],
            max_tickers=20,
            normalize_features=False
        )

        if len(test_raw_df) > 0:
            # 평가 시 이벤트 라벨 재계산 (학습 시와 동일한 threshold 사용)
            test_processed_df = test_detector._calculate_event_labels(test_raw_df)
            detector.evaluate(test_processed_df)
            detector.plot_event_distribution(test_processed_df, show=False)

        print("\n✅ Intraday range 기반 TCN Event Detector 완료!")
        print("📊 이벤트 라벨: max(high/open-1, open/low-1) >= 임계값 × ATR%")

    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
