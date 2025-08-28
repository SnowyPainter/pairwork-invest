#!/usr/bin/env python3
"""
Event Detector - M001 시리즈

실제 변동폭을 측정하여 5% 이상 등락 이벤트를 감지하는 모델
Direction Classifier와 함께 사용하여 더 정확한 예측 가능

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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# 프로젝트 모듈 임포트
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset_builder import build_dataset


# 변동폭 감지용 피처 세트
VOLATILITY_FEATURES = [
    # 가격 변동성 지표
    'atr14', 'atr5',           # Average True Range
    'parkinson20',         # Parkinson Volatility
    'vol_z20',              # Volume Z-score

    # 가격 위치 지표
    'rsi5',                    # RSI
    'stochk14', 'stochd14',             # Stochastic
    'willr14',                          # Williams %R

    # 이동평균 관련
    'ema5',           # Exponential Moving Average
    'sma5',           # Simple Moving Average

    # 모멘텀 지표
    'roc5',           # Rate of Change
    'macd_hist', # MACD

    # 거래량 지표
    'vwap20',                  # VWAP
    'obv',
    'cmf20',
    'cci20',            # Volume indicators
]

# 상관관계 기반 피처 세트 (단기 예측용)
CORRELATION_FEATURES = [
    # ATR-RSI 상관관계 기반
    'atr_rsi_corr_5', 'atr_rsi_corr_10', 'atr_rsi_corr_20',

    # MACD-볼린저 상관관계 기반
    'macd_bb_corr_5', 'macd_bb_corr_10', 'macd_bb_corr_20',

    # 변동성-모멘텀 상관관계
    'vol_momentum_corr_5', 'vol_momentum_corr_10',

    # 거래량-가격 상관관계
    'vol_price_corr_5', 'vol_price_corr_10', 'vol_price_corr_20',

    # RSI-스토캐스틱 상관관계
    'rsi_stoch_corr_5', 'rsi_stoch_corr_10',

    # VWAP-CMF 상관관계
    'vwap_cmf_corr_5', 'vwap_cmf_corr_10', 'vwap_cmf_corr_20',

    # 단기 변동성 조합 지표
    'atr_rsi_ratio', 'macd_vol_ratio', 'cmf_obv_ratio'
]

# 레짐 변경 감지용 고급 피처 세트 (단기 예측 강화)
REGIME_CHANGE_FEATURES = [
    # V-score 기반 레짐 감지 (멀티스케일 + 단기 강화)
    'v_score_5', 'v_score_10', 'v_score_15', 'v_score_20', 'v_score_40', 'v_score_60', 'v_score_90',
    'v2_score_5', 'v2_score_10', 'v2_score_15', 'v2_score_20', 'v2_score_40', 'v2_score_60', 'v2_score_90',

    # 변화율/곡률 피처 (단기 강화)
    'dcmf_3', 'dcmf_5', 'dcmf_10', 'd2cmf_3', 'd2cmf_5', 'd2cmf_10',
    'dobv_3', 'dobv_5', 'dobv_10', 'd2obv_3', 'd2obv_5', 'd2obv_10',

    # 전환점 감지 (단기 강화)
    'signflip_cmf_3', 'signflip_cmf_5', 'signflip_cmf_10',
    'signflip_obv_3', 'signflip_obv_5', 'signflip_obv_10',

    # 멀티스케일 앙상블 (단기 가중치 강화)
    'v_score_ens_short', 'v_score_ens', 'v_score_ens_long',
    'v2_score_ens_short', 'v2_score_ens', 'v2_score_ens_long',

    # Rolling slope/curvature (단기 강화)
    'cmf_slope_5', 'cmf_slope_10', 'cmf_slope_20', 'cmf_slope_40', 'cmf_slope_60',
    'obv_slope_5', 'obv_slope_10', 'obv_slope_20', 'obv_slope_40', 'obv_slope_60',
    'vwap_slope_5', 'vwap_slope_10', 'vwap_slope_20', 'vwap_slope_40', 'vwap_slope_60',

    # 단기 V-score 변이체
    'v_score_partial_5', 'v_score_partial_10', 'v_score_partial_15',
    'v2_score_partial_5', 'v2_score_partial_10', 'v2_score_partial_15'
]

# 통합 피처 세트
ENHANCED_FEATURES = VOLATILITY_FEATURES + REGIME_CHANGE_FEATURES + CORRELATION_FEATURES


class EventDetectorLGBM:
    """
    변동폭 이벤트 감지기 (LightGBM 기반)
    
    목표: 다음 날 5% 이상 변동 이벤트 예측
    """
    
    def __init__(self, threshold: float = 0.05, features: List[str] = None, 
                 use_enhanced_features: bool = True):
        """
        초기화
        
        Args:
            threshold: 이벤트 감지 임계값 (기본 5%)
            features: 사용할 피처 리스트 (None이면 기본 피처 사용)
            use_enhanced_features: 레짐 변경 감지 피처 사용 여부
        """
        self.threshold = threshold
        if features is not None:
            self.features = features
        elif use_enhanced_features:
            self.features = ENHANCED_FEATURES
        else:
            self.features = VOLATILITY_FEATURES
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance_ = None
        self.training_history = {}
        self.optimal_threshold_ = 0.5  # 최적 임계치 저장

    def _find_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray,
                               metric: str = 'f1') -> float:
        """
        최적 임계치 찾기

        Args:
            y_true: 실제 라벨
            y_proba: 예측 확률
            metric: 최적화할 메트릭 ('f1', 'precision', 'recall')

        Returns:
            최적 임계치
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_score = 0
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                from sklearn.metrics import precision_score
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                from sklearn.metrics import recall_score
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                continue

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold

    @staticmethod
    def _rolling_slope(x: pd.Series, win: int) -> pd.Series:
        """
        Rolling 선형회귀 기울기 계산
        
        Args:
            x: 입력 시계열
            win: 윈도우 크기
            
        Returns:
            Rolling 기울기
        """
        def slope_func(y):
            if len(y) < 2 or pd.isna(y).all():
                return np.nan
            t = np.arange(len(y))
            t = t - t.mean()
            den = (t**2).sum()
            if den == 0:
                return 0.0
            return (t * (y - y.mean())).sum() / den
        
        return x.rolling(win).apply(slope_func, raw=False)

    @staticmethod 
    def _partial_v_score(cmf: pd.Series, vwap: pd.Series, W: int, right: int = 10) -> pd.Series:
        """
        V-score 기반 레짐 변경 감지
        
        Args:
            cmf: CMF 또는 OBV/VWAP 시계열
            vwap: VWAP 시계열 (사용 안함, 호환성용)
            W: 장기 윈도우 크기
            right: 단기 윈도우 크기
            
        Returns:
            V-score 시계열
        """
        def v_score_func(y):
            if len(y) < right or pd.isna(y).all():
                return 0.0
            
            # 최소점 탐색
            min_idx = np.argmin(y.values)
            
            # 좌/우 기울기 근사
            left_vals = y.values[:min_idx+1] if min_idx > 0 else y.values[:1]
            right_vals = y.values[min_idx:] if min_idx < len(y)-1 else y.values[-1:]
            
            # 기울기 계산
            if len(left_vals) > 1:
                t_left = np.arange(len(left_vals))
                s_left = np.polyfit(t_left, left_vals, 1)[0] if len(left_vals) > 1 else 0
            else:
                s_left = 0
                
            if len(right_vals) > 1:
                t_right = np.arange(len(right_vals))
                s_right = np.polyfit(t_right, right_vals, 1)[0] if len(right_vals) > 1 else 0
            else:
                s_right = 0
            
            # 경과일/깊이 계산
            time_since_min = len(y) - 1 - min_idx
            depth = y.mean() - y.min() if not pd.isna(y).all() else 0
            
            # V-score 계산
            z = 1.2 * abs(s_left) + 1.0 * max(0, s_right) - 0.1 * time_since_min - 0.5 * depth
            return 1 / (1 + np.exp(-z))
        
        return cmf.rolling(W).apply(v_score_func, raw=False)

    def _generate_correlation_features(self, df_pd: pd.DataFrame) -> pd.DataFrame:
        """
        상관관계 기반 피처 생성 (단기 예측용)

        Args:
            df_pd: Pandas 데이터프레임

        Returns:
            상관관계 피처가 추가된 데이터프레임
        """
        print("  [상관관계 기반 피처 생성]")

        try:
            # 필수 컬럼 확인 및 기본값 설정
            required_cols = ['atr14', 'rsi5', 'macd_hist', 'close', 'volume', 'stochk14', 'cmf20', 'obv']
            for col in required_cols:
                if col not in df_pd.columns:
                    print(f"    [경고] 누락된 컬럼: {col}")
                    df_pd[col] = 0.0

            # 볼린저 밴드 계산 (없으면 기본값)
            if 'bb_lower' not in df_pd.columns:
                # 간단한 볼린저 밴드 근사 (SMA ± 2*rolling_std)
                if 'close' in df_pd.columns:
                    sma20 = df_pd['close'].rolling(20).mean()
                    std20 = df_pd['close'].rolling(20).std()
                    df_pd['bb_lower'] = sma20 - 2 * std20
                else:
                    df_pd['bb_lower'] = 0.0

            # 1) ATR-RSI 상관관계
            for period in [5, 10, 20]:
                df_pd[f'atr_rsi_corr_{period}'] = df_pd['atr14'].rolling(period).corr(df_pd['rsi5'])

            # 2) MACD-볼린저 상관관계
            for period in [5, 10, 20]:
                macd_bb_corr = df_pd['macd_hist'].rolling(period).corr(df_pd['bb_lower'])
                df_pd[f'macd_bb_corr_{period}'] = macd_bb_corr.fillna(0)

            # 3) 변동성-모멘텀 상관관계
            for period in [5, 10]:
                vol_momentum_corr = df_pd['atr14'].rolling(period).corr(df_pd['roc5'])
                df_pd[f'vol_momentum_corr_{period}'] = vol_momentum_corr.fillna(0)

            # 4) 거래량-가격 상관관계
            for period in [5, 10, 20]:
                if 'volume' in df_pd.columns and 'close' in df_pd.columns:
                    vol_price_corr = df_pd['volume'].rolling(period).corr(df_pd['close'])
                    df_pd[f'vol_price_corr_{period}'] = vol_price_corr.fillna(0)
                else:
                    df_pd[f'vol_price_corr_{period}'] = 0.0

            # 5) RSI-스토캐스틱 상관관계
            for period in [5, 10]:
                rsi_stoch_corr = df_pd['rsi5'].rolling(period).corr(df_pd['stochk14'])
                df_pd[f'rsi_stoch_corr_{period}'] = rsi_stoch_corr.fillna(0)

            # 6) VWAP-CMF 상관관계
            for period in [5, 10, 20]:
                if 'vwap20' in df_pd.columns:
                    vwap_cmf_corr = df_pd['vwap20'].rolling(period).corr(df_pd['cmf20'])
                    df_pd[f'vwap_cmf_corr_{period}'] = vwap_cmf_corr.fillna(0)
                else:
                    df_pd[f'vwap_cmf_corr_{period}'] = 0.0

            # 7) 단기 변동성 조합 지표
            df_pd['atr_rsi_ratio'] = df_pd['atr14'] / (df_pd['rsi5'] + 1e-8)
            df_pd['macd_vol_ratio'] = df_pd['macd_hist'] / (df_pd['atr14'] + 1e-8)
            df_pd['cmf_obv_ratio'] = df_pd['cmf20'] / (df_pd['obv'] + 1e-8)

            print(f"    생성된 상관관계 피처: {len(CORRELATION_FEATURES)}개")

        except Exception as e:
            print(f"    [경고] 상관관계 피처 생성 실패: {e}")
            # 실패 시 0으로 채움
            for feature in CORRELATION_FEATURES:
                if feature not in df_pd.columns:
                    df_pd[feature] = 0.0

        return df_pd

    def _generate_regime_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        레짐 변경 감지 피처 생성 (단기 예측 강화)

        Args:
            df: 입력 데이터프레임

        Returns:
            레짐 변경 피처가 추가된 데이터프레임
        """
        print("  [레짐 변경 피처 생성]")

        # Polars를 Pandas로 변환 (rolling 연산용)
        df_pd = df.to_pandas()

        # 필수 컬럼 확인
        required_cols = ['cmf20', 'obv', 'vwap20']
        missing_cols = [col for col in required_cols if col not in df_pd.columns]

        if missing_cols:
            print(f"    [경고] 누락된 컬럼: {missing_cols}")
            # 누락된 컬럼을 0으로 채움
            for col in missing_cols:
                df_pd[col] = 0.0

        try:
            # 1) V-score 기반 레짐 감지 (멀티스케일 + 단기 강화)
            for W in [5, 10, 15, 20, 40, 60, 90]:
                df_pd[f'v_score_{W}'] = self._partial_v_score(df_pd['cmf20'], df_pd['vwap20'], W, right=10)
                # OBV/VWAP 비율 기반
                obv_vwap_ratio = df_pd['obv'] / (df_pd['vwap20'] + 1e-8)
                df_pd[f'v2_score_{W}'] = self._partial_v_score(pd.Series(obv_vwap_ratio), df_pd['vwap20'], W, right=10)

            # 2) 단기 V-score 변이체 (partial 버전)
            for W in [5, 10, 15]:
                df_pd[f'v_score_partial_{W}'] = self._partial_v_score(df_pd['cmf20'], df_pd['vwap20'], W, right=5)
                obv_vwap_ratio = df_pd['obv'] / (df_pd['vwap20'] + 1e-8)
                df_pd[f'v2_score_partial_{W}'] = self._partial_v_score(pd.Series(obv_vwap_ratio), df_pd['vwap20'], W, right=5)

            # 3) 변화율/곡률 피처 (단기 강화)
            for k in [3, 5, 10]:
                df_pd[f'dcmf_{k}'] = df_pd['cmf20'].diff(k)
                df_pd[f'd2cmf_{k}'] = df_pd['cmf20'].diff().diff(k-1)
                df_pd[f'dobv_{k}'] = df_pd['obv'].diff(k)
                df_pd[f'd2obv_{k}'] = df_pd['obv'].diff().diff(k-1)

            # 4) 전환점 감지 (단기 강화)
            for k in [3, 5, 10]:
                df_pd[f'signflip_cmf_{k}'] = ((df_pd['cmf20'].shift(k) < 0) & (df_pd['cmf20'] >= 0)).astype(int)
                df_pd[f'signflip_obv_{k}'] = ((df_pd['obv'].shift(k) < 0) & (df_pd['obv'] >= 0)).astype(int)

            # 5) 멀티스케일 앙상블 (단기 가중치 강화)
            # 단기 앙상블 (5, 10, 15 위주)
            df_pd['v_score_ens_short'] = (0.5 * df_pd['v_score_5'] +
                                         0.3 * df_pd['v_score_10'] +
                                         0.2 * df_pd['v_score_15'])
            df_pd['v2_score_ens_short'] = (0.5 * df_pd['v2_score_5'] +
                                          0.3 * df_pd['v2_score_10'] +
                                          0.2 * df_pd['v2_score_15'])

            # 기존 앙상블 (20, 40, 60 위주)
            df_pd['v_score_ens'] = (0.5 * df_pd['v_score_20'] +
                                   0.3 * df_pd['v_score_40'] +
                                   0.2 * df_pd['v_score_60'])
            df_pd['v2_score_ens'] = (0.5 * df_pd['v2_score_20'] +
                                    0.3 * df_pd['v2_score_40'] +
                                    0.2 * df_pd['v2_score_60'])

            # 장기 앙상블 (60, 90 위주)
            df_pd['v_score_ens_long'] = (0.4 * df_pd['v_score_60'] +
                                        0.4 * df_pd['v_score_90'] +
                                        0.2 * df_pd['v_score_40'])
            df_pd['v2_score_ens_long'] = (0.4 * df_pd['v2_score_60'] +
                                         0.4 * df_pd['v2_score_90'] +
                                         0.2 * df_pd['v2_score_40'])

            # 6) Rolling slope/curvature (단기 강화)
            for win in [5, 10, 20, 40, 60]:
                df_pd[f'cmf_slope_{win}'] = self._rolling_slope(df_pd['cmf20'], win)
                df_pd[f'obv_slope_{win}'] = self._rolling_slope(df_pd['obv'], win)
                df_pd[f'vwap_slope_{win}'] = self._rolling_slope(df_pd['vwap20'], win)

            print(f"    생성된 레짐 변경 피처: {len(REGIME_CHANGE_FEATURES)}개")

        except Exception as e:
            print(f"    [경고] 레짐 변경 피처 생성 실패: {e}")
            # 실패 시 0으로 채움
            for feature in REGIME_CHANGE_FEATURES:
                if feature not in df_pd.columns:
                    df_pd[feature] = 0.0

        # 상관관계 기반 피처 생성
        df_pd = self._generate_correlation_features(df_pd)

        # 결측치 처리
        all_new_features = REGIME_CHANGE_FEATURES + CORRELATION_FEATURES
        for feature in all_new_features:
            if feature in df_pd.columns:
                df_pd[feature] = df_pd[feature].fillna(0.0)
            else:
                df_pd[feature] = 0.0

        # Pandas를 다시 Polars로 변환
        return pl.from_pandas(df_pd)

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
        이벤트 라벨 계산
        
        Args:
            df: 입력 데이터프레임
            
        Returns:
            이벤트 라벨이 추가된 데이터프레임
        """
        print(f"[이벤트 라벨 계산] 임계값: {self.threshold:.1%}")
        
        # 다양한 변동폭 지표 계산
        df_with_events = df.with_columns([
            # 1. 일중 최대 변동폭 (high-low) / open
            ((pl.col("high") - pl.col("low")) / pl.col("open")).alias("intraday_range"),
            
            # 2. 시가 대비 종가 변동률
            ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("open_to_close"),
            
            # 3. 전일 종가 대비 당일 최고가
            ((pl.col("high") - pl.col("close").shift(1)) / pl.col("close").shift(1)).alias("prev_close_to_high"),
            
            # 4. 전일 종가 대비 당일 최저가  
            ((pl.col("low") - pl.col("close").shift(1)) / pl.col("close").shift(1)).alias("prev_close_to_low"),
            
            # 5. 다음날 예측 타겟 (일중 최대 변동폭)
            ((pl.col("high").shift(-1) - pl.col("low").shift(-1)) / pl.col("open").shift(-1)).alias("next_day_range"),
            
            # 6. 다음날 방향성 (상승/하락)
            ((pl.col("close").shift(-1) - pl.col("open").shift(-1)) / pl.col("open").shift(-1)).alias("next_day_direction")
        ])
        
        # 이벤트 라벨 생성
        df_with_events = df_with_events.with_columns([
            # 대형 변동 이벤트 (임계값 이상)
            (pl.col("next_day_range") >= self.threshold).alias("big_move_event"),
            
            # 상승 이벤트 (임계값 이상 상승)
            ((pl.col("next_day_direction") >= self.threshold)).alias("big_up_event"),
            
            # 하락 이벤트 (임계값 이상 하락) 
            ((pl.col("next_day_direction") <= -self.threshold)).alias("big_down_event"),
            
            # 통합 이벤트 (상승 또는 하락)
            ((pl.col("next_day_direction").abs() >= self.threshold)).alias("big_directional_event")
        ])
        
        return df_with_events
    
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
        
        # 이벤트 라벨 계산
        df_with_events = self._calculate_event_labels(df)
        
        # 레짐 변경 피처 생성 (enhanced features 사용 시)
        if any(feature in self.features for feature in REGIME_CHANGE_FEATURES):
            df_with_events = self._generate_regime_features(df_with_events)
        
        # 이벤트 통계 출력
        event_stats = df_with_events.select([
            pl.col("big_move_event").sum().alias("big_moves"),
            pl.col("big_up_event").sum().alias("big_ups"), 
            pl.col("big_down_event").sum().alias("big_downs"),
            pl.col("big_directional_event").sum().alias("directional_events"),
            pl.len().alias("total_rows")
        ]).row(0)
        
        print(f"  이벤트 통계:")
        print(f"    대형 변동: {event_stats[0]:,}개 ({event_stats[0]/event_stats[4]:.1%})")
        print(f"    대형 상승: {event_stats[1]:,}개 ({event_stats[1]/event_stats[4]:.1%})")
        print(f"    대형 하락: {event_stats[2]:,}개 ({event_stats[2]/event_stats[4]:.1%})")
        print(f"    방향성 이벤트: {event_stats[3]:,}개 ({event_stats[3]/event_stats[4]:.1%})")
        
        return df_with_events
    
    def train(self, df: pl.DataFrame, target_col: str = "big_move_event", 
              test_size: float = 0.2, cv_folds: int = 5) -> Dict[str, Any]:
        """
        모델 학습
        
        Args:
            df: 학습 데이터
            target_col: 타겟 컬럼명
            test_size: 테스트 셋 비율
            cv_folds: 교차검증 폴드 수
            
        Returns:
            학습 결과
        """
        print(f"[모델 학습] 타겟: {target_col}")
        
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
        
        groups = train_df["ticker"].to_pandas()
        X_all = train_df.select(available_features).to_pandas()
        y_all = train_df.select(target_col).to_pandas().iloc[:, 0].astype(int)

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, valid_idx = next(gss.split(X_all, y_all, groups=groups))

        X_train, X_valid = X_all.iloc[train_idx], X_all.iloc[valid_idx]
        y_train, y_valid = y_all.iloc[train_idx], y_all.iloc[valid_idx]
        class_counts_all = y_all.value_counts()
        neg_all = int(class_counts_all.get(0, 0))
        pos_all = int(class_counts_all.get(1, 0))
        print("  [전체 라벨 분포]")
        print(f"    Negative (0): {neg_all:,}개")
        print(f"    Positive (1): {pos_all:,}개")
        print(f"    Positive 비율: {pos_all / max(neg_all + pos_all, 1):.1%}")

        # 클래스 불균형을 위한 scale_pos_weight 계산
        scale_pos_weight = neg_all / pos_all if pos_all > 0 else 1.0
        print(f"  Scale Positive Weight: {scale_pos_weight:.2f}")

        print(f"  훈련 셋: {len(X_train):,} 행")
        print(f"  테스트 셋: {len(X_valid):,} 행")

        # LightGBM 데이터셋 생성
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

        # 모델 학습 (scale_pos_weight 적용)
        params = self._get_default_params(scale_pos_weight=scale_pos_weight)
        
        print("  LightGBM 학습 중...")
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # 예측 및 평가
        y_pred_proba = self.model.predict(X_valid)

        # 최적 임계치 계산 (F1 기준)
        optimal_threshold = self._find_optimal_threshold(y_valid, y_pred_proba, metric='f1')
        self.optimal_threshold_ = optimal_threshold

        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # 성과 지표 계산
        accuracy = (y_pred == y_valid).mean()
        f1 = f1_score(y_valid, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_valid, y_pred_proba)

        # PR-AUC 계산
        precision, recall, _ = precision_recall_curve(y_valid, y_pred_proba)
        pr_auc = auc(recall, precision)

        print(f"\n[학습 결과]")
        print(f"  정확도: {accuracy:.3f}")
        print(f"  F1 스코어: {f1:.3f}")
        print(f"  최적 임계치: {optimal_threshold:.3f}")
        print(f"  ROC-AUC: {roc_auc:.3f}")
        print(f"  PR-AUC: {pr_auc:.3f}")
        
        # 피처 중요도 저장
        self.feature_importance_ = pd.DataFrame({
            'feature': available_features,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print(f"  상위 5개 중요 피처:")
        for i, row in self.feature_importance_.head().iterrows():
            print(f"    {i+1}. {row['feature']}: {row['importance']:.0f}")
        
        # 교차검증 (선택적)
        cv_scores = []
        if cv_folds > 0:
            print(f"\n  {cv_folds}-폴드 교차검증 수행 중...")
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            cv_scores = cross_val_score(rf_model, X_all, y_all, cv=cv_folds, scoring='roc_auc')
            print(f"  CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # 결과 저장
        results = {
            'model': self.model,
            'features': available_features,
            'target': target_col,
            'accuracy': accuracy,
            'f1_score': f1,
            'optimal_threshold': optimal_threshold,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'cv_scores': cv_scores,
            'feature_importance': self.feature_importance_,
            'class_distribution': class_counts_all,
            'threshold': self.threshold
        }
        
        self.training_history = results
        
        return results
    
    def predict(self, df: pl.DataFrame, threshold: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        예측 수행

        Args:
            df: 예측 데이터
            threshold: 예측 임계치 (None이면 저장된 최적 임계치 사용)

        Returns:
            (예측 클래스, 예측 확률)
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다!")

        # 임계치 설정 (기본값은 저장된 최적 임계치)
        if threshold is None:
            threshold = self.optimal_threshold_

        # 사용 가능한 피처만 추출
        available_features = [f for f in self.features if f in df.columns]

        if not available_features:
            print("  [경고] 사용 가능한 피처가 없습니다!")
            return np.zeros(len(df)), np.zeros(len(df))
        X = df[available_features]
        # 결측치 처리 (polars 버전에 따라 다른 메서드 사용)
        try:
            X = X.fill_nan(0.0)
        except AttributeError:
            X = X.fillna(0.0)

        # 예측
        y_pred_proba = self.model.predict(X)
        y_pred = (y_pred_proba >= threshold).astype(int)

        return y_pred, y_pred_proba
    
    def evaluate(self, df: pl.DataFrame, target_col: str = "big_move_event") -> Dict[str, Any]:
        """
        모델 평가
        
        Args:
            df: 평가 데이터
            target_col: 타겟 컬럼명
            
        Returns:
            평가 결과
        """
        print(f"[모델 평가] 타겟: {target_col}")
        
        # 결측치가 없는 데이터만 사용
        eval_df = df.filter(
            (~pl.col(target_col).is_null()) & 
            (~pl.any_horizontal(pl.col(self.features).is_null()))
        )
        
        if len(eval_df) == 0:
            print("  [경고] 평가 가능한 데이터가 없습니다!")
            return {'accuracy': 0.0, 'roc_auc': 0.0}
        
        print(f"  평가 데이터: {len(eval_df):,} 행")
        
        # 예측 (최적 임계치 사용)
        y_pred_proba = self.model.predict(eval_df[self.features]) if hasattr(self, 'model') and self.model else np.zeros(len(eval_df))
        y_true = eval_df.select(target_col).to_pandas().iloc[:, 0]

        # 최적 임계치 계산 (없으면 F1 기반으로 계산)
        if hasattr(self, 'optimal_threshold_') and self.optimal_threshold_ != 0.5:
            optimal_threshold = self.optimal_threshold_
        else:
            optimal_threshold = self._find_optimal_threshold(y_true, y_pred_proba, metric='f1')
            self.optimal_threshold_ = optimal_threshold

        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # 데이터 타입 확인 및 변환
        if y_true.dtype == bool:
            y_true = y_true.astype(int)
        
        # 클래스 분포 확인
        unique_classes = np.unique(y_true)
        print(f"  클래스 분포: {np.bincount(y_true)}")
        
        # 평가 지표
        accuracy = (y_pred == y_true).mean()

        # F1 스코어 계산
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # ROC-AUC와 PR-AUC 계산
        roc_auc = 0.0
        pr_auc = 0.0

        if len(unique_classes) > 1:
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba)

                # PR-AUC 계산
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                pr_auc = auc(recall, precision)

                print(f"  정확도: {accuracy:.3f}")
                print(f"  F1 스코어: {f1:.3f}")
                print(f"  최적 임계치: {optimal_threshold:.3f}")
                print(f"  ROC-AUC: {roc_auc:.3f}")
                print(f"  PR-AUC: {pr_auc:.3f}")
            except Exception as e:
                print(f"  정확도: {accuracy:.3f}")
                print(f"  F1 스코어: {f1:.3f}")
                print(f"  최적 임계치: {optimal_threshold:.3f}")
                print(f"  AUC 계산 불가 ({e})")
        else:
            print(f"  정확도: {accuracy:.3f}")
            print(f"  F1 스코어: {f1:.3f}")
            print(f"  최적 임계치: {optimal_threshold:.3f}")
            print(f"  AUC: 계산 불가 (단일 클래스)")
        
        # 분류 리포트
        try:
            print(f"\n  분류 리포트:")
            print(classification_report(y_true, y_pred))
        except Exception as e:
            print(f"  분류 리포트: 생성 불가 ({e})")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'optimal_threshold': optimal_threshold,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
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
        """이벤트 분포 시각화"""
        print("[이벤트 분포 분석]")
        
        # 이벤트 라벨이 없으면 계산
        if "big_move_event" not in df.columns:
            df = self._calculate_event_labels(df)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 일중 변동폭 분포
        axes[0, 0].hist(df['intraday_range'].to_pandas() * 100, bins=50, alpha=0.7)
        axes[0, 0].axvline(self.threshold * 100, color='red', linestyle='--', label=f'Threshold ({self.threshold:.1%})')
        axes[0, 0].set_xlabel('Intraday Range (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Intraday Range Distribution')
        axes[0, 0].legend()
        
        # 2. 다음날 변동폭 분포
        next_day_range = df['next_day_range'].to_pandas()
        axes[0, 1].hist(next_day_range * 100, bins=50, alpha=0.7)
        axes[0, 1].axvline(self.threshold * 100, color='red', linestyle='--', label=f'Threshold ({self.threshold:.1%})')
        axes[0, 1].set_xlabel('Next Day Range (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Next Day Range Distribution')
        axes[0, 1].legend()
        
        # 3. 이벤트별 월별 분포
        df_pd = df.to_pandas()
        df_pd['year_month'] = pd.to_datetime(df_pd['date']).dt.to_period('M')
        monthly_events = df_pd.groupby('year_month')['big_move_event'].sum()
        
        axes[1, 0].plot(monthly_events.index.astype(str), monthly_events.values)
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Event Count')
        axes[1, 0].set_title('Monthly Event Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 이벤트 크기별 분포
        event_sizes = next_day_range[next_day_range >= self.threshold] * 100
        axes[1, 1].hist(event_sizes, bins=30, alpha=0.7)
        axes[1, 1].set_xlabel('Event Size (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Event Size Distribution (Events Only)')
        
        plt.tight_layout()
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_model(self, path: str):
        """모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다!")
        
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # LightGBM 모델 저장
        self.model.save_model(str(model_path.with_suffix('.txt')))
        
        # 메타데이터 저장
        metadata = {
            'threshold': self.threshold,
            'optimal_threshold': self.optimal_threshold_,
            'features': self.features,
            'feature_importance': self.feature_importance_,
            'training_history': self.training_history
        }
        
        with open(model_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"[모델 저장] {model_path}")
    
    def load_model(self, path: str):
        """모델 로드"""
        model_path = Path(path)
        
        # LightGBM 모델 로드
        self.model = lgb.Booster(model_file=str(model_path.with_suffix('.txt')))
        
        # 메타데이터 로드
        with open(model_path.with_suffix('.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        self.threshold = metadata['threshold']
        self.optimal_threshold_ = metadata.get('optimal_threshold', 0.5)  # 호환성을 위해 기본값 설정
        self.features = metadata['features']
        self.feature_importance_ = metadata['feature_importance']
        self.training_history = metadata['training_history']
        
        print(f"[모델 로드] {model_path}")


def create_event_detector_model(market: str = "KR", years: List[int] = [2018, 2019, 2020],
                               threshold: float = 0.05, target: str = "big_move_event",
                               max_tickers: int = 100, save_model: bool = True,
                               use_enhanced_features: bool = True) -> EventDetectorLGBM:
    """
    Event Detector 모델 생성 및 학습
    
    Args:
        market: 시장 코드
        years: 학습 연도
        threshold: 이벤트 감지 임계값
        target: 예측 타겟 ("big_move_event", "big_directional_event" 등)
        max_tickers: 최대 종목 수
        save_model: 모델 저장 여부
        use_enhanced_features: 레짐 변경 감지 피처 사용 여부
        
    Returns:
        학습된 EventDetectorLGBM 모델
    """
    print(f"[Event Detector 생성]")
    print(f"  시장: {market}")
    print(f"  연도: {years}")
    print(f"  임계값: {threshold:.1%}")
    print(f"  타겟: {target}")
    print(f"  고급 피처 사용: {use_enhanced_features}")
    print("=" * 50)
    
    # 모델 생성
    detector = EventDetectorLGBM(threshold=threshold, use_enhanced_features=use_enhanced_features)
    
    # 데이터 로드
    df = detector.load_data(
        market=market, 
        years=years, 
        max_tickers=max_tickers,
        normalize_features=False  # 원본 값 사용
    )
    
    # 모델 학습
    results = detector.train(df, target_col=target)
    
    # 모델 저장
    if save_model:
        model_name = f"event_detector_{market}_{'_'.join(map(str, years))}_{int(threshold*100)}pct"
        save_path = f"models/saved/{model_name}"
        detector.save_model(save_path)
    
    print(f"\n[Event Detector 완성]")
    print(f"  최종 정확도: {results['accuracy']:.3f}")
    print(f"  최종 F1 스코어: {results['f1_score']:.3f}")
    print(f"  최적 임계치: {results['optimal_threshold']:.3f}")
    print(f"  최종 ROC-AUC: {results['roc_auc']:.3f}")
    print(f"  최종 PR-AUC: {results['pr_auc']:.3f}")
    
    return detector


if __name__ == "__main__":
    try:
        # 모델 생성 및 학습 (고급 피처 사용)
        detector = create_event_detector_model(
            market="KR",
            years=[2018, 2019, 2020],
            threshold=0.05,  # 5% 임계값
            target="big_move_event",
            max_tickers=50,
            save_model=True,
            use_enhanced_features=True  # 레짐 변경 감지 피처 사용
        )
        
        # 기존 모델 로드하려면 주석 해제
        # detector = EventDetectorLGBM(threshold=0.05, use_enhanced_features=True)
        # detector.load_model("models/saved/event_detector_KR_2018_2019_2020_5pct")
        
        # 피처 중요도 시각화
        detector.plot_feature_importance(show=False)
        
        # 테스트 데이터로 평가
        print("\n[테스트 데이터 평가]")
        test_detector = EventDetectorLGBM(threshold=0.05, use_enhanced_features=True)
        test_df = test_detector.load_data(
            market="KR",
            years=[2021],
            max_tickers=20,  # 더 작게
            normalize_features=False
        )
        
        if len(test_df) > 0:
            detector.evaluate(test_df)
        else:
            print("  테스트 데이터가 없습니다.")
        
        # 이벤트 분포 시각화
        detector.plot_event_distribution(test_df, show=False)
        
        print("✅ Event Detector 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
