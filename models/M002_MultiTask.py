from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

from data.dataset_builder import build_dataset
DEFAULT_FEATURES: Sequence[str] = (
    # Event primitives & combinations
    "event_local_vol_spike",
    "event_rebound_candidate",
    "event_volume_regain",
    "event_exhaustion_candidate",
    "event_breakdown_risk",
    "I_vr_and_vs",
    "I_bd_early",
    "I_bd_late",
    # Shape descriptors
    "curv_2",
    "price_slope",
    "price_accel",
    "pos_in_band_rel",
    "bb_width",
    "ema_spread_rel",
    "ema_spread_rel_slope",
    "atr_rel",
    "atr_slope",
    "volume_z",
    "volume_z_smooth",
    "local_vol_index",
    "rsi14",
    "delta_rsi",
    "delta_rsi_3",
    "rsi_smooth",
    "macd_hist",
    "delta_macd",
    "macd_smooth",
    "macd_signal_slope",
    "delta_atr_rel",
    # Context
    "event_volume_regain_freq20",
    "event_breakdown_freq20",
    "event_local_vol_freq20",
    "recovery_from_low_60",
    "distance_from_high_60",
    "days_since_volume_regain",
    "days_since_breakdown",
)


@dataclass
class M002TrainingConfig:
    market: str = "US"
    years: Sequence[int] = field(default_factory=lambda: list(range(2000, 2019)))
    horizon: int = 5
    rebound_thresh: float = 1.0
    drawdown_floor: float = -3.0
    feature_set: str = "m002"
    feature_columns: Sequence[str] = field(default_factory=lambda: DEFAULT_FEATURES)
    normalize_features: bool = True
    dropna_rows: bool = True
    valid_size: float = 0.2
    random_state: int = 42
    risk_aversion: float = 0.5  # λ in policy score
    use_walk_forward: bool = True
    walk_forward_train_ratio: float = 0.7
    walk_forward_valid_ratio: float = 0.1
    walk_forward_step_ratio: float = 0.1


class M002MultiTaskModel:
    def __init__(
        self,
        config: Optional[M002TrainingConfig] = None,
        *,
        classifier_params: Optional[Dict] = None,
        regressor_params: Optional[Dict] = None,
    ) -> None:
        self.config = config or M002TrainingConfig()
        self.classifier_params = classifier_params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "n_estimators": 400,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": self.config.random_state,
        }
        self.regressor_params = regressor_params or {
            "objective": "regression",
            "metric": "l1",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "n_estimators": 500,
            "reg_alpha": 0.05,
            "reg_lambda": 0.2,
            "random_state": self.config.random_state,
        }

        self.classifier: Optional[lgb.LGBMClassifier] = None
        self.regressor: Optional[MultiOutputRegressor] = None
        self.feature_columns: Sequence[str] = self.config.feature_columns

    def _build_dataframe(self) -> pd.DataFrame:
        dataset = build_dataset(
            years=self.config.years,
            market=self.config.market,
            feature_set=self.config.feature_set,
            label_horizon=self.config.horizon,
            label_task="regression",
            drop_na_rows=self.config.dropna_rows,
            normalize_features=self.config.normalize_features,
        )

        if isinstance(dataset, tuple):
            dataset = dataset[0]

        if isinstance(dataset, pl.LazyFrame):
            df_pl = dataset.collect()
        elif isinstance(dataset, pl.DataFrame):
            df_pl = dataset.clone()
        else:
            raise TypeError("Unsupported dataset type returned from build_dataset.")

        g = "ticker"
        horizon = int(self.config.horizon)
        rebound_thresh = float(self.config.rebound_thresh)
        drawdown_floor = float(self.config.drawdown_floor)

        future_close_expr = pl.col("close").shift(-horizon).over(g).alias("_future_close")
        min_future_expr = pl.min_horizontal(
            *[
                pl.col("close").shift(-offset).over(g)
                for offset in range(horizon + 1)
            ]
        ).alias("_min_future_close")

        ret_col = f"ret_{horizon}d_pct"
        dd_col = f"dd_{horizon}d_pct"

        df_pl = (
            df_pl.sort(["ticker", "date"])
            .with_columns([future_close_expr, min_future_expr])
            .with_columns([
                ((pl.col("_future_close") / pl.col("close") - 1.0) * 100.0).alias(ret_col),
                ((pl.col("_min_future_close") / pl.col("close") - 1.0) * 100.0).alias(dd_col),
            ])
            .with_columns([
                pl.when(
                    (pl.col(ret_col) >= rebound_thresh)
                    & (pl.col(dd_col) >= drawdown_floor)
                )
                .then(1)
                .otherwise(0)
                .cast(pl.Int8)
                .alias("label_rebound_bin")
            ])
            .drop(["_future_close", "_min_future_close"])
        )

        required_cols = list(self.feature_columns) + [ret_col, dd_col, "label_rebound_bin"]
        df_pl = df_pl.drop_nulls(required_cols)

        output_cols = ["ticker", "date"] + required_cols
        df_pd = df_pl.select(output_cols).to_pandas()
        return df_pd

    def _make_head_fold(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        feature_cols: Sequence[str],
        ret_col: str,
        dd_col: str,
    ) -> Dict[str, pd.DataFrame]:
        return {
            "X_train": train_df[feature_cols].reset_index(drop=True),
            "X_valid": valid_df[feature_cols].reset_index(drop=True),
            "y_cls_train": train_df["label_rebound_bin"].astype(int).reset_index(drop=True),
            "y_cls_valid": valid_df["label_rebound_bin"].astype(int).reset_index(drop=True),
            "y_reg_train": train_df[[ret_col, dd_col]].reset_index(drop=True),
            "y_reg_valid": valid_df[[ret_col, dd_col]].reset_index(drop=True),
        }

    def _build_single_split_fold(
        self,
        df_pd: pd.DataFrame,
        feature_cols: Sequence[str],
        ret_col: str,
        dd_col: str,
    ) -> List[Dict[str, pd.DataFrame]]:
        train_df, valid_df = train_test_split(
            df_pd,
            test_size=self.config.valid_size,
            shuffle=False,
        )
        return [self._make_head_fold(train_df, valid_df, feature_cols, ret_col, dd_col)]

    def _build_head_walk_forward_folds(
        self,
        df_pd: pd.DataFrame,
        feature_cols: Sequence[str],
        ret_col: str,
        dd_col: str,
    ) -> List[Dict[str, pd.DataFrame]]:
        if df_pd.empty:
            return []

        if not self.config.use_walk_forward:
            return self._build_single_split_fold(df_pd, feature_cols, ret_col, dd_col)

        n = len(df_pd)
        train_size = max(int(n * self.config.walk_forward_train_ratio), 1)
        valid_size = max(int(n * self.config.walk_forward_valid_ratio), 1)
        step = max(int(n * self.config.walk_forward_step_ratio), 1)
        step = max(step, valid_size)

        if train_size + valid_size > n:
            return self._build_single_split_fold(df_pd, feature_cols, ret_col, dd_col)

        folds: List[Dict[str, pd.DataFrame]] = []
        start = 0
        while start + train_size + valid_size <= n:
            train_slice = df_pd.iloc[start : start + train_size]
            valid_slice = df_pd.iloc[start + train_size : start + train_size + valid_size]
            folds.append(self._make_head_fold(train_slice, valid_slice, feature_cols, ret_col, dd_col))
            start += step

        if not folds:
            return self._build_single_split_fold(df_pd, feature_cols, ret_col, dd_col)
        return folds

    def train(self, data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        ret_col = f"ret_{self.config.horizon}d_pct"
        dd_col = f"dd_{self.config.horizon}d_pct"

        if data is None:
            data = self._build_dataframe()
        else:
            missing = [col for col in self.feature_columns if col not in data.columns]
            required_targets = {ret_col, dd_col, "label_rebound_bin"}
            missing_targets = [col for col in required_targets if col not in data.columns]
            if missing or missing_targets:
                raise ValueError(f"Provided data is missing required columns: features={missing}, targets={missing_targets}")
            data = data.copy()

        data = data.sort_values(["date", "ticker"]).reset_index(drop=True)
        folds = self._build_head_walk_forward_folds(data, self.feature_columns, ret_col, dd_col)
        if not folds:
            raise ValueError("No walk-forward folds could be generated for training.")

        fold_metrics: List[Dict[str, float]] = []
        latest_classifier: Optional[lgb.LGBMClassifier] = None
        latest_regressor: Optional[MultiOutputRegressor] = None

        for fold_idx, fold in enumerate(folds):
            X_train = fold["X_train"]
            X_valid = fold["X_valid"]
            y_train_cls = fold["y_cls_train"]
            y_valid_cls = fold["y_cls_valid"]
            y_train_reg = fold["y_reg_train"]
            y_valid_reg = fold["y_reg_valid"]

            nan_mask_train = y_train_reg.isna().any(axis=1)
            if nan_mask_train.any():
                X_train = X_train[~nan_mask_train]
                y_train_cls = y_train_cls[~nan_mask_train]
                y_train_reg = y_train_reg[~nan_mask_train]

            nan_mask_valid = y_valid_reg.isna().any(axis=1)
            if nan_mask_valid.any():
                X_valid = X_valid[~nan_mask_valid]
                y_valid_cls = y_valid_cls[~nan_mask_valid]
                y_valid_reg = y_valid_reg[~nan_mask_valid]

            classifier = lgb.LGBMClassifier(**self.classifier_params)
            classifier.fit(X_train, y_train_cls)

            reg_base = lgb.LGBMRegressor(**self.regressor_params)
            reg_multi = MultiOutputRegressor(reg_base)
            reg_multi.fit(X_train, y_train_reg)

            cls_pred_prob = classifier.predict_proba(X_valid)[:, 1]
            reg_pred = reg_multi.predict(X_valid)
            ret_pred = reg_pred[:, 0]
            dd_pred = reg_pred[:, 1]
            policy_score = cls_pred_prob * (ret_pred / 100.0) - self.config.risk_aversion * np.maximum(0.0, -(dd_pred / 100.0))

            fold_metric = {
                "fold": fold_idx,
                "valid_avg_precision": float(average_precision_score(y_valid_cls, cls_pred_prob)),
                "valid_mae_ret": float(mean_absolute_error(y_valid_reg.iloc[:, 0], ret_pred)),
                "valid_mae_dd": float(mean_absolute_error(y_valid_reg.iloc[:, 1], dd_pred)),
                "valid_policy_score_mean": float(np.nanmean(policy_score)),
                "classification_report": classification_report(
                    y_valid_cls,
                    (cls_pred_prob >= 0.5).astype(int),
                    target_names=["NoRebound", "Rebound"],
                    zero_division=0,
                    output_dict=False,
                ),
            }
            fold_metrics.append(fold_metric)

            if fold_idx == len(folds) - 1:
                latest_classifier = classifier
                latest_regressor = reg_multi

        if latest_classifier is None or latest_regressor is None:
            raise RuntimeError("Walk-forward training did not produce final models.")

        self.classifier = latest_classifier
        self.regressor = latest_regressor

        summary = dict(fold_metrics[-1])
        summary["fold_metrics"] = fold_metrics
        summary["n_rows"] = int(len(data))
        return summary

    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.classifier is None or self.regressor is None:
            raise RuntimeError("Model is not trained. Call train() first.")

        missing = [col for col in self.feature_columns if col not in features.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        X = features[list(self.feature_columns)]
        prob = self.classifier.predict_proba(X)[:, 1]
        reg_pred = self.regressor.predict(X)
        ret_pred, dd_pred = reg_pred[:, 0], reg_pred[:, 1]
        # 정책 점수 계산 시 퍼센트를 소수점으로 변환
        policy = prob * (ret_pred / 100.0) - self.config.risk_aversion * np.maximum(0.0, -(dd_pred / 100.0))
        return prob, ret_pred, policy


def main():
    """M002 모델 학습 실행"""
    import json
    from pathlib import Path

    # 설정
    config = M002TrainingConfig()

    print("=" * 60)
    print("M002 MultiTask 모델 학습")
    print("=" * 60)
    print(f"시장: {config.market}")
    print(f"학습 연도: {config.years}")
    print(f"예측 기간: {config.horizon}일")
    print(f"Rebound 임계값: {config.rebound_thresh}%")
    print(f"Drawdown 바닥: {config.drawdown_floor}%")
    print(f"리스크 회피도(λ): {config.risk_aversion}")
    print(f"특징 세트: {config.feature_set}")
    print(f"특징 수: {len(config.feature_columns)}")
    print("=" * 60)

    # 모델 생성 및 학습
    model = M002MultiTaskModel(config=config)
    metrics = model.train()

    print("\n학습 결과:")
    print(f"  유효성 검증 평균 정밀도: {metrics['valid_avg_precision']:.4f}")
    print(f"  수익률 MAE: {metrics['valid_mae_ret']:.4f}")
    print(f"  Drawdown MAE: {metrics['valid_mae_dd']:.4f}")
    print(f"  정책 점수 평균: {metrics['valid_policy_score_mean']:.4f}")

    print("\n분류 보고서:")
    print(metrics['classification_report'])

    # 모델 저장 (선택사항)
    save_dir = Path("models/saved")
    save_dir.mkdir(exist_ok=True)
    model_path = save_dir / f"m002_multitask_{config.market}_{'_'.join(map(str, config.years))}.pkl"

    try:
        import joblib
        joblib.dump(model, model_path)
        print(f"\n모델 저장됨: {model_path}")

        # 메타데이터 저장
        metadata = {
            "model_type": "M002_MultiTask",
            "market": config.market,
            "years": config.years,
            "horizon": config.horizon,
            "rebound_thresh": config.rebound_thresh,
            "drawdown_floor": config.drawdown_floor,
            "risk_aversion": config.risk_aversion,
            "feature_columns": config.feature_columns,
            "validation_metrics": metrics,
            "created_at": str(pd.Timestamp.now()),
        }

        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"메타데이터 저장됨: {metadata_path}")

    except Exception as e:
        print(f"모델 저장 실패: {e}")

    return metrics


if __name__ == "__main__":
    main()


__all__ = ["M002TrainingConfig", "M002MultiTaskModel"]
