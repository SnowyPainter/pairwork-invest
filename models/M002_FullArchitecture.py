from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

from data.dataset_builder import build_dataset
from models.M002_MultiTask import M002TrainingConfig, DEFAULT_FEATURES as BASELINE_FEATURES
from models.M002_RegimeClassifier import M002RegimeClassifier, RegimeConfig, DEFAULT_REGIME_FEATURES
from models.M002_constants import STATE_NAMES


# ---------------------------
# Feature sets
# ---------------------------

STATE_PROB_COLS: Sequence[str] = tuple(f"state_prob_{name}" for name in STATE_NAMES)

HEAD_FEATURES: Sequence[str] = tuple(
    sorted(
        set(BASELINE_FEATURES)
        | {
            "curv_2",
            "price_slope",
            "price_accel",
            "I_bd_early",
            "I_bd_late",
            "I_vr_and_vs",
            "event_local_vol_spike",
            "event_rebound_candidate",
            "event_volume_regain",
            "event_exhaustion_candidate",
            "event_breakdown_risk",
            "event_volume_regain_freq20",
            "event_breakdown_freq20",
            "event_local_vol_freq20",
            "recovery_from_low_60",
            "distance_from_high_60",
            "days_since_volume_regain",
            "days_since_breakdown",
            "atr_smooth",
        }
        | set(STATE_PROB_COLS)
    )
)


# ---------------------------
# Helpers
# ---------------------------

def _compute_future_returns(df: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """Attach future return and future drawdown (min-close path) over the next horizon days."""
    g = "ticker"
    future_close = pl.col("close").shift(-horizon).over(g).alias("_future_close")
    min_future = pl.min_horizontal(
        *[pl.col("close").shift(-o).over(g) for o in range(horizon + 1)]
    ).alias("_min_future_close")

    ret_col = f"ret_{horizon}d_pct"
    dd_col = f"dd_{horizon}d_pct"

    return (
        df.sort([g, "date"])
        .with_columns([future_close, min_future])
        .with_columns([
            ((pl.col("_future_close") / pl.col("close") - 1.0) * 100.0).alias(ret_col),
            ((pl.col("_min_future_close") / pl.col("close") - 1.0) * 100.0).alias(dd_col),
        ])
        .drop(["_future_close", "_min_future_close"])
    )


def _soft_policy_score(prob: np.ndarray, ret_pct: np.ndarray, dd_pct: np.ndarray, lam: float) -> np.ndarray:
    """
    Policy utility = P(rebound) * E[ret] - λ * max(0, -E[dd])
    Inputs are in percentage space; converted to ratio internally.
    """
    return prob * (ret_pct / 100.0) - lam * np.maximum(0.0, -(dd_pct / 100.0))


# ---------------------------
# Configs
# ---------------------------

@dataclass
class PolicyConfig:
    theta_long: float = 0.05
    theta_flat_low: float = 0.0
    theta_short: float = -0.05
    risk_aversion: float = 0.45
    size_k: float = 1.0
    size_max: float = 1.0
    ex_ante_vol_window: int = 10
    restrict_short_on_peak_prob: float = 0.4
    rescale_window: int = 60
    score_scale: float = 2.5
    use_rescaled_score: bool = True


@dataclass
class FullArchitectureConfig:
    multitask: M002TrainingConfig = field(default_factory=M002TrainingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    horizon: int = 5
    rebound_thresh: float = 1.0
    drawdown_floor: float = -3.0
    feature_set: str = "m002"
    normalize_features: bool = True
    verbose: bool = False  # minimal logging toggle

    @property
    def regime(self) -> RegimeConfig:
        # Share market/years/normalization with regime classifier
        return RegimeConfig(
            market=self.multitask.market,
            years=self.multitask.years,
            normalize_features=self.normalize_features,
        )


# ---------------------------
# Model
# ---------------------------

class M002FullArchitecture:
    """
    Three-tier stack:
      A) Regime classifier -> state probabilities
      B) Head: LightGBM classifier (rebound) + multi-output regressor (return, drawdown)
      C) Policy: soft utility + discrete action + position sizing
    """

    def __init__(self, config: Optional[FullArchitectureConfig] = None) -> None:
        self.config = config or FullArchitectureConfig()
        self.regime = M002RegimeClassifier(self.config.regime)
        self.policy_cfg = self.config.policy

        self.head_model: Optional[lgb.LGBMClassifier] = None
        self.head_multi: Optional[MultiOutputRegressor] = None
        self.head_features: Sequence[str] = HEAD_FEATURES

        self.trained_state: Dict[str, Dict] = {}

    # ---------- Data ----------

    def _load_base_dataframe(self) -> pl.DataFrame:
        """Load feature set via build_dataset and attach labels for the given horizon."""
        dataset = build_dataset(
            years=self.config.multitask.years,
            market=self.config.multitask.market,
            feature_set=self.config.feature_set,
            label_horizon=self.config.horizon,
            label_task="regression",
            drop_na_rows=False,
            normalize_features=self.config.normalize_features,
        )
        if isinstance(dataset, tuple):
            dataset = dataset[0]
        if isinstance(dataset, pl.LazyFrame):
            dataset = dataset.collect()
        if not isinstance(dataset, pl.DataFrame):
            raise TypeError("build_dataset must return a Polars DataFrame or LazyFrame")

        df = dataset.sort(["ticker", "date"])
        df = _compute_future_returns(df, horizon=self.config.horizon)

        # binary rebound label with a drawdown floor guard
        ret_col = f"ret_{self.config.horizon}d_pct"
        dd_col = f"dd_{self.config.horizon}d_pct"
        df = df.with_columns([
            pl.when(
                (pl.col(ret_col) >= self.config.rebound_thresh)
                & (pl.col(dd_col) >= self.config.drawdown_floor)
            )
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("label_rebound_bin")
        ])
        return df

    # ---------- Training ----------

    def train(self) -> Dict[str, Dict[str, float]]:
        """Train regime classifier + head models. Returns validation metrics."""
        if self.config.verbose:
            print("[A] Training regime classifier ...")
        regime_metrics = self.regime.train()

        if self.config.verbose:
            print("[B] Preparing head dataset ...")
        df = self._load_base_dataframe()

        # Join state probabilities
        state_probs = self.regime.predict_probabilities(
            df.select(["ticker", "date", *DEFAULT_REGIME_FEATURES])
        )
        df = df.join(state_probs, on=["ticker", "date"], how="left")

        # Keep NaNs: LightGBM handles feature NaNs; targets must be finite.
        ret_col = f"ret_{self.config.horizon}d_pct"
        dd_col = f"dd_{self.config.horizon}d_pct"

        # Filter to available features only (robust to missing columns)
        available_cols = set(df.columns)
        available_head_features: List[str] = [f for f in self.head_features if f in available_cols]
        if not available_head_features:
            raise ValueError("No head features available in the dataset after join.")

        # Build pandas frame
        use_cols = ["ticker", "date", "label_rebound_bin", ret_col, dd_col, *available_head_features]
        df_pd = (
            df.select(use_cols)
              .to_pandas()
              .replace([np.inf, -np.inf], np.nan)
        )

        # Targets cannot be NaN for sklearn
        yreg = df_pd[[ret_col, dd_col]]
        target_mask = yreg.notna().all(axis=1)
        df_pd = df_pd.loc[target_mask].reset_index(drop=True)

        if df_pd.empty:
            raise ValueError("No valid samples after removing NaN targets.")

        X = df_pd[available_head_features]
        y_cls = df_pd["label_rebound_bin"].astype(int)
        y_reg = df_pd[[ret_col, dd_col]]

        # Simple time-order split
        n = len(X)
        split_idx = max(int(n * 0.8), 1)
        X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
        y_cls_train, y_cls_valid = y_cls.iloc[:split_idx], y_cls.iloc[split_idx:]
        y_reg_train, y_reg_valid = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]

        if X_valid.empty:  # tiny dataset fallback
            X_valid, y_cls_valid, y_reg_valid = X_train.copy(), y_cls_train.copy(), y_reg_train.copy()

        # LightGBM params
        clf_params = dict(
            objective="binary",
            metric="binary_logloss",
            learning_rate=0.05,
            num_leaves=127,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=5,
            n_estimators=600,
            reg_alpha=0.05,
            reg_lambda=0.3,
            random_state=self.config.multitask.random_state,
        )
        reg_params = dict(
            objective="regression",
            metric="l1",
            learning_rate=0.05,
            num_leaves=127,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=5,
            n_estimators=700,
            reg_alpha=0.05,
            reg_lambda=0.2,
            random_state=self.config.multitask.random_state,
        )

        # Train classifier
        self.head_model = lgb.LGBMClassifier(**clf_params)
        self.head_model.fit(
            X_train,
            y_cls_train,
            eval_set=[(X_valid, y_cls_valid)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        # Train multi-output regressor
        base_reg = lgb.LGBMRegressor(**reg_params)
        self.head_multi = MultiOutputRegressor(base_reg)
        self.head_multi.fit(X_train, y_reg_train)

        # Metrics
        from sklearn.metrics import average_precision_score, mean_absolute_error

        prob_valid = self.head_model.predict_proba(X_valid)[:, 1]
        reg_valid = self.head_multi.predict(X_valid)
        ret_pred_valid, dd_pred_valid = reg_valid[:, 0], reg_valid[:, 1]
        policy_scores = _soft_policy_score(
            prob_valid, ret_pred_valid, dd_pred_valid, self.policy_cfg.risk_aversion
        )

        head_metrics = {
            "valid_avg_precision": float(average_precision_score(y_cls_valid, prob_valid)),
            "valid_mae_ret": float(mean_absolute_error(y_reg_valid.iloc[:, 0], ret_pred_valid)),
            "valid_mae_dd": float(mean_absolute_error(y_reg_valid.iloc[:, 1], dd_pred_valid)),
            "valid_policy_mean": float(np.nanmean(policy_scores)),
        }

        self.head_features = tuple(available_head_features)
        self.trained_state = {"regime_metrics": regime_metrics, "head_metrics": head_metrics}
        return self.trained_state

    # ---------- Policy ----------

    def _apply_policy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map soft score to {LONG, FLAT, SHORT} and size positions."""
        cfg = self.policy_cfg
        out = df.copy()

        # ex-ante volatility (ATR-smooth rolling mean)
        out["ex_ante_vol"] = (
            out.groupby("ticker")["atr_smooth"]
            .transform(lambda s: s.rolling(cfg.ex_ante_vol_window, min_periods=1).mean())
            .replace(0.0, np.nan)
        )

        if cfg.use_rescaled_score:
            window = max(cfg.rescale_window, 20)

            def _rolling_z(series: pd.Series) -> pd.Series:
                roll = series.rolling(window, min_periods=max(window // 3, 5))
                mean = roll.mean()
                std = roll.std(ddof=0).replace(0.0, np.nan)
                return ((series - mean) / std).fillna(0.0)

            out["policy_score_rescaled"] = (
                out.groupby("ticker")["policy_score"].transform(_rolling_z) * cfg.score_scale
            )
            effective_score = out["policy_score_rescaled"]
        else:
            effective_score = out["policy_score"]

        out["effective_score"] = effective_score

        # base position sizing with effective score
        base = cfg.size_k * effective_score / out["ex_ante_vol"]
        out["base_position_size"] = np.clip(base, -cfg.size_max, cfg.size_max).fillna(0.0)

        def decide(row: pd.Series) -> str:
            score = float(row["effective_score"])
            if cfg.theta_flat_low is not None and abs(score) < cfg.theta_flat_low:
                return "FLAT"

            if int(row.get("I_vr_and_vs", 0)) == 1:
                score *= 0.8
            if int(row.get("I_bd_late", 0)) == 1:
                score *= 1.1

            if score >= cfg.theta_long:
                return "LONG"

            peak_prob = float(row.get("state_prob_Peak", 0.0))
            if score <= cfg.theta_short and (int(row.get("I_bd_early", 0)) == 1 or peak_prob >= cfg.restrict_short_on_peak_prob):
                return "SHORT"

            return "FLAT"

        out["action"] = out.apply(decide, axis=1)
        out["position_size"] = np.where(out["action"] == "FLAT", 0.0, out["base_position_size"])
        out["position_size"] = np.clip(out["position_size"], -cfg.size_max, cfg.size_max)
        return out

    # ---------- Inference ----------

    def predict(self, df: pl.DataFrame) -> pd.DataFrame:
        """
        Predict per-row:
          - pred_rebound_prob
          - pred_expected_ret_pct
          - pred_expected_dd_pct
          - policy_score
          - action, position_size
        Returns pandas DataFrame aligned to input rows.
        """
        if self.head_model is None or self.head_multi is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        
        df = df.sort(["ticker", "date"])

        # Join state probabilities
        state_probs = self.regime.predict_probabilities(
            df.select(["ticker", "date", *DEFAULT_REGIME_FEATURES])
        )

        df = df.join(state_probs, on=["ticker", "date"], how="left")
        for col in STATE_PROB_COLS:
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null(0.0))

        missing = [c for c in self.head_features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns for prediction: {missing}")

        # 정책 적용에 필요한 컬럼들도 포함 (중복 제거)
        needed_for_policy = {"atr_smooth", "I_vr_and_vs", "I_bd_late", "I_bd_early", "state_prob_Peak"}
        cols_to_select = list(set(["ticker", "date"] + list(self.head_features) + list(needed_for_policy)))

        pdf = (
            df.select(cols_to_select)
              .to_pandas()
              .replace([np.inf, -np.inf], np.nan)
              .fillna(0.0)
        )

        # Head predictions
        probs = self.head_model.predict_proba(pdf[list(self.head_features)])[:, 1]
        reg_preds = self.head_multi.predict(pdf[list(self.head_features)])

        pdf["pred_rebound_prob"] = probs
        pdf["pred_expected_ret_pct"] = reg_preds[:, 0]
        pdf["pred_expected_dd_pct"] = reg_preds[:, 1]
        pdf["policy_score"] = _soft_policy_score(
            pdf["pred_rebound_prob"].to_numpy(),
            pdf["pred_expected_ret_pct"].to_numpy(),
            pdf["pred_expected_dd_pct"].to_numpy(),
            self.policy_cfg.risk_aversion,
        )

        # 정책 컬럼이 여전히 없는 경우에만 0으로 채움
        for col in needed_for_policy - set(pdf.columns):
            pdf[col] = 0.0

        return self._apply_policy(pdf)


# ---------------------------
# CLI entry (quiet by default)
# ---------------------------

def main() -> Dict[str, Dict[str, float]]:
    """
    Train the full architecture with current config and print a short summary
    only if verbose=True in the config. Returns metrics dict.
    """
    config = FullArchitectureConfig()  # set verbose=True if you want minimal logs
    model = M002FullArchitecture(config=config)
    metrics = model.train()

    if config.verbose:
        print("=== Regime ===")
        for k, v in metrics["regime_metrics"].items():
            print(f"{k}: {v:.4f}")
        print("=== Head ===")
        for k, v in metrics["head_metrics"].items():
            print(f"{k}: {v:.4f}")

    # Optional: persist
    try:
        from pathlib import Path
        import joblib
        save_dir = Path("models/saved")
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / f"m002_full_architecture_{config.multitask.market}_{'_'.join(map(str, config.multitask.years))}.pkl"
        joblib.dump(model, model_path)
    except Exception:
        # Silently ignore persistence failures in quiet mode
        if config.verbose:
            import traceback
            traceback.print_exc()

    return metrics


if __name__ == "__main__":
    main()


__all__ = [
    "M002FullArchitecture",
    "FullArchitectureConfig",
    "PolicyConfig",
]
