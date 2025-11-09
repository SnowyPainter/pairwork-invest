from __future__ import annotations

from copy import deepcopy
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


def _format_year_segment(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def years_to_slug(years: Sequence[int]) -> str:
    """Condense a collection of years into a compact slug (handles contiguous spans)."""
    unique_years = sorted({int(year) for year in years})
    if not unique_years:
        return "unknown-years"

    segments: List[str] = []
    segment_start = unique_years[0]
    previous = unique_years[0]

    for year in unique_years[1:]:
        if year == previous + 1:
            previous = year
            continue
        segments.append(_format_year_segment(segment_start, previous))
        segment_start = previous = year

    segments.append(_format_year_segment(segment_start, previous))
    return "_".join(segments)


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
    target_flat_ratio: float = 0.45
    target_short_ratio: float = 0.1
    score_scale: float = 2.5
    rescale_window: int = 60
    ex_ante_vol_window: int = 5
    size_k: float = 1.2
    size_max: float = 1.0
    restrict_short_on_peak_prob: float = 0.35
    use_rescaled_score: bool = True
    risk_aversion: float = 0.5



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
        if getattr(self.policy_cfg, "risk_aversion", None) is None:
            self.policy_cfg.risk_aversion = getattr(self.config.multitask, "risk_aversion", 0.5)

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

    def _make_head_fold(
        self,
        df_pd: pd.DataFrame,
        train_slice: slice,
        valid_slice: slice,
        feature_cols: Sequence[str],
        ret_col: str,
        dd_col: str,
    ) -> Dict[str, pd.DataFrame]:
        train_df = df_pd.iloc[train_slice].reset_index(drop=True)
        valid_df = df_pd.iloc[valid_slice].reset_index(drop=True)
        return {
            "X_train": train_df[list(feature_cols)],
            "X_valid": valid_df[list(feature_cols)],
            "y_cls_train": train_df["label_rebound_bin"].astype(int),
            "y_cls_valid": valid_df["label_rebound_bin"].astype(int),
            "y_reg_train": train_df[[ret_col, dd_col]],
            "y_reg_valid": valid_df[[ret_col, dd_col]],
        }

    def _build_head_folds(
        self,
        df_pd: pd.DataFrame,
        feature_cols: Sequence[str],
        ret_col: str,
        dd_col: str,
    ) -> List[Dict[str, pd.DataFrame]]:
        cfg = self.config.multitask
        n = len(df_pd)
        if n == 0:
            return []

        if not getattr(cfg, "use_walk_forward", False):
            split_idx = max(int(n * (1 - cfg.valid_size)), 1)
            train_slice = slice(0, split_idx)
            valid_slice = slice(split_idx, n)
            return [self._make_head_fold(df_pd, train_slice, valid_slice, feature_cols, ret_col, dd_col)]

        train_size = max(int(n * cfg.walk_forward_train_ratio), 1)
        valid_size = max(int(n * cfg.walk_forward_valid_ratio), 1)
        step = max(int(n * cfg.walk_forward_step_ratio), 1)
        step = max(step, valid_size)

        if train_size + valid_size > n:
            split_idx = max(int(n * (1 - cfg.valid_size)), 1)
            return [self._make_head_fold(df_pd, slice(0, split_idx), slice(split_idx, n), feature_cols, ret_col, dd_col)]

        folds: List[Dict[str, pd.DataFrame]] = []
        start = 0
        while start + train_size + valid_size <= n:
            train_slice = slice(start, start + train_size)
            valid_slice = slice(start + train_size, start + train_size + valid_size)
            folds.append(self._make_head_fold(df_pd, train_slice, valid_slice, feature_cols, ret_col, dd_col))
            start += step

        if not folds:
            split_idx = max(int(n * (1 - cfg.valid_size)), 1)
            folds.append(self._make_head_fold(df_pd, slice(0, split_idx), slice(split_idx, n), feature_cols, ret_col, dd_col))
        return folds

    def _prepare_head_dataset(self) -> Dict[str, pd.DataFrame]:
        df = self._load_base_dataframe()
        state_probs = self.regime.predict_probabilities(
            df.select(["ticker", "date", *DEFAULT_REGIME_FEATURES])
        )
        df = df.join(state_probs, on=["ticker", "date"], how="left")

        ret_col = f"ret_{self.config.horizon}d_pct"
        dd_col = f"dd_{self.config.horizon}d_pct"

        available_cols = set(df.columns)
        available_head_features: List[str] = [f for f in self.head_features if f in available_cols]
        if not available_head_features:
            raise ValueError("No head features available in the dataset after join.")

        use_cols = ["ticker", "date", "label_rebound_bin", ret_col, dd_col, *available_head_features]
        df_pd = (
            df.select(use_cols)
              .to_pandas()
              .replace([np.inf, -np.inf], np.nan)
        )

        targets = df_pd[[ret_col, dd_col]]
        mask = targets.notna().all(axis=1)
        df_pd = df_pd.loc[mask].sort_values(["date", "ticker"]).reset_index(drop=True)
        if df_pd.empty:
            raise ValueError("No valid samples after removing NaN targets.")

        folds = self._build_head_folds(df_pd, available_head_features, ret_col, dd_col)
        if not folds:
            raise ValueError("Failed to generate walk-forward folds for head dataset.")

        return {
            "folds": folds,
            "available_head_features": tuple(available_head_features),
        }

    def _default_head_params(self) -> Dict[str, Dict]:
        base = dict(
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
        reg = dict(
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
        return {"clf": base, "reg": reg}

    def _fit_head_models(
        self,
        fold: Dict[str, pd.DataFrame],
        clf_params: Dict,
        reg_params: Dict,
    ) -> Tuple[Dict[str, float], lgb.LGBMClassifier, MultiOutputRegressor]:
        X_train = fold["X_train"]
        X_valid = fold["X_valid"]
        y_cls_train = fold["y_cls_train"]
        y_cls_valid = fold["y_cls_valid"]
        y_reg_train = fold["y_reg_train"]
        y_reg_valid = fold["y_reg_valid"]

        head_model = lgb.LGBMClassifier(**clf_params)
        head_model.fit(
            X_train,
            y_cls_train,
            eval_set=[(X_valid, y_cls_valid)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        base_reg = lgb.LGBMRegressor(**reg_params)
        head_multi = MultiOutputRegressor(base_reg)
        head_multi.fit(X_train, y_reg_train)

        prob_valid = head_model.predict_proba(X_valid)[:, 1]
        reg_valid = head_multi.predict(X_valid)
        ret_pred_valid, dd_pred_valid = reg_valid[:, 0], reg_valid[:, 1]
        policy_scores = _soft_policy_score(
            prob_valid, ret_pred_valid, dd_pred_valid, self.policy_cfg.risk_aversion
        )

        from sklearn.metrics import average_precision_score, mean_absolute_error

        head_metrics = {
            "valid_avg_precision": float(average_precision_score(y_cls_valid, prob_valid)),
            "valid_mae_ret": float(mean_absolute_error(y_reg_valid.iloc[:, 0], ret_pred_valid)),
            "valid_mae_dd": float(mean_absolute_error(y_reg_valid.iloc[:, 1], dd_pred_valid)),
            "valid_policy_mean": float(np.nanmean(policy_scores)),
        }
        return head_metrics, head_model, head_multi

    def tune_head_hyperparams(
        self,
        n_trials: int = 40,
        timeout: Optional[int] = None,
        sampler: str = "tpe",
    ) -> Dict[str, Dict]:
        """AutoML-style tuning of LightGBM head via Optuna."""
        try:  # pragma: no cover - optional dependency
            import optuna
        except ImportError as exc:
            raise ImportError(
                "Optuna is required for hyperparameter tuning. Install with `pip install optuna`."
            ) from exc

        data = self._prepare_head_dataset()
        folds = data["folds"]
        if not folds:
            raise ValueError("No walk-forward folds available for head tuning.")
        self.head_features = data["available_head_features"]
        defaults = self._default_head_params()

        sampler_obj = (
            optuna.samplers.RandomSampler()
            if sampler.lower() == "random"
            else optuna.samplers.TPESampler()
        )

        study = optuna.create_study(direction="maximize", sampler=sampler_obj)

        def objective(trial: optuna.Trial) -> float:
            clf_params = deepcopy(defaults["clf"])
            reg_params = deepcopy(defaults["reg"])

            clf_params.update(
                learning_rate=trial.suggest_float("clf_learning_rate", 0.01, 0.2, log=True),
                num_leaves=trial.suggest_int("clf_num_leaves", 63, 255),
                feature_fraction=trial.suggest_float("clf_feature_fraction", 0.6, 1.0),
                bagging_fraction=trial.suggest_float("clf_bagging_fraction", 0.6, 1.0),
                bagging_freq=trial.suggest_int("clf_bagging_freq", 1, 10),
                n_estimators=trial.suggest_int("clf_n_estimators", 300, 900),
                reg_alpha=trial.suggest_float("clf_reg_alpha", 1e-3, 1.0, log=True),
                reg_lambda=trial.suggest_float("clf_reg_lambda", 1e-3, 1.0, log=True),
                min_child_samples=trial.suggest_int("clf_min_child_samples", 10, 80),
            )

            reg_params.update(
                learning_rate=trial.suggest_float("reg_learning_rate", 0.01, 0.2, log=True),
                num_leaves=trial.suggest_int("reg_num_leaves", 63, 255),
                feature_fraction=trial.suggest_float("reg_feature_fraction", 0.6, 1.0),
                bagging_fraction=trial.suggest_float("reg_bagging_fraction", 0.6, 1.0),
                bagging_freq=trial.suggest_int("reg_bagging_freq", 1, 10),
                n_estimators=trial.suggest_int("reg_n_estimators", 400, 1000),
                reg_alpha=trial.suggest_float("reg_reg_alpha", 1e-3, 1.0, log=True),
                reg_lambda=trial.suggest_float("reg_reg_lambda", 1e-3, 1.0, log=True),
                min_child_samples=trial.suggest_int("reg_min_child_samples", 10, 80),
            )

            metrics, _, _ = self._fit_head_models(folds[-1], clf_params, reg_params)
            score = metrics["valid_avg_precision"] + 0.1 * metrics["valid_policy_mean"]
            trial.set_user_attr("metrics", metrics)
            return score

        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best_trial = study.best_trial
        best_metrics = best_trial.user_attrs.get("metrics", {})

        merged_clf = deepcopy(defaults["clf"])
        merged_reg = deepcopy(defaults["reg"])
        merged_clf.update({k.replace("clf_", ""): v for k, v in best_trial.params.items() if k.startswith("clf_")})
        merged_reg.update({k.replace("reg_", ""): v for k, v in best_trial.params.items() if k.startswith("reg_")})

        self.best_head_params = {"clf": merged_clf, "reg": merged_reg}
        self.best_head_metrics = best_metrics
        self.best_head_study = study

        if self.config.verbose:
            print(f"[TUNE] Best objective: {best_trial.value:.6f}")
            print("[TUNE] Metrics:", best_metrics)

        return {
            "best_params": self.best_head_params,
            "best_metrics": best_metrics,
            "best_value": best_trial.value,
        }

    def train(self) -> Dict[str, Dict[str, float]]:
        """Train regime classifier + head models. Returns validation metrics."""
        if self.config.verbose:
            print("[A] Training regime classifier ...")
        regime_metrics = self.regime.train()

        if self.config.verbose:
            print("[B] Preparing head dataset ...")
        data = self._prepare_head_dataset()
        folds = data["folds"]
        if not folds:
            raise ValueError("Head dataset preparation produced no folds.")
        self.head_features = data["available_head_features"]

        # Always use default parameters (skip Optuna tuning)
        params = self._default_head_params()

        fold_metrics: List[Dict[str, float]] = []
        latest_head_model: Optional[lgb.LGBMClassifier] = None
        latest_head_multi: Optional[MultiOutputRegressor] = None

        for fold_idx, fold in enumerate(folds):
            metrics, head_model, head_multi = self._fit_head_models(fold, params["clf"], params["reg"])
            metrics_with_fold = dict(metrics)
            metrics_with_fold["fold"] = fold_idx
            fold_metrics.append(metrics_with_fold)

            if fold_idx == len(folds) - 1:
                latest_head_model = head_model
                latest_head_multi = head_multi

        if latest_head_model is None or latest_head_multi is None:
            raise RuntimeError("Failed to train head models on walk-forward folds.")

        self.head_model = latest_head_model
        self.head_multi = latest_head_multi

        self.trained_state = {
            "regime_metrics": regime_metrics,
            "head_metrics": fold_metrics,
            "head_params": params,
        }
        return self.trained_state

    # ---------- Policy ----------

    def _apply_policy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adaptive threshold policy application (balanced signal distribution)."""
        out = df.copy()

        # === 1️⃣ ex-ante volatility 계산 ===
        out["ex_ante_vol"] = (
            out.groupby("ticker")["atr_smooth"]
            .transform(lambda s: s.rolling(self.policy_cfg.ex_ante_vol_window, min_periods=1).mean())
            .replace(0.0, np.nan)
        )

        # === 2️⃣ score rescaling ===
        if self.policy_cfg.use_rescaled_score:
            window = max(self.policy_cfg.rescale_window, 20)

            def _rolling_z(series: pd.Series) -> pd.Series:
                roll = series.rolling(window, min_periods=max(window // 3, 5))
                mean = roll.mean()
                std = roll.std(ddof=0).replace(0.0, np.nan)
                return ((series - mean) / std).fillna(0.0)

            out["policy_score_rescaled"] = (
                out.groupby("ticker")["policy_score"].transform(_rolling_z) * self.policy_cfg.score_scale
            )
            effective_score = out["policy_score_rescaled"]
        else:
            effective_score = out["policy_score"]

        out["effective_score"] = effective_score

        # === 3️⃣ adaptive threshold 설정 ===
        valid_scores = out["effective_score"].replace([np.inf, -np.inf], np.nan).dropna()
        long_q = np.quantile(valid_scores, 1 - self.policy_cfg.target_short_ratio)
        short_q = np.quantile(valid_scores, self.policy_cfg.target_short_ratio)
        flat_low = np.quantile(valid_scores, 0.5 - self.policy_cfg.target_flat_ratio / 2)
        flat_high = np.quantile(valid_scores, 0.5 + self.policy_cfg.target_flat_ratio / 2)

        # === 4️⃣ decision logic ===
        def decide(row: pd.Series) -> str:
            score = float(row["effective_score"])
            peak_prob = float(row.get("state_prob_Peak", 0.0))

            # 이벤트 기반 가중치
            if int(row.get("I_vr_and_vs", 0)) == 1:
                score *= 0.8
            if int(row.get("I_bd_late", 0)) == 1:
                score *= 1.1
            if int(row.get("I_bd_early", 0)) == 1:
                score *= 1.1

            # adaptive threshold에 따라 결정
            if score >= long_q:
                return "LONG"
            elif score <= short_q or (peak_prob >= self.policy_cfg.restrict_short_on_peak_prob and score < flat_low):
                return "SHORT"
            elif flat_low < score < flat_high:
                return "FLAT"
            else:
                return "FLAT"

        out["action"] = out.apply(decide, axis=1)

        # === 5️⃣ 포지션 사이즈 계산 ===
        base = self.policy_cfg.size_k * effective_score / out["ex_ante_vol"]
        out["position_size"] = np.where(out["action"] == "FLAT", 0.0, base)
        out["position_size"] = np.clip(out["position_size"], -self.policy_cfg.size_max, self.policy_cfg.size_max).fillna(0.0)

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
        year_slug = years_to_slug(config.multitask.years)
        model_path = save_dir / f"m002_full_architecture_{config.multitask.market}_{year_slug}.pkl"
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
    "years_to_slug",
]
