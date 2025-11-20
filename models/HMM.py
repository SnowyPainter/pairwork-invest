"""
Simple Gaussian Hidden Markov Model utilities for market regime detection.

Designed for lightweight usage without external HMM dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class GaussianHMM:
    """Minimal multivariate Gaussian HMM with full EM training."""

    def __init__(self, n_states: int = 3, random_state: int = 42) -> None:
        self.n_states = int(n_states)
        self.random_state = int(random_state)
        self.scaler = StandardScaler()

        self.pi: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None

    # ----------------------------
    # helpers
    # ----------------------------
    def _init_params(self, X: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]

        self.pi = np.full(self.n_states, 1.0 / self.n_states)
        A = rng.random((self.n_states, self.n_states))
        self.A = A / A.sum(axis=1, keepdims=True)

        mu = rng.standard_normal((self.n_states, n_features))
        sigma = np.array([np.eye(n_features) for _ in range(self.n_states)])

        self.mu = mu
        self.sigma = sigma

    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Vectorized multivariate Gaussian pdf."""
        n_features = x.shape[1]
        cov = cov + np.eye(n_features) * 1e-6
        det = np.linalg.det(cov)
        if det <= 0:
            det = 1e-6
        inv = np.linalg.pinv(cov)
        diff = x - mean
        expo = np.einsum("...i,ij,...j->...", diff, inv, diff)
        norm = np.sqrt(((2 * np.pi) ** n_features) * det)
        return np.exp(-0.5 * expo) / norm

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        alpha = np.zeros((n_samples, self.n_states))
        c = np.zeros(n_samples)

        for i in range(self.n_states):
            alpha[0, i] = self.pi[i] * self._gaussian_pdf(X[0:1], self.mu[i], self.sigma[i])[0]
        c[0] = alpha[0].sum() or 1e-12
        alpha[0] /= c[0]

        for t in range(1, n_samples):
            for j in range(self.n_states):
                prob = self._gaussian_pdf(X[t:t+1], self.mu[j], self.sigma[j])[0]
                alpha[t, j] = prob * (alpha[t-1] @ self.A[:, j])
            c[t] = alpha[t].sum() or 1e-12
            alpha[t] /= c[t]
        return alpha, c

    def _backward(self, X: np.ndarray, c: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        beta = np.zeros((n_samples, self.n_states))
        beta[-1] = 1.0 / c[-1]

        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                probs = np.array([
                    self._gaussian_pdf(X[t+1:t+2], self.mu[j], self.sigma[j])[0]
                    for j in range(self.n_states)
                ])
                beta[t, i] = np.sum(self.A[i] * probs * beta[t + 1])
            beta[t] /= c[t]
        return beta

    # ----------------------------
    # API
    # ----------------------------
    def fit(self, X: np.ndarray, n_iter: int = 75, tol: float = 1e-4) -> "GaussianHMM":
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = np.asarray(X, dtype=float)
        if X.shape[0] < self.n_states:
            raise ValueError("Insufficient samples for HMM training.")

        X_scaled = self.scaler.fit_transform(X)
        self._init_params(X_scaled)

        prev_ll = -np.inf
        for _ in range(max(1, n_iter)):
            alpha, c = self._forward(X_scaled)
            beta = self._backward(X_scaled, c)

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)

            xi = np.zeros((X_scaled.shape[0] - 1, self.n_states, self.n_states))
            for t in range(X_scaled.shape[0] - 1):
                for i in range(self.n_states):
                    probs = np.array([
                        self._gaussian_pdf(X_scaled[t+1:t+2], self.mu[j], self.sigma[j])[0]
                        for j in range(self.n_states)
                    ])
                    xi[t, i] = alpha[t, i] * self.A[i] * probs * beta[t + 1]
                total = xi[t].sum() or 1e-12
                xi[t] /= total

            self.pi = gamma[0] / gamma[0].sum()
            self.A = xi.sum(axis=0)
            self.A /= self.A.sum(axis=1, keepdims=True)

            for j in range(self.n_states):
                weight = gamma[:, j][:, None]
                total_weight = weight.sum()
                if total_weight <= 0:
                    continue
                mean = (weight * X_scaled).sum(axis=0) / total_weight
                diff = X_scaled - mean
                weighted = diff * weight
                cov = (weighted.T @ diff) / total_weight
                self.mu[j] = mean
                self.sigma[j] = cov + np.eye(X_scaled.shape[1]) * 1e-6

            ll = np.sum(np.log(c))
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        return self

    def predict_states(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = self.scaler.transform(X)
        n_samples = X.shape[0]
        V = np.zeros((n_samples, self.n_states))
        back = np.zeros((n_samples, self.n_states), dtype=int)

        for i in range(self.n_states):
            V[0, i] = np.log(self.pi[i]) + np.log(self._gaussian_pdf(X[0:1], self.mu[i], self.sigma[i])[0])

        for t in range(1, n_samples):
            for j in range(self.n_states):
                probs = np.log(self.A[:, j]) + np.log(
                    self._gaussian_pdf(X[t:t+1], self.mu[j], self.sigma[j])[0]
                )
                best_prev = np.argmax(V[t-1] + probs)
                V[t, j] = V[t-1, best_prev] + probs[best_prev]
                back[t, j] = best_prev

        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(V[-1])
        for t in range(n_samples - 2, -1, -1):
            states[t] = back[t + 1, states[t + 1]]
        return states

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = self.scaler.transform(X)
        alpha, _ = self._forward(X)
        beta = self._backward(X, np.ones(X.shape[0]))
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma


# =============================================================================
# Market-facing wrapper
# =============================================================================

STATE_LABELS = ("UP", "DOWN", "FLAT")


@dataclass
class MarketRegimeHMM:
    n_states: int = 3
    random_state: int = 42

    def __post_init__(self) -> None:
        self.model = GaussianHMM(n_states=self.n_states, random_state=self.random_state)
        self._is_trained = False

    # feature engineering
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        price = df["close"].astype(float)
        volume = df["volume"].astype(float).replace(0, np.nan).ffill().fillna(1.0)

        returns = price.pct_change().fillna(0.0)
        vol = returns.rolling(20).std().fillna(returns.std())

        delta = price.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = 100 - 100 / (1 + rs)
        rsi = rsi.fillna(50.0)

        vol_ratio = volume / volume.rolling(20).mean()
        vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        features = np.column_stack([
            returns.to_numpy(),
            vol.to_numpy(),
            ((rsi - 50) / 50).to_numpy(),
            vol_ratio.pct_change().fillna(0.0).to_numpy(),
        ])
        return features

    def fit(self, df: pd.DataFrame, n_iter: int = 75) -> "MarketRegimeHMM":
        data = df.sort_values("date").reset_index(drop=True)
        X = self._extract_features(data)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        if X.shape[0] < self.n_states * 5:
            raise ValueError("Not enough history to train HMM.")
        self.model.fit(X, n_iter=n_iter)
        self._is_trained = True
        self._full_features = self._extract_features(data)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("HMM must be fitted before calling predict.")
        ordered = df.sort_values("date").reset_index(drop=True)
        features = self._extract_features(ordered)
        mask = ~np.isnan(features).any(axis=1)
        if not mask.any():
            raise ValueError("All feature rows contain NaN; cannot predict.")

        valid_features = features[mask]
        state_ids = self.model.predict_states(valid_features)
        probs = self.model.predict_proba(valid_features)

        out = ordered.loc[mask, ["date"]].copy()
        out["hmm_state_id"] = state_ids
        out["hmm_state"] = [STATE_LABELS[idx % len(STATE_LABELS)] for idx in state_ids]
        out["prob_up"] = probs[:, 0]
        out["prob_down"] = probs[:, 1] if probs.shape[1] > 1 else 0.0
        out["prob_flat"] = probs[:, 2] if probs.shape[1] > 2 else 1.0 - out["prob_up"] - out["prob_down"]
        return out


def detect_market_regimes(df: pd.DataFrame, n_states: int = 3) -> pd.DataFrame:
    model = MarketRegimeHMM(n_states=n_states)
    model.fit(df)
    return model.predict(df)
