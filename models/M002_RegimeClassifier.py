from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple
from tqdm import tqdm

from data.dataset_builder import build_dataset
from .M002_constants import STATE_NAMES, STATE_TO_ID


def balanced_collate_fn(batch: List[Tuple[Tensor, Tensor]], num_classes: int) -> Tuple[Tensor, Tensor]:
    """
    Collate function that creates balanced batches by oversampling minority classes within each batch.

    Args:
        batch: List of (sequence, label) tuples
        num_classes: Number of classes

    Returns:
        Tuple of (sequences, labels) with balanced class distribution
    """
    if not batch:
        raise ValueError("Empty batch")

    sequences, labels = zip(*batch)

    # Group samples by class
    class_samples = {i: [] for i in range(num_classes)}
    for seq, label in zip(sequences, labels):
        class_samples[label.item()].append(seq)

    # Find the maximum number of samples per class in this batch (only for classes with samples)
    available_classes = [i for i, samples in class_samples.items() if samples]
    if not available_classes:
        raise ValueError("No samples found in batch")

    max_samples_per_class = max(len(class_samples[i]) for i in available_classes)

    # Oversample each available class to match max_samples_per_class
    balanced_sequences = []
    balanced_labels = []

    for class_id in available_classes:
        class_seqs = class_samples[class_id]

        # Oversample by repeating existing samples
        oversampled_seqs = class_seqs * (max_samples_per_class // len(class_seqs))
        remaining = max_samples_per_class % len(class_seqs)
        if remaining > 0:
            oversampled_seqs.extend(class_seqs[:remaining])

        balanced_sequences.extend(oversampled_seqs)
        balanced_labels.extend([class_id] * max_samples_per_class)

    # Convert to tensors
    balanced_sequences_tensor = torch.stack(balanced_sequences)
    balanced_labels_tensor = torch.tensor(balanced_labels, dtype=torch.long)

    return balanced_sequences_tensor, balanced_labels_tensor


DEFAULT_REGIME_FEATURES: Sequence[str] = (
    "rsi_smooth",
    "macd_smooth",
    "atr_smooth",
    "volume_z_smooth",
    "ema_spread_rel",
    "ema_spread_rel_slope",
    "atr_rel",
    "delta_atr_rel",
    "volume_z",
    "pos_in_band_rel",
    "bb_width",
    "curv_2",
    "price_slope",
    "price_accel",
    "local_vol_index",
    "delta_rsi",
    "delta_rsi_3",
    "delta_macd",
)

def _assign_regime_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Balanced heuristic regime labeling (priority-based, vectorized, STATE_NAMES aligned)."""

    PRIORITIES = {
        "Accumulation": 4,
        "EarlyUp": 3,
        "Peak": 5,
        "Distribution": 1,
        "LateDown": 2,
    }

    # --- Conditions (slightly relaxed)
    cond_accum = (
        (pl.col("pos_in_band_rel") < 0.4)
        & (pl.col("atr_rel") < 1.2)
        & (pl.col("volume_z") < 1.0)
    )

    cond_early_up = (
        (pl.col("event_volume_regain") == 1)
        | (pl.col("event_rebound_candidate") == 1)
        | (
            (pl.col("ema_spread_rel") > -0.05)
            & (pl.col("price_slope") > -0.01)
            & (pl.col("rsi_smooth") > 40)
        )
    )

    cond_peak = (
        (pl.col("event_exhaustion_candidate") == 1)
        | (pl.col("I_bd_early") == 1)
        | (
            (pl.col("pos_in_band_rel") > 0.8)
            & (pl.col("rsi_smooth") > 55)
            & (pl.col("delta_rsi") < 0.5)
        )
    )

    cond_distribution = (
        (pl.col("event_breakdown_risk") == 1)
        | (
            (pl.col("pos_in_band_rel") > 0.55)
            & (pl.col("price_slope") < 0.0)
            & (pl.col("rsi_smooth") < 60)
        )
    )

    cond_late_down = (
        (pl.col("I_bd_late") == 1)
        | (
            (pl.col("event_rebound_candidate") == 1)
            & (pl.col("curv_2") > -0.5)     # 완화
            & (pl.col("ema_spread_rel") > -0.5)
            & (pl.col("price_slope") < 0.1) # 약한 하락 모멘텀 포함
        )
        | (
            (pl.col("pos_in_band_rel") < 0.45)
            & (pl.col("rsi_smooth") < 50)
            & (pl.col("delta_rsi") < 0)
        )
    )


    # --- Score columns (force to Int32)
    df = df.with_columns([
        pl.when(cond_accum).then(PRIORITIES["Accumulation"]).otherwise(0).cast(pl.Int32).alias("score_Accumulation"),
        pl.when(cond_early_up).then(PRIORITIES["EarlyUp"]).otherwise(0).cast(pl.Int32).alias("score_EarlyUp"),
        pl.when(cond_peak).then(PRIORITIES["Peak"]).otherwise(0).cast(pl.Int32).alias("score_Peak"),
        pl.when(cond_distribution).then(PRIORITIES["Distribution"]).otherwise(0).cast(pl.Int32).alias("score_Distribution"),
        pl.when(cond_late_down).then(PRIORITIES["LateDown"]).otherwise(0).cast(pl.Int32).alias("score_LateDown"),
    ])

    score_cols = [f"score_{s}" for s in STATE_NAMES]

    # --- Find max score index using struct and map_elements
    score_struct = pl.struct(score_cols)

    def get_max_score_index(score_struct):
        """Get index of state with maximum score."""
        scores = [score_struct[col] for col in score_cols]
        max_score = max(scores)
        max_indices = [i for i, score in enumerate(scores) if score == max_score]
        # Return the smallest index (earliest in STATE_NAMES) in case of ties
        return min(max_indices)

    df = df.with_columns([
        score_struct.map_elements(get_max_score_index, return_dtype=pl.Int32).alias("max_idx")
    ])

    df = df.with_columns([
        df["max_idx"].map_elements(lambda i: STATE_NAMES[int(i)], return_dtype=pl.Utf8).alias("regime_state")
    ])

    # Clean-up
    df = df.drop(score_cols + ["max_idx"])

    # Map to ID
    df = df.with_columns([
        pl.col("regime_state").replace(STATE_TO_ID).cast(pl.Int8).alias("regime_state_id")
    ])

    print("✅ Regime Label Distribution:")
    print(df["regime_state"].value_counts().sort("regime_state"))
    return df


@dataclass
class RegimeConfig:
    feature_columns: Sequence[str] = field(default_factory=lambda: DEFAULT_REGIME_FEATURES)
    seq_len: int = 20
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 128
    max_epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    valid_ratio: float = 0.2
    normalize_features: bool = True
    market: str = "US"
    years: Sequence[int] = field(default_factory=lambda: list(range(2000, 2019)))
    random_seed: int = 42
    balance_classes: bool = True
    class_weight_power: float = 0.5
    label_smoothing: float = 0.08
    use_walk_forward: bool = True
    walk_forward_train_ratio: float = 0.7
    walk_forward_valid_ratio: float = 0.1
    walk_forward_step_ratio: float = 0.1

    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RegimeSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.sequences = sequences.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return torch.from_numpy(self.sequences[idx]), torch.tensor(self.labels[idx])


class RegimeLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float, num_classes: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        logits = self.fc(self.dropout(last_hidden))
        return logits


class M002RegimeClassifier:
    def __init__(self, config: Optional[RegimeConfig] = None) -> None:
        self.config = config or RegimeConfig()
        self.model: Optional[RegimeLSTM] = None
        self.feature_columns = list(self.config.feature_columns)
        self.num_classes = len(STATE_NAMES)

    def _compute_class_weights(self, counts: np.ndarray) -> torch.Tensor:
        counts = counts.astype(np.float32)
        counts[counts == 0] = counts[counts > 0].min() if (counts > 0).any() else 1.0
        inv_freq = (counts.sum() / counts)
        inv_pow = inv_freq ** self.config.class_weight_power
        norm = inv_pow / inv_pow.mean()
        return torch.tensor(norm, dtype=torch.float32)

    def _prepare_dataframe(self) -> pl.DataFrame:
        dataset = build_dataset(
            years=self.config.years,
            market=self.config.market,
            feature_set="m002",
            label_horizon=self.config.seq_len,
            label_task="regression",
            drop_na_rows=False,
            normalize_features=self.config.normalize_features,
        )

        if isinstance(dataset, tuple):
            dataset = dataset[0]
        if isinstance(dataset, pl.LazyFrame):
            dataset = dataset.collect()
        elif not isinstance(dataset, pl.DataFrame):
            raise TypeError("Unsupported dataset type returned from build_dataset.")

        dataset = dataset.sort(["ticker", "date"])

        # Fill NaN values in feature columns using pandas for reliable filling
        print(f"Preprocessing {len(self.feature_columns)} feature columns...")
        df_pd = dataset.to_pandas()

        for col in self.feature_columns:
            nan_count = df_pd[col].isna().sum()
            if nan_count > 0:
                print(f"  Filling {nan_count} NaN values in {col}")
                # Forward fill, then backward fill, then fill with 0
                df_pd[col] = df_pd.groupby('ticker')[col].ffill().groupby(df_pd['ticker']).bfill().fillna(0.0)

        # Convert back to polars
        dataset = pl.DataFrame(df_pd)

        # Filter out tickers with too many NaN values
        print("Filtering tickers by data quality...")

        # 각 티커별 NaN 비율 계산
        ticker_nan_stats = []
        for ticker in dataset['ticker'].unique():
            ticker_data = dataset.filter(pl.col('ticker') == ticker)
            total_rows = ticker_data.shape[0]

            nan_counts = []
            for col in self.feature_columns:
                nan_count = ticker_data[col].is_null().sum()
                nan_counts.append(nan_count)

            avg_nan_ratio = sum(nan_counts) / (len(self.feature_columns) * total_rows) if total_rows > 0 else 1.0
            ticker_nan_stats.append((ticker, avg_nan_ratio, total_rows))

        # NaN 비율이 30% 미만이고 최소 50개 이상의 데이터 포인트가 있는 티커만 유지
        good_tickers = [
            ticker for ticker, nan_ratio, row_count in ticker_nan_stats
            if nan_ratio < 0.3 and row_count >= 50
        ]

        print(f"Total tickers: {len(ticker_nan_stats)}")
        print(f"Good quality tickers: {len(good_tickers)}")

        if len(good_tickers) < len(ticker_nan_stats) * 0.5:
            print("Warning: Too many tickers filtered out. Consider adjusting quality thresholds.")

        # 좋은 티커들만 유지
        dataset = dataset.filter(pl.col('ticker').is_in(good_tickers))

        # 남은 NaN 값들 처리
        initial_rows = dataset.shape[0]
        dataset = dataset.drop_nulls(self.feature_columns)
        final_rows = dataset.shape[0]
        dropped_rows = initial_rows - final_rows

        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} additional rows with NaN values")

        print(f"Final dataset: {final_rows} rows from {len(good_tickers)} tickers after quality filtering")

        # Assign regime labels
        print("Assigning regime labels...")
        dataset = _assign_regime_labels(dataset)

        # Check for NaN in regime_state_id and handle
        nan_regime_count = dataset["regime_state_id"].is_null().sum()
        if nan_regime_count > 0:
            print(f"Warning: {nan_regime_count} rows have NaN regime_state_id, dropping them")
            dataset = dataset.drop_nulls(["regime_state_id"])

        final_rows_after_labeling = dataset.shape[0]
        print(f"Final dataset after labeling: {final_rows_after_labeling} rows")

        return dataset

    def _build_sequences(
        self,
        df: pd.DataFrame,
        seq_len: int,
        feature_cols: Sequence[str],
        label_col: str,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
        sequences: List[np.ndarray] = []
        labels: List[int] = []
        meta: List[Tuple[str, pd.Timestamp]] = []

        for ticker, group in df.groupby("ticker"):
            group = group.sort_values("date")
            feats = group[feature_cols].to_numpy(dtype=np.float32)
            lab = group[label_col].to_numpy(dtype=np.int64)
            dates = group["date"].to_numpy()

            if feats.shape[0] < seq_len:
                continue
            for idx in range(seq_len - 1, feats.shape[0]):
                window = feats[idx - seq_len + 1 : idx + 1]
                sequences.append(window)
                labels.append(lab[idx])
                meta.append((ticker, dates[idx]))

        if not sequences:
            raise ValueError("Insufficient data to build regime sequences.")

        seq_array = np.stack(sequences)
        label_array = np.asarray(labels, dtype=np.int64)
        meta_df = pd.DataFrame(meta, columns=["ticker", "date"])
        order = np.argsort(meta_df["date"].to_numpy())
        seq_array = seq_array[order]
        label_array = label_array[order]
        meta_df = meta_df.iloc[order].reset_index(drop=True)
        class_counts = np.bincount(label_array, minlength=self.num_classes)
        return seq_array, label_array, meta_df, class_counts

    def _build_walk_forward_splits(self, num_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create rolling walk-forward train/valid index pairs."""
        if num_samples <= 1:
            return []

        if not self.config.use_walk_forward:
            split_idx = max(int(num_samples * (1 - self.config.valid_ratio)), 1)
            train_idx = np.arange(0, split_idx)
            valid_idx = np.arange(split_idx, num_samples)
            return [(train_idx, valid_idx)]

        train_size = max(int(num_samples * self.config.walk_forward_train_ratio), 1)
        valid_size = max(int(num_samples * self.config.walk_forward_valid_ratio), 1)
        step = max(int(num_samples * self.config.walk_forward_step_ratio), 1)
        step = max(step, valid_size)

        if train_size + valid_size > num_samples:
            split_idx = max(num_samples - valid_size, 1)
            train_idx = np.arange(0, split_idx)
            valid_idx = np.arange(split_idx, num_samples)
            return [(train_idx, valid_idx)]

        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        start = 0
        while start + train_size + valid_size <= num_samples:
            train_idx = np.arange(start, start + train_size)
            valid_idx = np.arange(start + train_size, start + train_size + valid_size)
            splits.append((train_idx, valid_idx))
            start += step

        if not splits:
            split_idx = max(int(num_samples * (1 - self.config.valid_ratio)), 1)
            splits.append((np.arange(0, split_idx), np.arange(split_idx, num_samples)))
        return splits

    def train(self) -> Dict[str, float]:
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        df_pl = self._prepare_dataframe()
        df_pd = df_pl.select(["ticker", "date", "regime_state_id", *self.feature_columns]).to_pandas()
        df_pd["date"] = pd.to_datetime(df_pd["date"])

        print("Validating training data...")
        X_check = df_pd[self.feature_columns].values
        y_check = df_pd["regime_state_id"].values

        if np.isnan(X_check).any():
            raise ValueError(f"NaN values found in features: {np.isnan(X_check).sum()} total NaN values")

        if np.isnan(y_check).any():
            raise ValueError(f"NaN values found in labels: {np.isnan(y_check).sum()} total NaN values")

        print(f"Data validation passed: {X_check.shape[0]} samples, {X_check.shape[1]} features")
        print(f"Label distribution: {np.bincount(y_check.astype(int))}")

        seq_array, label_array, _, class_counts = self._build_sequences(
            df_pd,
            seq_len=self.config.seq_len,
            feature_cols=self.feature_columns,
            label_col="regime_state_id",
        )

        # Validate sequences
        if np.isnan(seq_array).any():
            raise ValueError(f"NaN values found in sequences: {np.isnan(seq_array).sum()} total NaN values")

        if np.isnan(label_array).any():
            raise ValueError(f"NaN values found in sequence labels: {np.isnan(label_array).sum()} total NaN values")

        print(f"Sequence validation passed: {seq_array.shape[0]} sequences, shape {seq_array.shape}")

        print(f"Sequence label distribution: {class_counts}")

        num_samples = seq_array.shape[0]
        splits = self._build_walk_forward_splits(num_samples)
        if not splits:
            raise ValueError("Unable to create walk-forward splits for regime training.")

        device = self.config.device()
        fold_metrics: List[Dict[str, float]] = []
        best_overall_loss = math.inf
        best_overall_state: Optional[Dict[str, Tensor]] = None
        last_fold_state: Optional[Dict[str, Tensor]] = None

        for fold_idx, (train_idx, valid_idx) in enumerate(splits):
            train_seq, train_lbl = seq_array[train_idx], label_array[train_idx]
            valid_seq, valid_lbl = seq_array[valid_idx], label_array[valid_idx]

            if train_seq.size == 0 or valid_seq.size == 0:
                print(f"Skipping fold {fold_idx}: insufficient samples (train={train_seq.shape[0]}, valid={valid_seq.shape[0]})")
                continue

            train_dataset = RegimeSequenceDataset(train_seq, train_lbl)
            valid_dataset = RegimeSequenceDataset(valid_seq, valid_lbl)

            if self.config.balance_classes:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    drop_last=False,
                    collate_fn=lambda batch: balanced_collate_fn(batch, self.num_classes),
                )
            else:
                train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=False)

            valid_loader = DataLoader(valid_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=False)

            model = RegimeLSTM(
                input_dim=len(self.feature_columns),
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                num_classes=self.num_classes,
            ).to(device)

            criterion = nn.CrossEntropyLoss(
                weight=None,
                label_smoothing=self.config.label_smoothing,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

            fold_best_loss = math.inf
            fold_best_state: Optional[Dict[str, Tensor]] = None

            print(f"[Fold {fold_idx + 1}/{len(splits)}] train={train_seq.shape[0]}, valid={valid_seq.shape[0]} sequences")
            for epoch in tqdm(range(self.config.max_epochs), desc=f"Fold {fold_idx + 1}", unit="epoch"):
                model.train()
                total_loss = 0.0
                total_samples = 0

                for X, y in train_loader:
                    X = X.to(device)
                    y = y.to(device)
                    optimizer.zero_grad()
                    logits = model(X)

                    if torch.isnan(logits).any():
                        print(f"Warning: NaN detected in logits at batch {X.size(0)}")
                        continue

                    loss = criterion(logits, y)
                    if torch.isnan(loss):
                        print("Warning: NaN loss detected, skipping batch")
                        continue

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * X.size(0)
                    total_samples += X.size(0)

                avg_train_loss = total_loss / max(total_samples, 1)

                model.eval()
                val_loss = 0.0
                val_samples = 0
                with torch.no_grad():
                    for X, y in valid_loader:
                        X = X.to(device)
                        y = y.to(device)
                        logits = model(X)
                        loss = criterion(logits, y)
                        val_loss += loss.item() * X.size(0)
                        val_samples += X.size(0)

                avg_val_loss = val_loss / max(val_samples, 1)
                if avg_val_loss < fold_best_loss:
                    fold_best_loss = avg_val_loss
                    fold_best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                print(f"    Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

            if fold_best_state is None:
                fold_best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "train_size": int(train_seq.shape[0]),
                    "valid_size": int(valid_seq.shape[0]),
                    "best_valid_loss": float(fold_best_loss),
                }
            )

            if fold_best_loss < best_overall_loss:
                best_overall_loss = fold_best_loss
                best_overall_state = fold_best_state

            last_fold_state = fold_best_state

        final_state = last_fold_state or best_overall_state
        if final_state is None:
            raise RuntimeError("Regime classifier training did not produce a valid state.")

        self.model = RegimeLSTM(
            input_dim=len(self.feature_columns),
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            num_classes=self.num_classes,
        )
        self.model.load_state_dict(final_state)
        self.model.to(device)
        self.model.eval()

        metrics = {
            "num_sequences": int(num_samples),
            "class_counts": class_counts.tolist(),
            "fold_metrics": fold_metrics,
        }
        return metrics

    def _ensure_trained(self) -> RegimeLSTM:
        if self.model is None:
            raise RuntimeError("Regime classifier has not been trained yet.")
        return self.model

    def predict_probabilities(self, df: pl.DataFrame) -> pl.DataFrame:
        model = self._ensure_trained()
        device = self.config.device()
        feature_cols = self.feature_columns
        seq_len = self.config.seq_len

        df = df.sort(["ticker", "date"])
        df = df.drop_nulls(feature_cols)
        pdf = df.select(["ticker", "date", *feature_cols]).to_pandas()
        pdf["date"] = pd.to_datetime(pdf["date"])

        results: List[pd.DataFrame] = []
        model.eval()
        with torch.no_grad():
            for ticker, group in pdf.groupby("ticker"):
                group = group.sort_values("date").reset_index(drop=True)
                feats = group[feature_cols].to_numpy(dtype=np.float32)
                prob_matrix = np.full((group.shape[0], self.num_classes), np.nan, dtype=np.float32)

                if feats.shape[0] >= seq_len:
                    sequences = []
                    indices = []
                    for idx in range(seq_len - 1, feats.shape[0]):
                        window = feats[idx - seq_len + 1 : idx + 1]
                        sequences.append(window)
                        indices.append(idx)

                    if sequences:
                        tensor = torch.from_numpy(np.stack(sequences)).to(device)
                        logits = model(tensor)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                        prob_matrix[np.asarray(indices)] = probs

                if np.any(~np.isnan(prob_matrix)):  # 하나라도 유효한 예측이 있으면
                    valid_idx = np.where(~np.isnan(prob_matrix).any(axis=1))[0]
                    for idx in valid_idx:
                        row_data = {"ticker": ticker, "date": group["date"].iloc[idx]}
                        for i, name in enumerate(STATE_NAMES):
                            row_data[f"state_prob_{name}"] = float(prob_matrix[idx, i])
                        results.append(pd.DataFrame([row_data]))
                else:
                    # 전부 NaN이면 마지막 날짜만 NaN으로 채움
                    row_data = {"ticker": ticker, "date": group["date"].iloc[-1]}
                    for name in STATE_NAMES:
                        row_data[f"state_prob_{name}"] = np.nan
                    results.append(pd.DataFrame([row_data]))

        merged = pd.concat(results, axis=0, ignore_index=True)
        # Ensure all state probability columns are float64 in pandas before conversion
        state_prob_cols = [f"state_prob_{name}" for name in STATE_NAMES]
        for col in state_prob_cols:
            merged[col] = merged[col].astype('float64')

        merged_pl = pl.DataFrame(merged)
        # Convert date column back to Date type to match the original dataset
        merged_pl = merged_pl.with_columns([
            pl.col("date").cast(pl.Date)
        ])
        
        return merged_pl


__all__ = ["RegimeConfig", "M002RegimeClassifier", "DEFAULT_REGIME_FEATURES", "balanced_collate_fn"]
