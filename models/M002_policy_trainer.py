#!/usr/bin/env python3
"""
Utility CLI to recalibrate the policy component of the M002 Full Architecture.
Loads an existing trained model, rebuilds the dataset, and performs a quick
grid-search over policy hyperparameters (score_scale, size_k) while keeping the
regime & head models fixed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import joblib
import polars as pl

from data.dataset_builder import build_dataset
from models.M002_FullArchitecture import (
    M002FullArchitecture,
    FullArchitectureConfig,
    PolicyConfig,
    years_to_slug,
)


def _parse_float_list(arg: Optional[str]) -> Optional[List[float]]:
    if not arg:
        return None
    values = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values or None


def _parse_years(arg: Optional[str]) -> Optional[List[int]]:
    if not arg:
        return None
    years = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(token))
    return years or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-train the M002 policy layer only.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/saved") / f"m002_full_architecture_US_{years_to_slug(range(2000, 2019))}.pkl",
        help="Path to the trained M002 full architecture model.",
    )
    parser.add_argument(
        "--output-model-path",
        type=Path,
        default=None,
        help="Where to save the updated model (defaults to --model-path).",
    )
    parser.add_argument(
        "--market",
        type=str,
        default=None,
        help="Market to use when rebuilding the dataset (default: model config value).",
    )
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="Comma-separated years or ranges (e.g., '2010-2013,2015'). Defaults to model config years.",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=500,
        help="Optional cap on tickers when rebuilding the dataset.",
    )
    parser.add_argument(
        "--score-grid",
        type=str,
        default=None,
        help="Comma-separated list of score_scale candidates.",
    )
    parser.add_argument(
        "--size-grid",
        type=str,
        default=None,
        help="Comma-separated list of size_k candidates.",
    )
    parser.add_argument(
        "--target-flat",
        type=float,
        default=None,
        help="Override target_flat_ratio before tuning.",
    )
    parser.add_argument(
        "--target-short",
        type=float,
        default=None,
        help="Override target_short_ratio before tuning.",
    )
    parser.add_argument(
        "--policy-reset",
        action="store_true",
        help="Reset policy config to fresh defaults before tuning.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose policy debugging logs.",
    )
    args = parser.parse_args()

    if args.debug:
        os.environ["M002_POLICY_DEBUG"] = "1"

    print(f"[Policy Trainer] Loading model from {args.model_path}")
    model: M002FullArchitecture = joblib.load(args.model_path)

    if args.policy_reset:
        print("[Policy Trainer] Resetting policy config to defaults.")
        model.policy_cfg = PolicyConfig()

    if args.target_flat is not None:
        model.policy_cfg.target_flat_ratio = float(args.target_flat)
    if args.target_short is not None:
        model.policy_cfg.target_short_ratio = float(args.target_short)

    market = args.market or model.config.multitask.market
    years = _parse_years(args.years) or list(model.config.multitask.years)
    feature_set = model.config.feature_set
    horizon = model.config.horizon
    normalize = model.config.normalize_features

    print(f"[Policy Trainer] Building dataset market={market}, years={years}, feature_set={feature_set}")
    dataset = build_dataset(
        years=years,
        market=market,
        max_tickers=args.max_tickers,
        feature_set=feature_set,
        label_horizon=horizon,
        label_task="regression",
        verbose=False,
        normalize_features=normalize,
    )
    if isinstance(dataset, tuple):
        dataset = dataset[0]
    if isinstance(dataset, pl.LazyFrame):
        dataset = dataset.collect()

    score_grid = _parse_float_list(args.score_grid)
    size_grid = _parse_float_list(args.size_grid)

    print("[Policy Trainer] Re-calibrating policy hyperparameters...")
    result = model.train_policy_only(
        df=dataset,
        score_scale_grid=score_grid,
        size_k_grid=size_grid,
    )

    best_summary = result.get("best_summary", {})
    best_cfg = result.get("best_config", model.policy_cfg)
    print("[Policy Trainer] Best policy config:")
    print(
        json.dumps(
            {
                "score_scale": best_cfg.score_scale,
                "size_k": best_cfg.size_k,
                "target_short_ratio": best_cfg.target_short_ratio,
                "target_flat_ratio": best_cfg.target_flat_ratio,
                "long_ratio": best_summary.get("long_ratio"),
                "short_ratio": best_summary.get("short_ratio"),
                "flat_ratio": best_summary.get("flat_ratio"),
                "long_size_mean": best_summary.get("long_size_mean"),
                "policy_mean": best_summary.get("policy_mean"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    output_path = args.output_model_path or args.model_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"[Policy Trainer] Saved updated model to {output_path}")


if __name__ == "__main__":
    main()
