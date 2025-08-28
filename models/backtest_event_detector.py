#!/usr/bin/env python3
"""
Event Detector Backtesting Script

Integrates trained EventDetector with the backtester for performance evaluation
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import polars as pl
import numpy as np
import torch
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.event_detector import EventDetector, EventDetectorTrainer, create_event_detector
from data.dataset_builder import build_dataset
from backtester.backtester import (
    BacktestConfig, UniverseRule, SignalRule, ExecutionRule, PortfolioRule,
    backtest, plot_equity, plot_drawdown, plot_monthly_heatmap,
    plot_rolling_sharpe, plot_contrib_by_ticker
)


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest Event Detector Model")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--market", type=str, default="KR", choices=["KR", "US"],
                       help="Market to analyze")
    parser.add_argument("--test_years", type=str, default="2020",
                       help="Test years (comma-separated)")
    parser.add_argument("--max_tickers", type=int, default=3000,
                       help="Maximum number of tickers")
    
    # Backtesting parameters
    parser.add_argument("--top_n", type=int, default=20,
                       help="Number of top predictions to select")
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="Minimum probability threshold")
    parser.add_argument("--fee_bps", type=float, default=8.0,
                       help="Trading fees in basis points")
    parser.add_argument("--slippage_bps", type=float, default=5.0,
                       help="Slippage in basis points")
    
    parser.add_argument("--output_dir", type=str, default="reports/backtest_event_detector",
                       help="Output directory for results")
    
    return parser.parse_args()


def prepare_signals(df: pl.DataFrame, model_path: str) -> pl.DataFrame:
    """
    Generate trading signals using trained EventDetector
    """
    print("📊 Loading trained model...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    feature_names = checkpoint['feature_names']
    
    # Create and load model
    model = create_event_detector(**model_config)
    trainer = EventDetectorTrainer(model, device=device)
    trainer.load_model(model_path)
    
    print(f"Model features: {feature_names}")
    
    # Prepare features
    features, labels, categorical_features, _ = trainer.prepare_data(df)
    
    # Make predictions
    print("🎯 Generating predictions...")
    predictions, probabilities = trainer.predict(features, categorical_features)
    
    # Add predictions to dataframe
    df_with_signals = df.with_columns([
        pl.Series("event_pred", predictions),  # 0, 1, 2 -> Down, Normal, Up
        pl.Series("prob_down", probabilities[:, 0]),
        pl.Series("prob_normal", probabilities[:, 1]),
        pl.Series("prob_up", probabilities[:, 2])
    ])
    
    # Create trading signals
    # Strategy 1: Long only on high-confidence up predictions
    df_with_signals = df_with_signals.with_columns([
        # Long signal: high probability of up event
        pl.col("prob_up").alias("signal_long"),
        
        # Short signal: high probability of down event  
        pl.col("prob_down").alias("signal_short"),
        
        # Combined signal: up prob - down prob
        (pl.col("prob_up") - pl.col("prob_down")).alias("signal_combined"),
        
        # Event signal: max(up_prob, down_prob) for any event
        pl.max_horizontal(["prob_up", "prob_down"]).alias("signal_event")
    ])
    
    return df_with_signals


def run_backtest_strategy(df: pl.DataFrame, signal_col: str, args) -> Dict[str, Any]:
    """Run backtest for a specific signal strategy"""
    
    output_dir = Path(args.output_dir) / signal_col
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure backtest
    config = BacktestConfig(
        label_col="futret_1",
        signal_col=signal_col,
        universe=UniverseRule(
            top_k_per_day=1000,  # Pre-filter universe
            min_turnover=1e6,    # Minimum liquidity
            min_price=1000       # Minimum price (KRW)
        ),
        signal=SignalRule(
            select_top_n=args.top_n,
            min_threshold=args.threshold,
            long_only=True
        ),
        execution=ExecutionRule(mode="next_open_to_close"),
        portfolio=PortfolioRule(
            weighting="equal",
            max_gross_leverage=1.0,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps
        ),
        outdir=output_dir
    )
    
    print(f"\n🚀 Running backtest for {signal_col}...")
    print(f"  Signal threshold: {args.threshold}")
    print(f"  Top N selections: {args.top_n}")
    print(f"  Fees: {args.fee_bps} bps, Slippage: {args.slippage_bps} bps")
    
    # Run backtest
    results = backtest(df, config)
    
    # Generate plots
    print(f"📈 Generating plots...")
    plot_equity(results, show=False)
    plot_drawdown(results, show=False)
    plot_monthly_heatmap(results, show=False)
    plot_rolling_sharpe(results, window=60, show=False)
    plot_contrib_by_ticker(results, top=20, show=False)
    
    return results


def main():
    args = parse_args()
    
    # Parse test years
    test_years = [int(y.strip()) for y in args.test_years.split(",")]
    
    print("=" * 80)
    print("🔍 Event Detector Backtesting")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Market: {args.market}")
    print(f"Test Years: {test_years}")
    print(f"Max Tickers: {args.max_tickers}")
    print("=" * 80)
    
    # Load test data
    print("\n📊 Loading test dataset...")
    df = build_dataset(
        years=test_years,
        market=args.market,
        max_tickers=args.max_tickers,
        feature_set="v2",
        label_horizon=1,
        label_task="classification",
        label_thresh=0.05,
        verbose=False
    )
    
    print(f"Test dataset shape: {df.shape}")
    
    # Generate signals
    df_with_signals = prepare_signals(df, args.model_path)
    
    # Analyze signal distribution
    print(f"\n📊 Signal Analysis:")
    for signal_col in ["signal_long", "signal_short", "signal_combined", "signal_event"]:
        signal_stats = df_with_signals.select([
            pl.col(signal_col).mean().alias("mean"),
            pl.col(signal_col).std().alias("std"),
            pl.col(signal_col).min().alias("min"),
            pl.col(signal_col).max().alias("max"),
            (pl.col(signal_col) > args.threshold).mean().alias("above_threshold")
        ]).to_pandas().iloc[0]
        
        print(f"  {signal_col}:")
        print(f"    Mean: {signal_stats['mean']:.4f}, Std: {signal_stats['std']:.4f}")
        print(f"    Range: [{signal_stats['min']:.4f}, {signal_stats['max']:.4f}]")
        print(f"    Above threshold ({args.threshold}): {signal_stats['above_threshold']:.1%}")
    
    # Run backtests for different strategies
    strategies = {
        "signal_long": "Long Only (Up Events)",
        "signal_event": "Event Detection (Any Event)",
        "signal_combined": "Long/Short (Up - Down)"
    }
    
    all_results = {}
    
    for signal_col, description in strategies.items():
        print(f"\n" + "="*60)
        print(f"Strategy: {description}")
        print("="*60)
        
        try:
            results = run_backtest_strategy(df_with_signals, signal_col, args)
            all_results[signal_col] = results
            
            # Print summary
            summary = results['summary']
            print(f"\n📈 Results Summary:")
            print(f"  Days: {summary['days']}")
            print(f"  Annual Return: {summary['ret_annual']:.1%}")
            print(f"  Annual Volatility: {summary['vol_annual']:.1%}")
            print(f"  Sharpe Ratio: {summary['sharpe']:.2f}")
            print(f"  Max Drawdown: {summary['max_drawdown']:.1%}")
            print(f"  Final Equity: {summary['final_equity']:.2f}x")
            print(f"  Avg Positions: {summary['avg_n_positions']:.1f}")
            
        except Exception as e:
            print(f"❌ Error running backtest for {signal_col}: {e}")
            continue
    
    # Compare strategies
    if len(all_results) > 1:
        print(f"\n" + "="*80)
        print("📊 Strategy Comparison")
        print("="*80)
        
        comparison_df = []
        for signal_col, results in all_results.items():
            summary = results['summary']
            comparison_df.append({
                'Strategy': strategies[signal_col],
                'Return': f"{summary['ret_annual']:.1%}",
                'Volatility': f"{summary['vol_annual']:.1%}",
                'Sharpe': f"{summary['sharpe']:.2f}",
                'Max DD': f"{summary['max_drawdown']:.1%}",
                'Final Equity': f"{summary['final_equity']:.2f}x"
            })
        
        comparison_df = pd.DataFrame(comparison_df)
        print(comparison_df.to_string(index=False))
    
    print(f"\n✅ Backtesting completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    
    return all_results


if __name__ == "__main__":
    results = main()
