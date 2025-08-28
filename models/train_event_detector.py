#!/usr/bin/env python3
"""
Event Detector Training Script

Usage:
    python models/train_event_detector.py --market KR --years 2018,2019,2020 --max_tickers 100
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import polars as pl

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.event_detector import EventDetector, EventDetectorTrainer, create_event_detector
from data.dataset_builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train Event Detector Model")
    
    # Data parameters
    parser.add_argument("--market", type=str, default="KR", choices=["KR", "US"], 
                       help="Market to analyze")
    parser.add_argument("--years", type=str, default="2018,2019,2020", 
                       help="Years to include (comma-separated)")
    parser.add_argument("--max_tickers", type=int, default=100, 
                       help="Maximum number of tickers")
    parser.add_argument("--feature_set", type=str, default="v2", 
                       help="Feature set version")
    parser.add_argument("--event_thresh", type=float, default=0.05, 
                       help="Event threshold (5% = 0.05)")
    
    # Model parameters
    parser.add_argument("--d_model", type=int, default=192, 
                       help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=8, 
                       help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=512, 
                       help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, 
                       help="Dropout rate")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, 
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, 
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, 
                       help="Weight decay")
    parser.add_argument("--val_split", type=float, default=0.2, 
                       help="Validation split")
    parser.add_argument("--early_stopping", type=int, default=15, 
                       help="Early stopping patience")
    
    # Output parameters
    parser.add_argument("--model_name", type=str, default="event_detector", 
                       help="Model name for saving")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints", 
                       help="Output directory")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse years
    years = [int(y.strip()) for y in args.years.split(",")]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🚀 Event Detector Training")
    print("=" * 80)
    print(f"Market: {args.market}")
    print(f"Years: {years}")
    print(f"Max Tickers: {args.max_tickers}")
    print(f"Event Threshold: {args.event_thresh * 100}%")
    print(f"Model: FT-Transformer (d_model={args.d_model}, heads={args.n_heads})")
    print(f"Training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print("=" * 80)
    
    # Load dataset
    print("\n📊 Loading dataset...")
    df = build_dataset(
        years=years,
        market=args.market,
        max_tickers=args.max_tickers,
        feature_set=args.feature_set,
        label_horizon=1,
        label_task="classification",
        label_thresh=args.event_thresh,
        verbose=False
    )
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns}")
    
    # Check event distribution
    event_dist = df.select(pl.col("label_1d_cls").value_counts()).to_pandas()
    print(f"\nEvent distribution:")
    print(event_dist)
    
    # Use comprehensive feature set (will be filtered in EventDetector)
    print(f"\n🎯 Using comprehensive feature set (auto-filtered)")
    print(f"   Available columns: {len(df.columns)}")
    available_features = [
        'rel_range', 'obv', 'parkinson20', 'macd', 'macd_hist', 'rsi10',
        'atr5', 'tr14', 'vol_roc5', 'vol_z20', 'stochd14', 'rsi6', 'roc10', 'roc20' 
    ]
    
    # Create trainer first (model will be created during training)
    print(f"\n🧠 Initializing EventDetector trainer...")
    n_tickers = df.select("ticker").n_unique() if "ticker" in df.columns else 3000
    print(f"   Unique tickers: {n_tickers}")
    
    # Create a placeholder model to initialize trainer
    temp_model = create_event_detector(
        n_features=50,  # Placeholder, will be updated
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        n_tickers=n_tickers
    )
    
    trainer = EventDetectorTrainer(temp_model)
    
    # Train model (model will be recreated with correct dimensions)
    print(f"\n Training model...")
    history = trainer.train(
        df=df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping,
        feature_cols=available_features,
        train_years=years[:-1],  # Last year is validation
        val_years=[years[-1]]
    )
    
    # Evaluate on full dataset
    print(f"\n📈 Evaluating model...")
    results = trainer.evaluate(df)
    
    # Save final model
    model_path = output_dir / f"{args.model_name}_final.pth"
    trainer.save_model(str(model_path))
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = output_dir / f"{args.model_name}_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to {history_path}")
    
    # Save results
    results_path = output_dir / f"{args.model_name}_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("🎉 Training Completed Successfully!")
    print("=" * 80)
    print(f"Best Model: {model_path}")
    print(f"Final AUC: {results['auc']:.4f}")
    print(f"Final Accuracy: {results['accuracy']:.4f}")
    print(f"Macro Precision: {results['macro_precision']:.4f}")
    print(f"Macro Recall: {results['macro_recall']:.4f}")
    print(f"Macro F1-Score: {results['macro_f1']:.4f}")
    print("=" * 80)
    
    return trainer, results, history


if __name__ == "__main__":
    trainer, results, history = main()
