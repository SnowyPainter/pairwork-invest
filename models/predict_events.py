#!/usr/bin/env python3
"""
Event Detection Inference Script

Usage:
    python models/predict_events.py --model_path models/checkpoints/event_detector_final.pth --data_path data.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import polars as pl
import numpy as np
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.event_detector import EventDetector, EventDetectorTrainer, create_event_detector


def parse_args():
    parser = argparse.ArgumentParser(description="Event Detection Inference")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to data file (CSV or Parquet)")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output path for predictions (default: auto-generate)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Prediction threshold (default: 0.5)")
    parser.add_argument("--batch_size", type=int, default=1000,
                       help="Batch size for inference")
    
    return parser.parse_args()


def load_data(data_path: str) -> pl.DataFrame:
    """Load data from file"""
    data_path = Path(data_path)
    
    if data_path.suffix == '.csv':
        df = pl.read_csv(data_path)
    elif data_path.suffix == '.parquet':
        df = pl.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    return df


def main():
    args = parse_args()
    
    print("=" * 80)
    print("🔍 Event Detection Inference")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Threshold: {args.threshold}")
    print("=" * 80)
    
    # Load data
    print("\n📊 Loading data...")
    df = load_data(args.data_path)
    print(f"Data shape: {df.shape}")
    
    # Load model
    print("\n🧠 Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load checkpoint to get model configuration
    checkpoint = torch.load(args.model_path, map_location=device)
    model_config = checkpoint['model_config']
    feature_names = checkpoint['feature_names']
    
    print(f"Model features ({len(feature_names)}):")
    for i, feat in enumerate(feature_names, 1):
        print(f"  {i:2d}. {feat}")
    
    # Create model with loaded configuration
    model = create_event_detector(**model_config)
    trainer = EventDetectorTrainer(model, device=device)
    trainer.load_model(args.model_path)
    
    # Check if all required features are available
    missing_features = [feat for feat in feature_names if feat not in df.columns]
    if missing_features:
        print(f"\n⚠️  Warning: Missing features in data:")
        for feat in missing_features:
            print(f"    - {feat}")
        print("Proceeding with available features only...")
        
        # Filter to available features
        available_features = [feat for feat in feature_names if feat in df.columns]
        feature_names = available_features
    
    # Extract features
    print(f"\n🔄 Extracting features...")
    features_df = df.select(feature_names).to_pandas()
    features = features_df.values.astype(np.float32)
    
    # Handle NaN values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Features shape: {features.shape}")
    print(f"NaN values handled: ✓")
    
    # Make predictions
    print(f"\n🎯 Making predictions...")
    predictions, probabilities = trainer.predict(features)
    
    # Apply custom threshold
    if args.threshold != 0.5:
        predictions = (probabilities >= args.threshold).astype(int)
        print(f"Applied custom threshold: {args.threshold}")
    
    # Add predictions to dataframe
    df_with_predictions = df.with_columns([
        pl.Series("event_prediction", predictions),
        pl.Series("event_probability", probabilities)
    ])
    
    # Summary statistics
    n_total = len(predictions)
    n_events = predictions.sum()
    event_rate = n_events / n_total
    avg_prob = probabilities.mean()
    max_prob = probabilities.max()
    
    print(f"\n📈 Prediction Summary:")
    print(f"  Total samples: {n_total:,}")
    print(f"  Predicted events: {n_events:,} ({event_rate:.1%})")
    print(f"  Average probability: {avg_prob:.3f}")
    print(f"  Max probability: {max_prob:.3f}")
    
    # Show some examples
    print(f"\n🔍 Sample Predictions:")
    sample_df = df_with_predictions.select([
        "date", "ticker", "event_prediction", "event_probability"
    ]).head(10).to_pandas()
    print(sample_df.to_string(index=False))
    
    # Save results
    if args.output_path is None:
        data_path = Path(args.data_path)
        output_path = data_path.parent / f"{data_path.stem}_predictions{data_path.suffix}"
    else:
        output_path = Path(args.output_path)
    
    print(f"\n💾 Saving predictions...")
    if output_path.suffix == '.csv':
        df_with_predictions.write_csv(output_path)
    elif output_path.suffix == '.parquet':
        df_with_predictions.write_parquet(output_path)
    else:
        # Default to parquet
        output_path = output_path.with_suffix('.parquet')
        df_with_predictions.write_parquet(output_path)
    
    print(f"Predictions saved to: {output_path}")
    
    # High-probability events
    high_prob_events = df_with_predictions.filter(
        pl.col("event_probability") >= 0.8
    ).select([
        "date", "ticker", "event_prediction", "event_probability"
    ]).sort("event_probability", descending=True)
    
    if len(high_prob_events) > 0:
        print(f"\n🚨 High-Confidence Events (prob ≥ 0.8): {len(high_prob_events)}")
        print(high_prob_events.head(20).to_pandas().to_string(index=False))
    
    print("\n" + "=" * 80)
    print("✅ Event Detection Completed Successfully!")
    print("=" * 80)
    
    return df_with_predictions


if __name__ == "__main__":
    predictions_df = main()
