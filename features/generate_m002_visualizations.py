#!/usr/bin/env python3
"""
Generate M002 feature visualizations - Combined mode with all events.

Usage:
    python features/generate_m002_visualizations.py --tickers AAPL MSFT GOOGL --start 2023-01-01
    python features/generate_m002_visualizations.py --tickers AAPL MSFT --start 2023-01-01 --end 2023-12-31
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.M002_FeatureExplorer import generate_combined_visualizations


def main():
    parser = argparse.ArgumentParser(description="Generate M002 combined feature visualizations")
    parser.add_argument(
        "--tickers", "-t",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL"],  # Default tickers for demo
        help="Ticker symbols (e.g., AAPL MSFT GOOGL, default: AAPL MSFT GOOGL)"
    )
    parser.add_argument(
        "--start", "-s",
        default="2023-01-01",  # Default start date
        help="Start date (YYYY-MM-DD, default: 2023-01-01)"
    )
    parser.add_argument(
        "--end", "-e",
        help="End date (YYYY-MM-DD), optional"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="reports/m002",
        help="Output directory (default: reports/m002)"
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Data interval (default: 1d)"
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show download progress"
    )
    parser.add_argument(
        "--dropna",
        action="store_true",
        help="Drop rows with NaN feature values"
    )

    args = parser.parse_args()

    # Generate combined visualizations (all events together)
    generate_combined_visualizations(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        interval=args.interval,
        progress=args.progress,
        dropna=args.dropna,
    )


if __name__ == "__main__":
    main()
