#!/usr/bin/env python3
"""
Generate simple publication-ready price charts with 5 event markers for M002 features.

Usage:
    python features/generate_simple_event_plots.py --tickers AAPL MSFT --output-dir reports/simple_events
    python features/generate_simple_event_plots.py --tickers ALLY --start 2020-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from M002_FeatureExplorer import M002FeatureExplorer


def generate_simple_event_plots(
    tickers: Sequence[str],
    start: str = "2020-01-01",
    end: str | None = None,
    output_dir: str = "reports/simple_events",
    height: int = 600,
    width: int = 1000,
    show_legend: bool = True,
    font_size: int = 14,
    use_numeric_xaxis: bool = True,
) -> None:
    """
    Generate simple price charts with 5 event markers for publication.

    Args:
        tickers: List of ticker symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD), optional
        output_dir: Output directory for plots
        height: Plot height in pixels
        width: Plot width in pixels
        show_legend: Whether to show legend
        font_size: Base font size
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"🎯 Generating simple event plots for {len(tickers)} tickers...")
    print(f"📁 Output directory: {output_path.absolute()}")
    print(f"📅 Date range: {start} to {end or 'latest'}")

    # Initialize explorer (mode doesn't matter for event generation)
    explorer = M002FeatureExplorer(
        tickers=tickers,
        start=start,
        end=end,
        mode="buyer",  # All events are calculated regardless of mode
    )

    # Build features
    print("🔄 Building features and detecting events...")
    features_df = explorer.build_features()
    print(f"✅ Features built: {len(features_df)} rows")

    # Generate plots for each ticker
    successful_plots = 0
    for ticker in tickers:
        ticker_upper = ticker.upper()
        print(f"📊 Processing {ticker_upper}...")

        try:
            # Generate simple event plot
            fig = explorer.plot_simple_price_with_events(
                ticker=ticker_upper,
                height=height,
                width=width,
                show_legend=show_legend,
                font_size=font_size,
                use_numeric_xaxis=use_numeric_xaxis,
            )

            # Save as HTML (interactive)
            html_file = output_path / f"{ticker_upper}_simple_events.html"
            fig.write_html(str(html_file))
            print(f"  ✅ Saved HTML: {html_file.name}")

            # Save as PNG (static image for publications)
            png_file = output_path / f"{ticker_upper}_simple_events.png"
            fig.write_image(str(png_file), scale=2)  # High DPI for publications
            print(f"  ✅ Saved PNG: {png_file.name}")

            successful_plots += 1

        except Exception as e:
            print(f"  ❌ Failed to process {ticker_upper}: {e}")
            continue

    # Generate summary
    print("\n📋 Summary:")
    print(f"  • Total tickers processed: {len(tickers)}")
    print(f"  • Successful plots: {successful_plots}")
    print(f"  • Output directory: {output_path.absolute()}")

    # Event statistics
    event_counts = {}
    for ticker in tickers:
        ticker_upper = ticker.upper()
        subset = features_df[features_df["ticker"] == ticker_upper]

        if not subset.empty:
            event_counts[ticker_upper] = {}
            for event_col in [
                "event_local_vol_spike",
                "event_rebound_candidate",
                "event_volume_regain",
                "event_exhaustion_candidate",
                "event_breakdown_risk"
            ]:
                if event_col in subset.columns:
                    count = int(subset[event_col].sum())
                    event_counts[ticker_upper][event_col] = count

    if event_counts:
        print("\n🎯 Event counts:")
        for ticker, events in event_counts.items():
            total_events = sum(events.values())
            print(f"  • {ticker}: {total_events} total events")
            for event_name, count in events.items():
                if count > 0:
                    print(f"    - {event_name}: {count}")

    print("\n✨ Done! Check the output directory for your publication-ready plots.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate simple price charts with 5 event markers for M002 features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python features/generate_simple_event_plots.py --tickers AAPL MSFT GOOGL
  python features/generate_simple_event_plots.py --tickers ALLY --start 2020-01-01 --end 2023-12-31 --output-dir reports/ally_events
  python features/generate_simple_event_plots.py --tickers SPY QQQ --no-legend --font-size 12
        """
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="Ticker symbols to process (e.g., AAPL MSFT GOOGL)"
    )

    parser.add_argument(
        "--start",
        default="2020-01-01",
        help="Start date (YYYY-MM-DD, default: 2020-01-01)"
    )

    parser.add_argument(
        "--end",
        help="End date (YYYY-MM-DD, default: latest available)"
    )

    parser.add_argument(
        "--output-dir",
        default="reports/simple_events",
        help="Output directory for plots (default: reports/simple_events)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="Plot height in pixels (default: 600)"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1000,
        help="Plot width in pixels (default: 1000)"
    )

    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Hide legend in plots"
    )

    parser.add_argument(
        "--font-size",
        type=int,
        default=14,
        help="Base font size (default: 14)"
    )

    parser.add_argument(
        "--use-numeric-xaxis",
        action="store_true",
        default=True,
        help="Use numeric indices for x-axis instead of dates (default: True)"
    )

    parser.add_argument(
        "--use-date-xaxis",
        action="store_true",
        help="Use actual dates for x-axis (overrides --use-numeric-xaxis)"
    )

    args = parser.parse_args()

    generate_simple_event_plots(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        show_legend=not args.no_legend,
        font_size=args.font_size,
        use_numeric_xaxis=args.use_date_xaxis is False,
    )


if __name__ == "__main__":
    main()
