from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(font_scale=1.3, font='NanumGothic', style='whitegrid')

DEFAULT_OUTPUT = Path("reports/paper_plots/figures/compare_methods.png")
DEFAULT_CACHE = Path("reports/paper_plots/figures/.aapl_cache.csv")
LOOKBACK_DAYS = 90  # 3 months for clearer visualization


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------
def _download_aapl_data(lookback_days: int = LOOKBACK_DAYS, use_cache: bool = True) -> pd.DataFrame:
    """Download AAPL data from Yahoo Finance with caching."""
    # Try to use cache first
    if use_cache and DEFAULT_CACHE.exists():
        print(f"Using cached data from {DEFAULT_CACHE}")
        df = pd.read_csv(DEFAULT_CACHE, parse_dates=['date'])
        if len(df) >= lookback_days:
            return df.tail(lookback_days).reset_index(drop=True)
    
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Install `yfinance` to pull AAPL history.")

    print("Downloading fresh data from Yahoo Finance...")
    end = datetime.utcnow()
    start = end - timedelta(days=int(lookback_days * 1.8))
    
    try:
        df = yf.download(
            "AAPL",
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            timeout=10,
        )
    except Exception as e:
        print(f"Warning: Yahoo Finance download failed: {e}")
        if DEFAULT_CACHE.exists():
            print("Falling back to cached data...")
            df = pd.read_csv(DEFAULT_CACHE, parse_dates=['date'])
            if len(df) >= lookback_days:
                return df.tail(lookback_days).reset_index(drop=True)
        raise ValueError("No AAPL data available (download failed and no cache found)")
    
    if df.empty:
        if DEFAULT_CACHE.exists():
            print("Empty response, falling back to cached data...")
            df = pd.read_csv(DEFAULT_CACHE, parse_dates=['date'])
            if len(df) >= lookback_days:
                return df.tail(lookback_days).reset_index(drop=True)
        raise ValueError("No AAPL data returned from Yahoo Finance.")

    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = (
        df.reset_index()
        .rename(columns=str.lower)
        .sort_values("date")
        .reset_index(drop=True)
    )
    
    # Save to cache
    DEFAULT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEFAULT_CACHE, index=False)
    print(f"Cached data saved to {DEFAULT_CACHE}")
    
    return df.tail(lookback_days).reset_index(drop=True)


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features for regime detection."""
    df = df.copy()
    
    # Price-based features
    df['slope'] = df['close'].pct_change(3)
    df['curvature'] = df['slope'].diff()
    df['volatility'] = df['close'].pct_change().rolling(10).std()
    
    # Volume features
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Events (simplified from )
    df['event_momentum_burst'] = (df['slope'] > df['slope'].quantile(0.85)).astype(int)
    df['event_mean_revert'] = (df['curvature'] < df['curvature'].quantile(0.15)).astype(int)
    df['event_volume_surge'] = (df['volume_ratio'] > 1.5).astype(int)
    df['event_breakdown'] = ((df['slope'] < df['slope'].quantile(0.15)) & 
                             (df['rsi'] < 35)).astype(int)
    
    df = df.fillna(0.0)
    return df


def _simple_markov_regimes(df: pd.DataFrame) -> np.ndarray:
    """Simple 3-state Markov regime with random transitions."""
    np.random.seed(42)
    transition = np.array([
        [0.75, 0.15, 0.10],  # Bull -> Bull/Range/Shock
        [0.20, 0.65, 0.15],  # Range -> Bull/Range/Shock
        [0.25, 0.25, 0.50],  # Shock -> Bull/Range/Shock
    ])
    
    states = np.zeros(len(df), dtype=int)
    state = 0  # Start with Bull
    
    for i in range(len(df)):
        states[i] = state
        state = np.random.choice(3, p=transition[state])
    
    return states


def _morphological_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """5-state morphological regime classification (-inspired)."""
    df = df.copy()
    
    # Regime conditions based on _RegimeClassifier logic
    # Accumulation: low price position, low volatility, low volume
    cond_accum = (
        (df['close'] < df['close'].rolling(60).quantile(0.4)) &
        (df['volatility'] < df['volatility'].median()) &
        (df['volume_ratio'] < 1.0)
    )
    
    # EarlyUp: momentum burst or mean revert event
    cond_early_up = (
        (df['event_momentum_burst'] == 1) |
        (df['event_mean_revert'] == 1) |
        ((df['slope'] > 0) & (df['rsi'] > 40) & (df['rsi'] < 60))
    )
    
    # Peak: high price, high RSI, declining momentum
    cond_peak = (
        (df['close'] > df['close'].rolling(60).quantile(0.8)) &
        (df['rsi'] > 60) &
        (df['curvature'] < 0)
    )
    
    # Distribution: high price but declining
    cond_distribution = (
        (df['close'] > df['close'].rolling(60).quantile(0.6)) &
        (df['slope'] < 0) &
        (df['rsi'] < 60)
    )
    
    # LateDown: breakdown or low price with declining momentum
    cond_late_down = (
        (df['event_breakdown'] == 1) |
        ((df['close'] < df['close'].rolling(60).quantile(0.45)) &
         (df['rsi'] < 50))
    )
    
    # Priority-based regime assignment (matching  priorities)
    priorities = {
        'Accumulation': 4,
        'EarlyUp': 3,
        'Peak': 5,
        'Distribution': 1,
        'LateDown': 2,
    }
    
    df['score_Accumulation'] = cond_accum.astype(int) * priorities['Accumulation']
    df['score_EarlyUp'] = cond_early_up.astype(int) * priorities['EarlyUp']
    df['score_Peak'] = cond_peak.astype(int) * priorities['Peak']
    df['score_Distribution'] = cond_distribution.astype(int) * priorities['Distribution']
    df['score_LateDown'] = cond_late_down.astype(int) * priorities['LateDown']
    
    score_cols = ['score_Accumulation', 'score_EarlyUp', 'score_Peak', 
                  'score_Distribution', 'score_LateDown']
    
    df['regime'] = df[score_cols].idxmax(axis=1).str.replace('score_', '')
    df.loc[df[score_cols].sum(axis=1) == 0, 'regime'] = 'Accumulation'  # Default
    
    return df


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
def _draw_candles(ax: plt.Axes, df: pd.DataFrame, width: float = 0.6) -> None:
    """Draw candlestick chart."""
    dates = mdates.date2num(df["date"])
    
    for x, (_, row) in zip(dates, df.iterrows()):
        open_, high, low, close = row['open'], row['high'], row['low'], row['close']
        color = "#2a9d8f" if close >= open_ else "#e63946"
        
        # Wick
        ax.plot([x, x], [low, high], color=color, linewidth=1.2, solid_capstyle="round")
        
        # Body
        body_height = max(abs(close - open_), 0.001)
        lower = min(open_, close)
        rect = Rectangle(
            (x - width / 2, lower),
            width,
            body_height,
            facecolor=color,
            edgecolor=color,
            alpha=0.85,
        )
        ax.add_patch(rect)
    
    ax.set_xlim(dates[0] - 0.6, dates[-1] + 0.6)


def _mark_event_points(mask: pd.Series) -> pd.Series:
    """Mark event transition points."""
    mask = mask.fillna(False)
    shifted = mask.shift(1, fill_value=False)
    return mask & ~shifted


def draw_markov_panel(axes, df: pd.DataFrame):
    """Draw Markov regime switching (top 2 subplots)."""
    states = _simple_markov_regimes(df)
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    labels = ["Bull", "Range", "Shock"]
    state_names = {0: "Bull", 1: "Range", 2: "Shock"}
    
    # Price panel
    ax_price = axes[0]
    _draw_candles(ax_price, df)
    
    # Draw regime background spans
    last_state = states[0]
    start_idx = 0
    
    for idx in range(1, len(states)):
        if states[idx] != states[idx - 1]:
            ax_price.axvspan(
                df['date'].iloc[start_idx],
                df['date'].iloc[idx],
                color=colors[last_state],
                alpha=0.15,
            )
            start_idx = idx
            last_state = states[idx]
    
    ax_price.axvspan(
        df['date'].iloc[start_idx],
        df['date'].iloc[-1],
        color=colors[last_state],
        alpha=0.15,
    )
    
    ax_price.set_ylabel("Price ($)", fontsize=11)
    ax_price.set_title(
        "① Markov Regime Switching (Hidden States, Random Transitions)",
        fontsize=13,
        fontweight="bold"
    )
    ax_price.grid(alpha=0.25)
    
    # Add legend with colored patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], alpha=0.4, label=labels[i]) 
                      for i in range(3)]
    ax_price.legend(handles=legend_elements, loc="upper left", framealpha=0.9)
    
    # State transition panel
    ax_states = axes[1]
    state_y = [0, 1, 2]
    ax_states.plot(df['date'], states, color='#34495e', linewidth=2, drawstyle='steps-post')
    ax_states.set_yticks(state_y)
    ax_states.set_yticklabels(labels)
    ax_states.set_ylabel("State", fontsize=11)
    ax_states.grid(alpha=0.25)

def draw_morphological_panel(axes, df: pd.DataFrame):
    """Draw  morphological regime classification (bottom 3 subplots)."""
    df = _morphological_regimes(df)
    
    # Regime colors (matching _constants.STATE_NAMES order)
    regime_colors = {
        'Accumulation': '#4d908e',
        'EarlyUp': '#577590',
        'Peak': '#f9c74f',
        'Distribution': '#f9844a',
        'LateDown': '#d62828',
    }
    
    # Price panel with regimes
    ax_price = axes[0]
    _draw_candles(ax_price, df)
    
    # Draw regime background spans
    last_regime = df['regime'].iloc[0]
    start_idx = 0
    
    for idx in range(1, len(df)):
        if df['regime'].iloc[idx] != df['regime'].iloc[idx - 1]:
            ax_price.axvspan(
                df['date'].iloc[start_idx],
                df['date'].iloc[idx],
                color=regime_colors[last_regime],
                alpha=0.2,
            )
            start_idx = idx
            last_regime = df['regime'].iloc[idx]
    
    ax_price.axvspan(
        df['date'].iloc[start_idx],
        df['date'].iloc[-1],
        color=regime_colors[last_regime],
        alpha=0.2,
    )
    
    # Event markers
    event_configs = {
        'event_momentum_burst': ('#ff6b6b', 'Momentum↑'),
        'event_mean_revert': ('#4d908e', 'Mean-Revert'),
        'event_volume_surge': ('#f9c74f', 'Volume↑'),
        'event_breakdown': ('#e63946', 'Breakdown'),
    }
    
    for event_col, (color, label) in event_configs.items():
        markers = _mark_event_points(df[event_col].astype(bool))
        if markers.any():
            ax_price.scatter(
                df.loc[markers, 'date'],
                df.loc[markers, 'close'],
                color=color,
                s=60,
                edgecolor='white',
                linewidth=1.2,
                zorder=5,
                alpha=0.9,
            )
    
    ax_price.set_ylabel("Price ($)", fontsize=11)
    ax_price.set_title(
        "②  Morphological Labeling (5 Regimes + Events + LSTM Probabilities)",
        fontsize=13,
        fontweight="bold"
    )
    ax_price.grid(alpha=0.25)
    
    # Legend for regimes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=regime_colors[name], alpha=0.5, label=name) 
                      for name in regime_colors.keys()]
    ax_price.legend(handles=legend_elements, loc="upper left", framealpha=0.9, ncol=2)
    
    # Event panel
    ax_events = axes[1]
    event_series = pd.DataFrame({
        'Momentum↑': df['event_momentum_burst'],
        'Mean-Revert': df['event_mean_revert'],
        'Volume↑': df['event_volume_surge'],
        'Breakdown': df['event_breakdown'],
    })
    
    for idx, (name, series) in enumerate(event_series.items()):
        event_times = df.loc[series == 1, 'date']
        ax_events.scatter(event_times, [idx] * len(event_times), 
                         s=80, marker='|', linewidths=2.5,
                         label=name, alpha=0.8)
    
    ax_events.set_yticks(range(len(event_series.columns)))
    ax_events.set_yticklabels(event_series.columns)
    ax_events.set_ylabel("Events", fontsize=11)
    ax_events.set_xlabel("Date", fontsize=11)
    ax_events.grid(alpha=0.25)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    print("Downloading AAPL data...")
    df = _download_aapl_data(LOOKBACK_DAYS)
    
    print("Computing features...")
    df = _compute_features(df)
    
    print("Creating comparison plot...")
    fig = plt.figure(figsize=(14, 12))
    
    # Create grid: 2 rows for Markov, 3 rows for Morphological
    gs = fig.add_gridspec(5, 1, height_ratios=[2, 1, 2, 1, 1], hspace=0.25)
    
    axes = [
        fig.add_subplot(gs[0, 0]),  # Markov price
        fig.add_subplot(gs[1, 0]),  # Markov states
        fig.add_subplot(gs[2, 0]),  # Morphological price
        fig.add_subplot(gs[3, 0]),  # Morphological signals
    ]
    
    draw_markov_panel(axes[0:2], df)
    draw_morphological_panel(axes[2:5], df)
    
    # Format date axes
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    for ax in [axes[1], axes[3]]:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    
    # Hide x-axis labels for all but bottom subplot
    for ax in axes[:-1]:
        ax.set_xticklabels([])
    
    fig.suptitle(
        "Regime Classification: Markov Switching vs. Morphological Labeling",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )
    
    fig.savefig(DEFAULT_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved to {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
