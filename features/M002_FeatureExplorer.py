from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Literal, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FeatureMode = Literal["buyer", "seller"]

BUYER_EVENT_COLS: Sequence[str] = (
    "event_local_vol_spike",
    "event_rebound_candidate",
    "event_volume_regain",
)

SELLER_EVENT_COLS: Sequence[str] = (
    "event_local_vol_spike",
    "event_exhaustion_candidate",
    "event_breakdown_risk",
)

BUYER_LSTM_FEATURES: Sequence[str] = (
    "rsi_smooth",
    "delta_rsi_rel",
    "macd_smooth",
    "pos_in_band_rel",
    "atr_rel",
    "volume_z",
    "vol_roc",
    "ema_spread_rel",
    "bb_width",
)

SELLER_LSTM_FEATURES: Sequence[str] = (
    "atr_rel",
    "delta_atr_rel",
    "ema_spread_rel",
    "rsi_smooth",
    "macd_smooth",
    "pos_in_band_rel",
    "volume_z",
    "bb_width",
    "local_vol_index",
)

MODE_FEATURE_PRIORITIES: Dict[FeatureMode, Dict[str, Sequence[str]]] = {
    "buyer": {
        "emphasize": ("rsi_smooth", "macd_smooth", "pos_in_band_rel", "atr_rel"),
        "deemphasize": ("ema_spread_rel", "delta_atr_rel"),
    },
    "seller": {
        "emphasize": ("atr_rel", "ema_spread_rel", "macd_smooth", "delta_atr_rel"),
        "deemphasize": ("volume_z", "delta_rsi_rel"),
    },
}


class M002FeatureExplorer:
    """
    Fetches price data from Yahoo Finance, engineers the M002 feature set,
    and offers visual helpers for analysis and event inspection.
    """

    def __init__(
        self,
        tickers: Sequence[str],
        start: str,
        end: Optional[str] = None,
        *,
        interval: str = "1d",
        mode: FeatureMode = "seller",
        auto_adjust: bool = True,
        progress: bool = False,
        dropna: bool = False,
        volume_z_window: int = 60,
        event_window_pre: int = 5,
        event_window_post: int = 5,
        gaussian_sigma: float = 2.0,
    ) -> None:
        self.tickers = self._normalize_tickers(tickers)
        if not self.tickers:
            raise ValueError("At least one ticker symbol is required.")
        self.start = start
        self.end = end
        self.interval = interval
        self.mode = mode
        self.auto_adjust = auto_adjust
        self.progress = progress
        self.dropna = dropna
        self.volume_z_window = volume_z_window
        self.event_window_pre = max(0, int(event_window_pre))
        self.event_window_post = max(0, int(event_window_post))
        self.gaussian_sigma = gaussian_sigma

        self._raw_df: Optional[pd.DataFrame] = None
        self._feature_df: Optional[pd.DataFrame] = None

    @staticmethod
    def _normalize_tickers(tickers: Sequence[str]) -> List[str]:
        unique: List[str] = []
        for ticker in tickers:
            if not ticker:
                continue
            canonical = ticker.strip().upper()
            if canonical and canonical not in unique:
                unique.append(canonical)
        return unique

    def fetch_raw_prices(self, *, force: bool = False) -> pd.DataFrame:
        """
        Download OHLCV data for all configured tickers.
        """
        if self._raw_df is not None and not force:
            return self._raw_df.copy()

        frames: List[pd.DataFrame] = []

        for ticker in self.tickers:
            data = yf.download(
                ticker,
                start=self.start,
                end=self.end,
                interval=self.interval,
                auto_adjust=self.auto_adjust,
                progress=self.progress,
            )
            if data.empty:
                continue

            frame = data.reset_index()
            # Handle MultiIndex columns from Yahoo Finance
            if isinstance(frame.columns, pd.MultiIndex):
                # Flatten MultiIndex columns
                frame.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in frame.columns]
            else:
                frame = frame.rename(columns=str.lower)
            frame["ticker"] = ticker
            if "adj close" in frame.columns and "adj_close" not in frame.columns:
                frame = frame.rename(columns={"adj close": "adj_close"})

            frames.append(frame)

        if not frames:
            raise ValueError("No price data could be downloaded from Yahoo Finance.")

        raw = pd.concat(frames, ignore_index=True)
        raw = raw.rename(columns={"date": "date"})
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.sort_values(["ticker", "date"]).reset_index(drop=True)
        self._raw_df = raw
        return raw.copy()

    def build_features(self, *, force: bool = False) -> pd.DataFrame:
        """
        Compute feature set for all tickers. Uses cached result unless force=True.
        """
        if self._feature_df is not None and not force:
            return self._feature_df.copy()

        raw = self.fetch_raw_prices()
        feature_frames = [self._compute_features_for_single(group) for _, group in raw.groupby("ticker", sort=False)]
        features = pd.concat(feature_frames, ignore_index=True)

        if self.dropna:
            feature_cols = [col for col in features.columns if col.startswith("log_") or col.startswith("rsi") or col.startswith("macd") or col.startswith("atr") or col.startswith("ema") or col.startswith("bb_") or col.startswith("local_") or col.startswith("trend_") or col.startswith("vol_") or col.startswith("pos_in_band")]
            features = features.dropna(subset=feature_cols)

        self._feature_df = features.reset_index(drop=True)
        return self._feature_df.copy()

    def get_feature_frame(self) -> pd.DataFrame:
        """
        Ensure features are built and return a copy.
        """
        if self._feature_df is None:
            return self.build_features()
        return self._feature_df.copy()

    def available_event_columns(self) -> List[str]:
        frame = self.get_feature_frame()
        return [col for col in frame.columns if col.startswith("event_")]

    def get_lstm_feature_columns(self) -> Sequence[str]:
        if self.mode == "buyer":
            return BUYER_LSTM_FEATURES
        return SELLER_LSTM_FEATURES

    def get_mode_feature_priorities(self) -> Dict[str, Sequence[str]]:
        return MODE_FEATURE_PRIORITIES[self.mode]

    def build_lstm_sequences(
        self,
        ticker: str,
        *,
        seq_len: int = 30,
        feature_columns: Optional[Sequence[str]] = None,
        dropna: bool = True,
        target_column: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Construct sliding window sequences tailored for LSTM training.

        Returns
        -------
        X : np.ndarray
            Array shaped (n_samples, seq_len, n_features).
        y : np.ndarray
            Target array aligned to the last element of each sequence.
        index : pd.DatetimeIndex
            End-date for each sequence window.
        """
        frame = self.get_feature_frame()
        subset = frame[frame["ticker"] == ticker.upper()].sort_values("date").reset_index(drop=True)
        if subset.empty:
            raise ValueError(f"No rows available for ticker {ticker!r}. Call build_features() first.")

        feature_cols = list(feature_columns or self.get_lstm_feature_columns())
        missing = [col for col in feature_cols if col not in subset.columns]
        if missing:
            raise ValueError(f"Missing feature columns for LSTM preparation: {missing}")

        target_col = target_column
        if target_col is None:
            target_col = "label_buy_soft" if self.mode == "buyer" else "label_sell_soft"
        if target_col not in subset.columns:
            raise ValueError(f"Target column {target_col!r} not available in feature frame.")

        feature_matrix = subset[feature_cols].to_numpy(dtype=float)
        targets = subset[target_col].to_numpy(dtype=float)
        dates = subset["date"].to_numpy()

        windows: List[np.ndarray] = []
        window_targets: List[float] = []
        window_dates: List[pd.Timestamp] = []

        for end_idx in range(seq_len - 1, len(subset)):
            start_idx = end_idx - seq_len + 1
            window_slice = feature_matrix[start_idx : end_idx + 1]
            if dropna and np.isnan(window_slice).any():
                continue
            windows.append(window_slice)
            window_targets.append(targets[end_idx])
            window_dates.append(pd.Timestamp(dates[end_idx]))

        if not windows:
            raise ValueError(
                f"No LSTM windows could be built for ticker {ticker!r}. "
                "Consider lowering seq_len or allowing NaNs."
            )

        X = np.stack(windows)
        y = np.asarray(window_targets, dtype=float)
        idx = pd.DatetimeIndex(window_dates, name="window_end")
        return X, y, idx

    def plot_feature_panels(
        self,
        ticker: str,
        feature_columns: Sequence[str],
        *,
        include_price: bool = True,
        height_per_panel: int = 220,
        smoothing_window: int = 5,
        show_inflection_points: bool = True,
    ) -> go.Figure:
        """
        Plot selected feature time-series for a single ticker with smoothing and normalized overlays.

        Features are grouped by category and overlaid on normalized scales for better inflection point analysis.
        """
        frame = self.get_feature_frame()
        subset = frame[frame["ticker"] == ticker.upper()]
        if subset.empty:
            raise ValueError(f"No feature rows available for ticker {ticker!r}.")

        subset = subset.sort_values("date").copy()

        # Apply smoothing to all numeric columns
        numeric_cols = subset.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['date', 'ticker']:  # Don't smooth categorical columns
                subset[f"{col}_smooth"] = subset[col].rolling(window=smoothing_window, center=True).mean()

        # Define feature groups for overlay plotting with enhanced categorization
        price_features = [f for f in feature_columns if any(x in f.lower() for x in [
            'ema', 'ma', 'close', 'price', 'return', 'log_return', 'spread', 'macd',
            'pos_in_band_rel', 'ema_spread_rel', 'ema_spread_smooth', 'macd_smooth',
            'macd_hist_rel', 'trend_momentum_divergence'
        ])]
        volume_features = [f for f in feature_columns if any(x in f.lower() for x in [
            'vol', 'volume', 'volume_z', 'volume_z_smooth', 'vol_roc', 'volume_adjusted_atr'
        ])]
        momentum_features = [f for f in feature_columns if any(x in f.lower() for x in [
            'rsi', 'momentum', 'stoch', 'williams', 'delta_rsi', 'delta_rsi_rel',
            'rsi_smooth', 'rsi_local_norm', 'rsi_local_z'
        ])]
        volatility_features = [f for f in feature_columns if any(x in f.lower() for x in [
            'atr', 'bb_', 'std', 'volatility', 'atr_rel', 'atr_smooth', 'delta_atr_rel',
            'atr_local_norm', 'bb_width', 'bb_width_delta_5', 'local_vol_index'
        ])]
        event_features = [f for f in feature_columns if f.startswith('event_') or f.startswith('label_')]
        other_features = [f for f in feature_columns if f not in price_features + volume_features + momentum_features + volatility_features + event_features]

        # Create subplots: one for each group
        subplot_titles = []
        if include_price or price_features:
            subplot_titles.append("Price & Trend Indicators")
        if volume_features:
            subplot_titles.append("Volume Indicators")
        if momentum_features:
            subplot_titles.append("Momentum Indicators")
        if volatility_features:
            subplot_titles.append("Volatility Indicators")
        if event_features:
            subplot_titles.append("Event Signals & Labels")
        if other_features:
            subplot_titles.append("Other Features")

        rows = len(subplot_titles)
        if rows == 0:
            raise ValueError("No valid feature groups to plot.")

        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles
        )

        current_row = 1

        # Price & Trend Indicators (normalized to price scale)
        if include_price or price_features:
            # Add base price line
            if include_price:
                price_scaled = self._minmax_scale(subset["close"])
                fig.add_trace(
                    go.Scatter(
                        x=subset["date"],
                        y=price_scaled,
                        name="Price (normalized)",
                        mode="lines",
                        line=dict(color="black", width=2),
                    ),
                    row=current_row, col=1
                )

            # Add price-related features (normalized)
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            for i, feature in enumerate(price_features):
                if feature in subset.columns:
                    # Normalize feature to [0,1] range for overlay
                    feature_norm = self._minmax_scale(subset[feature])
                    smooth_feature = f"{feature}_smooth"
                    if smooth_feature in subset.columns:
                        feature_norm_smooth = self._minmax_scale(subset[smooth_feature])
                        fig.add_trace(
                            go.Scatter(
                                x=subset["date"],
                                y=feature_norm_smooth,
                                name=f"{feature} (smooth)",
                                mode="lines",
                                line=dict(color=colors[i % len(colors)], width=2),
                            ),
                            row=current_row, col=1
                        )
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=subset["date"],
                                y=feature_norm,
                                name=feature,
                                mode="lines",
                                line=dict(color=colors[i % len(colors)], dash="dot"),
                            ),
                            row=current_row, col=1
                        )

            # Add inflection point markers if requested
            if show_inflection_points and include_price:
                # Simple inflection detection: second derivative approximation
                price_smooth = subset["close_smooth"] if "close_smooth" in subset.columns else subset["close"]
                second_deriv = np.gradient(np.gradient(price_smooth.values))

                # Find inflection points (zero crossings of second derivative)
                inflection_mask = (second_deriv[:-1] * second_deriv[1:]) < 0
                inflection_dates = subset["date"].iloc[1:][inflection_mask]
                inflection_prices = price_scaled.iloc[1:][inflection_mask]
                inflection_types = ["concave → convex (BUY)" if second_deriv[i+1] > 0 else "convex → concave (SELL)"
                                  for i in range(len(second_deriv)-1) if (second_deriv[i] * second_deriv[i+1]) < 0]

                for date, price, inf_type in zip(inflection_dates, inflection_prices, inflection_types):
                    color = "green" if "BUY" in inf_type else "red"
                    symbol = "triangle-up" if "BUY" in inf_type else "triangle-down"
                    fig.add_trace(
                        go.Scatter(
                            x=[date], y=[price],
                            mode="markers",
                            marker=dict(symbol=symbol, size=12, color=color),
                            name=inf_type,
                            showlegend=False,  # Don't clutter legend
                        ),
                        row=current_row, col=1
                    )

            fig.update_yaxes(title_text="Normalized Scale [0-1]", row=current_row, col=1)
            current_row += 1

        # Volume Indicators (normalized)
        if volume_features:
            colors = ['darkblue', 'navy', 'royalblue', 'steelblue', 'lightblue']
            for i, feature in enumerate(volume_features):
                if feature in subset.columns:
                    feature_norm = self._minmax_scale(subset[feature])
                    smooth_feature = f"{feature}_smooth"
                    if smooth_feature in subset.columns:
                        feature_norm_smooth = self._minmax_scale(subset[smooth_feature])
                        fig.add_trace(
                            go.Scatter(
                                x=subset["date"],
                                y=feature_norm_smooth,
                                name=f"{feature} (smooth)",
                                mode="lines",
                                line=dict(color=colors[i % len(colors)], width=2),
                            ),
                            row=current_row, col=1
                        )
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=subset["date"],
                                y=feature_norm,
                                name=feature,
                                mode="lines",
                                line=dict(color=colors[i % len(colors)], dash="dot"),
                            ),
                            row=current_row, col=1
                        )
            fig.update_yaxes(title_text="Volume Scale [0-1]", row=current_row, col=1)
            current_row += 1

        # Momentum Indicators (normalized)
        if momentum_features:
            colors = ['darkred', 'firebrick', 'tomato', 'coral', 'orange']
            for i, feature in enumerate(momentum_features):
                if feature in subset.columns:
                    feature_norm = self._minmax_scale(subset[feature])
                    smooth_feature = f"{feature}_smooth"
                    if smooth_feature in subset.columns:
                        feature_norm_smooth = self._minmax_scale(subset[smooth_feature])
                        fig.add_trace(
                            go.Scatter(
                                x=subset["date"],
                                y=feature_norm_smooth,
                                name=f"{feature} (smooth)",
                                mode="lines",
                                line=dict(color=colors[i % len(colors)], width=2),
                            ),
                            row=current_row, col=1
                        )
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=subset["date"],
                                y=feature_norm,
                                name=feature,
                                mode="lines",
                                line=dict(color=colors[i % len(colors)], dash="dot"),
                            ),
                            row=current_row, col=1
                        )
            fig.update_yaxes(title_text="Momentum Scale [0-1]", row=current_row, col=1)
            current_row += 1

        # Volatility Indicators (normalized)
        if volatility_features:
            colors = ['darkgreen', 'forestgreen', 'seagreen', 'mediumseagreen', 'lightgreen']
            for i, feature in enumerate(volatility_features):
                if feature in subset.columns:
                    feature_norm = self._minmax_scale(subset[feature])
                    smooth_feature = f"{feature}_smooth"
                    if smooth_feature in subset.columns:
                        feature_norm_smooth = self._minmax_scale(subset[smooth_feature])
                        fig.add_trace(
                            go.Scatter(
                                x=subset["date"],
                                y=feature_norm_smooth,
                                name=f"{feature} (smooth)",
                                mode="lines",
                                line=dict(color=colors[i % len(colors)], width=2),
                            ),
                            row=current_row, col=1
                        )
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=subset["date"],
                                y=feature_norm,
                                name=feature,
                                mode="lines",
                                line=dict(color=colors[i % len(colors)], dash="dot"),
                            ),
                            row=current_row, col=1
                        )
            fig.update_yaxes(title_text="Volatility Scale [0-1]", row=current_row, col=1)
            current_row += 1

        # Event Signals & Labels (0-1 scale, special handling)
        if event_features:
            colors = ['lime', 'springgreen', 'chartreuse', 'palegreen', 'lightgreen',
                     'red', 'tomato', 'coral', 'orange', 'gold']
            for i, feature in enumerate(event_features):
                if feature in subset.columns:
                    # Event features are already 0/1, but labels might be continuous
                    if feature.startswith('label_'):
                        # Soft labels - already in 0-1 range
                        fig.add_trace(
                            go.Scatter(
                                x=subset["date"],
                                y=subset[feature],
                                name=f"{feature} (soft)",
                                mode="lines",
                                line=dict(color=colors[i % len(colors)], width=2, dash="solid"),
                                fill='tozeroy',
                                fillcolor=colors[i % len(colors)].replace('1.0', '0.3'),
                            ),
                            row=current_row, col=1
                        )
                    else:
                        # Binary event signals - show as bars/steps
                        fig.add_trace(
                            go.Scatter(
                                x=subset["date"],
                                y=subset[feature],
                                name=feature,
                                mode="lines",
                                line=dict(color=colors[i % len(colors)], width=3, shape='hv'),
                                fill='tozeroy',
                                fillcolor=colors[i % len(colors)].replace('1.0', '0.2'),
                            ),
                            row=current_row, col=1
                        )
            fig.update_yaxes(title_text="Event Signals [0-1]", row=current_row, col=1)
            current_row += 1

        # Other Features
        if other_features:
            colors = ['purple', 'magenta', 'violet', 'indigo', 'darkviolet']
            for i, feature in enumerate(other_features):
                if feature in subset.columns:
                    feature_norm = self._minmax_scale(subset[feature])
                    smooth_feature = f"{feature}_smooth"
                    if smooth_feature in subset.columns:
                        feature_norm_smooth = self._minmax_scale(subset[smooth_feature])
                        fig.add_trace(
                            go.Scatter(
                                x=subset["date"],
                                y=feature_norm_smooth,
                                name=f"{feature} (smooth)",
                                mode="lines",
                                line=dict(color=colors[i % len(colors)], width=2),
                            ),
                            row=current_row, col=1
                        )
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=subset["date"],
                                y=feature_norm,
                                name=feature,
                                mode="lines",
                                line=dict(color=colors[i % len(colors)], dash="dot"),
                            ),
                            row=current_row, col=1
                        )
            fig.update_yaxes(title_text="Other Features [0-1]", row=current_row, col=1)

        # Update layout
        fig.update_layout(
            height=max(600, rows * 300),
            title=f"{ticker.upper()} — Feature Panels ({self.mode.title()} mode) | 변곡점 매매 분석",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                x=0,
                font=dict(size=10)
            ),
            hovermode="x unified",
        )

        # Add annotation about inflection points
        if show_inflection_points and include_price:
            fig.add_annotation(
                text="🟢 BUY: 오목→볼록 변곡점 | 🔴 SELL: 볼록→오목 변곡점",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=12, color="darkblue"),
                bgcolor="lightyellow",
                bordercolor="black",
                borderwidth=1,
            )

        return fig

    def plot_simple_price_with_events(
        self,
        ticker: str,
        *,
        height: int = 600,
        width: int = 1000,
        show_legend: bool = True,
        font_size: int = 14,
        use_numeric_xaxis: bool = True,
    ) -> go.Figure:
        """
        Create a simple price chart with 5 event markers for publication-ready visualization.

        Shows price line with colored markers for each of the 5 event types:
        - Local Vol Spike (Orange star)
        - Rebound Candidate (Green triangle-up)
        - Volume Regain (Lime circle)
        - Exhaustion Candidate (Red triangle-down)
        - Breakdown Risk (Dark red X)
        """
        frame = self.get_feature_frame()
        subset = frame[frame["ticker"] == ticker.upper()].sort_values("date").copy()

        if subset.empty:
            raise ValueError(f"No feature rows available for ticker {ticker!r}.")

        # Prepare x-axis data
        if use_numeric_xaxis:
            x_data = list(range(len(subset)))  # Use numeric indices 0, 1, 2, ...
            x_label = "Trading Days"
        else:
            x_data = subset["date"]
            x_label = "Date"

        # Create figure
        fig = go.Figure()

        # Add price line
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=subset["close"],
                mode="lines",
                line=dict(color="black", width=2),
                name="Price",
                showlegend=show_legend
            )
        )

        # Event configurations
        event_configs = {
            "event_local_vol_spike": {
                "name": "Local Vol Spike",
                "color": "orange",
                "symbol": "star",
                "size": 12,
                "description": "High volatility spike period"
            },
            "event_rebound_candidate": {
                "name": "Rebound Candidate",
                "color": "green",
                "symbol": "triangle-up",
                "size": 12,
                "description": "Potential rebound signal"
            },
            "event_volume_regain": {
                "name": "Volume Regain",
                "color": "lime",
                "symbol": "circle",
                "size": 10,
                "description": "Volume recovery signal"
            },
            "event_exhaustion_candidate": {
                "name": "Exhaustion Candidate",
                "color": "red",
                "symbol": "triangle-down",
                "size": 12,
                "description": "Selling exhaustion signal"
            },
            "event_breakdown_risk": {
                "name": "Breakdown Risk",
                "color": "darkred",
                "symbol": "x",
                "size": 12,
                "description": "Downside risk signal"
            }
        }

        # Add event markers
        legend_tracker = {}
        for event_col, config in event_configs.items():
            if event_col in subset.columns:
                mask = (subset[event_col].fillna(0) > 0).values
                if mask.any():
                    event_name = config["name"]
                    if event_name not in legend_tracker:
                        legend_tracker[event_name] = False

                    # Get x data for events (numeric indices or dates)
                    event_x_data = [x_data[i] for i in range(len(subset)) if mask[i]]

                    fig.add_trace(
                        go.Scatter(
                            x=event_x_data,
                            y=subset.loc[mask, "close"],
                            mode="markers",
                            marker=dict(
                                symbol=config["symbol"],
                                size=config["size"],
                                color=config["color"],
                                line=dict(width=2, color="black")
                            ),
                            name=f"{event_name}<br><i>{config['description']}</i>",
                            showlegend=show_legend and not legend_tracker[event_name],
                            hovertemplate=f"<b>{event_name}</b><br>{config['description']}<br>Day: %{{x}}<br>Price: %{{y:.2f}}<extra></extra>" if use_numeric_xaxis else f"<b>{event_name}</b><br>{config['description']}<br>Date: %{{x}}<br>Price: %{{y:.2f}}<extra></extra>"
                        )
                    )
                    legend_tracker[event_name] = True

        # Update layout for publication quality
        fig.update_layout(
            height=height,
            width=width,
            xaxis=dict(
                title=x_label,
                title_font=dict(size=font_size),
                tickfont=dict(size=font_size - 2),
                tickformat="%Y-%m-%d" if not use_numeric_xaxis else None,
                dtick="M12" if not use_numeric_xaxis else None,  # Show tick every 12 months (yearly)
                tickmode="auto" if use_numeric_xaxis else None
            ),
            yaxis=dict(
                title="Price ($)",
                title_font=dict(size=font_size),
                tickfont=dict(size=font_size - 2)
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.08,
                x=0.5,
                xanchor="center",
                font=dict(size=font_size - 2),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="black",
                borderwidth=1
            ) if show_legend else None,
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=50, r=50, t=120, b=80)
        )

        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

        return fig

    def plot_scaled_price_with_events(
        self,
        *,
        tickers: Optional[Sequence[str]] = None,
        event_columns: Optional[Sequence[str]] = None,
        price_column: str = "close_scaled",
        height: int = 500,
    ) -> go.Figure:
        """
        Plot scaled close prices and overlay selected event features as markers.
        """
        frame = self.get_feature_frame()

        if tickers is not None:
            wanted = {ticker.strip().upper() for ticker in tickers}
            frame = frame[frame["ticker"].isin(wanted)]

        if frame.empty:
            raise ValueError("No rows available for the requested tickers.")

        if price_column not in frame.columns:
            frame = frame.copy()
            frame[price_column] = frame.groupby("ticker")["close"].transform(self._minmax_scale)

        if event_columns is None:
            event_columns = [col for col in frame.columns if col.startswith("event_")]

        if not event_columns:
            raise ValueError("No event columns found. Build features first or specify event_columns explicitly.")

        fig = go.Figure()
        legend_tracker: Dict[str, bool] = {col: False for col in event_columns}

        for ticker, group in frame.groupby("ticker"):
            group = group.sort_values("date")
            fig.add_trace(
                go.Scatter(
                    x=group["date"],
                    y=group[price_column],
                    name=f"{ticker} price",
                    mode="lines",
                    legendgroup=f"{ticker}_price",
                )
            )

            for event_col in event_columns:
                if event_col not in group.columns:
                    continue
                mask = (group[event_col].fillna(0) > 0).values
                if not mask.any():
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=group.loc[mask, "date"],
                        y=group.loc[mask, price_column],
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=11),
                        name=event_col if not legend_tracker[event_col] else event_col + f" ({ticker})",
                        showlegend=not legend_tracker[event_col],
                        legendgroup=event_col,
                    )
                )
                legend_tracker[event_col] = True

        fig.update_layout(
            height=height,
            title=f"Scaled price with event markers ({self.mode.title()} mode)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            yaxis_title="Scaled price",
        )
        return fig

    def plot_feature_correlation(
        self,
        ticker: str,
        *,
        feature_columns: Optional[Sequence[str]] = None,
        absolute: bool = False,
    ) -> go.Figure:
        """
        Plot a heatmap of feature correlations for a single ticker.
        """
        frame = self.get_feature_frame()
        subset = frame[frame["ticker"] == ticker.upper()]
        if subset.empty:
            raise ValueError(f"No feature rows available for ticker {ticker!r}.")

        if feature_columns is None:
            feature_columns = [
                col
                for col in subset.columns
                if col
                not in {
                    "date",
                    "ticker",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj_close",
                    "volume",
                    "close_scaled",
                    "local_episode_id",
                    "local_episode_label",
                }
                and not col.startswith("event_")
                and subset[col].dtype in ['float64', 'float32', 'int64', 'int32']
            ]

        corr = subset.set_index("date")[list(feature_columns)].dropna().corr()
        matrix = corr.abs() if absolute else corr

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix.values,
                x=matrix.columns.tolist(),
                y=matrix.index.tolist(),
                colorscale="RdBu",
                zmid=0 if not absolute else None,
                reversescale=True,
                colorbar=dict(title="|ρ|" if absolute else "ρ"),
            )
        )
        fig.update_layout(
            title=f"{ticker.upper()} feature correlations ({self.mode.title()} mode)",
            height=400 + 20 * len(feature_columns),
        )
        return fig

    def _compute_features_for_single(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])

        numeric_cols = [col for col in ["open", "high", "low", "close", "adj_close", "volume"] if col in df.columns]
        for col in numeric_cols:
            if col in df.columns:
                try:
                    # Ensure it's a Series
                    if isinstance(df[col], pd.Series):
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    else:
                        print(f"Warning: Column {col} is not a Series, setting to NaN")
                        df[col] = np.nan
                except (TypeError, ValueError) as e:
                    print(f"Warning: Could not convert column {col} to numeric: {e}")
                    df[col] = np.nan

        df["return"] = self._safe_pct_change(df["close"])
        df["log_return"] = np.log(df["close"].replace(0, np.nan) / df["close"].shift(1))
        df["vol_roc"] = self._safe_pct_change(df["volume"])
        df["vol_roc_neg"] = (df["vol_roc"] < 0).astype(float)

        df["ema_5"] = self._ema(df["close"], 5)
        df["ema_20"] = self._ema(df["close"], 20)
        df["ema_spread"] = df["ema_5"] - df["ema_20"]
        df["delta_ema_spread"] = df["ema_spread"].diff()

        df["ema_12"] = self._ema(df["close"], 12)
        df["ema_26"] = self._ema(df["close"], 26)
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = self._ema(df["macd"], 9)
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["delta_macd"] = df["macd"].diff()

        df["rsi_14"] = self._rsi(df["close"], 14)
        df["rsi_7"] = self._rsi(df["close"], 7)
        df["delta_rsi"] = df["rsi_14"].diff()
        df["delta_rsi_3"] = df["rsi_14"].diff(3)

        df["true_range"] = self._true_range(df["high"], df["low"], df["close"])
        df["atr_14"] = self._atr(df["true_range"], 14)
        df["delta_atr_3"] = df["atr_14"].diff(3)
        df["delta_atr_5"] = df["atr_14"].diff(5)

        bb_window = 20
        bb_mid = df["close"].rolling(bb_window).mean()
        bb_std = df["close"].rolling(bb_window).std(ddof=0)
        df["bb_mid"] = bb_mid
        df["bb_upper"] = bb_mid + 2 * bb_std
        df["bb_lower"] = bb_mid - 2 * bb_std
        band_width = df["bb_upper"] - df["bb_lower"]
        with np.errstate(divide="ignore", invalid="ignore"):
            pos_in_band = (df["close"] - df["bb_lower"]) / band_width
        df["pos_in_band"] = pos_in_band.where(band_width.abs() > 1e-9)
        df["pos_in_band"] = df["pos_in_band"].clip(lower=0.0, upper=1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_width_delta_5"] = self._safe_divide(df["bb_width"] - df["bb_width"].shift(5), df["bb_width"].shift(5))

        df["volume_z"] = self._rolling_zscore(df["volume"], window=self.volume_z_window)
        df["atr_over_ema20"] = self._safe_divide(df["atr_14"], df["ema_20"])
        df["delta_rsi_x_delta_macd"] = df["delta_rsi_3"] * df["delta_macd"]
        df["delta_ema_spread_x_delta_atr"] = df["delta_ema_spread"] * df["delta_atr_5"]

        returns = df["close"].pct_change()
        short_vol = returns.rolling(5).std(ddof=0)
        long_vol = returns.rolling(20).std(ddof=0)
        df["local_vol_index"] = self._safe_divide(short_vol, long_vol)
        df["trend_momentum_divergence"] = df["macd"] - df["rsi_14"]
        df["ema_atr_corr_20"] = df["ema_5"].rolling(20).corr(df["atr_14"])
        df["volume_adjusted_atr"] = self._safe_divide(df["atr_14"], np.log(df["volume"].replace(0, np.nan)))
        df["rsi_local_z"] = self._rolling_zscore(df["rsi_14"], window=20)

        # Event-centred normalization helpers
        df["pos_in_band_rel"] = self._safe_divide(df["close"] - df["bb_mid"], band_width)
        df["atr_rel"] = self._safe_divide(
            df["atr_14"],
            df["atr_14"].rolling(20, min_periods=5).mean(),
        )
        df["delta_atr_rel"] = self._safe_divide(
            df["delta_atr_3"],
            df["atr_14"].rolling(10, min_periods=4).std(ddof=0),
        )
        df["delta_rsi_rel"] = self._safe_divide(
            df["delta_rsi"],
            df["rsi_14"].rolling(10, min_periods=4).std(ddof=0),
        )
        df["ema_spread_rel"] = self._safe_divide(df["ema_spread"], df["ema_20"])
        df["macd_hist_rel"] = self._safe_divide(
            df["macd_hist"],
            df["macd_hist"].rolling(15, min_periods=5).std(ddof=0),
        )

        df["close_scaled"] = self._minmax_scale(df["close"])
        if "volume" in df.columns:
            df["volume_scaled"] = self._minmax_scale(df["volume"])

        df["event_local_vol_spike"] = (df["local_vol_index"] > 1.5).astype(float)

        # Compute ALL events regardless of mode (for comprehensive analysis)
        # Buyer events
        df["event_rebound_candidate"] = (
            (df["pos_in_band"] < 0.35)
            & (df["delta_rsi_3"] > 0)
            & (df["delta_macd"] > 0)
            & (df["volume_z"] > 0)
        ).astype(float)

        df["event_volume_regain"] = (
            (df["volume_z"] > 0.8)
            & (df["vol_roc"] > 0)
            & (df["delta_rsi"] > 0)
        ).astype(float)

        # Seller events
        df["event_exhaustion_candidate"] = (
            (df["pos_in_band"] > 0.85)
            & (df["delta_ema_spread"] < 0)
            & (df["delta_atr_5"] > 0)
            & (df["volume_z"] < 0.5)
        ).astype(float)

        df["event_breakdown_risk"] = (
            (df["local_vol_index"] > 1.2)
            & (df["macd_hist"] < 0)
            & (df["delta_rsi"] < 0)
        ).astype(float)

        # Add local_vol_spike event (common to both modes)
        df["event_local_vol_spike"] = (df["local_vol_index"] > 1.5).astype(float)

        # Local episode assignment around ALL events for normalization
        all_event_columns = [
            "event_local_vol_spike",
            "event_rebound_candidate",
            "event_volume_regain",
            "event_exhaustion_candidate",
            "event_breakdown_risk"
        ]
        episode_id, episode_label = self._build_local_episodes(df, all_event_columns)
        df["local_episode_id"] = episode_id
        df["local_episode_label"] = episode_label

        # Local normalization within event windows
        df["rsi_local_norm"] = self._apply_episode_zscore(
            df, "rsi_14", episode_id, fallback_window=15
        )
        df["atr_local_norm"] = self._apply_episode_zscore(
            df, "atr_rel", episode_id, fallback_window=15
        )
        df["macd_local_norm"] = self._apply_episode_zscore(
            df, "macd_hist", episode_id, fallback_window=15
        )
        df["ema_spread_local_norm"] = self._apply_episode_zscore(
            df, "ema_spread_rel", episode_id, fallback_window=15
        )

        # Noise filtering & smoothing
        df["rsi_smooth"] = df["rsi_14"].ewm(span=5, adjust=False).mean().rolling(3, min_periods=1).median()
        df["atr_smooth"] = df["atr_rel"].ewm(span=3, adjust=False).mean()
        df["macd_smooth"] = df["macd_hist"].ewm(span=4, adjust=False).mean()
        df["volume_z_smooth"] = df["volume_z"].ewm(span=4, adjust=False).mean()
        df["ema_spread_smooth"] = df["ema_spread_rel"].ewm(span=4, adjust=False).mean()

        # Soft labels for event windows
        df["label_buy_soft"] = self._gaussian_smooth(df["event_rebound_candidate"], sigma=self.gaussian_sigma)
        df["label_sell_soft"] = self._gaussian_smooth(df["event_breakdown_risk"], sigma=self.gaussian_sigma)

        df = df.drop(columns=["ema_12", "ema_26", "true_range"], errors="ignore")
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def _mode_event_columns(self) -> Sequence[str]:
        return BUYER_EVENT_COLS if self.mode == "buyer" else SELLER_EVENT_COLS

    def _build_local_episodes(
        self,
        df: pd.DataFrame,
        event_columns: Sequence[str],
    ) -> Tuple[pd.Series, pd.Series]:
        length = len(df)
        episode_id = np.full(length, -1, dtype=int)
        episode_label = np.full(length, "", dtype=object)
        current_id = 0

        for event_col in event_columns:
            event_flags = df.get(
                event_col,
                pd.Series(0, index=df.index, dtype=float),
            ).fillna(0).to_numpy()
            event_indices = np.flatnonzero(event_flags > 0)
            for idx in event_indices:
                start = max(0, idx - self.event_window_pre)
                end = min(length - 1, idx + self.event_window_post)
                label = f"{event_col}:{current_id}"
                for position in range(start, end + 1):
                    if episode_id[position] == -1:
                        episode_id[position] = current_id
                        episode_label[position] = label
                current_id += 1

        return (
            pd.Series(episode_id, index=df.index, name="local_episode_id"),
            pd.Series(episode_label, index=df.index, name="local_episode_label"),
        )

    def _apply_episode_zscore(
        self,
        df: pd.DataFrame,
        column: str,
        episode_id: pd.Series,
        *,
        fallback_window: int,
    ) -> pd.Series:
        series = df[column].astype(float)
        result = pd.Series(np.nan, index=df.index, dtype=float)

        valid_mask = episode_id >= 0
        if valid_mask.any():
            groups = episode_id[valid_mask]
            grouped = series[valid_mask].groupby(groups, group_keys=False)
            result.loc[valid_mask] = grouped.transform(self._zscore_transform)

        fallback = self._rolling_zscore(series, window=fallback_window)
        return result.fillna(fallback)

    @staticmethod
    def _zscore_transform(series: pd.Series) -> pd.Series:
        std = series.std(ddof=0)
        if std is None or np.isclose(std, 0.0):
            return pd.Series(0.0, index=series.index, dtype=float)
        mean = series.mean()
        return (series - mean) / std

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.clip(0, 100)

    @staticmethod
    def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_close = close.shift(1)
        comp = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        return comp.max(axis=1)

    @staticmethod
    def _atr(true_range: pd.Series, period: int) -> pd.Series:
        return true_range.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
        if min_periods is None:
            min_periods = max(5, window // 3)
        mean = series.rolling(window, min_periods=min_periods).mean()
        std = series.rolling(window, min_periods=min_periods).std(ddof=0)
        z = (series - mean) / std
        return z.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
        return series.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _minmax_scale(series: pd.Series) -> pd.Series:
        min_val = series.min()
        max_val = series.max()
        denom = max_val - min_val
        if pd.isna(min_val) or pd.isna(max_val) or np.isclose(denom, 0):
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / denom

    @staticmethod
    def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = numerator / denominator
        return result.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _gaussian_smooth(series: pd.Series, sigma: float, truncate: float = 3.0) -> pd.Series:
        values = series.fillna(0).astype(float).to_numpy()
        if sigma <= 0:
            return pd.Series(values, index=series.index, dtype=float)

        radius = max(1, int(truncate * sigma))
        kernel_index = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (kernel_index / sigma) ** 2)
        kernel_sum = kernel.sum()
        if np.isclose(kernel_sum, 0.0):
            return pd.Series(values, index=series.index, dtype=float)
        kernel /= kernel_sum

        smoothed = np.convolve(values, kernel, mode="same")
        smoothed = np.clip(smoothed, 0.0, 1.0)
        return pd.Series(smoothed, index=series.index, dtype=float)


def build_buyer_feature_frame(
    tickers: Sequence[str],
    start: str,
    end: Optional[str] = None,
    *,
    interval: str = "1d",
    **kwargs,
) -> pd.DataFrame:
    explorer = M002FeatureExplorer(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        mode="buyer",
        **kwargs,
    )
    return explorer.build_features()


def build_seller_feature_frame(
    tickers: Sequence[str],
    start: str,
    end: Optional[str] = None,
    *,
    interval: str = "1d",
    **kwargs,
) -> pd.DataFrame:
    explorer = M002FeatureExplorer(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        mode="seller",
        **kwargs,
    )
    return explorer.build_features()


def generate_visualizations(
    tickers: Sequence[str],
    start: str,
    end: Optional[str] = None,
    *,
    mode: FeatureMode = "buyer",
    output_dir: str = "reports/m002",
    **kwargs,
) -> None:
    """
    Generate comprehensive visualizations for M002 feature analysis.

    Args:
        tickers: List of ticker symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD), optional
        mode: 'buyer' or 'seller'
        output_dir: Output directory for visualizations
        **kwargs: Additional arguments for M002FeatureExplorer
    """
    import os
    from pathlib import Path

    # Create output directory
    output_path = Path(output_dir) / mode
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating {mode} mode visualizations for {len(tickers)} tickers...")

    # Initialize explorer
    explorer = M002FeatureExplorer(
        tickers=tickers,
        start=start,
        end=end,
        mode=mode,
        **kwargs
    )

    # Build features
    features_df = explorer.build_features()
    print(f"Built features for {len(tickers)} tickers with {len(features_df)} rows")

    # Get available event columns
    event_columns = explorer.available_event_columns()
    print(f"Available event columns: {event_columns}")

    # Generate visualizations for each ticker
    for ticker in tickers:
        ticker_upper = ticker.upper()
        print(f"Processing {ticker_upper}...")

        try:
            # 1. Feature panels (technical indicators)
            tech_features = ["rsi_14", "macd", "ema_spread", "bb_mid", "pos_in_band", "volume_z"]
            available_tech = [f for f in tech_features if f in features_df.columns]

            if available_tech:
                fig_panels = explorer.plot_feature_panels(
                    ticker_upper,
                    available_tech[:6],  # Limit to 6 features for readability
                    include_price=True
                )
                fig_panels.write_html(str(output_path / f"{ticker_upper}_feature_panels.html"))
                print(f"  ✓ Saved feature panels for {ticker_upper}")

            # 2. Scaled price with events
            if event_columns:
                fig_events = explorer.plot_scaled_price_with_events(
                    tickers=[ticker_upper],
                    event_columns=event_columns[:5],  # Limit to 5 event types
                    height=600
                )
                fig_events.write_html(str(output_path / f"{ticker_upper}_events.html"))
                print(f"  ✓ Saved events visualization for {ticker_upper}")

            # 3. Feature correlation heatmap
            try:
                fig_corr = explorer.plot_feature_correlation(
                    ticker_upper,
                    absolute=True
                )
                fig_corr.write_html(str(output_path / f"{ticker_upper}_correlations.html"))
                print(f"  ✓ Saved correlation heatmap for {ticker_upper}")
            except Exception as e:
                print(f"  ⚠ Failed to generate correlation for {ticker_upper}: {e}")

        except Exception as e:
            print(f"  ✗ Failed to process {ticker_upper}: {e}")
            continue

    # Generate summary statistics
    try:
        summary_stats = features_df.groupby("ticker").agg({
            "close": ["count", "mean", "std", "min", "max"],
            "volume": ["mean", "sum"] if "volume" in features_df.columns else "count",
        }).round(2)

        # Save summary as CSV
        summary_stats.to_csv(output_path / "summary_statistics.csv")
        print(f"✓ Saved summary statistics to {output_path / 'summary_statistics.csv'}")

        # Save event counts
        if event_columns:
            event_summary = features_df.groupby("ticker")[event_columns].sum()
            event_summary.to_csv(output_path / "event_summary.csv")
            print(f"✓ Saved event summary to {output_path / 'event_summary.csv'}")

    except Exception as e:
        print(f"⚠ Failed to generate summary statistics: {e}")

    # Save feature info
    feature_info = {
        "tickers": tickers,
        "date_range": f"{start} to {end or 'latest'}",
        "mode": mode,
        "total_rows": len(features_df),
        "event_columns": event_columns,
        "feature_columns": [col for col in features_df.columns if not col.startswith("event_") and col not in ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]],
    }

    import json
    with open(output_path / "feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=2, default=str)

    print(f"✓ Completed {mode} mode visualizations. Files saved to {output_path}")


def generate_combined_visualizations(
    tickers: Sequence[str],
    start: str,
    end: Optional[str] = None,
    *,
    output_dir: str = "reports/m002",
    **kwargs,
) -> None:
    """
    Generate visualizations comparing buyer and seller events side by side.

    Creates combined charts showing both buyer and seller event signals
    with colored regions for different event types.
    """
    import os
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating combined all-events visualizations for {len(tickers)} tickers...")

    # Create single explorer with all events
    explorer = M002FeatureExplorer(
        tickers=tickers,
        start=start,
        end=end,
        mode="buyer",  # mode doesn't matter now, all events are calculated
        **kwargs
    )

    # Build features (all events included)
    features = explorer.build_features()

    print(f"Built features with all events ({len(features)} rows)")

    # Generate visualizations for each ticker
    for ticker in tickers:
        ticker_upper = ticker.upper()
        print(f"Processing combined view for {ticker_upper}...")

        try:
            # Create combined event chart with all events
            fig = _create_combined_event_chart(
                ticker_upper, features, explorer
            )
            fig.write_html(str(output_path / f"{ticker_upper}_all_events.html"))
            print(f"  ✓ Saved all events chart for {ticker_upper}")

            # Create local episode visualization
            fig_episodes = _create_episode_visualization(
                ticker_upper, features
            )
            fig_episodes.write_html(str(output_path / f"{ticker_upper}_episodes.html"))
            print(f"  ✓ Saved episode visualization for {ticker_upper}")

        except Exception as e:
            print(f"  ✗ Failed to process {ticker_upper}: {e}")
            continue

    # Extract and save episode information
    episode_data = extract_episode_data(tickers, features)

    # Save combined metadata
    combined_info = {
        "tickers": tickers,
        "date_range": f"{start} to {end or 'latest'}",
        "all_events": [
            "event_local_vol_spike",
            "event_rebound_candidate",
            "event_volume_regain",
            "event_exhaustion_candidate",
            "event_breakdown_risk"
        ],
        "total_rows": len(features),
        "episode_summary": episode_data["summary"]
    }

    import json
    with open(output_path / "combined_info.json", "w") as f:
        json.dump(combined_info, f, indent=2, default=str)

    # Save detailed episode data
    with open(output_path / "episodes_detailed.json", "w") as f:
        json.dump(episode_data["detailed"], f, indent=2, default=str)

    print(f"✓ Completed combined visualizations. Files saved to {output_path}")
    print(f"✓ Episode data saved: {len(episode_data['detailed']['episodes'])} episodes found")

    # Automatically run episode analysis
    print("\n🚀 Running automatic episode analysis...")
    try:
        from features.analyze_m002_episodes import main as analyze_main
        analyze_main()
        print("✓ Episode analysis completed!")
    except Exception as e:
        print(f"✗ Episode analysis failed: {e}")

    print(f"\n🎉 All processing complete! Check {output_path} for results.")


def extract_episode_data(tickers: Sequence[str], features_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract detailed episode information from combined feature dataframe.

    Returns:
        dict with 'summary' and 'detailed' keys containing episode statistics and details.
    """
    episodes = []
    summary_stats = {
        "total_episodes": 0,
        "by_event_type": {},
        "by_ticker": {},
        "avg_episode_length": 0,
        "episode_lengths": []
    }

    # Process each ticker
    for ticker in tickers:
        ticker_data = features_df[features_df["ticker"] == ticker].copy()

        if ticker_data.empty:
            continue

        summary_stats["by_ticker"][ticker] = 0

        # Extract episodes
        episode_ids = ticker_data["local_episode_id"].unique()
        episode_ids = episode_ids[episode_ids >= 0]  # Exclude -1 (no episode)

        for episode_id in episode_ids:
            episode_mask = ticker_data["local_episode_id"] == episode_id
            episode_data = ticker_data[episode_mask].copy()

            if episode_data.empty:
                continue

            # Episode basic info
            episode_label = episode_data["local_episode_label"].iloc[0]
            start_date = episode_data["date"].min()
            end_date = episode_data["date"].max()
            duration = len(episode_data)

            # Extract event type from label
            event_type = episode_label.split(":")[0] if ":" in episode_label else episode_label

            # Calculate episode statistics
            price_change = (episode_data["close"].iloc[-1] - episode_data["close"].iloc[0]) / episode_data["close"].iloc[0] * 100
            max_price_change = ((episode_data["high"].max() - episode_data["low"].min()) / episode_data["low"].min()) * 100

            # Key feature statistics during episode
            feature_stats = {}
            key_features = [
                "close", "volume_z", "rsi_smooth", "macd_smooth", "atr_smooth",
                "pos_in_band_rel", "ema_spread_rel", "local_vol_index"
            ]

            for feature in key_features:
                    if feature in episode_data.columns:
                        values = episode_data[feature].dropna()
                        if not values.empty:
                            feature_stats[feature] = {
                                "mean": float(values.mean()),
                                "std": float(values.std()) if len(values) > 1 else 0.0,
                                "min": float(values.min()),
                                "max": float(values.max()),
                                "start": float(values.iloc[0]),
                                "end": float(values.iloc[-1])
                            }

            # Event signals during episode
            event_signals = {}
            event_columns = [col for col in ticker_data.columns if col.startswith('event_') or col.startswith('label_')]
            for event_col in event_columns:
                if event_col in episode_data.columns:
                    signal_values = episode_data[event_col].dropna()
                    if not signal_values.empty:
                        event_signals[event_col] = {
                            "active_periods": int(signal_values.sum()),
                            "max_intensity": float(signal_values.max()),
                            "avg_intensity": float(signal_values.mean())
                        }

                # Create episode record
                episode_record = {
                    "episode_id": int(episode_id),
                    "ticker": ticker,
                    "event_type": event_type,
                    "episode_label": episode_label,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "duration_days": duration,
                    "price_change_pct": float(price_change),
                    "max_price_range_pct": float(max_price_change),
                    "feature_statistics": feature_stats,
                    "event_signals": event_signals,
                    "data_points": len(episode_data)
                }

                episodes.append(episode_record)

                # Update summary statistics
                summary_stats["total_episodes"] += 1
                summary_stats["by_ticker"][ticker] += 1

                if event_type not in summary_stats["by_event_type"]:
                    summary_stats["by_event_type"][event_type] = 0
                summary_stats["by_event_type"][event_type] += 1

                summary_stats["episode_lengths"].append(duration)

    # Calculate average episode length
    if summary_stats["episode_lengths"]:
        summary_stats["avg_episode_length"] = sum(summary_stats["episode_lengths"]) / len(summary_stats["episode_lengths"])
    else:
        summary_stats["avg_episode_length"] = 0

    # Sort episodes by date
    episodes.sort(key=lambda x: (x["ticker"], x["start_date"], x["episode_id"]))

    return {
        "summary": summary_stats,
        "detailed": {
            "episodes": episodes,
            "metadata": {
                "total_episodes": len(episodes),
                "date_generated": pd.Timestamp.now().isoformat(),
                "tickers_analyzed": list(tickers)
            }
        }
    }


def _create_combined_event_chart(
    ticker: str,
    features_df: pd.DataFrame,
    explorer: M002FeatureExplorer,
) -> go.Figure:
    """Create a chart showing all events with colored regions and event markers."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Filter data for ticker
    subset = features_df[features_df["ticker"] == ticker].copy()

    if subset.empty:
        raise ValueError(f"No data available for ticker {ticker}")

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            "Price Chart with Event Regions",
            "All Events Markers",
            "Event Signals & Labels"
        ]
    )

    # Common date range
    dates = subset["date"]

    # 1. Price chart with colored regions
    price_scaled = explorer._minmax_scale(subset["close"])

    # Create price line segments with different colors for event periods
    event_line_colors = {
        "event_local_vol_spike": "orange",
        "event_rebound_candidate": "green",
        "event_volume_regain": "lime",
        "event_exhaustion_candidate": "red",
        "event_breakdown_risk": "darkred"
    }

    event_names = {
        "event_local_vol_spike": "Local Vol Spike",
        "event_rebound_candidate": "Rebound Candidate",
        "event_volume_regain": "Volume Regain",
        "event_exhaustion_candidate": "Exhaustion Candidate",
        "event_breakdown_risk": "Breakdown Risk"
    }

    # Add price line segments for each event type
    for event_col, color in event_line_colors.items():
        if event_col in subset.columns:
            event_mask = subset[event_col] > 0
            if event_mask.any():
                event_dates = dates[event_mask]
                event_prices = price_scaled[event_mask]
                fig.add_trace(
                    go.Scatter(
                        x=event_dates,
                        y=event_prices,
                        mode="lines",
                        line=dict(color=color, width=3),
                        name=f"Price during {event_names[event_col]}",
                        showlegend=True
                    ),
                    row=1, col=1
                )

    # Add default price line (non-event periods)
    non_event_mask = ~subset[[col for col in event_line_colors.keys() if col in subset.columns]].any(axis=1)
    if non_event_mask.any():
        fig.add_trace(
            go.Scatter(
                x=dates[non_event_mask],
                y=price_scaled[non_event_mask],
                mode="lines",
                line=dict(color="black", width=2),
                name="Price (normal periods)",
                showlegend=True
            ),
            row=1, col=1
        )

    # 2. All Events Markers - Show event points on price chart
    # Add price line to the markers subplot as well
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=price_scaled,
            mode="lines",
            line=dict(color="gray", width=2, dash="dot"),
            name="Price (reference)",
            showlegend=True
        ),
        row=2, col=1
    )

    event_marker_colors = {
        "event_local_vol_spike": "orange",
        "event_rebound_candidate": "green",
        "event_volume_regain": "lime",
        "event_exhaustion_candidate": "red",
        "event_breakdown_risk": "darkred"
    }

    event_symbols = {
        "event_local_vol_spike": "star",
        "event_rebound_candidate": "triangle-up",
        "event_volume_regain": "circle",
        "event_exhaustion_candidate": "triangle-down",
        "event_breakdown_risk": "x"
    }

    for event_col in event_marker_colors.keys():
        if event_col in subset.columns:
            event_mask = subset[event_col] > 0
            if event_mask.any():
                event_dates = dates[event_mask]
                event_prices = price_scaled[event_mask]
                fig.add_trace(
                    go.Scatter(
                        x=event_dates,
                        y=event_prices,
                        mode="markers",
                        marker=dict(
                            color=event_marker_colors[event_col],
                            symbol=event_symbols[event_col],
                            size=8,
                            line=dict(width=1, color='black')
                        ),
                        name=f'{event_names[event_col]} Points',
                        showlegend=True
                    ),
                    row=2, col=1
                )

    # 3. Event Signals & Labels - Show soft labels and signals
    # Add price reference line to the signals subplot
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=price_scaled,
            mode="lines",
            line=dict(color="gray", width=2, dash="dot"),
            name="Price (reference)",
            showlegend=True
        ),
        row=3, col=1
    )

    # Add soft labels (smoothed signals)
    if "label_buy_soft" in subset.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=subset["label_buy_soft"],
                mode="lines",
                line=dict(color="green", width=2, dash="dot"),
                name="Buy Soft Label",
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.1)'
            ),
            row=3, col=1
        )

    if "label_sell_soft" in subset.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=subset["label_sell_soft"],
                mode="lines",
                line=dict(color="red", width=2, dash="dot"),
                name="Sell Soft Label",
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.1)'
            ),
            row=3, col=1
        )

    # Add buy/sell signals from inflection points
    if "buy_signal" in subset.columns and subset["buy_signal"].any():
        buy_dates = dates[subset["buy_signal"] == 1]
        buy_prices = price_scaled[subset["buy_signal"] == 1]
        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode="markers",
                marker=dict(
                    color="green",
                    symbol="triangle-up",
                    size=12,
                    line=dict(width=2, color='darkgreen')
                ),
                name="Buy Signals (Inflection)",
                showlegend=True
            ),
            row=3, col=1
        )

    if "sell_signal" in subset.columns and subset["sell_signal"].any():
        sell_dates = dates[subset["sell_signal"] == -1]
        sell_prices = price_scaled[subset["sell_signal"] == -1]
        fig.add_trace(
            go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode="markers",
                marker=dict(
                    color="red",
                    symbol="triangle-down",
                    size=12,
                    line=dict(width=2, color='darkred')
                ),
                name="Sell Signals (Inflection)",
                showlegend=True
            ),
            row=3, col=1
        )

    # Update layout
    fig.update_layout(
        height=900,
        title=f"{ticker} — All Events Analysis with Signals",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0
        )
    )

    # Update y-axes
    fig.update_yaxes(title_text="Price (0-1)", row=1, col=1)
    fig.update_yaxes(title_text="Buyer Events (0-1)", row=2, col=1)
    fig.update_yaxes(title_text="Seller Events (0-1)", row=3, col=1)

    return fig


def _create_episode_visualization(
    ticker: str,
    features_df: pd.DataFrame,
) -> go.Figure:
    """Create visualization showing local episodes for all events."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Filter data for ticker
    subset = features_df[features_df["ticker"] == ticker].copy()

    if subset.empty:
        raise ValueError(f"No data available for ticker {ticker}")

    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            "Price Chart with Episode Regions",
            "All Events - Episode Analysis"
        ]
    )

    dates = subset["date"]

    # Color mapping for episodes by event type
    event_colors = {
        "event_local_vol_spike": 'rgba(255, 165, 0, 0.2)',  # Orange
        "event_rebound_candidate": 'rgba(0, 255, 0, 0.2)',   # Green
        "event_volume_regain": 'rgba(0, 255, 0, 0.25)',     # Darker Green
        "event_exhaustion_candidate": 'rgba(255, 0, 0, 0.2)', # Red
        "event_breakdown_risk": 'rgba(255, 0, 0, 0.25)'      # Darker Red
    }

    event_names = {
        "event_local_vol_spike": "Local Vol Spike",
        "event_rebound_candidate": "Rebound Candidate",
        "event_volume_regain": "Volume Regain",
        "event_exhaustion_candidate": "Exhaustion Candidate",
        "event_breakdown_risk": "Breakdown Risk"
    }

    price_scaled = subset["close"] / subset["close"].max()  # Simple scaling

    # Add base price line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=price_scaled,
            mode="lines",
            line=dict(color="black", width=1),
            name="Price (normalized)",
            showlegend=True
        ),
        row=1, col=1
    )

    # Get all unique episode IDs
    all_episode_ids = subset["local_episode_id"].unique()
    all_episode_ids = all_episode_ids[all_episode_ids >= 0]  # Exclude -1

    # Group episodes by event type and show top episodes for each
    episodes_by_event = {}
    for episode_id in all_episode_ids:
        mask = subset["local_episode_id"] == episode_id
        if mask.any():
            episode_label = subset.loc[mask, "local_episode_label"].iloc[0]
            event_type = episode_label.split(':')[0]  # Extract event type from label
            if event_type not in episodes_by_event:
                episodes_by_event[event_type] = []
            episodes_by_event[event_type].append(episode_id)

    # Show top 2-3 episodes for each event type
    for event_type, episode_ids in episodes_by_event.items():
        color = event_colors.get(event_type, 'rgba(128, 128, 128, 0.2)')  # Default gray
        for episode_id in episode_ids[:2]:  # Limit to 2 episodes per event type
            mask = subset["local_episode_id"] == episode_id
            if mask.any():
                episode_dates = dates[mask]
                episode_prices = price_scaled[mask]
                episode_label = subset.loc[mask, "local_episode_label"].iloc[0]

                fig.add_trace(
                    go.Scatter(
                        x=episode_dates,
                        y=episode_prices,
                        fill='tozeroy',
                        mode='lines',
                        line=dict(width=2),
                        fillcolor=color,
                        name=f'{event_type} Episode {episode_id}',
                        showlegend=True
                    ),
                    row=1, col=1
                )

    # Second subplot: Show event signals over time
    for event_col, color in event_colors.items():
        if event_col in subset.columns:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=subset[event_col],
                    mode="lines",
                    line=dict(color=color.replace('0.2', '1.0'), width=2, shape='hv'),
                    fill='tozeroy',
                    fillcolor=color,
                    name=f'{event_names.get(event_col, event_col)} Signal',
                    showlegend=True
                ),
                row=2, col=1
            )

    # Update layout
    fig.update_layout(
        height=800,
        title=f"{ticker} — Local Episodes Analysis",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0
        )
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price (normalized)", row=1, col=1)
    fig.update_yaxes(title_text="Event Signals (0-1)", row=2, col=1)

    return fig


__all__ = [
    "M002FeatureExplorer",
    "build_buyer_feature_frame",
    "build_seller_feature_frame",
    "generate_visualizations",
    "generate_combined_visualizations",
]
