"""
Feature Builder — Transforms raw OHLCV into relative, AI-friendly features.

All features are RELATIVE (ratios, normalized values) so they are:
- Scale-invariant (works across symbols with different price levels)
- Stationary (no trends in feature space)
- Bounded (prevents extreme values from destabilizing training)

Zero hardcoded parameters — all thresholds from config.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import polars as pl


# ─────────────────────────────────────────────
# Candle Ratio Features
# ─────────────────────────────────────────────

def candle_ratios(df: pl.DataFrame) -> pl.DataFrame:
    """
    Decompose each candle into relative body/wick ratios.

    Output columns:
    - body_ratio: |close - open| / range  (0 to 1)
    - upper_wick_ratio: upper_wick / range  (0 to 1)
    - lower_wick_ratio: lower_wick / range  (0 to 1)
    - candle_direction: +1 (bullish) or -1 (bearish) or 0 (doji)

    All ratios sum to ≈ 1.0 for each candle.
    """
    total_range = (df["high"] - df["low"]).alias("_range")

    return df.with_columns([
        # Body
        (
            (df["close"] - df["open"]).abs() / total_range.clip(lower_bound=1e-10)
        ).clip(0.0, 1.0).alias("body_ratio"),

        # Upper wick
        (
            (df["high"] - pl.max_horizontal(df["open"], df["close"]))
            / total_range.clip(lower_bound=1e-10)
        ).clip(0.0, 1.0).alias("upper_wick_ratio"),

        # Lower wick
        (
            (pl.min_horizontal(df["open"], df["close"]) - df["low"])
            / total_range.clip(lower_bound=1e-10)
        ).clip(0.0, 1.0).alias("lower_wick_ratio"),

        # Direction: +1 bullish, -1 bearish, 0 doji
        pl.when(df["close"] > df["open"]).then(1.0)
        .when(df["close"] < df["open"]).then(-1.0)
        .otherwise(0.0)
        .alias("candle_direction"),
    ])


# ─────────────────────────────────────────────
# Volume Features
# ─────────────────────────────────────────────

def relative_volume(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """
    Calculate Relative Volume (RVol): current volume / rolling average.

    RVol > 1.0 means above-average volume (potential breakout).
    RVol < 0.5 means low volume (potential ranging/consolidation).
    """
    vol_mean = df["volume"].rolling_mean(window_size=window).alias("_vol_mean")
    return df.with_columns(
        (df["volume"] / vol_mean.clip(lower_bound=1e-10))
        .alias("relative_volume")
    )


# ─────────────────────────────────────────────
# Time Encoding Features
# ─────────────────────────────────────────────

def time_encoding(df: pl.DataFrame) -> pl.DataFrame:
    """
    Cyclical time encoding using sin/cos for hour-of-day and day-of-week.

    This preserves the cyclical nature (23:00 is close to 00:00).
    Output: sin_hour, cos_hour, sin_dow, cos_dow (all in [-1, 1])
    """
    TWO_PI = 2.0 * math.pi

    # Extract to numpy, calculate, return as Polars columns
    hour_np = df["time"].dt.hour().to_numpy().astype(np.float64)
    dow_np = df["time"].dt.weekday().to_numpy().astype(np.float64)

    hour_rad = hour_np * TWO_PI / 24.0
    dow_rad = dow_np * TWO_PI / 5.0

    return df.with_columns([
        pl.Series("sin_hour", np.sin(hour_rad)),
        pl.Series("cos_hour", np.cos(hour_rad)),
        pl.Series("sin_dow", np.sin(dow_rad)),
        pl.Series("cos_dow", np.cos(dow_rad)),
    ])


def time_encoding_numpy(
    hours: np.ndarray,
    days_of_week: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Pure numpy time encoding (for use outside Polars context).

    Returns:
        Dict with sin_hour, cos_hour, sin_dow, cos_dow arrays.
    """
    TWO_PI = 2.0 * np.pi
    return {
        "sin_hour": np.sin(hours * TWO_PI / 24.0),
        "cos_hour": np.cos(hours * TWO_PI / 24.0),
        "sin_dow": np.sin(days_of_week * TWO_PI / 5.0),
        "cos_dow": np.cos(days_of_week * TWO_PI / 5.0),
    }


# ─────────────────────────────────────────────
# Returns & Volatility Features
# ─────────────────────────────────────────────

def returns_features(
    df: pl.DataFrame,
    atr_window: int = 14,
    vol_window: int = 20,
) -> pl.DataFrame:
    """
    Calculate log returns and ATR-normalized volatility.

    Output columns:
    - log_return: ln(close / prev_close)
    - atr_normalized: ATR / close (relative, scale-invariant)
    - rolling_volatility: std(log_returns) over vol_window
    """
    # Log return
    log_ret = (df["close"] / df["close"].shift(1)).log().alias("log_return")

    # True Range components
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - df["close"].shift(1)).abs()
    low_prev_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pl.max_horizontal(high_low, high_prev_close, low_prev_close)

    # ATR normalized by close price
    atr = true_range.rolling_mean(window_size=atr_window)
    atr_norm = (atr / df["close"].clip(lower_bound=1e-10)).alias("atr_normalized")

    # Rolling volatility of log returns
    rolling_vol = log_ret.rolling_std(window_size=vol_window).alias("rolling_volatility")

    return df.with_columns([log_ret, atr_norm, rolling_vol])


# ─────────────────────────────────────────────
# Price Position Features
# ─────────────────────────────────────────────

def price_position(df: pl.DataFrame, window: int = 50) -> pl.DataFrame:
    """
    Calculate where price is relative to recent range (0.0 = bottom, 1.0 = top).

    Output columns:
    - price_position: (close - rolling_low) / (rolling_high - rolling_low)
    - ma_distance: (close - SMA) / SMA (normalized distance from mean)
    """
    rolling_high = df["close"].rolling_max(window_size=window)
    rolling_low = df["close"].rolling_min(window_size=window)
    rolling_range = (rolling_high - rolling_low).clip(lower_bound=1e-10)
    sma = df["close"].rolling_mean(window_size=window)

    return df.with_columns([
        ((df["close"] - rolling_low) / rolling_range)
        .clip(0.0, 1.0)
        .alias("price_position"),

        ((df["close"] - sma) / sma.clip(lower_bound=1e-10))
        .alias("ma_distance"),
    ])


# ─────────────────────────────────────────────
# Spread Features
# ─────────────────────────────────────────────

def spread_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize spread relative to price.

    Output columns:
    - spread_normalized: spread / close (relative to price level)
    - spread_relative: spread / rolling_mean_spread (relative to typical spread)
    """
    if "spread" not in df.columns:
        return df

    spread_mean = df["spread"].rolling_mean(window_size=100).clip(lower_bound=1e-10)

    return df.with_columns([
        (df["spread"] / df["close"].clip(lower_bound=1e-10))
        .alias("spread_normalized"),

        (df["spread"] / spread_mean)
        .alias("spread_relative"),
    ])


# ─────────────────────────────────────────────
# Master Feature Pipeline
# ─────────────────────────────────────────────

# All feature column names the pipeline outputs
FEATURE_COLUMNS: list[str] = [
    # Candle ratios
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "candle_direction",
    # Volume
    "relative_volume",
    # Time encoding
    "sin_hour", "cos_hour", "sin_dow", "cos_dow",
    # Returns & volatility
    "log_return", "atr_normalized", "rolling_volatility",
    # Price position
    "price_position", "ma_distance",
]


def build_features(
    df: pl.DataFrame,
    vol_window: int = 20,
    atr_window: int = 14,
    position_window: int = 50,
    include_spread: bool = True,
) -> pl.DataFrame:
    """
    Apply full feature pipeline to raw OHLCV DataFrame.

    Args:
        df: DataFrame with columns [time, open, high, low, close, volume]
        vol_window: Window for relative volume calculation
        atr_window: Window for ATR calculation
        position_window: Window for price position calculation
        include_spread: Whether to include spread features

    Returns:
        DataFrame with all original columns plus feature columns.
        First `max(windows)` rows will have NaN values due to rolling calculations.
    """
    result = df.clone()

    # Apply each feature group
    result = candle_ratios(result)
    result = relative_volume(result, window=vol_window)
    result = time_encoding(result)
    result = returns_features(result, atr_window=atr_window, vol_window=vol_window)
    result = price_position(result, window=position_window)

    if include_spread and "spread" in result.columns:
        result = spread_features(result)

    # Drop rows with NaN (from rolling calculations warmup)
    warmup = max(vol_window, atr_window, position_window)
    result = result.slice(warmup)

    return result
