"""
Feature Builder — SMC + Volume + Price Action features for DRL trading.

Architecture:
- SMC (Smart Money Concepts): Structure, Order Blocks, FVG, Liquidity
- Volume: Delta, Climax, Relative Volume
- Price Action: Pin Bars, Engulfing, Inside Bars, Candle Ratios
- Time Encoding: Cyclical hour/dow

NO traditional indicators (RSI, ATR, Bollinger, MA, etc.)
All features are RELATIVE (ratios, booleans) — scale-invariant.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import polars as pl


# ─────────────────────────────────────────────
# PRICE ACTION — Candle Decomposition
# ─────────────────────────────────────────────

def candle_ratios(df: pl.DataFrame) -> pl.DataFrame:
    """
    Decompose each candle into relative body/wick ratios.

    Output: body_ratio, upper_wick_ratio, lower_wick_ratio, candle_direction
    All ratios bounded [0, 1], sum ≈ 1.0.
    """
    total_range = (df["high"] - df["low"]).alias("_range")

    return df.with_columns([
        (
            (df["close"] - df["open"]).abs() / total_range.clip(lower_bound=1e-10)
        ).clip(0.0, 1.0).alias("body_ratio"),
        (
            (df["high"] - pl.max_horizontal(df["open"], df["close"]))
            / total_range.clip(lower_bound=1e-10)
        ).clip(0.0, 1.0).alias("upper_wick_ratio"),
        (
            (pl.min_horizontal(df["open"], df["close"]) - df["low"])
            / total_range.clip(lower_bound=1e-10)
        ).clip(0.0, 1.0).alias("lower_wick_ratio"),
        pl.when(df["close"] > df["open"]).then(1.0)
        .when(df["close"] < df["open"]).then(-1.0)
        .otherwise(0.0)
        .alias("candle_direction"),
    ])


# ─────────────────────────────────────────────
# PRICE ACTION — Pattern Detection
# ─────────────────────────────────────────────

def pin_bar(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detect Pin Bar (rejection candle).

    Bullish pin bar: lower wick > 2x body, small upper wick
    Bearish pin bar: upper wick > 2x body, small lower wick

    Output: pin_bar_bull (+1/0), pin_bar_bear (-1/0)
    """
    body = (df["close"] - df["open"]).abs()
    total = (df["high"] - df["low"]).clip(lower_bound=1e-10)
    upper_wick = df["high"] - pl.max_horizontal(df["open"], df["close"])
    lower_wick = pl.min_horizontal(df["open"], df["close"]) - df["low"]

    return df.with_columns([
        # Bullish pin: long lower wick, small body, small upper wick
        pl.when(
            (lower_wick > body * 2.0) & (upper_wick < body * 0.5) & (body / total < 0.33)
        ).then(1.0).otherwise(0.0).alias("pin_bar_bull"),

        # Bearish pin: long upper wick, small body, small lower wick
        pl.when(
            (upper_wick > body * 2.0) & (lower_wick < body * 0.5) & (body / total < 0.33)
        ).then(1.0).otherwise(0.0).alias("pin_bar_bear"),
    ])


def engulfing(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detect Engulfing patterns (momentum reversal/continuation).

    Bullish engulfing: bearish candle followed by bullish candle that fully engulfs it.
    Bearish engulfing: bullish candle followed by bearish candle that fully engulfs it.

    Output: engulfing_bull (+1/0), engulfing_bear (-1/0)
    """
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_body = (prev_close - prev_open).abs()
    curr_body = (df["close"] - df["open"]).abs()

    return df.with_columns([
        # Bullish: prev bearish, curr bullish, curr body engulfs prev
        pl.when(
            (prev_close < prev_open)  # prev bearish
            & (df["close"] > df["open"])  # curr bullish
            & (df["open"] <= prev_close)  # curr open <= prev close
            & (df["close"] >= prev_open)  # curr close >= prev open
            & (curr_body > prev_body)  # curr body larger
        ).then(1.0).otherwise(0.0).alias("engulfing_bull"),

        # Bearish: prev bullish, curr bearish, curr body engulfs prev
        pl.when(
            (prev_close > prev_open)  # prev bullish
            & (df["close"] < df["open"])  # curr bearish
            & (df["open"] >= prev_close)  # curr open >= prev close
            & (df["close"] <= prev_open)  # curr close <= prev open
            & (curr_body > prev_body)  # curr body larger
        ).then(1.0).otherwise(0.0).alias("engulfing_bear"),
    ])


def inside_bar(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detect Inside Bar (range contraction / consolidation).

    Inside bar: current bar's high < prev high AND current low > prev low.
    Signals indecision — market compressing before breakout.

    Output: inside_bar (1.0 = inside bar, 0.0 = not)
    """
    return df.with_columns(
        pl.when(
            (df["high"] < df["high"].shift(1))
            & (df["low"] > df["low"].shift(1))
        ).then(1.0).otherwise(0.0).alias("inside_bar")
    )


# ─────────────────────────────────────────────
# VOLUME — Analysis
# ─────────────────────────────────────────────

def relative_volume(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """
    Relative Volume: current vol / rolling avg.
    RVol > 1.5 = above-average (potential breakout/climax).
    """
    vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
    vol_mean = df[vol_col].rolling_mean(window_size=window).alias("_vol_mean")
    return df.with_columns(
        (df[vol_col] / vol_mean.clip(lower_bound=1e-10))
        .alias("relative_volume")
    )


def volume_delta(df: pl.DataFrame) -> pl.DataFrame:
    """
    Estimate Volume Delta from candle position.

    Logic: If close > open (bullish), buying pressure dominates.
    Delta = volume * (2 * position_in_range - 1), where position = (close-low)/(high-low)
    Normalized to [-1, +1] range.

    Output: vol_delta (continuous [-1, +1])
    """
    vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
    total_range = (df["high"] - df["low"]).clip(lower_bound=1e-10)
    close_position = (df["close"] - df["low"]) / total_range  # 0 = low, 1 = high

    # Delta ratio: +1 if close at high (buying), -1 if close at low (selling)
    delta_ratio = (2.0 * close_position - 1.0).clip(-1.0, 1.0)

    return df.with_columns(
        delta_ratio.alias("vol_delta")
    )


def climax_volume(df: pl.DataFrame, window: int = 20, threshold: float = 2.0) -> pl.DataFrame:
    """
    Detect Climax Volume: extreme volume spike with direction.

    Climax = volume > threshold × rolling avg.
    +1 = bullish climax (big volume + bullish candle)
    -1 = bearish climax (big volume + bearish candle)
    0 = no climax

    Output: climax_vol (-1/0/+1)
    """
    vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
    vol_mean = df[vol_col].rolling_mean(window_size=window).clip(lower_bound=1e-10)
    is_climax = df[vol_col] > (vol_mean * threshold)

    return df.with_columns(
        pl.when(is_climax & (df["close"] > df["open"])).then(1.0)
        .when(is_climax & (df["close"] < df["open"])).then(-1.0)
        .otherwise(0.0)
        .alias("climax_vol")
    )


# ─────────────────────────────────────────────
# SMC — Smart Money Concepts
# ─────────────────────────────────────────────

def swing_structure(df: pl.DataFrame, lookback: int = 5) -> pl.DataFrame:
    """
    Detect Swing Highs/Lows and market structure trend.

    Swing High: high[i] > all highs in [i-lookback, i+lookback]
    Swing Low:  low[i] < all lows in [i-lookback, i+lookback]

    Trend: +1 if HH + HL (uptrend), -1 if LH + LL (downtrend), 0 if unclear.

    Output: swing_high (0/1), swing_low (0/1), swing_trend (+1/0/-1)
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(highs)

    sh = np.zeros(n, dtype=np.float32)
    sl = np.zeros(n, dtype=np.float32)
    trend = np.zeros(n, dtype=np.float32)

    # Detect swing points
    for i in range(lookback, n - lookback):
        window_highs = highs[i - lookback: i + lookback + 1]
        window_lows = lows[i - lookback: i + lookback + 1]
        if highs[i] == np.max(window_highs):
            sh[i] = 1.0
        if lows[i] == np.min(window_lows):
            sl[i] = 1.0

    # Build trend from recent swing points
    last_sh_val = 0.0
    last_sl_val = 0.0
    prev_sh_val = 0.0
    prev_sl_val = 0.0

    for i in range(n):
        if sh[i] == 1.0:
            prev_sh_val = last_sh_val
            last_sh_val = highs[i]
        if sl[i] == 1.0:
            prev_sl_val = last_sl_val
            last_sl_val = lows[i]

        # Uptrend: Higher High + Higher Low
        if last_sh_val > prev_sh_val > 0 and last_sl_val > prev_sl_val > 0:
            trend[i] = 1.0
        # Downtrend: Lower High + Lower Low
        elif 0 < last_sh_val < prev_sh_val and 0 < last_sl_val < prev_sl_val:
            trend[i] = -1.0
        else:
            trend[i] = 0.0

    return df.with_columns([
        pl.Series("swing_high", sh),
        pl.Series("swing_low", sl),
        pl.Series("swing_trend", trend),
    ])


def bos_choch(df: pl.DataFrame, lookback: int = 5) -> pl.DataFrame:
    """
    Detect BOS (Break of Structure) and CHoCH (Change of Character).

    BOS: Price breaks above last Swing High (bullish) or below Swing Low (bearish)
         in the SAME trend direction = continuation.
    CHoCH: Price breaks structure AGAINST the trend = reversal signal.

    Output: bos (+1 bullish / -1 bearish / 0), choch (+1 / -1 / 0)
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(highs)

    bos = np.zeros(n, dtype=np.float32)
    choch = np.zeros(n, dtype=np.float32)

    # Find swing levels
    last_sh = 0.0
    last_sl = float("inf")
    trend = 0  # 1 = up, -1 = down

    for i in range(lookback, n - lookback):
        # Check for swing high/low at position i-lookback (confirmed)
        pos = i - lookback
        window_h = highs[pos - lookback: pos + lookback + 1] if pos >= lookback else highs[:pos + lookback + 1]
        window_l = lows[pos - lookback: pos + lookback + 1] if pos >= lookback else lows[:pos + lookback + 1]

        if len(window_h) > 0 and highs[pos] == np.max(window_h):
            last_sh = highs[pos]
        if len(window_l) > 0 and lows[pos] == np.min(window_l):
            last_sl = lows[pos]

        # BOS: break above swing high in uptrend, or below swing low in downtrend
        if last_sh > 0 and closes[i] > last_sh:
            if trend >= 0:
                bos[i] = 1.0  # Bullish BOS (continuation)
            else:
                choch[i] = 1.0  # Bullish CHoCH (reversal)
            trend = 1
            last_sh = highs[i]

        if last_sl < float("inf") and closes[i] < last_sl:
            if trend <= 0:
                bos[i] = -1.0  # Bearish BOS (continuation)
            else:
                choch[i] = -1.0  # Bearish CHoCH (reversal)
            trend = -1
            last_sl = lows[i]

    return df.with_columns([
        pl.Series("bos", bos),
        pl.Series("choch", choch),
    ])


def order_blocks(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detect Order Blocks (OB) — last opposite candle before impulsive move.

    Bullish OB: Last bearish candle before a strong bullish move (3+ candles up).
    Bearish OB: Last bullish candle before a strong bearish move (3+ candles down).

    Output: ob_bull_dist (distance to nearest bullish OB, normalized),
            ob_bear_dist (distance to nearest bearish OB, normalized)
            Values in [0, 1]: 0 = at OB, 1 = far from OB
    """
    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(opens)

    ob_bull = np.ones(n, dtype=np.float32)  # Distance to bullish OB (1 = far)
    ob_bear = np.ones(n, dtype=np.float32)  # Distance to bearish OB (1 = far)

    # Track order block levels
    bull_ob_levels: list[tuple[float, float]] = []  # (ob_high, ob_low)
    bear_ob_levels: list[tuple[float, float]] = []

    for i in range(3, n):
        # Check for impulsive bullish move (3 consecutive bullish candles)
        if all(closes[i-j] > opens[i-j] for j in range(3)):
            # The bearish candle just before = bullish OB
            if i >= 4 and closes[i-3] < opens[i-3]:
                bull_ob_levels.append((highs[i-3], lows[i-3]))
                if len(bull_ob_levels) > 10:
                    bull_ob_levels.pop(0)

        # Check for impulsive bearish move
        if all(closes[i-j] < opens[i-j] for j in range(3)):
            if i >= 4 and closes[i-3] > opens[i-3]:
                bear_ob_levels.append((highs[i-3], lows[i-3]))
                if len(bear_ob_levels) > 10:
                    bear_ob_levels.pop(0)

        # Calculate distance to nearest OB
        price = closes[i]
        price_range = highs[max(0, i-50):i+1].max() - lows[max(0, i-50):i+1].min()
        price_range = max(price_range, 1e-10)

        if bull_ob_levels:
            distances = [abs(price - (h + l) / 2) / price_range for h, l in bull_ob_levels]
            ob_bull[i] = min(min(distances), 1.0)

        if bear_ob_levels:
            distances = [abs(price - (h + l) / 2) / price_range for h, l in bear_ob_levels]
            ob_bear[i] = min(min(distances), 1.0)

    return df.with_columns([
        pl.Series("ob_bull_dist", ob_bull),
        pl.Series("ob_bear_dist", ob_bear),
    ])


def fair_value_gaps(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detect Fair Value Gaps (FVG) — imbalance zones.

    Bullish FVG: candle[i-2].high < candle[i].low (gap up, unfilled)
    Bearish FVG: candle[i-2].low > candle[i].high (gap down, unfilled)

    Output: fvg_bull_active (1 if price near unfilled bullish FVG),
            fvg_bear_active (1 if price near unfilled bearish FVG)
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(highs)

    fvg_bull = np.zeros(n, dtype=np.float32)
    fvg_bear = np.zeros(n, dtype=np.float32)

    # Track active FVGs (midpoint, top, bottom)
    active_bull_fvgs: list[tuple[float, float]] = []  # (gap_low=candle[i-2].high, gap_high=candle[i].low)
    active_bear_fvgs: list[tuple[float, float]] = []

    for i in range(2, n):
        # Detect new bullish FVG
        if highs[i-2] < lows[i]:
            active_bull_fvgs.append((highs[i-2], lows[i]))
            if len(active_bull_fvgs) > 5:
                active_bull_fvgs.pop(0)

        # Detect new bearish FVG
        if lows[i-2] > highs[i]:
            active_bear_fvgs.append((highs[i], lows[i-2]))
            if len(active_bear_fvgs) > 5:
                active_bear_fvgs.pop(0)

        # Check if price is near an active FVG
        price = closes[i]

        # Remove filled FVGs
        active_bull_fvgs = [(lo, hi) for lo, hi in active_bull_fvgs if price > lo]
        active_bear_fvgs = [(lo, hi) for lo, hi in active_bear_fvgs if price < hi]

        # Signal if price touching FVG zone
        for lo, hi in active_bull_fvgs:
            if lo <= price <= hi:
                fvg_bull[i] = 1.0
                break

        for lo, hi in active_bear_fvgs:
            if lo <= price <= hi:
                fvg_bear[i] = 1.0
                break

    return df.with_columns([
        pl.Series("fvg_bull_active", fvg_bull),
        pl.Series("fvg_bear_active", fvg_bear),
    ])


def liquidity_zones(df: pl.DataFrame, window: int = 20, tolerance_pct: float = 0.001) -> pl.DataFrame:
    """
    Detect Liquidity Zones — equal highs/lows where stop-losses cluster.

    Equal highs: Multiple swing highs at similar price = buy-side liquidity
    Equal lows: Multiple swing lows at similar price = sell-side liquidity

    Output: liq_above (normalized distance to nearest buy-side liquidity),
            liq_below (normalized distance to nearest sell-side liquidity)
            Values: 0 = at liquidity, 1 = far from it
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(highs)

    liq_above = np.ones(n, dtype=np.float32)
    liq_below = np.ones(n, dtype=np.float32)

    for i in range(window, n):
        window_highs = highs[i - window: i]
        window_lows = lows[i - window: i]
        price = closes[i]
        price_range = window_highs.max() - window_lows.min()
        if price_range < 1e-10:
            continue

        tol = price * tolerance_pct

        # Find equal highs (buy-side liquidity above)
        above_levels = []
        for h in window_highs:
            if h > price:
                matched = False
                for lvl in above_levels:
                    if abs(h - lvl) < tol:
                        matched = True
                        break
                if not matched:
                    # Count touches
                    touches = sum(1 for wh in window_highs if abs(wh - h) < tol)
                    if touches >= 2:
                        above_levels.append(h)

        if above_levels:
            nearest = min(abs(price - lvl) for lvl in above_levels)
            liq_above[i] = min(nearest / price_range, 1.0)

        # Find equal lows (sell-side liquidity below)
        below_levels = []
        for lo in window_lows:
            if lo < price:
                matched = False
                for lvl in below_levels:
                    if abs(lo - lvl) < tol:
                        matched = True
                        break
                if not matched:
                    touches = sum(1 for wl in window_lows if abs(wl - lo) < tol)
                    if touches >= 2:
                        below_levels.append(lo)

        if below_levels:
            nearest = min(abs(price - lvl) for lvl in below_levels)
            liq_below[i] = min(nearest / price_range, 1.0)

    return df.with_columns([
        pl.Series("liq_above", liq_above),
        pl.Series("liq_below", liq_below),
    ])


# ─────────────────────────────────────────────
# TIME — Cyclical Encoding
# ─────────────────────────────────────────────

def time_encoding(df: pl.DataFrame) -> pl.DataFrame:
    """Cyclical time encoding: sin/cos for hour and day-of-week."""
    TWO_PI = 2.0 * math.pi
    hour_np = df["time"].dt.hour().to_numpy().astype(np.float64)
    dow_np = df["time"].dt.weekday().to_numpy().astype(np.float64)

    return df.with_columns([
        pl.Series("sin_hour", np.sin(hour_np * TWO_PI / 24.0)),
        pl.Series("cos_hour", np.cos(hour_np * TWO_PI / 24.0)),
        pl.Series("sin_dow", np.sin(dow_np * TWO_PI / 5.0)),
        pl.Series("cos_dow", np.cos(dow_np * TWO_PI / 5.0)),
    ])


def time_encoding_numpy(
    hours: np.ndarray,
    days_of_week: np.ndarray,
) -> dict[str, np.ndarray]:
    """Pure numpy time encoding."""
    TWO_PI = 2.0 * np.pi
    return {
        "sin_hour": np.sin(hours * TWO_PI / 24.0),
        "cos_hour": np.cos(hours * TWO_PI / 24.0),
        "sin_dow": np.sin(days_of_week * TWO_PI / 5.0),
        "cos_dow": np.cos(days_of_week * TWO_PI / 5.0),
    }


# ─────────────────────────────────────────────
# LOG RETURN (raw, not an indicator)
# ─────────────────────────────────────────────

def log_return(df: pl.DataFrame) -> pl.DataFrame:
    """Simple log return: ln(close / prev_close). Raw market data, not indicator."""
    return df.with_columns(
        (df["close"] / df["close"].shift(1)).log().alias("log_return")
    )


# ─────────────────────────────────────────────
# Master Feature Pipeline
# ─────────────────────────────────────────────

FEATURE_COLUMNS: list[str] = [
    # Price Action — Candle decomposition
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "candle_direction",
    # Price Action — Patterns
    "pin_bar_bull", "pin_bar_bear",
    "engulfing_bull", "engulfing_bear",
    "inside_bar",
    # Volume
    "relative_volume", "vol_delta", "climax_vol",
    # SMC — Structure
    "swing_high", "swing_low", "swing_trend",
    "bos", "choch",
    # SMC — Zones
    "ob_bull_dist", "ob_bear_dist",
    "fvg_bull_active", "fvg_bear_active",
    "liq_above", "liq_below",
    # Time encoding
    "sin_hour", "cos_hour", "sin_dow", "cos_dow",
    # Raw return
    "log_return",
]


def build_features(
    df: pl.DataFrame,
    vol_window: int = 20,
    swing_lookback: int = 5,
    include_spread: bool = False,
) -> pl.DataFrame:
    """
    Apply full SMC + Volume + Price Action feature pipeline.

    Args:
        df: DataFrame with columns [time, open, high, low, close, tick_volume/volume]
        vol_window: Window for relative volume and climax detection
        swing_lookback: Window for swing high/low detection

    Returns:
        DataFrame with all feature columns. First rows may have NaN from rolling.
    """
    result = df.clone()

    # Price Action
    result = candle_ratios(result)
    result = pin_bar(result)
    result = engulfing(result)
    result = inside_bar(result)

    # Volume
    result = relative_volume(result, window=vol_window)
    result = volume_delta(result)
    result = climax_volume(result, window=vol_window)

    # SMC
    result = swing_structure(result, lookback=swing_lookback)
    result = bos_choch(result, lookback=swing_lookback)
    result = order_blocks(result)
    result = fair_value_gaps(result)
    result = liquidity_zones(result, window=vol_window)

    # Time + Returns
    result = time_encoding(result)
    result = log_return(result)

    # Drop warmup rows
    warmup = max(vol_window, swing_lookback * 2, 5)
    result = result.slice(warmup)

    return result
