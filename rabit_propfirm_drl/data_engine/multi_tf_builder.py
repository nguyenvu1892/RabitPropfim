"""
Multi-Timeframe Feature Builder — Resamples and aligns features across timeframes.

The Cross-Attention Transformer needs:
- M15 features as QUERY (primary decision timeframe)
- H1/H4 features as CONTEXT (higher-timeframe structure)

This module handles:
1. Resampling M1 → M5, M15, H1, H4
2. Building features for each timeframe
3. Aligning timestamps so each M15 row has corresponding H1/H4 context
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import polars as pl

from data_engine.feature_builder import build_features

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# OHLCV Resampling
# ─────────────────────────────────────────────

# Polars duration strings for each timeframe
TF_DURATIONS = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
}


def resample_ohlcv(df: pl.DataFrame, target_tf: str) -> pl.DataFrame:
    """
    Resample M1 OHLCV data to a higher timeframe.

    Args:
        df: M1 DataFrame with [time, open, high, low, close, volume] columns
        target_tf: Target timeframe (M5, M15, H1, H4, D1)

    Returns:
        Resampled OHLCV DataFrame
    """
    duration = TF_DURATIONS.get(target_tf)
    if duration is None:
        raise ValueError(f"Unknown timeframe: {target_tf}. Use one of {list(TF_DURATIONS)}")

    resampled = (
        df.sort("time")
        .group_by_dynamic("time", every=duration)
        .agg([
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        ])
        .sort("time")
    )

    # Carry spread forward if present
    if "spread" in df.columns:
        spread_resampled = (
            df.sort("time")
            .group_by_dynamic("time", every=duration)
            .agg([pl.col("spread").mean().alias("spread")])
            .sort("time")
        )
        resampled = resampled.join(spread_resampled, on="time", how="left")

    logger.info(
        "Resampled %d M1 bars → %d %s bars",
        len(df), len(resampled), target_tf,
    )
    return resampled


# ─────────────────────────────────────────────
# Multi-TF Feature Construction
# ─────────────────────────────────────────────

def build_multi_tf_features(
    m1_data: pl.DataFrame,
    timeframes: list[str] | None = None,
) -> dict[str, pl.DataFrame]:
    """
    Build features for multiple timeframes from M1 data.

    Args:
        m1_data: Raw M1 OHLCV DataFrame
        timeframes: List of timeframes to build. Default: ["M5", "M15", "H1", "H4"]

    Returns:
        Dict mapping timeframe → feature-enriched DataFrame
        Example: {"M15": df_m15, "H1": df_h1, "H4": df_h4}
    """
    if timeframes is None:
        timeframes = ["M5", "M15", "H1", "H4"]

    result: dict[str, pl.DataFrame] = {}

    for tf in timeframes:
        if tf == "M1":
            resampled = m1_data
        else:
            resampled = resample_ohlcv(m1_data, tf)

        featured = build_features(resampled)
        result[tf] = featured
        logger.info("Built %d features for %s (%d rows)", len(featured.columns), tf, len(featured))

    return result


# ─────────────────────────────────────────────
# Timestamp Alignment
# ─────────────────────────────────────────────

def align_context_to_query(
    query_df: pl.DataFrame,
    context_df: pl.DataFrame,
    context_prefix: str,
    lookback: int,
) -> pl.DataFrame:
    """
    For each row in query_df, find the most recent context row and align.

    Uses asof join: for each query timestamp, finds the latest context timestamp
    that is <= query timestamp. This prevents look-ahead bias.

    Args:
        query_df: Primary timeframe DataFrame (e.g., M15) with 'time' column
        context_df: Higher timeframe DataFrame (e.g., H4) with 'time' column
        context_prefix: Prefix for context columns (e.g., "h4_")
        lookback: Not used for asof join, kept for API compatibility

    Returns:
        query_df with context columns joined
    """
    # Prefix all non-time columns in context
    context_renamed = context_df.rename(
        {col: f"{context_prefix}{col}" for col in context_df.columns if col != "time"}
    )

    # Asof join: for each query time, get latest context that is <= query time
    aligned = query_df.sort("time").join_asof(
        context_renamed.sort("time"),
        on="time",
        strategy="backward",  # Use latest context AT OR BEFORE query time
    )

    logger.info(
        "Aligned %s context to query: %d rows, %d new columns",
        context_prefix, len(aligned),
        len(context_renamed.columns) - 1,  # -1 for 'time'
    )
    return aligned


def build_aligned_dataset(
    m1_data: pl.DataFrame,
    query_tf: str = "M15",
    context_tfs: list[str] | None = None,
    context_lookbacks: dict[str, int] | None = None,
) -> pl.DataFrame:
    """
    Build the complete multi-timeframe aligned dataset.

    Creates query features (M15 by default) and aligns context features
    from higher timeframes (H1, H4 by default) using asof joins.

    Args:
        m1_data: Raw M1 data
        query_tf: Primary decision timeframe
        context_tfs: Higher timeframes for context
        context_lookbacks: Lookback bars per context TF (for API compat)

    Returns:
        Single DataFrame with query features + aligned context features
    """
    if context_tfs is None:
        context_tfs = ["H1", "H4"]
    if context_lookbacks is None:
        context_lookbacks = {"H1": 48, "H4": 30}

    # Build all TF features
    all_tfs = build_multi_tf_features(m1_data, [query_tf] + context_tfs)

    # Start with query TF
    result = all_tfs[query_tf]

    # Align each context TF
    for ctx_tf in context_tfs:
        prefix = f"{ctx_tf.lower()}_"
        lookback = context_lookbacks.get(ctx_tf, 30)
        result = align_context_to_query(
            result,
            all_tfs[ctx_tf],
            context_prefix=prefix,
            lookback=lookback,
        )

    # Drop rows with nulls from alignment
    result = result.drop_nulls()

    logger.info(
        "Final aligned dataset: %d rows, %d columns",
        len(result), len(result.columns),
    )
    return result
