"""
Tests for Feature Builder (T1.3.3).

Validates:
- Candle ratios are bounded [0, 1] and sum ≈ 1.0
- Relative volume is always positive
- Sin/cos time encoding is bounded [-1, 1]
- Returns features produce valid log returns
- Multi-TF alignment produces correct row counts
- Full pipeline runs without errors
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from data_engine.feature_builder import (
    FEATURE_COLUMNS,
    build_features,
    candle_ratios,
    price_position,
    relative_volume,
    returns_features,
    time_encoding,
    time_encoding_numpy,
)
from data_engine.multi_tf_builder import (
    build_multi_tf_features,
    resample_ohlcv,
)


# ─────────────────────────────────────────────
# Test Data Generator
# ─────────────────────────────────────────────

def _make_ohlcv(n_bars: int = 1000, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic M1 OHLCV data with valid candle relationships."""
    rng = np.random.default_rng(seed)

    # Simulate random walk price
    base_price = 1.1000
    returns = rng.normal(0, 0.0002, n_bars)
    mid_prices = base_price + np.cumsum(returns)

    start_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    times = [start_time + timedelta(minutes=i) for i in range(n_bars)]

    # Generate valid OHLCV: high >= max(open,close), low <= min(open,close)
    opens = mid_prices + rng.normal(0, 0.0001, n_bars)
    closes = mid_prices + rng.normal(0, 0.0001, n_bars)
    highs = np.maximum(opens, closes) + rng.uniform(0.0001, 0.0005, n_bars)
    lows = np.minimum(opens, closes) - rng.uniform(0.0001, 0.0005, n_bars)
    volumes = rng.uniform(100, 5000, n_bars)
    spreads = rng.uniform(1.0, 3.0, n_bars)

    return pl.DataFrame({
        "time": pl.Series(times).cast(pl.Datetime("us", time_zone="UTC")),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "spread": spreads,
    })


# ─────────────────────────────────────────────
# TEST: Candle Ratios
# ─────────────────────────────────────────────

class TestCandleRatios:

    def test_ratios_bounded_0_1(self) -> None:
        df = _make_ohlcv()
        result = candle_ratios(df)
        for col in ["body_ratio", "upper_wick_ratio", "lower_wick_ratio"]:
            values = result[col].to_numpy()
            assert np.all(values >= 0.0), f"{col} has values < 0"
            assert np.all(values <= 1.0), f"{col} has values > 1"

    def test_ratios_sum_approximately_one(self) -> None:
        df = _make_ohlcv()
        result = candle_ratios(df)
        total = (
            result["body_ratio"] + result["upper_wick_ratio"] + result["lower_wick_ratio"]
        ).to_numpy()
        # Allow small floating point errors
        assert np.allclose(total, 1.0, atol=0.01), f"Ratios don't sum to 1: range {total.min()}-{total.max()}"

    def test_candle_direction_values(self) -> None:
        df = _make_ohlcv()
        result = candle_ratios(df)
        directions = result["candle_direction"].to_numpy()
        assert set(np.unique(directions)).issubset({-1.0, 0.0, 1.0})


# ─────────────────────────────────────────────
# TEST: Relative Volume
# ─────────────────────────────────────────────

class TestRelativeVolume:

    def test_relative_volume_positive(self) -> None:
        df = _make_ohlcv()
        result = relative_volume(df, window=20)
        # Skip first 20 rows (NaN from rolling)
        rvol = result["relative_volume"].slice(20).to_numpy()
        assert np.all(rvol > 0), "Relative volume should always be positive"

    def test_relative_volume_around_one(self) -> None:
        df = _make_ohlcv()
        result = relative_volume(df, window=20)
        rvol = result["relative_volume"].slice(20).to_numpy()
        # Mean should be roughly around 1.0
        assert 0.5 < np.mean(rvol) < 2.0, f"Mean RVol = {np.mean(rvol)}, expected ~1.0"


# ─────────────────────────────────────────────
# TEST: Time Encoding
# ─────────────────────────────────────────────

class TestTimeEncoding:

    def test_sin_cos_bounded(self) -> None:
        df = _make_ohlcv()
        result = time_encoding(df)
        for col in ["sin_hour", "cos_hour", "sin_dow", "cos_dow"]:
            values = result[col].to_numpy()
            assert np.all(values >= -1.0 - 1e-7), f"{col} has values < -1"
            assert np.all(values <= 1.0 + 1e-7), f"{col} has values > 1"

    def test_time_encoding_numpy(self) -> None:
        hours = np.array([0, 6, 12, 18, 23])
        dows = np.array([0, 1, 2, 3, 4])
        result = time_encoding_numpy(hours, dows)
        for key in ["sin_hour", "cos_hour", "sin_dow", "cos_dow"]:
            assert key in result
            assert np.all(result[key] >= -1.0 - 1e-7)
            assert np.all(result[key] <= 1.0 + 1e-7)


# ─────────────────────────────────────────────
# TEST: Returns Features
# ─────────────────────────────────────────────

class TestReturnsFeatures:

    def test_log_return_finite(self) -> None:
        df = _make_ohlcv()
        result = returns_features(df)
        log_ret = result["log_return"].slice(1).to_numpy()  # First row is NaN
        assert np.all(np.isfinite(log_ret)), "Log returns should be finite"

    def test_atr_normalized_positive(self) -> None:
        df = _make_ohlcv()
        result = returns_features(df, atr_window=14)
        atr = result["atr_normalized"].slice(14).to_numpy()
        valid = atr[np.isfinite(atr)]
        assert np.all(valid >= 0), "ATR normalized should be non-negative"


# ─────────────────────────────────────────────
# TEST: Full Pipeline
# ─────────────────────────────────────────────

class TestBuildFeatures:

    def test_full_pipeline_runs(self) -> None:
        df = _make_ohlcv(n_bars=500)
        result = build_features(df)
        assert len(result) > 0, "Pipeline should produce rows"

    def test_feature_columns_present(self) -> None:
        df = _make_ohlcv(n_bars=500)
        result = build_features(df)
        for col in FEATURE_COLUMNS:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_no_infinities(self) -> None:
        df = _make_ohlcv(n_bars=500)
        result = build_features(df)
        for col in FEATURE_COLUMNS:
            values = result[col].to_numpy()
            finite = np.isfinite(values)
            assert np.all(finite), f"{col} has non-finite values"


# ─────────────────────────────────────────────
# TEST: Multi-TF Resampling
# ─────────────────────────────────────────────

class TestMultiTF:

    def test_resample_reduces_rows(self) -> None:
        df = _make_ohlcv(n_bars=1000)
        m15 = resample_ohlcv(df, "M15")
        h1 = resample_ohlcv(df, "H1")
        assert len(m15) < len(df), "M15 should have fewer rows than M1"
        assert len(h1) < len(m15), "H1 should have fewer rows than M15"

    def test_build_multi_tf(self) -> None:
        df = _make_ohlcv(n_bars=5000)
        result = build_multi_tf_features(df, ["M15", "H1"])
        assert "M15" in result
        assert "H1" in result
        assert len(result["M15"]) > len(result["H1"])
