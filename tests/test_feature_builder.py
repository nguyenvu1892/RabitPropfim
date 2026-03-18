"""
Tests for Feature Builder — SMC + Volume + Price Action.

Validates:
- Candle ratios bounded [0, 1] and sum ≈ 1.0
- Pin bar, engulfing, inside bar detection produces valid signals
- Volume delta in [-1, +1], climax vol in {-1, 0, +1}
- SMC swing structure detects swing highs/lows
- BOS/CHoCH produce valid signals
- Order blocks and FVG detection runs without errors
- Full pipeline produces all expected feature columns
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from data_engine.feature_builder import (
    FEATURE_COLUMNS,
    bos_choch,
    build_features,
    candle_ratios,
    climax_volume,
    engulfing,
    fair_value_gaps,
    inside_bar,
    liquidity_zones,
    log_return,
    order_blocks,
    pin_bar,
    relative_volume,
    swing_structure,
    time_encoding,
    time_encoding_numpy,
    volume_delta,
)
from data_engine.multi_tf_builder import (
    build_multi_tf_features,
    resample_ohlcv,
)


# ─────────────────────────────────────────────
# Test Data Generator
# ─────────────────────────────────────────────

def _make_ohlcv(n_bars: int = 1000, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic M5 OHLCV data with valid candle relationships."""
    rng = np.random.default_rng(seed)
    base_price = 1.1000
    returns = rng.normal(0, 0.0003, n_bars)
    mid_prices = base_price + np.cumsum(returns)

    start_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    times = [start_time + timedelta(minutes=5 * i) for i in range(n_bars)]

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
# TEST: Candle Ratios (Price Action)
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
        assert np.allclose(total, 1.0, atol=0.01)

    def test_candle_direction_values(self) -> None:
        df = _make_ohlcv()
        result = candle_ratios(df)
        directions = result["candle_direction"].to_numpy()
        assert set(np.unique(directions)).issubset({-1.0, 0.0, 1.0})


# ─────────────────────────────────────────────
# TEST: Price Action Patterns
# ─────────────────────────────────────────────

class TestPriceActionPatterns:

    def test_pin_bar_values(self) -> None:
        df = _make_ohlcv()
        result = pin_bar(df)
        for col in ["pin_bar_bull", "pin_bar_bear"]:
            values = result[col].to_numpy()
            assert set(np.unique(values)).issubset({0.0, 1.0})

    def test_engulfing_values(self) -> None:
        df = _make_ohlcv()
        result = engulfing(df)
        for col in ["engulfing_bull", "engulfing_bear"]:
            vals = result[col].drop_nulls().to_numpy()
            assert set(np.unique(vals)).issubset({0.0, 1.0})

    def test_inside_bar_values(self) -> None:
        df = _make_ohlcv()
        result = inside_bar(df)
        vals = result["inside_bar"].drop_nulls().to_numpy()
        assert set(np.unique(vals)).issubset({0.0, 1.0})

    def test_inside_bar_detects_some(self) -> None:
        """With random data, at least some inside bars should appear."""
        df = _make_ohlcv(n_bars=2000)
        result = inside_bar(df)
        count = result["inside_bar"].sum()
        assert count > 0, "No inside bars detected in 2000 random candles"


# ─────────────────────────────────────────────
# TEST: Volume Features
# ─────────────────────────────────────────────

class TestVolumeFeatures:

    def test_relative_volume_positive(self) -> None:
        df = _make_ohlcv()
        result = relative_volume(df, window=20)
        rvol = result["relative_volume"].slice(20).to_numpy()
        assert np.all(rvol > 0)

    def test_volume_delta_bounded(self) -> None:
        df = _make_ohlcv()
        result = volume_delta(df)
        vd = result["vol_delta"].to_numpy()
        assert np.all(vd >= -1.0 - 1e-7)
        assert np.all(vd <= 1.0 + 1e-7)

    def test_climax_vol_values(self) -> None:
        df = _make_ohlcv()
        result = climax_volume(df, window=20)
        vals = result["climax_vol"].slice(20).to_numpy()
        assert set(np.unique(vals)).issubset({-1.0, 0.0, 1.0})


# ─────────────────────────────────────────────
# TEST: SMC Features
# ─────────────────────────────────────────────

class TestSMCFeatures:

    def test_swing_structure_values(self) -> None:
        df = _make_ohlcv()
        result = swing_structure(df, lookback=5)
        for col in ["swing_high", "swing_low"]:
            vals = result[col].to_numpy()
            assert set(np.unique(vals)).issubset({0.0, 1.0})
        trend = result["swing_trend"].to_numpy()
        assert set(np.unique(trend)).issubset({-1.0, 0.0, 1.0})

    def test_swing_detects_some(self) -> None:
        df = _make_ohlcv(n_bars=2000)
        result = swing_structure(df, lookback=5)
        assert result["swing_high"].sum() > 0
        assert result["swing_low"].sum() > 0

    def test_bos_choch_values(self) -> None:
        df = _make_ohlcv()
        result = bos_choch(df, lookback=5)
        for col in ["bos", "choch"]:
            vals = result[col].to_numpy()
            assert set(np.unique(vals)).issubset({-1.0, 0.0, 1.0})

    def test_order_blocks_bounded(self) -> None:
        df = _make_ohlcv(n_bars=500)
        result = order_blocks(df)
        for col in ["ob_bull_dist", "ob_bear_dist"]:
            vals = result[col].to_numpy()
            assert np.all(vals >= 0.0)
            assert np.all(vals <= 1.0 + 1e-7)

    def test_fvg_values(self) -> None:
        df = _make_ohlcv(n_bars=500)
        result = fair_value_gaps(df)
        for col in ["fvg_bull_active", "fvg_bear_active"]:
            vals = result[col].to_numpy()
            assert set(np.unique(vals)).issubset({0.0, 1.0})

    def test_liquidity_zones_bounded(self) -> None:
        df = _make_ohlcv(n_bars=500)
        result = liquidity_zones(df, window=20)
        for col in ["liq_above", "liq_below"]:
            vals = result[col].to_numpy()
            assert np.all(vals >= 0.0)
            assert np.all(vals <= 1.0 + 1e-7)


# ─────────────────────────────────────────────
# TEST: Time Encoding
# ─────────────────────────────────────────────

class TestTimeEncoding:

    def test_sin_cos_bounded(self) -> None:
        df = _make_ohlcv()
        result = time_encoding(df)
        for col in ["sin_hour", "cos_hour", "sin_dow", "cos_dow"]:
            values = result[col].to_numpy()
            assert np.all(values >= -1.0 - 1e-7)
            assert np.all(values <= 1.0 + 1e-7)

    def test_time_encoding_numpy(self) -> None:
        hours = np.array([0, 6, 12, 18, 23])
        dows = np.array([0, 1, 2, 3, 4])
        result = time_encoding_numpy(hours, dows)
        for key in ["sin_hour", "cos_hour", "sin_dow", "cos_dow"]:
            assert key in result
            assert np.all(result[key] >= -1.0 - 1e-7)
            assert np.all(result[key] <= 1.0 + 1e-7)


# ─────────────────────────────────────────────
# TEST: Full Pipeline
# ─────────────────────────────────────────────

class TestBuildFeatures:

    def test_full_pipeline_runs(self) -> None:
        df = _make_ohlcv(n_bars=500)
        result = build_features(df)
        assert len(result) > 0

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
            # Allow NaN in first few rows, check no inf
            finite_mask = np.isfinite(values)
            nan_mask = np.isnan(values)
            inf_only = ~finite_mask & ~nan_mask
            assert not np.any(inf_only), f"{col} has infinite values"


# ─────────────────────────────────────────────
# TEST: Multi-TF Resampling
# ─────────────────────────────────────────────

class TestMultiTF:

    def test_resample_reduces_rows(self) -> None:
        df = _make_ohlcv(n_bars=1000)
        m15 = resample_ohlcv(df, "M15")
        h1 = resample_ohlcv(df, "H1")
        assert len(m15) < len(df)
        assert len(h1) < len(m15)

    def test_build_multi_tf(self) -> None:
        df = _make_ohlcv(n_bars=5000)
        result = build_multi_tf_features(df, ["M15", "H1"])
        assert "M15" in result
        assert "H1" in result
        assert len(result["M15"]) > len(result["H1"])
