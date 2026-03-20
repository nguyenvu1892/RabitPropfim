"""
KnowledgeExtractor — Semantic Market Feature Engineering.

Converts raw OHLCV + indicator data into 22 semantically meaningful
variables that encode TRADER KNOWLEDGE, not just numbers.

Three variable groups:
    A. SMC Variables  (7):  distance_to_ob, is_in_fvg, trend_state, etc.
    B. PA Variables   (8):  is_pinbar, is_doji, is_engulfing, etc.
    C. Volume Variables (5): vol_anomaly, vol_exhaustion, vol_climax, etc.
    D. Context Variables (2): session time encoding

Total output: 22-dim vector per bar, all values in [-1, 1] or [0, 1].

Performance: Full NumPy vectorization — designed to run in <1ms per bar
on real-time tick data. No Python loops on the hot path.

Usage:
    extractor = KnowledgeExtractor()
    knowledge = extractor.extract(ohlcv_window, atr_value)
    # knowledge.shape == (22,) or (N, 22) for batch
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ─── Constants ──────────────────────────────────────────────────────
N_SMC_VARS = 7
N_PA_VARS = 8
N_VOL_VARS = 5
N_CTX_VARS = 2
N_KNOWLEDGE_VARS = N_SMC_VARS + N_PA_VARS + N_VOL_VARS + N_CTX_VARS  # 22

# Feature indices in output vector
SMC_START = 0
PA_START = N_SMC_VARS                        # 7
VOL_START = N_SMC_VARS + N_PA_VARS           # 15
CTX_START = N_SMC_VARS + N_PA_VARS + N_VOL_VARS  # 20

FEATURE_NAMES = [
    # SMC (7)
    "distance_to_ob_bull", "distance_to_ob_bear", "is_in_fvg",
    "fvg_fill_pct", "trend_state", "liquidity_grab", "swing_distance",
    # PA (8)
    "is_pinbar", "is_doji", "is_inside_bar", "is_engulfing_bull",
    "is_engulfing_bear", "is_hammer", "is_shooting_star", "candle_strength",
    # Volume (5)
    "vol_anomaly", "vol_exhaustion", "vol_climax", "vol_trend", "delta_approx",
    # Context (2)
    "session_sin", "session_cos",
]


@dataclass
class OBZone:
    """Order Block zone for SMC analysis."""
    price_high: float
    price_low: float
    direction: int   # +1 bullish (demand), -1 bearish (supply)
    strength: float  # 0-1, based on volume and wick at creation


@dataclass
class FVGZone:
    """Fair Value Gap zone."""
    high: float      # Top of the gap
    low: float       # Bottom of the gap
    direction: int   # +1 bullish FVG, -1 bearish FVG
    fill_pct: float  # How much of the gap has been filled (0-1)


class KnowledgeExtractor:
    """
    Extracts 22 semantic market knowledge variables from raw data.

    All computations are vectorized with NumPy. For single-bar extraction
    (real-time), pass a window of data and the last row is the "current" bar.

    Args:
        vol_sma_period: Period for volume moving average (default: 20)
        swing_lookback: Bars to look back for swing high/low detection (default: 5)
        vol_anomaly_threshold: Volume / SMA ratio to flag anomaly (default: 2.0)
        vol_exhaustion_threshold: Low volume threshold ratio (default: 0.5)
        vol_climax_threshold: Climax volume threshold ratio (default: 3.0)
    """

    def __init__(
        self,
        vol_sma_period: int = 20,
        swing_lookback: int = 5,
        vol_anomaly_threshold: float = 2.0,
        vol_exhaustion_threshold: float = 0.5,
        vol_climax_threshold: float = 3.0,
    ) -> None:
        self.vol_sma_period = vol_sma_period
        self.swing_lookback = swing_lookback
        self.vol_anomaly_threshold = vol_anomaly_threshold
        self.vol_exhaustion_threshold = vol_exhaustion_threshold
        self.vol_climax_threshold = vol_climax_threshold

    def extract(
        self,
        ohlcv: np.ndarray,
        atr: float,
        ob_zones: Optional[list[OBZone]] = None,
        fvg_zones: Optional[list[FVGZone]] = None,
        bar_index: int = -1,
    ) -> np.ndarray:
        """
        Extract 22-dim knowledge vector for a single bar.

        Args:
            ohlcv:     (N, 5) array — [open, high, low, close, volume]
                       N should be >= vol_sma_period + swing_lookback
            atr:       Average True Range in price units (for normalization)
            ob_zones:  Optional list of active Order Block zones
            fvg_zones: Optional list of active Fair Value Gap zones
            bar_index: Which bar to extract for (-1 = last bar)

        Returns:
            (22,) float32 array — knowledge variables
        """
        if atr <= 0:
            atr = 1e-8  # Safety

        o = ohlcv[:, 0]
        h = ohlcv[:, 1]
        l = ohlcv[:, 2]
        c = ohlcv[:, 3]
        v = ohlcv[:, 4] if ohlcv.shape[1] > 4 else np.ones(len(ohlcv))

        idx = bar_index if bar_index >= 0 else len(ohlcv) - 1
        result = np.zeros(N_KNOWLEDGE_VARS, dtype=np.float32)

        # ── Group A: SMC Variables (7) ──
        result[SMC_START:SMC_START + N_SMC_VARS] = self._extract_smc(
            o, h, l, c, idx, atr, ob_zones, fvg_zones,
        )

        # ── Group B: PA Variables (8) ──
        result[PA_START:PA_START + N_PA_VARS] = self._extract_pa(
            o, h, l, c, v, idx,
        )

        # ── Group C: Volume Variables (5) ──
        result[VOL_START:VOL_START + N_VOL_VARS] = self._extract_volume(
            o, h, l, c, v, idx,
        )

        # ── Group D: Context Variables (2) ──
        result[CTX_START:CTX_START + N_CTX_VARS] = self._extract_context(idx)

        return result

    def extract_batch(
        self,
        ohlcv: np.ndarray,
        atr_array: np.ndarray,
        ob_zones: Optional[list[OBZone]] = None,
        fvg_zones: Optional[list[FVGZone]] = None,
        start_idx: int = 0,
    ) -> np.ndarray:
        """
        Extract knowledge for multiple consecutive bars (batch mode).

        Args:
            ohlcv:     (N, 5) — full OHLCV history
            atr_array: (N,) — ATR per bar
            ob_zones:  Optional OB zones (static for this batch)
            fvg_zones: Optional FVG zones (static for this batch)
            start_idx: First bar to extract (needs lookback behind it)

        Returns:
            (N - start_idx, 22) float32 array
        """
        n_bars = len(ohlcv) - start_idx
        result = np.zeros((n_bars, N_KNOWLEDGE_VARS), dtype=np.float32)

        for i in range(n_bars):
            idx = start_idx + i
            atr_val = float(atr_array[idx]) if idx < len(atr_array) else 1e-8
            result[i] = self.extract(ohlcv, atr_val, ob_zones, fvg_zones, idx)

        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # GROUP A: SMC Variables (7)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _extract_smc(
        self,
        o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
        idx: int, atr: float,
        ob_zones: Optional[list[OBZone]],
        fvg_zones: Optional[list[FVGZone]],
    ) -> np.ndarray:
        """Extract 7 SMC variables."""
        result = np.zeros(N_SMC_VARS, dtype=np.float32)
        price = c[idx]

        # [0] distance_to_ob_bull — normalized by ATR
        # [1] distance_to_ob_bear — normalized by ATR
        if ob_zones:
            bull_obs = [z for z in ob_zones if z.direction > 0]
            bear_obs = [z for z in ob_zones if z.direction < 0]

            if bull_obs:
                closest_bull = min(bull_obs, key=lambda z: abs(price - z.price_high))
                result[0] = np.clip((price - closest_bull.price_high) / atr, -5, 5) / 5
            if bear_obs:
                closest_bear = min(bear_obs, key=lambda z: abs(z.price_low - price))
                result[1] = np.clip((closest_bear.price_low - price) / atr, -5, 5) / 5

        # [2] is_in_fvg — 1.0 if price is inside any active FVG
        # [3] fvg_fill_pct — fill percentage of the nearest FVG
        if fvg_zones:
            for fvg in fvg_zones:
                if fvg.low <= price <= fvg.high:
                    result[2] = 1.0
                    result[3] = np.clip(fvg.fill_pct, 0, 1)
                    break
            if result[2] == 0:
                # Not inside FVG — find closest
                if fvg_zones:
                    closest_fvg = min(
                        fvg_zones,
                        key=lambda z: min(abs(price - z.high), abs(price - z.low)),
                    )
                    result[3] = np.clip(closest_fvg.fill_pct, 0, 1)

        # [4] trend_state — derived from BOS/CHoCH in recent bars
        #     +1.0 = BOS up, +0.5 = CHoCH up, 0 = neutral,
        #     -0.5 = CHoCH down, -1.0 = BOS down
        result[4] = self._compute_trend_state(h, l, c, idx)

        # [5] liquidity_grab — wick swept swing high/low then reversed
        result[5] = self._detect_liquidity_grab(o, h, l, c, idx)

        # [6] swing_distance — min distance to swing H/L, normalized by ATR
        result[6] = self._compute_swing_distance(h, l, c, idx, atr)

        return result

    def _compute_trend_state(
        self, h: np.ndarray, l: np.ndarray, c: np.ndarray, idx: int,
    ) -> float:
        """
        Compute trend state from recent price structure.

        Logic:
        - Find recent swing highs/lows
        - BOS up   = higher high + higher low  → +1.0
        - CHoCH up = higher low after lower low → +0.5
        - BOS down = lower low + lower high     → -1.0
        - CHoCH dn = lower high after higher H  → -0.5
        """
        lb = min(self.swing_lookback * 4, idx)
        if lb < 10:
            return 0.0

        recent_h = h[idx - lb:idx + 1]
        recent_l = l[idx - lb:idx + 1]

        # Find local swing highs/lows
        swing_highs = []
        swing_lows = []
        for i in range(2, len(recent_h) - 2):
            if recent_h[i] >= recent_h[i-1] and recent_h[i] >= recent_h[i-2] \
               and recent_h[i] >= recent_h[i+1] and recent_h[i] >= recent_h[i+2]:
                swing_highs.append(recent_h[i])
            if recent_l[i] <= recent_l[i-1] and recent_l[i] <= recent_l[i-2] \
               and recent_l[i] <= recent_l[i+1] and recent_l[i] <= recent_l[i+2]:
                swing_lows.append(recent_l[i])

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 0.0

        # Compare last two swings
        hh = swing_highs[-1] > swing_highs[-2]  # Higher High
        hl = swing_lows[-1] > swing_lows[-2]     # Higher Low
        ll = swing_lows[-1] < swing_lows[-2]     # Lower Low
        lh = swing_highs[-1] < swing_highs[-2]   # Lower High

        if hh and hl:
            return 1.0    # BOS Up — strong uptrend
        elif hl and not hh:
            return 0.5    # CHoCH Up — potential reversal to bullish
        elif ll and lh:
            return -1.0   # BOS Down — strong downtrend
        elif lh and not ll:
            return -0.5   # CHoCH Down — potential reversal to bearish
        return 0.0

    def _detect_liquidity_grab(
        self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
        idx: int,
    ) -> float:
        """
        Detect liquidity sweep — wick beyond swing then close inside.

        A liquidity grab occurs when price briefly exceeds a swing level
        (sweeping stop losses) then reverses back. Classic institutional move.
        """
        lb = min(self.swing_lookback * 3, idx)
        if lb < 6:
            return 0.0

        # Find swing high/low in lookback (excluding current bar)
        window_h = h[idx - lb:idx]
        window_l = l[idx - lb:idx]
        swing_high = np.max(window_h)
        swing_low = np.min(window_l)

        current_h = h[idx]
        current_l = l[idx]
        current_c = c[idx]
        current_o = o[idx]

        # Bull liquidity grab: wick below swing low but close above
        if current_l < swing_low and current_c > swing_low:
            if current_c > current_o:  # Bullish close
                return 1.0

        # Bear liquidity grab: wick above swing high but close below
        if current_h > swing_high and current_c < swing_high:
            if current_c < current_o:  # Bearish close
                return -1.0

        return 0.0

    def _compute_swing_distance(
        self, h: np.ndarray, l: np.ndarray, c: np.ndarray,
        idx: int, atr: float,
    ) -> float:
        """
        Min distance from current price to nearest swing high or swing low.
        Normalized by ATR, clipped to [-1, 1].
        Negative = below swing low, Positive = above swing high.
        """
        lb = min(self.swing_lookback * 4, idx)
        if lb < 6:
            return 0.0

        window_h = h[idx - lb:idx]
        window_l = l[idx - lb:idx]
        swing_high = np.max(window_h)
        swing_low = np.min(window_l)
        price = c[idx]

        dist_to_high = (price - swing_high) / atr
        dist_to_low = (price - swing_low) / atr

        # Return whichever is closer, sign-preserving
        if abs(dist_to_high) < abs(dist_to_low):
            return float(np.clip(dist_to_high, -1, 1))
        else:
            return float(np.clip(dist_to_low, -1, 1))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # GROUP B: Price Action Variables (8)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _extract_pa(
        self,
        o: np.ndarray, h: np.ndarray, l: np.ndarray,
        c: np.ndarray, v: np.ndarray, idx: int,
    ) -> np.ndarray:
        """Extract 8 Price Action variables for bar at idx."""
        result = np.zeros(N_PA_VARS, dtype=np.float32)

        co = c[idx]
        op = o[idx]
        hi = h[idx]
        lo = l[idx]

        body = abs(co - op)
        full_range = hi - lo + 1e-10
        body_ratio = body / full_range

        is_bullish = co >= op
        upper_wick = (hi - co) if is_bullish else (hi - op)
        lower_wick = (op - lo) if is_bullish else (co - lo)
        upper_wick_ratio = upper_wick / full_range
        lower_wick_ratio = lower_wick / full_range

        # [0] is_pinbar — long lower wick, small body (bullish reversal signal)
        result[0] = 1.0 if (lower_wick_ratio > 0.6 and body_ratio < 0.3) else 0.0

        # [1] is_doji — tiny body, indecision
        result[1] = 1.0 if body_ratio < 0.1 else 0.0

        # [2] is_inside_bar — current bar entirely within previous bar
        if idx > 0:
            result[2] = 1.0 if (hi <= h[idx - 1] and lo >= l[idx - 1]) else 0.0

        # [3] is_engulfing_bull — bullish candle engulfs previous bearish candle
        if idx > 0:
            prev_body = abs(c[idx - 1] - o[idx - 1])
            result[3] = 1.0 if (
                is_bullish
                and body > prev_body
                and co > o[idx - 1]
                and op < c[idx - 1]
                and c[idx - 1] < o[idx - 1]  # previous was bearish
            ) else 0.0

        # [4] is_engulfing_bear — bearish candle engulfs previous bullish candle
        if idx > 0:
            result[4] = 1.0 if (
                not is_bullish
                and body > abs(c[idx - 1] - o[idx - 1])
                and co < o[idx - 1]
                and op > c[idx - 1]
                and c[idx - 1] > o[idx - 1]  # previous was bullish
            ) else 0.0

        # [5] is_hammer — pinbar at a swing low (bullish reversal)
        is_at_low = False
        if idx >= self.swing_lookback:
            recent_low = np.min(l[idx - self.swing_lookback:idx])
            is_at_low = lo <= recent_low * 1.001  # within 0.1%
        result[5] = 1.0 if (result[0] > 0 and is_at_low and is_bullish) else 0.0

        # [6] is_shooting_star — upper wick long at swing high (bearish reversal)
        is_at_high = False
        if idx >= self.swing_lookback:
            recent_high = np.max(h[idx - self.swing_lookback:idx])
            is_at_high = hi >= recent_high * 0.999
        result[6] = 1.0 if (
            upper_wick_ratio > 0.6 and body_ratio < 0.3
            and is_at_high and not is_bullish
        ) else 0.0

        # [7] candle_strength — body dominance × relative volume [0, 1]
        vol_ratio = 1.0
        if idx >= self.vol_sma_period:
            vol_mean = np.mean(v[idx - self.vol_sma_period:idx]) + 1e-10
            vol_ratio = min(v[idx] / vol_mean, 3.0) / 3.0
        result[7] = float(np.clip(body_ratio * vol_ratio, 0.0, 1.0))

        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # GROUP C: Volume Variables (5)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _extract_volume(
        self,
        o: np.ndarray, h: np.ndarray, l: np.ndarray,
        c: np.ndarray, v: np.ndarray, idx: int,
    ) -> np.ndarray:
        """Extract 5 Volume variables."""
        result = np.zeros(N_VOL_VARS, dtype=np.float32)

        if idx < self.vol_sma_period:
            return result

        vol_window = v[idx - self.vol_sma_period:idx]
        vol_mean = np.mean(vol_window) + 1e-10
        current_vol = v[idx]
        vol_ratio = current_vol / vol_mean

        body = abs(c[idx] - o[idx])
        full_range = h[idx] - l[idx] + 1e-10
        wick_ratio = 1.0 - (body / full_range)

        # [0] vol_anomaly — spike detection, normalized [0, 1]
        #     0.0 = normal volume, 1.0 = extreme spike (>= threshold)
        result[0] = float(np.clip(
            (vol_ratio - 1.0) / (self.vol_anomaly_threshold - 1.0), 0.0, 1.0,
        ))

        # [1] vol_exhaustion — low volume + big wick = supply/demand exhaustion
        #     High value = potential reversal signal
        is_low_vol = vol_ratio < self.vol_exhaustion_threshold
        result[1] = 1.0 if (is_low_vol and wick_ratio > 0.6) else 0.0

        # [2] vol_climax — extreme volume spike (climactic move, often ends)
        result[2] = 1.0 if vol_ratio >= self.vol_climax_threshold else 0.0

        # [3] vol_trend — short-term vs long-term volume ratio
        #     > 0 = volume increasing, < 0 = volume decreasing
        if idx >= self.vol_sma_period:
            vol_short = np.mean(v[max(0, idx - 5):idx + 1]) + 1e-10
            vol_long = vol_mean
            result[3] = float(np.clip((vol_short / vol_long) - 1.0, -1.0, 1.0))

        # [4] delta_approx — approximate buy/sell pressure from candle position
        #     Close near high → buyers dominated, close near low → sellers dominated
        result[4] = float((c[idx] - l[idx]) / full_range - 0.5) * 2.0  # [-1, 1]

        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # GROUP D: Context Variables (2)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _extract_context(bar_index: int) -> np.ndarray:
        """
        Extract session time context (2 variables).

        Uses cyclical encoding so 23:00 and 01:00 are "close" in feature space.
        Since we don't have actual timestamps in this method, we approximate
        hour from bar_index assuming M5 bars (12 bars per hour).

        For real-time use, the caller should provide actual hour.
        """
        result = np.zeros(N_CTX_VARS, dtype=np.float32)

        # Approximate hour (M5: 12 bars/hour, M1: 60 bars/hour)
        hour_approx = (bar_index // 12) % 24

        result[0] = float(math.sin(2 * math.pi * hour_approx / 24))
        result[1] = float(math.cos(2 * math.pi * hour_approx / 24))

        return result

    def extract_context_with_hour(self, hour: int) -> np.ndarray:
        """Extract context variables with known hour (for real-time use)."""
        result = np.zeros(N_CTX_VARS, dtype=np.float32)
        result[0] = float(math.sin(2 * math.pi * hour / 24))
        result[1] = float(math.cos(2 * math.pi * hour / 24))
        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # UTILITY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def compute_atr(
        h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14,
    ) -> np.ndarray:
        """
        Compute ATR array for entire series. Vectorized.

        Args:
            h, l, c: High, Low, Close arrays (N,)
            period: ATR lookback period

        Returns:
            (N,) ATR array (first `period` values are approximate)
        """
        n = len(h)
        tr = np.zeros(n, dtype=np.float32)

        # First bar: simple range
        tr[0] = h[0] - l[0]

        # Subsequent bars: True Range
        tr[1:] = np.maximum(
            h[1:] - l[1:],
            np.maximum(
                np.abs(h[1:] - c[:-1]),
                np.abs(l[1:] - c[:-1]),
            ),
        )

        # SMA of True Range
        atr = np.convolve(tr, np.ones(period) / period, mode="same")

        # Fix edges (convolve "same" mode causes boundary artifacts)
        for i in range(min(period, n)):
            if i > 0:
                atr[i] = np.mean(tr[:i + 1])

        return atr.astype(np.float32)

    @staticmethod
    def get_feature_names() -> list[str]:
        """Return ordered list of 22 feature names."""
        return FEATURE_NAMES.copy()
