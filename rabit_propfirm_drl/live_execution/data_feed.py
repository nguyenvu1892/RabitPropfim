"""
DataFeedManager — Real-time multi-timeframe OHLCV polling from MT5.

Responsibility:
    - Poll MT5 every POLL_INTERVAL_SEC for new M5 candles
    - Maintain rolling buffers of M5, H1, H4 OHLCV data
    - Detect new bar close → trigger signal for inference
    - Validate data integrity (gaps, NaN, stale data)

Design Decisions:
    - POLLING (not callback): MT5 Python API has no event-driven tick API.
      We poll every 5s which is fast enough for M5 bars (300s each).
    - BUFFER SIZES: Sized to match model input requirements from training.
      M5=96 (8h), H1=48 (2d), H4=24 (4d).
    - STALE DATA PROTECTION: If the latest bar timestamp is older than
      expected, we flag data as stale and refuse to trigger inference.

Thread Safety:
    This class is NOT thread-safe. Use from the main trading loop only.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Graceful MT5 import
try:
    import MetaTrader5 as mt5  # type: ignore

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None  # type: ignore
    logger.warning("MetaTrader5 not installed. DataFeedManager disabled.")


# ─── Constants ─────────────────────────────────────────────────────
# OHLCV column indices (from mt5.copy_rates_from_pos structured array)
COL_TIME = 0
COL_OPEN = 1
COL_HIGH = 2
COL_LOW = 3
COL_CLOSE = 4
COL_TICK_VOLUME = 5
COL_SPREAD = 6
COL_REAL_VOLUME = 7

# Number of standard OHLCV columns we extract
N_OHLCV_COLS = 6  # time, open, high, low, close, tick_volume


# ─── Data Types ────────────────────────────────────────────────────
@dataclass
class BarData:
    """Container for multi-timeframe OHLCV data."""

    m5: np.ndarray       # (N_m5, 6) — [time, O, H, L, C, volume]
    h1: np.ndarray       # (N_h1, 6)
    h4: np.ndarray       # (N_h4, 6)
    timestamp: datetime   # Timestamp of latest M5 bar
    symbol: str


@dataclass
class FeedHealth:
    """Health status of the data feed."""

    is_healthy: bool
    last_bar_time: Optional[datetime] = None
    bars_received_m5: int = 0
    bars_received_h1: int = 0
    bars_received_h4: int = 0
    staleness_seconds: float = 0.0
    error: str = ""


class DataFeedError(Exception):
    """Raised when data feed encounters an unrecoverable error."""


# ─── DataFeedManager ──────────────────────────────────────────────
class DataFeedManager:
    """
    Manages real-time multi-timeframe OHLCV data from MT5.

    Usage:
        feed = DataFeedManager("XAUUSD")
        has_new, data = feed.poll()
        if has_new and data is not None:
            # Process data.m5, data.h1, data.h4
            ...

    Poll Interval:
        Call poll() every POLL_INTERVAL_SEC (5s default).
        Returns (True, BarData) only when a NEW M5 bar has closed.
        Otherwise returns (False, None).
    """

    # Class-level constants (match design doc)
    POLL_INTERVAL_SEC: float = 5.0
    M5_BARS_NEEDED: int = 96     # 96 × 5min = 8 hours lookback
    H1_BARS_NEEDED: int = 48     # 48 × 1h   = 2 days lookback
    H4_BARS_NEEDED: int = 24     # 24 × 4h   = 4 days lookback

    # Stale data threshold: if latest bar is older than this, refuse inference
    STALENESS_THRESHOLD_SEC: float = 600.0  # 10 minutes

    # MT5 timeframe mappings
    _TF_MAP = None  # Lazily initialized (MT5 may not be importable)

    def __init__(self, symbol: str) -> None:
        """
        Args:
            symbol: MT5 symbol name (e.g., "XAUUSD", "US100.cash")
        """
        if not MT5_AVAILABLE:
            raise DataFeedError(
                "MetaTrader5 package not installed. Cannot create DataFeedManager."
            )

        self.symbol = symbol

        # State tracking
        self._last_m5_time: Optional[datetime] = None
        self._poll_count: int = 0
        self._error_count: int = 0
        self._consecutive_errors: int = 0

        # OHLCV caches (preserved between polls for fast access)
        self._m5_cache: Optional[np.ndarray] = None
        self._h1_cache: Optional[np.ndarray] = None
        self._h4_cache: Optional[np.ndarray] = None

        logger.info(
            "DataFeedManager initialized: symbol=%s, "
            "buffers=[M5×%d, H1×%d, H4×%d], poll_interval=%.1fs",
            self.symbol,
            self.M5_BARS_NEEDED,
            self.H1_BARS_NEEDED,
            self.H4_BARS_NEEDED,
            self.POLL_INTERVAL_SEC,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PUBLIC API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def poll(self) -> tuple[bool, Optional[BarData]]:
        """
        Poll MT5 for new M5 bar data.

        Returns:
            (has_new_bar, BarData | None)
            - (True, BarData) when a new M5 candle has closed
            - (False, None) when no new bar detected or error occurred

        This method is designed to be called in a tight loop.
        It handles all internal errors gracefully (logs + returns False).
        """
        self._poll_count += 1

        try:
            # ── Ensure symbol is selected in MT5 ──
            if not self._ensure_symbol_selected():
                return False, None

            # ── Fetch M5 bars ──
            rates_m5 = self._fetch_rates(mt5.TIMEFRAME_M5, self.M5_BARS_NEEDED)
            if rates_m5 is None:
                return False, None

            # ── Check if we have a NEW M5 bar ──
            latest_time = self._extract_bar_time(rates_m5[-1])
            if not self._is_new_bar(latest_time):
                return False, None

            # ── NEW BAR DETECTED — refresh all timeframes ──
            self._last_m5_time = latest_time
            self._m5_cache = self._structured_to_ohlcv(rates_m5)

            # Fetch H1 and H4
            rates_h1 = self._fetch_rates(mt5.TIMEFRAME_H1, self.H1_BARS_NEEDED)
            if rates_h1 is not None:
                self._h1_cache = self._structured_to_ohlcv(rates_h1)
            else:
                logger.warning(
                    "[%s] H1 data unavailable — using cached data", self.symbol
                )

            rates_h4 = self._fetch_rates(mt5.TIMEFRAME_H4, self.H4_BARS_NEEDED)
            if rates_h4 is not None:
                self._h4_cache = self._structured_to_ohlcv(rates_h4)
            else:
                logger.warning(
                    "[%s] H4 data unavailable — using cached data", self.symbol
                )

            # ── Validate caches ──
            if not self._validate_caches():
                return False, None

            # ── Staleness check ──
            staleness = self._compute_staleness(latest_time)
            if staleness > self.STALENESS_THRESHOLD_SEC:
                logger.warning(
                    "[%s] Data is stale (%.0fs old). "
                    "Market may be closed — skipping inference.",
                    self.symbol,
                    staleness,
                )
                return False, None

            # ── Build result ──
            bar_data = BarData(
                m5=self._m5_cache.copy(),
                h1=self._h1_cache.copy(),
                h4=self._h4_cache.copy(),
                timestamp=latest_time,
                symbol=self.symbol,
            )

            self._consecutive_errors = 0
            logger.debug(
                "[%s] New M5 bar @ %s | M5=%d H1=%d H4=%d",
                self.symbol,
                latest_time.strftime("%H:%M:%S"),
                len(self._m5_cache),
                len(self._h1_cache),
                len(self._h4_cache),
            )
            return True, bar_data

        except Exception as e:
            self._error_count += 1
            self._consecutive_errors += 1
            logger.error(
                "[%s] DataFeed poll error (#%d, consecutive=%d): %s",
                self.symbol,
                self._error_count,
                self._consecutive_errors,
                e,
                exc_info=self._consecutive_errors <= 3,  # Full trace for first 3
            )
            return False, None

    def get_health(self) -> FeedHealth:
        """Get current health status of the data feed."""
        staleness = 0.0
        if self._last_m5_time is not None:
            staleness = self._compute_staleness(self._last_m5_time)

        is_healthy = (
            self._consecutive_errors < 5
            and staleness < self.STALENESS_THRESHOLD_SEC
            and self._m5_cache is not None
        )

        return FeedHealth(
            is_healthy=is_healthy,
            last_bar_time=self._last_m5_time,
            bars_received_m5=len(self._m5_cache) if self._m5_cache is not None else 0,
            bars_received_h1=len(self._h1_cache) if self._h1_cache is not None else 0,
            bars_received_h4=len(self._h4_cache) if self._h4_cache is not None else 0,
            staleness_seconds=staleness,
            error="" if is_healthy else f"errors={self._consecutive_errors}",
        )

    def reset(self) -> None:
        """Reset all internal state and caches. Use when switching symbol."""
        self._last_m5_time = None
        self._m5_cache = None
        self._h1_cache = None
        self._h4_cache = None
        self._consecutive_errors = 0
        logger.info("[%s] DataFeedManager reset", self.symbol)

    @property
    def last_bar_time(self) -> Optional[datetime]:
        """Timestamp of the last detected M5 bar."""
        return self._last_m5_time

    @property
    def poll_count(self) -> int:
        """Total number of poll() calls made."""
        return self._poll_count

    @property
    def error_count(self) -> int:
        """Total number of errors encountered."""
        return self._error_count

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PRIVATE — MT5 Data Fetching
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _ensure_symbol_selected(self) -> bool:
        """Ensure the symbol is selected in MT5 MarketWatch."""
        try:
            info = mt5.symbol_info(self.symbol)
            if info is None:
                logger.error("[%s] Symbol not found in MT5", self.symbol)
                return False

            if not info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    logger.error(
                        "[%s] Failed to select symbol in MarketWatch", self.symbol
                    )
                    return False

            return True
        except Exception as e:
            logger.error("[%s] symbol_info error: %s", self.symbol, e)
            return False

    def _fetch_rates(
        self, timeframe: int, count: int
    ) -> Optional[np.ndarray]:
        """
        Fetch OHLCV rates from MT5.

        Args:
            timeframe: MT5 timeframe constant (mt5.TIMEFRAME_M5, etc.)
            count: Number of bars to fetch

        Returns:
            Structured numpy array from MT5, or None on error.
        """
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)

            if rates is None:
                error_code = mt5.last_error()
                logger.warning(
                    "[%s] copy_rates_from_pos returned None "
                    "(tf=%d, count=%d, error=%s)",
                    self.symbol,
                    timeframe,
                    count,
                    error_code,
                )
                return None

            if len(rates) == 0:
                logger.warning(
                    "[%s] copy_rates_from_pos returned 0 bars (tf=%d)",
                    self.symbol,
                    timeframe,
                )
                return None

            # Sanity: warn if we got fewer bars than requested
            if len(rates) < count:
                logger.debug(
                    "[%s] Got %d/%d bars (tf=%d) — may be normal for new symbol",
                    self.symbol,
                    len(rates),
                    count,
                    timeframe,
                )

            return rates

        except Exception as e:
            logger.error(
                "[%s] _fetch_rates error (tf=%d): %s", self.symbol, timeframe, e
            )
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PRIVATE — Data Transformation & Validation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _structured_to_ohlcv(rates: np.ndarray) -> np.ndarray:
        """
        Convert MT5 structured array to plain OHLCV numpy array.

        MT5 returns: dtype=[('time','<i8'), ('open','<f8'), ('high','<f8'),
                            ('low','<f8'), ('close','<f8'), ('tick_volume','<i8'),
                            ('spread','<i4'), ('real_volume','<i8')]

        We extract: [time, open, high, low, close, tick_volume] → (N, 6)
        """
        ohlcv = np.column_stack([
            rates['time'].astype(np.float64),
            rates['open'].astype(np.float64),
            rates['high'].astype(np.float64),
            rates['low'].astype(np.float64),
            rates['close'].astype(np.float64),
            rates['tick_volume'].astype(np.float64),
        ])
        return ohlcv

    @staticmethod
    def _extract_bar_time(rate_entry) -> datetime:
        """
        Extract datetime from a single MT5 rate entry.

        MT5 returns time as Unix timestamp (int64).
        """
        timestamp = int(rate_entry['time'])
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    def _is_new_bar(self, latest_time: datetime) -> bool:
        """Check if this bar timestamp is newer than the last seen one."""
        if self._last_m5_time is None:
            return True  # First poll — always treat as new
        return latest_time > self._last_m5_time

    def _validate_caches(self) -> bool:
        """
        Validate that all OHLCV caches contain valid data.

        Checks:
        - All caches are non-None
        - Minimum bar counts met
        - No NaN values in price columns
        - Prices are positive
        """
        if self._m5_cache is None:
            logger.error("[%s] M5 cache is None after fetch", self.symbol)
            return False

        if self._h1_cache is None:
            logger.error("[%s] H1 cache is None", self.symbol)
            return False

        if self._h4_cache is None:
            logger.error("[%s] H4 cache is None", self.symbol)
            return False

        # Minimum bar counts
        min_m5 = 20  # Need at least 20 bars for ATR and features
        min_h1 = 10
        min_h4 = 5

        if len(self._m5_cache) < min_m5:
            logger.error(
                "[%s] M5 bars too few: %d < %d",
                self.symbol, len(self._m5_cache), min_m5,
            )
            return False

        if len(self._h1_cache) < min_h1:
            logger.error(
                "[%s] H1 bars too few: %d < %d",
                self.symbol, len(self._h1_cache), min_h1,
            )
            return False

        if len(self._h4_cache) < min_h4:
            logger.error(
                "[%s] H4 bars too few: %d < %d",
                self.symbol, len(self._h4_cache), min_h4,
            )
            return False

        # NaN check on price columns (indices 1-4: O, H, L, C)
        for name, cache in [
            ("M5", self._m5_cache),
            ("H1", self._h1_cache),
            ("H4", self._h4_cache),
        ]:
            prices = cache[:, 1:5]  # Open, High, Low, Close
            if np.any(np.isnan(prices)):
                logger.error("[%s] NaN found in %s price data!", self.symbol, name)
                return False
            if np.any(prices <= 0):
                logger.error(
                    "[%s] Non-positive prices in %s data!", self.symbol, name
                )
                return False

        return True

    @staticmethod
    def _compute_staleness(bar_time: datetime) -> float:
        """Compute how many seconds old a bar is relative to now."""
        now = datetime.now(timezone.utc)
        delta = (now - bar_time).total_seconds()
        return max(0.0, delta)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # REPR
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def __repr__(self) -> str:
        health = self.get_health()
        return (
            f"DataFeedManager(symbol={self.symbol!r}, "
            f"healthy={health.is_healthy}, "
            f"polls={self._poll_count}, "
            f"errors={self._error_count}, "
            f"last_bar={self._last_m5_time})"
        )
