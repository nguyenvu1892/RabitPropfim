"""
ConnectionGuard — 3-tier MT5 connection protection with auto-reconnect.

Architecture:
    Tier 1: Fast Retry (3 attempts, 1s apart)
        → For transient errors (network hiccup, MT5 busy)

    Tier 2: Full Reconnect (shutdown + reinitialize, 5 attempts)
        → For connection drops (MT5 terminal restart, VPS issue)

    Tier 3: Exponential Backoff (2s → 4s → 8s → ... → max 300s)
        → For extended outages (sàn maintenance, server down)

    After all tiers fail → raise MT5ConnectionError
    → Engine must handle: close all positions + alert operator

Usage:
    guard = ConnectionGuard()

    # Wrap any MT5 operation:
    result = guard.execute_with_guard(mt5.account_info)

    # Or with arguments:
    result = guard.execute_with_guard(
        mt5.copy_rates_from_pos, "XAUUSD", mt5.TIMEFRAME_M5, 0, 96
    )

Thread Safety:
    NOT thread-safe. Use from main trading loop only.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Graceful MT5 import
try:
    import MetaTrader5 as mt5  # type: ignore

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None  # type: ignore


# ─── Custom Exceptions ────────────────────────────────────────────
class MT5ConnectionError(Exception):
    """
    Raised when MT5 connection cannot be restored after all retry tiers.

    The caller (MT5ExecutionEngine) must:
    1. Close all open positions (if possible)
    2. Send emergency alert (Telegram)
    3. Stop the trading loop
    """


class MT5OperationError(Exception):
    """
    Raised when an MT5 operation fails but connection is still alive.

    This is less severe than MT5ConnectionError — the engine can
    continue operating, but the specific operation should be retried
    or skipped.
    """


# ─── Connection Health Tracker ────────────────────────────────────
class ConnectionHealth:
    """
    Tracks connection health metrics for monitoring.

    Provides running stats on connection quality:
    - Uptime / downtime
    - Reconnect count
    - Error frequency
    """

    def __init__(self) -> None:
        self.total_operations: int = 0
        self.successful_operations: int = 0
        self.failed_operations: int = 0
        self.reconnect_count: int = 0
        self.last_error: str = ""
        self.last_error_time: Optional[str] = None
        self.last_success_time: Optional[str] = None
        self.longest_outage_sec: float = 0.0
        self._outage_start: Optional[float] = None

    def record_success(self) -> None:
        """Record a successful MT5 operation."""
        self.total_operations += 1
        self.successful_operations += 1
        self.last_success_time = datetime.now(timezone.utc).isoformat()

        # End outage tracking
        if self._outage_start is not None:
            outage_duration = time.time() - self._outage_start
            self.longest_outage_sec = max(self.longest_outage_sec, outage_duration)
            self._outage_start = None

    def record_failure(self, error: str) -> None:
        """Record a failed MT5 operation."""
        self.total_operations += 1
        self.failed_operations += 1
        self.last_error = error
        self.last_error_time = datetime.now(timezone.utc).isoformat()

        # Start outage tracking
        if self._outage_start is None:
            self._outage_start = time.time()

    def record_reconnect(self) -> None:
        """Record a successful reconnection."""
        self.reconnect_count += 1

    @property
    def success_rate(self) -> float:
        """Operation success rate (0.0 – 1.0)."""
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations

    def to_dict(self) -> dict[str, Any]:
        """Serialize health metrics to dict."""
        return {
            "total_operations": self.total_operations,
            "success_rate": round(self.success_rate, 4),
            "reconnect_count": self.reconnect_count,
            "failed_operations": self.failed_operations,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time,
            "longest_outage_sec": round(self.longest_outage_sec, 1),
        }


# ─── ConnectionGuard ──────────────────────────────────────────────
class ConnectionGuard:
    """
    3-tier MT5 connection protection with auto-reconnect.

    Wraps any MT5 API call with retry logic:
        Tier 1: Fast retry (transient errors)
        Tier 2: Full reconnect (connection drops)
        Tier 3: Exponential backoff (extended outages)

    Args:
        max_fast_retries:  Number of quick retries (Tier 1). Default: 3
        max_reconnects:    Number of reconnect attempts (Tier 2). Default: 5
        fast_retry_delay:  Seconds between fast retries. Default: 1.0
        backoff_base:      Base seconds for exponential backoff. Default: 2.0
        backoff_max:       Maximum backoff seconds. Default: 300.0 (5 min)
        alert_callback:    Optional function(title, msg) for alerts
    """

    def __init__(
        self,
        max_fast_retries: int = 3,
        max_reconnects: int = 5,
        fast_retry_delay: float = 1.0,
        backoff_base: float = 2.0,
        backoff_max: float = 300.0,
        alert_callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        if not MT5_AVAILABLE:
            raise MT5ConnectionError(
                "MetaTrader5 package not installed. Cannot create ConnectionGuard."
            )

        self.max_fast_retries = max_fast_retries
        self.max_reconnects = max_reconnects
        self.fast_retry_delay = fast_retry_delay
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._alert_callback = alert_callback

        # State
        self._is_connected: bool = False
        self.health = ConnectionHealth()

        logger.info(
            "ConnectionGuard initialized: fast_retries=%d, "
            "reconnects=%d, backoff=[%.1fs → %.1fs]",
            max_fast_retries,
            max_reconnects,
            backoff_base,
            backoff_max,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PUBLIC API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def execute_with_guard(
        self, operation: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """
        Execute an MT5 operation with full 3-tier retry protection.

        Args:
            operation: Any callable MT5 function (e.g., mt5.account_info)
            *args:     Positional arguments for the operation
            **kwargs:  Keyword arguments for the operation

        Returns:
            The result of the operation call.

        Raises:
            MT5ConnectionError: After ALL retry tiers are exhausted.
                The caller must handle this as a critical failure.
        """
        # ── Tier 1: Fast Retry ──
        last_error = None
        for attempt in range(self.max_fast_retries):
            try:
                result = operation(*args, **kwargs)

                # MT5 functions return None on some errors without raising
                if result is None:
                    mt5_error = mt5.last_error()
                    if mt5_error and mt5_error[0] != 1:  # 1 = RES_S_OK
                        raise MT5OperationError(
                            f"MT5 returned None: error={mt5_error}"
                        )

                self.health.record_success()
                self._is_connected = True
                return result

            except MT5OperationError as e:
                last_error = e
                logger.warning(
                    "Tier 1 — fast retry %d/%d failed: %s",
                    attempt + 1,
                    self.max_fast_retries,
                    e,
                )
                time.sleep(self.fast_retry_delay)

            except Exception as e:
                last_error = e
                logger.warning(
                    "Tier 1 — fast retry %d/%d exception: %s",
                    attempt + 1,
                    self.max_fast_retries,
                    e,
                )
                time.sleep(self.fast_retry_delay)

        # ── Tier 2 + 3: Reconnect with Backoff ──
        logger.error(
            "Tier 1 exhausted. Escalating to Tier 2 (reconnect). "
            "Last error: %s",
            last_error,
        )
        self._send_alert(
            "⚠️ MT5 CONNECTION ISSUE",
            f"Fast retries exhausted. Attempting reconnect...\n"
            f"Error: {last_error}",
        )

        for reconnect_attempt in range(self.max_reconnects):
            try:
                # Fully shutdown and reinitialize
                self._reconnect()

                # Try the operation again
                result = operation(*args, **kwargs)
                if result is not None:
                    self.health.record_success()
                    self.health.record_reconnect()
                    logger.info(
                        "Reconnect successful on attempt %d/%d",
                        reconnect_attempt + 1,
                        self.max_reconnects,
                    )
                    self._send_alert(
                        "✅ MT5 RECONNECTED",
                        f"Connection restored after {reconnect_attempt + 1} "
                        f"reconnect attempt(s).",
                    )
                    return result

                # Result is None — treat as failure
                raise MT5OperationError("Operation returned None after reconnect")

            except Exception as e:
                last_error = e
                self.health.record_failure(str(e))

                # Tier 3: Exponential backoff
                wait_time = min(
                    self.backoff_base * (2 ** reconnect_attempt),
                    self.backoff_max,
                )
                logger.error(
                    "Tier 2/3 — reconnect %d/%d failed: %s. "
                    "Backoff: %.0fs",
                    reconnect_attempt + 1,
                    self.max_reconnects,
                    e,
                    wait_time,
                )
                time.sleep(wait_time)

        # ── ALL TIERS EXHAUSTED ──
        self._is_connected = False
        error_msg = (
            f"MT5 unreachable after {self.max_fast_retries} fast retries + "
            f"{self.max_reconnects} reconnect attempts. "
            f"Last error: {last_error}"
        )
        self.health.record_failure(error_msg)

        self._send_alert(
            "🚨 MT5 UNRECOVERABLE",
            f"All reconnect attempts failed!\n{error_msg}",
        )

        raise MT5ConnectionError(error_msg)

    def ensure_connected(self) -> bool:
        """
        Verify MT5 connection is alive. Attempt reconnect if not.

        Returns:
            True if connected (possibly after reconnect)

        Raises:
            MT5ConnectionError: If connection cannot be restored
        """
        try:
            info = mt5.terminal_info()
            if info is not None:
                self._is_connected = True
                return True
        except Exception:
            pass

        # Connection appears dead — try to restore
        logger.warning("MT5 connection check failed. Attempting restore...")
        self._is_connected = False

        try:
            self._reconnect()
            # Verify
            info = mt5.terminal_info()
            if info is not None:
                self._is_connected = True
                self.health.record_reconnect()
                logger.info("MT5 connection restored via ensure_connected()")
                return True
        except Exception as e:
            logger.error("ensure_connected() reconnect failed: %s", e)

        raise MT5ConnectionError(
            "MT5 connection dead and restore failed in ensure_connected()"
        )

    @property
    def is_connected(self) -> bool:
        """Check if MT5 is currently connected (cached state)."""
        return self._is_connected

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PRIVATE — Reconnection Logic
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _reconnect(self) -> None:
        """
        Perform a full MT5 shutdown + reinitialize cycle.

        This is the nuclear option — used when fast retries fail.
        MT5 terminal may need a moment between shutdown/init.
        """
        logger.info("Performing MT5 shutdown → reinitialize cycle")

        try:
            mt5.shutdown()
        except Exception as e:
            logger.debug("mt5.shutdown() during reconnect: %s", e)

        # Brief pause to let MT5 terminal release resources
        time.sleep(2.0)

        if not mt5.initialize():
            mt5_error = mt5.last_error()
            raise MT5ConnectionError(
                f"mt5.initialize() failed during reconnect: {mt5_error}"
            )

        self._is_connected = True
        logger.info("MT5 reinitialized successfully")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PRIVATE — Alerts
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _send_alert(self, title: str, message: str) -> None:
        """Send alert via configured callback (Telegram/Discord)."""
        if self._alert_callback is not None:
            try:
                self._alert_callback(title, message)
            except Exception as e:
                logger.error("ConnectionGuard alert send failed: %s", e)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # REPR
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def __repr__(self) -> str:
        return (
            f"ConnectionGuard("
            f"connected={self._is_connected}, "
            f"success_rate={self.health.success_rate:.1%}, "
            f"reconnects={self.health.reconnect_count})"
        )
