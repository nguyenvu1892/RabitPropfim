"""
Killswitch & Risk Watchdog — Triple-layer protection system.

Layer 1: Killswitch DD threshold (from config)
Layer 2: Forced SL on all positions (environment handles this)
Layer 3: Watchdog monitors equity in real-time, alerts via Telegram

All thresholds from prop_rules.yaml.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class SafetyEvent:
    """Record of a safety event."""
    timestamp: str
    event_type: str       # "killswitch" | "dd_warning" | "watchdog_alert"
    details: str
    current_dd: float
    equity: float


class Killswitch:
    """
    Triple-layer risk protection.

    Layer 1: Soft killswitch — reduces position size, alerts operator
    Layer 2: Hard killswitch — closes all positions, blocks new trades
    Layer 3: Emergency shutdown — disconnects and alerts via all channels
    """

    def __init__(self, config: dict) -> None:
        self.soft_threshold = config.get("killswitch_dd_threshold", 0.045)
        self.hard_threshold = config.get("max_daily_drawdown", 0.05)
        self.emergency_threshold = config.get("max_total_drawdown", 0.10)

        self.is_soft_triggered = False
        self.is_hard_triggered = False
        self.is_emergency = False

        self.events: list[SafetyEvent] = []
        self._alert_callback: Optional[Callable[[str, str], None]] = None

    def set_alert_callback(
        self, callback: Callable[[str, str], None]
    ) -> None:
        """Set callback for sending alerts (e.g. Telegram bot)."""
        self._alert_callback = callback

    def check(
        self,
        daily_dd: float,
        total_dd: float,
        equity: float,
    ) -> str:
        """
        Check all 3 layers of protection.

        Args:
            daily_dd: Current daily drawdown (decimal)
            total_dd: Current total drawdown (decimal)
            equity: Current account equity

        Returns:
            "normal" | "soft" | "hard" | "emergency"
        """
        now = datetime.now(timezone.utc).isoformat()

        # Layer 3: Emergency
        if total_dd >= self.emergency_threshold:
            if not self.is_emergency:
                self.is_emergency = True
                event = SafetyEvent(
                    timestamp=now,
                    event_type="emergency_shutdown",
                    details=f"Total DD {total_dd:.2%} >= {self.emergency_threshold:.2%}",
                    current_dd=total_dd,
                    equity=equity,
                )
                self.events.append(event)
                self._send_alert("🚨 EMERGENCY SHUTDOWN", event.details)
                logger.critical("EMERGENCY: %s", event.details)
            return "emergency"

        # Layer 2: Hard killswitch
        if daily_dd >= self.hard_threshold:
            if not self.is_hard_triggered:
                self.is_hard_triggered = True
                event = SafetyEvent(
                    timestamp=now,
                    event_type="hard_killswitch",
                    details=f"Daily DD {daily_dd:.2%} >= {self.hard_threshold:.2%}",
                    current_dd=daily_dd,
                    equity=equity,
                )
                self.events.append(event)
                self._send_alert("🛑 HARD KILLSWITCH", event.details)
                logger.error("HARD KILLSWITCH: %s", event.details)
            return "hard"

        # Layer 1: Soft killswitch
        if daily_dd >= self.soft_threshold:
            if not self.is_soft_triggered:
                self.is_soft_triggered = True
                event = SafetyEvent(
                    timestamp=now,
                    event_type="soft_killswitch",
                    details=f"Daily DD {daily_dd:.2%} approaching limit. Reducing exposure.",
                    current_dd=daily_dd,
                    equity=equity,
                )
                self.events.append(event)
                self._send_alert("⚠️ DD WARNING", event.details)
                logger.warning("SOFT KILLSWITCH: %s", event.details)
            return "soft"

        return "normal"

    def reset_daily(self) -> None:
        """Reset daily killswitch state (new trading day)."""
        self.is_soft_triggered = False
        self.is_hard_triggered = False
        logger.info("Daily killswitch reset")

    def _send_alert(self, title: str, details: str) -> None:
        if self._alert_callback:
            try:
                self._alert_callback(title, details)
            except Exception as e:
                logger.error("Failed to send alert: %s", e)


class EquityWatchdog:
    """
    Continuous equity monitoring with configurable check interval.

    Runs as a background monitor, checking equity at regular intervals
    and triggering alerts when thresholds are breached.
    """

    def __init__(
        self,
        killswitch: Killswitch,
        check_interval_seconds: float = 5.0,
    ) -> None:
        self.killswitch = killswitch
        self.check_interval = check_interval_seconds
        self._last_check = 0.0
        self.equity_history: list[tuple[float, float]] = []  # (timestamp, equity)

    def tick(
        self,
        daily_dd: float,
        total_dd: float,
        equity: float,
    ) -> str:
        """
        Check equity on each tick.

        Returns the killswitch status.
        """
        now = time.time()
        self.equity_history.append((now, equity))

        # Trim history to last 1 hour
        cutoff = now - 3600
        self.equity_history = [
            (t, e) for t, e in self.equity_history if t > cutoff
        ]

        # Only run killswitch check at intervals
        if now - self._last_check >= self.check_interval:
            self._last_check = now
            return self.killswitch.check(daily_dd, total_dd, equity)

        return "normal"

    @property
    def equity_drawdown_curve(self) -> list[tuple[float, float]]:
        """Return equity curve (timestamps, equities) for plotting."""
        return list(self.equity_history)
