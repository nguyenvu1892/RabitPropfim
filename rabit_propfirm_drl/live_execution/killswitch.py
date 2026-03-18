"""
Killswitch & Risk Watchdog — Triple-layer protection system.

Layer 1: Killswitch DD threshold (from config)
Layer 2: Forced SL on all positions (environment handles this)
Layer 3: Watchdog monitors equity in real-time, alerts via Telegram

Additional:
- DailyLossGate: max 0.3% loss per trade, 3% daily loss cooldown
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class SafetyEvent:
    """Record of a safety event."""
    timestamp: str
    event_type: str       # "killswitch" | "dd_warning" | "watchdog_alert" | "cooldown"
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
                self._send_alert("EMERGENCY SHUTDOWN", event.details)
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
                self._send_alert("HARD KILLSWITCH", event.details)
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
                self._send_alert("DD WARNING", event.details)
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


class DailyLossGate:
    """
    Per-trade and daily loss limiter.

    Rules:
    - Max loss per single trade = 0.3% of balance
    - If cumulative daily loss >= 3% -> cooldown (stop trading until next day)
    - Auto-resets on new calendar day

    This is a SEPARATE layer from the killswitch (which protects Prop Firm
    hard limits). DailyLossGate protects internal risk management.
    """

    def __init__(self, config: dict) -> None:
        self.max_loss_per_trade_pct = config.get("max_loss_per_trade_pct", 0.003)
        self.daily_cooldown_pct = config.get("daily_loss_cooldown_pct", 0.03)

        self._today: date | None = None
        self._daily_loss: float = 0.0  # Cumulative daily loss as decimal
        self._daily_start_balance: float = 0.0
        self._is_cooled_down: bool = False
        self._alert_callback: Optional[Callable[[str, str], None]] = None

        self.trade_history: list[dict] = []

    def set_alert_callback(self, callback: Callable[[str, str], None]) -> None:
        self._alert_callback = callback

    @property
    def is_cooled_down(self) -> bool:
        """Check if daily cooldown is active (auto-resets on new day)."""
        self._check_day_rollover()
        return self._is_cooled_down

    def start_day(self, balance: float) -> None:
        """Mark the start of a new trading day."""
        self._today = date.today()
        self._daily_start_balance = balance
        self._daily_loss = 0.0
        self._is_cooled_down = False
        logger.info("DailyLossGate: day started, balance=%.2f", balance)

    def max_sl_amount(self, balance: float) -> float:
        """
        Calculate maximum allowable SL amount in currency.

        Returns the max dollar amount a single trade can lose.
        E.g. balance=100000, max_loss_pct=0.003 -> $300
        """
        return balance * self.max_loss_per_trade_pct

    def can_trade(self) -> bool:
        """Check if trading is allowed (not cooled down)."""
        self._check_day_rollover()
        return not self._is_cooled_down

    def record_trade_result(self, pnl: float, balance: float) -> bool:
        """
        Record the PnL of a completed trade and check cooldown.

        Args:
            pnl: Trade profit/loss in currency (negative = loss)
            balance: Current account balance after trade

        Returns:
            True if still allowed to trade, False if cooldown triggered
        """
        self._check_day_rollover()

        if self._daily_start_balance == 0:
            self._daily_start_balance = balance - pnl

        # Only track losses
        if pnl < 0:
            loss_pct = abs(pnl) / self._daily_start_balance
            self._daily_loss += loss_pct

        self.trade_history.append({
            "pnl": pnl,
            "daily_loss_cumulative": self._daily_loss,
            "balance_after": balance,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Check cooldown (epsilon for float precision)
        if self._daily_loss >= self.daily_cooldown_pct - 1e-9:
            if not self._is_cooled_down:
                self._is_cooled_down = True
                msg = (
                    f"Daily loss {self._daily_loss:.2%} >= {self.daily_cooldown_pct:.2%}. "
                    f"Trading stopped until tomorrow."
                )
                logger.warning("DAILY COOLDOWN: %s", msg)
                if self._alert_callback:
                    self._alert_callback("DAILY LOSS COOLDOWN", msg)
            return False

        return True

    @property
    def daily_loss_pct(self) -> float:
        """Current cumulative daily loss as decimal."""
        self._check_day_rollover()
        return self._daily_loss

    @property
    def remaining_daily_risk(self) -> float:
        """How much more can be lost today before cooldown (as decimal)."""
        self._check_day_rollover()
        return max(0.0, self.daily_cooldown_pct - self._daily_loss)

    def _check_day_rollover(self) -> None:
        """Auto-reset if it's a new calendar day."""
        today = date.today()
        if self._today is not None and today > self._today:
            logger.info("DailyLossGate: new day detected, resetting cooldown")
            self._today = today
            self._daily_loss = 0.0
            self._is_cooled_down = False
