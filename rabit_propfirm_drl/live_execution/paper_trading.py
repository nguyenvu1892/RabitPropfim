"""
Paper Trading Orchestrator — Manages the full paper trading validation loop.

Orchestrates:
1. MT5 Connector (data + orders)
2. Feature Pipeline (build + normalize)
3. Inference Pipeline (model → decisions)
4. Killswitch & Watchdog (safety)
5. Alert Bot (Telegram notifications)
6. Session logging & performance tracking

Runs on M15 bar intervals, checking for new bars and executing decisions.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradingSession:
    """Tracks metrics for a single trading session (day)."""
    date: str
    start_balance: float
    end_balance: float = 0.0
    peak_equity: float = 0.0
    max_daily_dd: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    inferences: int = 0
    decisions: dict[str, int] = field(default_factory=dict)
    killswitch_events: int = 0

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def daily_return_pct(self) -> float:
        if self.start_balance == 0:
            return 0.0
        return (self.end_balance - self.start_balance) / self.start_balance * 100


@dataclass
class PaperTradingReport:
    """Full paper trading validation report."""
    start_date: str
    end_date: str
    total_days: int
    sessions: list[TradingSession]
    initial_balance: float
    final_balance: float
    total_return_pct: float
    max_daily_dd: float
    max_total_dd: float
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    prop_firm_pass: bool
    failure_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_days": self.total_days,
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "total_return_pct": round(self.total_return_pct, 2),
            "max_daily_dd": round(self.max_daily_dd, 4),
            "max_total_dd": round(self.max_total_dd, 4),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 3),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "prop_firm_pass": self.prop_firm_pass,
            "failure_reason": self.failure_reason,
            "sessions": [asdict(s) for s in self.sessions],
        }

    def save(self, path: Path | str) -> None:
        """Save report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Paper trading report saved to %s", path)


class PaperTradingOrchestrator:
    """
    Manages the full paper trading validation process.

    Coordinates all components for a 5-day minimum validation run
    on a demo account before live deployment.
    """

    def __init__(
        self,
        config: dict,
        model_fn: Callable[[np.ndarray], np.ndarray],
        initial_balance: float = 10_000.0,
    ) -> None:
        """
        Args:
            config: Dict from prop_rules.yaml
            model_fn: Trained model inference function
            initial_balance: Starting balance for paper trading
        """
        self.config = config
        self.model_fn = model_fn
        self.initial_balance = initial_balance

        # Prop Firm rules
        self.max_daily_dd = config.get("max_daily_drawdown", 0.05)
        self.max_total_dd = config.get("max_total_drawdown", 0.10)
        self.profit_target = config.get("profit_target", 0.08)
        self.min_trading_days = config.get("min_trading_days", 5)

        # Session tracking
        self.sessions: list[TradingSession] = []
        self.current_session: TradingSession | None = None

        # Overall tracking
        self.peak_equity = initial_balance
        self.current_balance = initial_balance

        self._alert_callback: Optional[Callable[[str, str], None]] = None

    def set_alert_callback(
        self, callback: Callable[[str, str], None]
    ) -> None:
        self._alert_callback = callback

    def start_session(self, balance: float | None = None) -> TradingSession:
        """Start a new daily trading session."""
        if balance is None:
            balance = self.current_balance

        session = TradingSession(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            start_balance=balance,
            peak_equity=balance,
        )
        self.current_session = session
        logger.info("Paper trading session started: %s (balance=%.2f)",
                     session.date, balance)
        return session

    def end_session(self, final_balance: float) -> TradingSession:
        """End current trading session."""
        if self.current_session is None:
            raise RuntimeError("No active session to end")

        session = self.current_session
        session.end_balance = final_balance
        self.current_balance = final_balance
        self.peak_equity = max(self.peak_equity, final_balance)

        self.sessions.append(session)
        self.current_session = None

        logger.info(
            "Session ended: %s | PnL: %.2f | DD: %.2f%% | Trades: %d (WR: %.1f%%)",
            session.date, session.total_pnl, session.max_daily_dd * 100,
            session.total_trades, session.win_rate * 100,
        )

        self._send_alert(
            f"📊 Session {session.date}",
            f"PnL: ${session.total_pnl:.2f} | DD: {session.max_daily_dd:.2%} "
            f"| Trades: {session.total_trades} | WR: {session.win_rate:.0%}",
        )

        return session

    def record_trade(
        self, pnl: float, is_winner: bool
    ) -> None:
        """Record a completed trade in the current session."""
        if self.current_session:
            self.current_session.total_trades += 1
            self.current_session.total_pnl += pnl
            if is_winner:
                self.current_session.winning_trades += 1

    def update_equity(self, equity: float) -> None:
        """Update equity tracking for drawdown calculation."""
        if self.current_session:
            self.current_session.peak_equity = max(
                self.current_session.peak_equity, equity
            )
            if self.current_session.peak_equity > 0:
                dd = (self.current_session.peak_equity - equity) / self.current_session.peak_equity
                self.current_session.max_daily_dd = max(
                    self.current_session.max_daily_dd, dd
                )

        self.peak_equity = max(self.peak_equity, equity)

    def generate_report(self) -> PaperTradingReport:
        """
        Generate comprehensive paper trading report.

        Evaluates against Prop Firm rules to determine pass/fail.
        """
        if not self.sessions:
            return PaperTradingReport(
                start_date="", end_date="", total_days=0,
                sessions=[], initial_balance=self.initial_balance,
                final_balance=self.current_balance,
                total_return_pct=0, max_daily_dd=0, max_total_dd=0,
                total_trades=0, win_rate=0, sharpe_ratio=0,
                prop_firm_pass=False, failure_reason="No sessions completed",
            )

        # Calculate metrics
        total_trades = sum(s.total_trades for s in self.sessions)
        winning = sum(s.winning_trades for s in self.sessions)
        max_daily = max(s.max_daily_dd for s in self.sessions)
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance

        # Total DD
        max_total_dd = (self.peak_equity - self.current_balance) / max(self.peak_equity, 1)
        max_total_dd = max(0, max_total_dd)

        # Sharpe ratio from daily returns
        daily_returns = [s.daily_return_pct for s in self.sessions]
        if len(daily_returns) >= 2:
            mean_ret = float(np.mean(daily_returns))
            std_ret = float(np.std(daily_returns)) + 1e-8
            sharpe = mean_ret / std_ret * np.sqrt(252)  # Annualized
        else:
            sharpe = 0.0

        # Prop Firm pass/fail check
        prop_pass = True
        failure_reason = ""

        if len(self.sessions) < self.min_trading_days:
            prop_pass = False
            failure_reason = f"Not enough trading days: {len(self.sessions)} < {self.min_trading_days}"
        elif max_daily > self.max_daily_dd:
            prop_pass = False
            failure_reason = f"Daily DD exceeded: {max_daily:.2%} > {self.max_daily_dd:.2%}"
        elif max_total_dd > self.max_total_dd:
            prop_pass = False
            failure_reason = f"Total DD exceeded: {max_total_dd:.2%} > {self.max_total_dd:.2%}"
        elif total_return < self.profit_target:
            prop_pass = False
            failure_reason = f"Profit target not met: {total_return:.2%} < {self.profit_target:.2%}"

        return PaperTradingReport(
            start_date=self.sessions[0].date,
            end_date=self.sessions[-1].date,
            total_days=len(self.sessions),
            sessions=list(self.sessions),
            initial_balance=self.initial_balance,
            final_balance=self.current_balance,
            total_return_pct=total_return * 100,
            max_daily_dd=max_daily,
            max_total_dd=max_total_dd,
            total_trades=total_trades,
            win_rate=winning / max(total_trades, 1),
            sharpe_ratio=float(sharpe),
            prop_firm_pass=prop_pass,
            failure_reason=failure_reason,
        )

    def _send_alert(self, title: str, details: str) -> None:
        if self._alert_callback:
            try:
                self._alert_callback(title, details)
            except Exception as e:
                logger.error("Alert failed: %s", e)
