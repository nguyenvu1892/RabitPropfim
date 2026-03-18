"""
Reward Engine — Multi-component reward function for DRL trading agent.

8 Components:
1. realized_pnl         — Actual profit/loss when closing a trade
2. unrealized_shaping   — Mark-to-market shaping (delta per step)
3. dd_penalty           — Exponential drawdown penalty
4. overnight_penalty    — Penalty for holding past session end
5. spread_commission    — Execution cost deduction
6. rr_bonus             — Bonus for good risk/reward ratio
7. overtrading_penalty  — Penalty for excessive trades per day
8. inaction_nudge       — Gentle nudge if idle too long

All weights/thresholds from prop_rules.yaml — zero hardcoding.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components for logging/debugging."""
    realized_pnl: float = 0.0
    unrealized_shaping: float = 0.0
    dd_penalty: float = 0.0
    overnight_penalty: float = 0.0
    spread_commission: float = 0.0
    rr_bonus: float = 0.0
    overtrading_penalty: float = 0.0
    inaction_nudge: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.realized_pnl
            + self.unrealized_shaping
            + self.dd_penalty
            + self.overnight_penalty
            + self.spread_commission
            + self.rr_bonus
            + self.overtrading_penalty
            + self.inaction_nudge
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "realized_pnl": self.realized_pnl,
            "unrealized_shaping": self.unrealized_shaping,
            "dd_penalty": self.dd_penalty,
            "overnight_penalty": self.overnight_penalty,
            "spread_commission": self.spread_commission,
            "rr_bonus": self.rr_bonus,
            "overtrading_penalty": self.overtrading_penalty,
            "inaction_nudge": self.inaction_nudge,
            "total": self.total,
        }


class RewardEngine:
    """
    Multi-component reward function with all params from config.

    Usage:
        engine = RewardEngine(config)
        breakdown = engine.calculate(state_dict)
        reward = breakdown.total
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: Dict from prop_rules.yaml (validated by Pydantic)
        """
        # Reward component weights
        self.unrealized_weight = config.get("unrealized_shaping_weight", 0.1)
        self.overnight_pen = config.get("overnight_penalty", -5.0)
        self.rr_threshold = config.get("rr_bonus_threshold", 1.5)
        self.rr_coeff = config.get("rr_bonus_coefficient", 0.3)
        self.overtrading_pen = config.get("overtrading_penalty", -0.5)
        self.inaction_pen = config.get("inaction_nudge", -0.01)
        self.inaction_threshold = config.get("inaction_threshold_steps", 500)
        self.max_trades_per_day = config.get("max_trades_per_day", 15)

        # DD penalty params
        self.dd_alpha = config.get("dd_penalty_alpha", 2.0)
        self.dd_beta = config.get("dd_penalty_beta", 3.0)
        self.dd_start = config.get("dd_penalty_start", 0.02)
        self.max_daily_dd = config.get("max_daily_drawdown", 0.05)

        # Trading hours
        self.trading_end_utc = config.get("trading_end_utc", 21)

    def calculate(
        self,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        prev_unrealized_pnl: float = 0.0,
        current_dd: float = 0.0,
        hour_utc: int = 12,
        has_open_positions: bool = False,
        spread_cost: float = 0.0,
        commission: float = 0.0,
        risk_reward_ratio: float = 0.0,
        trades_today: int = 0,
        steps_since_last_trade: int = 0,
        account_balance: float = 10000.0,
        trade_just_opened: bool = False,
        trade_just_closed: bool = False,
    ) -> RewardBreakdown:
        """
        Calculate all 8 reward components.

        Args:
            realized_pnl: PnL from trade just closed (0 if no close this step)
            unrealized_pnl: Current mark-to-market of open positions
            prev_unrealized_pnl: Mark-to-market from previous step
            current_dd: Current drawdown as decimal (0.03 = 3%)
            hour_utc: Current hour in UTC
            has_open_positions: Whether any positions are open
            spread_cost: Spread cost paid this step (when opening trade)
            commission: Commission paid this step
            risk_reward_ratio: R/R ratio of closed trade (0 if no close)
            trades_today: Number of trades opened today
            steps_since_last_trade: Steps since last trade action
            account_balance: Current account balance for normalization
            trade_just_opened: Whether a trade was opened this step
            trade_just_closed: Whether a trade was closed this step

        Returns:
            RewardBreakdown with all 8 components
        """
        breakdown = RewardBreakdown()

        # Normalize by balance to make reward scale-invariant
        balance_norm = max(account_balance, 1.0)

        # ─── Component 1: Realized PnL ───
        if trade_just_closed and realized_pnl != 0:
            breakdown.realized_pnl = realized_pnl / balance_norm * 100  # As percentage

        # ─── Component 2: Unrealized PnL Shaping ───
        # Delta-based: reward the CHANGE in unrealized PnL (not absolute)
        # This prevents gaming by just holding profitable positions
        delta_unrealized = unrealized_pnl - prev_unrealized_pnl
        breakdown.unrealized_shaping = (
            self.unrealized_weight * delta_unrealized / balance_norm * 100
        )

        # ─── Component 3: Exponential Drawdown Penalty ───
        if current_dd > self.dd_start:
            # Penalty grows exponentially as DD approaches limit
            dd_ratio = current_dd / self.max_daily_dd
            breakdown.dd_penalty = -self.dd_alpha * math.exp(
                self.dd_beta * dd_ratio
            )

        # ─── Component 4: Overnight Penalty ───
        if has_open_positions and hour_utc >= self.trading_end_utc:
            breakdown.overnight_penalty = self.overnight_pen

        # ─── Component 5: Spread & Commission Cost ───
        if trade_just_opened:
            total_cost = spread_cost + commission
            breakdown.spread_commission = -total_cost / balance_norm * 100

        # ─── Component 6: Risk/Reward Bonus ───
        if trade_just_closed and risk_reward_ratio > self.rr_threshold:
            breakdown.rr_bonus = self.rr_coeff * (risk_reward_ratio - 1.0)

        # ─── Component 7: Overtrading Penalty ───
        if trades_today > self.max_trades_per_day:
            excess = trades_today - self.max_trades_per_day
            breakdown.overtrading_penalty = self.overtrading_pen * excess

        # ─── Component 8: Inaction Nudge ───
        if steps_since_last_trade > self.inaction_threshold and not has_open_positions:
            breakdown.inaction_nudge = self.inaction_pen

        return breakdown

    def is_episode_done(
        self,
        daily_dd: float,
        total_dd: float,
        max_total_dd: float = 0.10,
    ) -> tuple[bool, str]:
        """
        Check if episode should terminate due to Prop Firm rule violation.

        Returns:
            (done, reason) tuple
        """
        if daily_dd >= self.max_daily_dd:
            return True, f"Daily DD {daily_dd:.2%} >= limit {self.max_daily_dd:.2%}"

        if total_dd >= max_total_dd:
            return True, f"Total DD {total_dd:.2%} >= limit {max_total_dd:.2%}"

        return False, ""
