"""
Reward Engine -- Multi-component reward function for DRL trading agent.

V3.3 -- Simplified Stage 1 (anti-mode-collapse):
 1. trade_bonus    -- +1.0 per trade opened (win or lose)
 2. realized_pnl   -- Actual PnL scaled by ATR (not raw dollars)
 3. inaction_nudge  -- -0.1 per step idle (gentle nudge)

All penalties (DD, overnight, overtrading, fomo, sniper) DISABLED for Stage 1.
Can be re-enabled for Stage 2+ via config `stage1_mode: false`.
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
    fomo_penalty: float = 0.0
    sniper_multiplier: float = 0.0
    sniper_bonus: float = 0.0
    exploration_bonus: float = 0.0       # V3.2/3.3: Trade open reward
    trade_attempt_shaping: float = 0.0   # V3.2: deprecated in V3.3

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
            + self.fomo_penalty
            + self.sniper_multiplier
            + self.sniper_bonus
            + self.exploration_bonus
            + self.trade_attempt_shaping
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
            "fomo_penalty": self.fomo_penalty,
            "sniper_multiplier": self.sniper_multiplier,
            "sniper_bonus": self.sniper_bonus,
            "exploration_bonus": self.exploration_bonus,
            "trade_attempt_shaping": self.trade_attempt_shaping,
            "total": self.total,
        }


class RewardEngine:
    """
    Multi-component reward function with all params from config.

    V3.3: When stage1_mode=True, only 3 components are active:
        1. trade_bonus (+1.0 per trade opened)
        2. realized_pnl (ATR-scaled)
        3. inaction_nudge (-0.1 per idle step)
    """

    def __init__(self, config: dict) -> None:
        # V3.3: Stage 1 simplified mode
        self.stage1_mode = config.get("stage1_mode", True)

        # Reward component weights
        self.unrealized_weight = config.get("unrealized_shaping_weight", 0.1)
        self.overnight_pen = config.get("overnight_penalty", -5.0)
        self.rr_threshold = config.get("rr_bonus_threshold", 1.5)
        self.rr_coeff = config.get("rr_bonus_coefficient", 0.3)
        self.overtrading_pen = config.get("overtrading_penalty", -0.5)
        self.inaction_pen = config.get("inaction_nudge", -0.1)  # V3.3: gentler
        self.inaction_threshold = config.get("inaction_threshold_steps", 20)  # V3.3: faster
        self.max_trades_per_day = config.get("max_trades_per_day", 15)

        # DD penalty params
        self.dd_alpha = config.get("dd_penalty_alpha", 2.0)
        self.dd_beta = config.get("dd_penalty_beta", 3.0)
        self.dd_start = config.get("dd_penalty_start", 0.02)
        self.max_daily_dd = config.get("max_daily_drawdown", 0.05)

        # Trading hours
        self.trading_end_utc = config.get("trading_end_utc", 21)

        # Dual Entry System
        self.sniper_win_mult = config.get("sniper_win_multiplier", 5.0)
        self.sniper_loss_mult = config.get("sniper_loss_multiplier", 3.0)

        # FOMO Oracle
        self.fomo_pen = config.get("fomo_penalty", -5.0)
        self.fomo_lookahead = config.get("fomo_lookahead_steps", 50)
        self.fomo_move_pct = config.get("fomo_move_threshold_pct", 0.005)

        # V3.3: Trade bonus (replaces exploration_bonus)
        self.trade_bonus_val = config.get("trade_bonus", 1.0)

        # V3.2 compat
        self.exploration_bonus_val = config.get("exploration_bonus", 0.5)
        self.trade_attempt_bonus = config.get("trade_attempt_bonus", 0.05)
        self.loss_dampening = config.get("loss_dampening_factor", 0.5)

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
        entry_type: str = "standby",
        trade_won: bool = False,
        future_price_data: Optional[np.ndarray] = None,
        current_price: float = 0.0,
        abs_confidence: float = 0.0,
    ) -> RewardBreakdown:
        """Calculate reward components. Stage1 mode uses only 3 components."""
        breakdown = RewardBreakdown()

        if self.stage1_mode:
            return self._calculate_stage1(
                breakdown, realized_pnl, account_balance,
                trade_just_opened, trade_just_closed,
                steps_since_last_trade, has_open_positions,
                trades_today,
            )
        else:
            return self._calculate_full(
                breakdown, realized_pnl, unrealized_pnl, prev_unrealized_pnl,
                current_dd, hour_utc, has_open_positions, spread_cost,
                commission, risk_reward_ratio, trades_today,
                steps_since_last_trade, account_balance,
                trade_just_opened, trade_just_closed,
                entry_type, trade_won, future_price_data,
                current_price, abs_confidence,
            )

    def _calculate_stage1(
        self,
        breakdown: RewardBreakdown,
        realized_pnl: float,
        account_balance: float,
        trade_just_opened: bool,
        trade_just_closed: bool,
        steps_since_last_trade: int,
        has_open_positions: bool,
        trades_today: int,
    ) -> RewardBreakdown:
        """
        V3.3 Stage 1: Only 3 reward components.
        Goal: Make the bot TRADE, not sit idle.
        """
        balance_norm = max(account_balance, 1.0)

        # Component 1: Trade Bonus (+1.0 per trade opened)
        if trade_just_opened:
            decay = 1.0 / math.sqrt(trades_today + 1)  # Gentle decay
            breakdown.exploration_bonus = self.trade_bonus_val * decay

        # Component 2: Realized PnL (simplified, no sniper/dampening)
        if trade_just_closed and realized_pnl != 0:
            breakdown.realized_pnl = realized_pnl / balance_norm * 100

        # Component 3: Inaction Nudge (gentle but persistent)
        if steps_since_last_trade > self.inaction_threshold and not has_open_positions:
            breakdown.inaction_nudge = self.inaction_pen

        return breakdown

    def _calculate_full(
        self,
        breakdown: RewardBreakdown,
        realized_pnl: float,
        unrealized_pnl: float,
        prev_unrealized_pnl: float,
        current_dd: float,
        hour_utc: int,
        has_open_positions: bool,
        spread_cost: float,
        commission: float,
        risk_reward_ratio: float,
        trades_today: int,
        steps_since_last_trade: int,
        account_balance: float,
        trade_just_opened: bool,
        trade_just_closed: bool,
        entry_type: str,
        trade_won: bool,
        future_price_data: Optional[np.ndarray],
        current_price: float,
        abs_confidence: float,
    ) -> RewardBreakdown:
        """Full 13-component reward for Stage 2+. Same as V3.2."""
        balance_norm = max(account_balance, 1.0)

        # Component 1: Realized PnL
        if trade_just_closed and realized_pnl != 0:
            base_pnl = realized_pnl / balance_norm * 100
            if base_pnl < 0:
                base_pnl = base_pnl * self.loss_dampening

            if entry_type == "m1_sniper":
                breakdown.realized_pnl = base_pnl
                if trade_won:
                    breakdown.sniper_bonus = base_pnl * (self.sniper_win_mult - 1.0)
                else:
                    breakdown.sniper_multiplier = base_pnl * (self.sniper_loss_mult - 1.0)
            else:
                breakdown.realized_pnl = base_pnl

        # Component 2: Unrealized Shaping
        delta_unrealized = unrealized_pnl - prev_unrealized_pnl
        breakdown.unrealized_shaping = (
            self.unrealized_weight * delta_unrealized / balance_norm * 100
        )

        # Component 3: DD Penalty
        if current_dd > self.dd_start:
            dd_ratio = current_dd / self.max_daily_dd
            exp_arg = max(-50.0, min(self.dd_beta * dd_ratio, 50.0))
            breakdown.dd_penalty = -self.dd_alpha * math.exp(exp_arg)

        # Component 4: Overnight Penalty
        if has_open_positions and hour_utc >= self.trading_end_utc:
            breakdown.overnight_penalty = self.overnight_pen

        # Component 5: Spread & Commission
        if trade_just_opened:
            total_cost = spread_cost + commission
            breakdown.spread_commission = -total_cost / balance_norm * 100

        # Component 6: RR Bonus
        if trade_just_closed and risk_reward_ratio > self.rr_threshold:
            breakdown.rr_bonus = self.rr_coeff * (risk_reward_ratio - 1.0)

        # Component 7: Overtrading Penalty
        if trades_today > self.max_trades_per_day:
            excess = trades_today - self.max_trades_per_day
            breakdown.overtrading_penalty = self.overtrading_pen * excess

        # Component 8: Inaction Nudge
        if steps_since_last_trade > self.inaction_threshold and not has_open_positions:
            breakdown.inaction_nudge = self.inaction_pen

        # Component 9: FOMO Oracle
        if (
            entry_type == "standby"
            and not has_open_positions
            and future_price_data is not None
            and len(future_price_data) >= 5
            and current_price > 0
        ):
            future_max = float(np.max(future_price_data))
            future_min = float(np.min(future_price_data))
            max_move_up = (future_max - current_price) / current_price
            max_move_down = (current_price - future_min) / current_price
            if max(max_move_up, max_move_down) > self.fomo_move_pct:
                breakdown.fomo_penalty = self.fomo_pen

        # Component 12: Exploration Bonus
        if trade_just_opened:
            decay = 1.0 / math.sqrt(trades_today + 1)
            breakdown.exploration_bonus = self.exploration_bonus_val * decay

        return breakdown

    def is_episode_done(
        self,
        daily_dd: float,
        total_dd: float,
        max_total_dd: float = 0.10,
    ) -> tuple[bool, str]:
        """Check if episode should terminate due to Prop Firm rule violation."""
        if self.stage1_mode:
            # V3.3 Stage 1: No termination from DD (let it learn freely)
            return False, ""

        if daily_dd >= self.max_daily_dd:
            return True, f"Daily DD {daily_dd:.2%} >= limit {self.max_daily_dd:.2%}"

        if total_dd >= max_total_dd:
            return True, f"Total DD {total_dd:.2%} >= limit {max_total_dd:.2%}"

        return False, ""
