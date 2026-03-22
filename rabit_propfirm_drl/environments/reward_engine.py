"""
Reward Engine -- Multi-component reward function for DRL trading agent.

11 Components:
 1. realized_pnl         -- Actual profit/loss when closing a trade
 2. unrealized_shaping   -- Mark-to-market shaping (delta per step)
 3. dd_penalty           -- Exponential drawdown penalty
 4. overnight_penalty    -- Penalty for holding past session end
 5. spread_commission    -- Execution cost deduction
 6. rr_bonus             -- Bonus for good risk/reward ratio
 7. overtrading_penalty  -- Penalty for excessive trades per day
 8. inaction_nudge       -- Strong penalty if idle too long (-0.5/step)
 9. fomo_penalty         -- PENALTY for missing obvious M1 setups (oracle)
10. sniper_multiplier    -- 3x loss penalty for failed Sniper (M1) entries
11. sniper_bonus         -- 5x win bonus for successful Sniper (M1) entries

All weights/thresholds from prop_rules.yaml -- zero hardcoding.
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

        # Dual Entry System -- Asymmetric Reward
        self.sniper_win_mult = config.get("sniper_win_multiplier", 5.0)
        self.sniper_loss_mult = config.get("sniper_loss_multiplier", 3.0)

        # FOMO Oracle (training only)
        self.fomo_pen = config.get("fomo_penalty", -5.0)
        self.fomo_lookahead = config.get("fomo_lookahead_steps", 50)
        self.fomo_move_pct = config.get("fomo_move_threshold_pct", 0.005)

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
        # Dual Entry System params
        entry_type: str = "standby",  # "standby", "m5_normal", "m1_sniper"
        trade_won: bool = False,
        future_price_data: Optional[np.ndarray] = None,  # Oracle lookahead
        current_price: float = 0.0,
    ) -> RewardBreakdown:
        """
        Calculate all 11 reward components.

        Args:
            ... (standard args) ...
            entry_type: "standby", "m5_normal", or "m1_sniper"
            trade_won: Whether the closed trade was profitable
            future_price_data: Next N bars of price data (TRAINING ONLY oracle)
            current_price: Current close price for FOMO calculation

        Returns:
            RewardBreakdown with all 11 components
        """
        breakdown = RewardBreakdown()

        # Normalize by balance to make reward scale-invariant
        balance_norm = max(account_balance, 1.0)

        # --- Component 1: Realized PnL (with Asymmetric Sniper Multiplier) ---
        if trade_just_closed and realized_pnl != 0:
            base_pnl = realized_pnl / balance_norm * 100  # As percentage

            if entry_type == "m1_sniper":
                if trade_won:
                    # Component 11: Sniper WIN bonus (5x)
                    breakdown.realized_pnl = base_pnl
                    breakdown.sniper_bonus = base_pnl * (self.sniper_win_mult - 1.0)
                else:
                    # Component 10: Sniper LOSS penalty (3x)
                    breakdown.realized_pnl = base_pnl
                    breakdown.sniper_multiplier = base_pnl * (self.sniper_loss_mult - 1.0)
            else:
                # M5 Normal: standard 1x reward
                breakdown.realized_pnl = base_pnl

        # --- Component 2: Unrealized PnL Shaping ---
        delta_unrealized = unrealized_pnl - prev_unrealized_pnl
        breakdown.unrealized_shaping = (
            self.unrealized_weight * delta_unrealized / balance_norm * 100
        )

        # --- Component 3: Exponential Drawdown Penalty ---
        if current_dd > self.dd_start:
            dd_ratio = current_dd / self.max_daily_dd
            # Clamp exponent to prevent OverflowError (math range error)
            exp_arg = max(-50.0, min(self.dd_beta * dd_ratio, 50.0))
            breakdown.dd_penalty = -self.dd_alpha * math.exp(exp_arg)

        # --- Component 4: Overnight Penalty ---
        if has_open_positions and hour_utc >= self.trading_end_utc:
            breakdown.overnight_penalty = self.overnight_pen

        # --- Component 5: Spread & Commission Cost ---
        if trade_just_opened:
            total_cost = spread_cost + commission
            breakdown.spread_commission = -total_cost / balance_norm * 100

        # --- Component 6: Risk/Reward Bonus ---
        if trade_just_closed and risk_reward_ratio > self.rr_threshold:
            breakdown.rr_bonus = self.rr_coeff * (risk_reward_ratio - 1.0)

        # --- Component 7: Overtrading Penalty ---
        if trades_today > self.max_trades_per_day:
            excess = trades_today - self.max_trades_per_day
            breakdown.overtrading_penalty = self.overtrading_pen * excess

        # --- Component 8: Inaction Nudge ("Bleeding" penalty) ---
        if steps_since_last_trade > self.inaction_threshold and not has_open_positions:
            breakdown.inaction_nudge = self.inaction_pen

        # --- Component 9: FOMO Oracle Penalty (TRAINING ONLY) ---
        # If agent chose Standby but price moved significantly,
        # penalize for missing the opportunity.
        if (
            entry_type == "standby"
            and not has_open_positions
            and future_price_data is not None
            and len(future_price_data) >= 5
            and current_price > 0
        ):
            # Check if price moved > threshold in next N bars
            future_max = float(np.max(future_price_data))
            future_min = float(np.min(future_price_data))
            max_move_up = (future_max - current_price) / current_price
            max_move_down = (current_price - future_min) / current_price
            max_move = max(max_move_up, max_move_down)

            if max_move > self.fomo_move_pct:
                # Agent missed an obvious setup -> FOMO penalty
                breakdown.fomo_penalty = self.fomo_pen

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
