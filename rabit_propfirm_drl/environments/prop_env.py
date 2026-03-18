"""
PropFirm Trading Environment — Custom Gymnasium environment for DRL training.

Simulates realistic Prop Firm intraday trading with:
- Multi-component reward engine (8 components)
- Physics simulation (variable spread, slippage, latency)
- Prop Firm rules (DD limits, session hours, position limits)
- Continuous action space: [confidence, risk_fraction, sl_mult, tp_mult]
- Action gating: |confidence| < threshold → HOLD

State: concatenated feature vector from feature_builder + normalizer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environments.physics_sim import MarketPhysics, ExecutionResult
from environments.reward_engine import RewardBreakdown, RewardEngine

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Position Tracking
# ─────────────────────────────────────────────

@dataclass
class Position:
    """Represents an open trading position."""
    ticket: int
    direction: int            # +1 = LONG, -1 = SHORT
    entry_price: float
    lots: float
    sl_price: float           # Stop loss price
    tp_price: float           # Take profit price
    entry_step: int           # Step when opened
    spread_cost: float = 0.0  # Cost paid at entry


# ─────────────────────────────────────────────
# Gymnasium Environment
# ─────────────────────────────────────────────

class PropFirmTradingEnv(gym.Env):
    """
    Custom Gymnasium environment for Prop Firm intraday trading.

    Action Space (continuous, Box):
        [confidence, risk_fraction, sl_mult, tp_mult]
        - confidence: [-1, 1] — direction + conviction
          * |confidence| < threshold → HOLD
          * confidence > threshold → BUY
          * confidence < -threshold → SELL
        - risk_fraction: [0, 1] — fraction of max allowed lot size
        - sl_mult: [0.5, 3.0] — SL distance multiplier (× ATR)
        - tp_mult: [0.5, 5.0] — TP distance multiplier (× ATR)

    Observation Space (continuous, Box):
        Feature vector of shape (n_features,) — normalized by RunningNormalizer

    Rewards:
        Multi-component from RewardEngine (8 components)

    Termination:
        - Daily DD exceeds max_daily_drawdown
        - Total DD exceeds max_total_drawdown
        - Episode length reached
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: np.ndarray,
        config: dict,
        feature_names: list[str] | None = None,
        initial_balance: float = 10_000.0,
        episode_length: int = 2000,
        pip_value: float = 0.0001,
        lot_value: float = 100_000.0,
        commission_per_lot: float = 7.0,
        render_mode: str | None = None,
    ) -> None:
        """
        Args:
            data: Feature array of shape (n_timesteps, n_features).
                  Must include 'close' price and features from feature_builder.
            config: Dict from prop_rules.yaml
            feature_names: Names of feature columns (for debugging)
            initial_balance: Starting account balance
            episode_length: Max steps per episode
            pip_value: Price value of 1 pip (0.0001 for forex)
            lot_value: Contract size per lot (100,000 for forex)
            commission_per_lot: Round-trip commission per lot in account currency
            render_mode: Gymnasium render mode
        """
        super().__init__()

        # ─── Data ───
        self.data = data.astype(np.float32)
        self.n_timesteps, self.n_features = self.data.shape
        self.feature_names = feature_names
        self.render_mode = render_mode

        # ─── Config ───
        self.config = config
        self.initial_balance = initial_balance
        self.episode_length = min(episode_length, self.n_timesteps - 1)
        self.pip_value = pip_value
        self.lot_value = lot_value
        self.commission_per_lot = commission_per_lot

        # Load Prop Firm rules from config
        self.max_daily_dd = config.get("max_daily_drawdown", 0.05)
        self.max_total_dd = config.get("max_total_drawdown", 0.10)
        self.max_lots = config.get("max_lots_per_trade", 10.0)
        self.max_positions = config.get("max_open_positions", 5)
        self.confidence_threshold = config.get("confidence_threshold", 0.3)
        self.trading_start = config.get("trading_start_utc", 1)
        self.trading_end = config.get("trading_end_utc", 21)

        # Find close price column index (assumed to be at index 4 or named)
        self._close_idx = self._find_close_index()

        # ─── Components ───
        self.physics = MarketPhysics(config)
        self.reward_engine = RewardEngine(config)

        # ─── Spaces ───
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.n_features,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.5, 0.5], dtype=np.float32),
            high=np.array([1.0, 1.0, 3.0, 5.0], dtype=np.float32),
            dtype=np.float32,
        )

        # ─── State (initialized in reset) ───
        self._reset_state()

    def _find_close_index(self) -> int:
        """Find the index of the 'close' price in feature array."""
        if self.feature_names and "close" in self.feature_names:
            return self.feature_names.index("close")
        # Default: assume standard OHLCV ordering (time,open,high,low,close,...)
        return 4 if self.n_features > 4 else 0

    def _reset_state(self) -> None:
        """Initialize or reset all episode state."""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.daily_peak = self.initial_balance

        self.positions: list[Position] = []
        self.trade_history: list[dict] = []
        self.current_step = 0
        self.start_step = 0
        self.trades_today = 0
        self.steps_since_last_trade = 0
        self.prev_unrealized = 0.0
        self._next_ticket = 1

    # ─────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment for a new episode."""
        super().reset(seed=seed)

        self._reset_state()

        # Random start point (avoid look-ahead bias)
        max_start = self.n_timesteps - self.episode_length - 1
        if max_start > 0:
            self.start_step = self.np_random.integers(0, max_start)
        else:
            self.start_step = 0
        self.current_step = self.start_step

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step.

        Args:
            action: [confidence, risk_fraction, sl_mult, tp_mult]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        self.steps_since_last_trade += 1

        # Current price
        price = self._get_current_price()
        hour_utc = self._get_simulated_hour()

        # Parse action
        confidence = float(np.clip(action[0], -1.0, 1.0))
        risk_fraction = float(np.clip(action[1], 0.0, 1.0))
        sl_mult = float(np.clip(action[2], 0.5, 3.0))
        tp_mult = float(np.clip(action[3], 0.5, 5.0))

        # ─── Check existing positions (SL/TP hits) ───
        realized_pnl = 0.0
        rr_ratio = 0.0
        trade_closed = False

        positions_to_close = []
        for pos in self.positions:
            hit, pnl, rr = self._check_sl_tp(pos, price)
            if hit:
                positions_to_close.append((pos, pnl, rr))

        for pos, pnl, rr in positions_to_close:
            realized_pnl += pnl
            rr_ratio = max(rr_ratio, rr)
            trade_closed = True
            self.balance += pnl
            self.positions.remove(pos)
            self.trade_history.append({
                "ticket": pos.ticket,
                "direction": pos.direction,
                "entry": pos.entry_price,
                "exit": price,
                "pnl": pnl,
                "lots": pos.lots,
                "duration": self.current_step - pos.entry_step,
            })

        # ─── Action Gating ───
        trade_opened = False
        spread_cost = 0.0

        if abs(confidence) >= self.confidence_threshold:
            direction = 1 if confidence > 0 else -1

            # Check: can we open a new position?
            if (
                len(self.positions) < self.max_positions
                and self.trading_start <= hour_utc < self.trading_end
            ):
                # Calculate lot size
                scaled_confidence = (abs(confidence) - self.confidence_threshold) / (
                    1.0 - self.confidence_threshold
                )
                lot_size = risk_fraction * self.max_lots * scaled_confidence
                lot_size = max(0.01, min(lot_size, self.max_lots))

                # ATR estimate for SL/TP (use rolling volatility from features if available)
                atr_pips = self._estimate_atr_pips(price)

                sl_pips = atr_pips * sl_mult
                tp_pips = atr_pips * tp_mult

                # Execute through physics
                exec_result = self.physics.execute_order(
                    price=price,
                    direction=direction,
                    lot_size=lot_size,
                    hour_utc=hour_utc,
                    pip_value=self.pip_value,
                )

                if exec_result.filled:
                    # Calculate SL/TP prices
                    if direction > 0:  # BUY
                        sl_price = exec_result.fill_price - sl_pips * self.pip_value
                        tp_price = exec_result.fill_price + tp_pips * self.pip_value
                    else:  # SELL
                        sl_price = exec_result.fill_price + sl_pips * self.pip_value
                        tp_price = exec_result.fill_price - tp_pips * self.pip_value

                    # Commission
                    commission = self.commission_per_lot * exec_result.fill_lots
                    self.balance -= commission

                    # Create position
                    pos = Position(
                        ticket=self._next_ticket,
                        direction=direction,
                        entry_price=exec_result.fill_price,
                        lots=exec_result.fill_lots,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        entry_step=self.current_step,
                        spread_cost=exec_result.spread_pips * self.pip_value * exec_result.fill_lots * self.lot_value,
                    )
                    self._next_ticket += 1
                    self.positions.append(pos)
                    self.trades_today += 1
                    self.steps_since_last_trade = 0
                    trade_opened = True
                    spread_cost = pos.spread_cost

        # ─── Update equity ───
        unrealized = self._total_unrealized_pnl(price)
        self.equity = self.balance + unrealized
        self.peak_equity = max(self.peak_equity, self.equity)
        self.daily_peak = max(self.daily_peak, self.equity)

        # ─── Calculate drawdown ───
        daily_dd = max(0, (self.daily_peak - self.equity) / self.daily_peak)
        total_dd = max(0, (self.peak_equity - self.equity) / self.peak_equity)

        # ─── Calculate reward ───
        breakdown = self.reward_engine.calculate(
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized,
            prev_unrealized_pnl=self.prev_unrealized,
            current_dd=daily_dd,
            hour_utc=hour_utc,
            has_open_positions=len(self.positions) > 0,
            spread_cost=spread_cost,
            commission=self.commission_per_lot * (lot_size if trade_opened else 0),
            risk_reward_ratio=rr_ratio,
            trades_today=self.trades_today,
            steps_since_last_trade=self.steps_since_last_trade,
            account_balance=self.balance,
            trade_just_opened=trade_opened,
            trade_just_closed=trade_closed,
        )

        self.prev_unrealized = unrealized
        reward = float(breakdown.total)

        # ─── Check termination ───
        terminated = False
        truncated = False

        done, reason = self.reward_engine.is_episode_done(
            daily_dd, total_dd, self.max_total_dd
        )
        if done:
            terminated = True
            logger.debug("Episode terminated: %s", reason)

        # Truncate if episode length reached
        steps_elapsed = self.current_step - self.start_step
        if steps_elapsed >= self.episode_length:
            truncated = True

        # Truncate if out of data
        if self.current_step >= self.n_timesteps - 1:
            truncated = True

        obs = self._get_observation()
        info = self._get_info()
        info["reward_breakdown"] = breakdown.to_dict()

        return obs, reward, terminated, truncated, info

    # ─────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """Get current observation (feature vector)."""
        idx = min(self.current_step, self.n_timesteps - 1)
        return self.data[idx].copy()

    def _get_current_price(self) -> float:
        """Get current close price."""
        idx = min(self.current_step, self.n_timesteps - 1)
        return float(self.data[idx, self._close_idx])

    def _get_simulated_hour(self) -> int:
        """Simulate hour of day from step index."""
        # Map steps to trading hours (roughly 1 step = 15 min = M15)
        minutes = (self.current_step * 15) % (24 * 60)
        return minutes // 60

    def _estimate_atr_pips(self, price: float) -> float:
        """Estimate ATR in pips from recent price moves."""
        lookback = min(14, self.current_step - self.start_step)
        if lookback < 2:
            return 20.0  # Default 20 pips

        start = max(0, self.current_step - lookback)
        end = self.current_step + 1
        prices = self.data[start:end, self._close_idx]
        returns = np.abs(np.diff(prices))
        atr = float(np.mean(returns))
        atr_pips = atr / self.pip_value

        return max(atr_pips, 5.0)  # Minimum 5 pips

    def _total_unrealized_pnl(self, current_price: float) -> float:
        """Calculate total unrealized PnL across all open positions."""
        total = 0.0
        for pos in self.positions:
            price_diff = (current_price - pos.entry_price) * pos.direction
            pnl = price_diff * pos.lots * self.lot_value
            total += pnl
        return total

    def _check_sl_tp(
        self, pos: Position, current_price: float
    ) -> tuple[bool, float, float]:
        """
        Check if SL or TP hit for a position.

        Returns:
            (hit, pnl, risk_reward_ratio)
        """
        if pos.direction > 0:  # LONG
            if current_price <= pos.sl_price:
                pnl = (pos.sl_price - pos.entry_price) * pos.lots * self.lot_value
                return True, pnl, 0.0  # SL hit, RR = 0

            if current_price >= pos.tp_price:
                pnl = (pos.tp_price - pos.entry_price) * pos.lots * self.lot_value
                risk = abs(pos.entry_price - pos.sl_price)
                reward = abs(pos.tp_price - pos.entry_price)
                rr = reward / max(risk, 1e-10)
                return True, pnl, rr
        else:  # SHORT
            if current_price >= pos.sl_price:
                pnl = (pos.entry_price - pos.sl_price) * pos.lots * self.lot_value
                return True, pnl, 0.0

            if current_price <= pos.tp_price:
                pnl = (pos.entry_price - pos.tp_price) * pos.lots * self.lot_value
                risk = abs(pos.sl_price - pos.entry_price)
                reward = abs(pos.entry_price - pos.tp_price)
                rr = reward / max(risk, 1e-10)
                return True, pnl, rr

        return False, 0.0, 0.0

    def _get_info(self) -> dict[str, Any]:
        """Build info dict for current step."""
        return {
            "balance": self.balance,
            "equity": self.equity,
            "daily_dd": max(0, (self.daily_peak - self.equity) / max(self.daily_peak, 1)),
            "total_dd": max(0, (self.peak_equity - self.equity) / max(self.peak_equity, 1)),
            "open_positions": len(self.positions),
            "trades_today": self.trades_today,
            "step": self.current_step,
            "total_trades": len(self.trade_history),
        }
