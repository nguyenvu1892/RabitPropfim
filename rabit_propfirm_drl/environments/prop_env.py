"""
PropFirm Trading Environment — Custom Gymnasium environment for DRL training.

v2.0 — Cognitive Architecture:
    - Multi-TF observations: dict of {m1, m5, m15, h1} arrays
    - 4 timeframes: M1 (128 bars), M5 (64 bars), M15 (48 bars), H1 (24 bars)
    - 50-dim features per bar (28 raw + 22 knowledge)
    - H4 removed entirely
    - Session-based trading (intraday close by 22:00)

State: Dict observation space:
    m1:  (128, 50) — sniper entry
    m5:  (64, 50)  — normal entry
    m15: (48, 50)  — structure context
    h1:  (24, 50)  — trend bias

Action: [confidence, risk_fraction, sl_mult, tp_mult] ∈ continuous
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environments.physics_sim import MarketPhysics, ExecutionResult
from environments.reward_engine import RewardBreakdown, RewardEngine

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Position Tracking
# ─────────────────────────────────────────

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


# ─────────────────────────────────────────
# Multi-TF Trading Environment
# ─────────────────────────────────────────

class MultiTFTradingEnv(gym.Env):
    """
    Multi-Timeframe Trading Environment for Cognitive Architecture.

    Accepts 4 separate OHLCV+feature arrays and returns dict observations
    with fixed-size windows per timeframe.

    Observation Space (Dict):
        {
            "m1":  Box(128, 50),  — M1 window (sniper entry)
            "m5":  Box(64, 50),   — M5 window (normal entry)
            "m15": Box(48, 50),   — M15 window (structure)
            "h1":  Box(24, 50),   — H1 window (trend bias)
        }

    Action Space (continuous, Box):
        [confidence, risk_fraction, sl_mult, tp_mult]

    Stepping uses M5 as the primary timeframe — each step() advances
    one M5 bar, and M1/M15/H1 bars are aligned by timestamp.
    """

    metadata = {"render_modes": ["human"]}

    # Default lookback windows per timeframe
    LOOKBACK_M1: int = 128
    LOOKBACK_M5: int = 64
    LOOKBACK_M15: int = 48
    LOOKBACK_H1: int = 24

    def __init__(
        self,
        data_m1: np.ndarray,
        data_m5: np.ndarray,
        data_m15: np.ndarray,
        data_h1: np.ndarray,
        config: dict,
        n_features: int = 50,
        initial_balance: float = 10_000.0,
        episode_length: int = 2000,
        pip_value: float = 0.0001,
        lot_value: float = 100_000.0,
        commission_per_lot: float = 7.0,
        close_idx: int = 4,
        render_mode: str | None = None,
    ) -> None:
        """
        Args:
            data_m1:  (N_m1, n_features) — M1 feature array
            data_m5:  (N_m5, n_features) — M5 feature array (primary TF)
            data_m15: (N_m15, n_features) — M15 feature array
            data_h1:  (N_h1, n_features) — H1 feature array
            config:    Dict from prop_rules.yaml
            n_features: Number of features per bar (28 raw + 22 knowledge = 50)
            initial_balance: Starting account balance
            episode_length:  Max steps per episode (in M5 bars)
            pip_value:  Price value of 1 pip
            lot_value:  Contract size per lot
            commission_per_lot: Round-trip commission per lot
            close_idx:  Column index of close price in feature arrays
            render_mode: Gymnasium render mode
        """
        super().__init__()

        # ─── Data (4 timeframes) ───
        self.data_m1 = data_m1.astype(np.float32)
        self.data_m5 = data_m5.astype(np.float32)
        self.data_m15 = data_m15.astype(np.float32)
        self.data_h1 = data_h1.astype(np.float32)

        self.n_features = n_features
        self.close_idx = close_idx
        self.render_mode = render_mode

        # Time alignment ratios (how many M1 bars per M5 bar, etc.)
        # M1:M5 = 5:1, M5:M15 = 3:1, M15:H1 = 4:1
        self.m1_per_m5 = 5
        self.m5_per_m15 = 3
        self.m15_per_h1 = 4

        # Total M5 bars available (primary TF drives stepping)
        self.n_m5_bars = len(self.data_m5)

        # ─── Config ───
        self.config = config
        self.initial_balance = initial_balance
        self.episode_length = min(episode_length, self.n_m5_bars - self.LOOKBACK_M5 - 1)
        self.pip_value = pip_value
        self.lot_value = lot_value
        self.commission_per_lot = commission_per_lot

        # Load Prop Firm rules from config
        self.max_daily_dd = config.get("max_daily_drawdown", 0.05)
        self.max_total_dd = config.get("max_total_drawdown", 0.10)
        self.max_lots = config.get("max_lots_per_trade", 10.0)
        self.max_positions = config.get("max_open_positions", 5)
        self.confidence_threshold = config.get("confidence_threshold", 0.15)
        self.m5_threshold = config.get("m5_normal_threshold", 0.50)
        self.m1_threshold = config.get("m1_sniper_threshold", 0.85)
        self.trading_start = config.get("trading_start_utc", 1)
        self.trading_end = config.get("trading_end_utc", 22)  # Intraday close by 22:00

        # ─── Components ───
        self.physics = MarketPhysics(config)
        self.reward_engine = RewardEngine(config)

        # ─── Spaces ───
        self.observation_space = spaces.Dict({
            "m1": spaces.Box(
                low=-10.0, high=10.0,
                shape=(self.LOOKBACK_M1, n_features),
                dtype=np.float32,
            ),
            "m5": spaces.Box(
                low=-10.0, high=10.0,
                shape=(self.LOOKBACK_M5, n_features),
                dtype=np.float32,
            ),
            "m15": spaces.Box(
                low=-10.0, high=10.0,
                shape=(self.LOOKBACK_M15, n_features),
                dtype=np.float32,
            ),
            "h1": spaces.Box(
                low=-10.0, high=10.0,
                shape=(self.LOOKBACK_H1, n_features),
                dtype=np.float32,
            ),
        })

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.5, 0.5], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 3.0, 5.0], dtype=np.float32),
            dtype=np.float32,
        )

        # ─── State (initialized in reset) ───
        self._reset_state()

    def _reset_state(self) -> None:
        """Initialize or reset all episode state."""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.daily_peak = self.initial_balance

        self.positions: list[Position] = []
        self.trade_history: list[dict] = []
        self.current_m5_step = 0       # Index into data_m5
        self.start_m5_step = 0
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
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Reset environment for a new episode."""
        super().reset(seed=seed)

        self._reset_state()

        # Ensure enough lookback for all TFs
        min_start_m5 = self.LOOKBACK_M5
        max_start_m5 = self.n_m5_bars - self.episode_length - 1

        if max_start_m5 > min_start_m5:
            self.start_m5_step = self.np_random.integers(min_start_m5, max_start_m5)
        else:
            self.start_m5_step = min_start_m5

        self.current_m5_step = self.start_m5_step

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """
        Execute one M5 time step.

        Args:
            action: [confidence, risk_fraction, sl_mult, tp_mult]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.current_m5_step += 1
        self.steps_since_last_trade += 1

        # Current price (from M5 close)
        price = self._get_current_price()
        hour_utc = self._get_simulated_hour()

        # Parse action (5-dim: confidence, entry_type, risk_frac, sl_mult, tp_mult)
        confidence = float(np.clip(action[0], -1.0, 1.0))
        entry_type_raw = float(np.clip(action[1], -1.0, 1.0))  # <0 = M5, >0 = M1
        risk_fraction = float(np.clip(action[2], 0.0, 1.0))
        sl_mult = float(np.clip(action[3], 0.5, 3.0))
        tp_mult = float(np.clip(action[4], 0.5, 5.0))

        # Determine entry type with DUAL THRESHOLDS
        entry_type_str = "standby"  # Default: no trade
        abs_conf = abs(confidence)
        if entry_type_raw > 0 and abs_conf >= self.m1_threshold:
            entry_type_str = "m1_sniper"
        elif entry_type_raw <= 0 and abs_conf >= self.m5_threshold:
            entry_type_str = "m5_normal"

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
                "duration": self.current_m5_step - pos.entry_step,
            })

        # ─── Intraday session close ───
        # Force close all positions at session end
        if hour_utc >= self.trading_end and self.positions:
            for pos in list(self.positions):
                pnl = self._calc_position_pnl(pos, price)
                realized_pnl += pnl
                trade_closed = True
                self.balance += pnl
                self.trade_history.append({
                    "ticket": pos.ticket,
                    "direction": pos.direction,
                    "entry": pos.entry_price,
                    "exit": price,
                    "pnl": pnl,
                    "lots": pos.lots,
                    "duration": self.current_m5_step - pos.entry_step,
                    "close_reason": "SESSION_END",
                })
            self.positions.clear()

        # ─── Action Gating ───
        trade_opened = False
        spread_cost = 0.0
        lot_size = 0.0

        if entry_type_str in ("m5_normal", "m1_sniper"):
            direction = 1 if confidence > 0 else -1

            # Check: can we open a new position?
            if (
                len(self.positions) < self.max_positions
                and self.trading_start <= hour_utc < self.trading_end
            ):
                # Calculate lot size
                scaled_confidence = (abs_conf - self.confidence_threshold) / (
                    1.0 - self.confidence_threshold
                )
                lot_size = risk_fraction * self.max_lots * scaled_confidence
                lot_size = max(0.01, min(lot_size, self.max_lots))

                # ATR estimate for SL/TP
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
                        entry_step=self.current_m5_step,
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
            daily_dd, total_dd, self.max_total_dd,
        )
        if done:
            terminated = True
            logger.debug("Episode terminated: %s", reason)

        # Truncate if episode length reached
        steps_elapsed = self.current_m5_step - self.start_m5_step
        if steps_elapsed >= self.episode_length:
            truncated = True

        # Truncate if out of M5 data
        if self.current_m5_step >= self.n_m5_bars - 1:
            truncated = True

        obs = self._get_observation()
        info = self._get_info()
        info["reward_breakdown"] = breakdown.to_dict()

        return obs, reward, terminated, truncated, info

    # ─────────────────────────────────────────
    # Observation Construction
    # ─────────────────────────────────────────

    def _get_observation(self) -> dict[str, np.ndarray]:
        """
        Build dict observation from 4 timeframes.

        Each TF returns a fixed-size window ending at the current aligned bar.
        If not enough bars available, zero-pad on the left.
        """
        m5_idx = min(self.current_m5_step, self.n_m5_bars - 1)

        # M5 window
        m5_obs = self._get_window(self.data_m5, m5_idx, self.LOOKBACK_M5)

        # M1: 5 M1 bars per M5 bar
        m1_idx = min(m5_idx * self.m1_per_m5, len(self.data_m1) - 1)
        m1_obs = self._get_window(self.data_m1, m1_idx, self.LOOKBACK_M1)

        # M15: 1 M15 bar per 3 M5 bars
        m15_idx = min(m5_idx // self.m5_per_m15, len(self.data_m15) - 1)
        m15_obs = self._get_window(self.data_m15, m15_idx, self.LOOKBACK_M15)

        # H1: 1 H1 bar per 12 M5 bars (= 4 M15 bars)
        h1_idx = min(m5_idx // (self.m5_per_m15 * self.m15_per_h1), len(self.data_h1) - 1)
        h1_obs = self._get_window(self.data_h1, h1_idx, self.LOOKBACK_H1)

        return {
            "m1": m1_obs,
            "m5": m5_obs,
            "m15": m15_obs,
            "h1": h1_obs,
        }

    def _get_window(
        self, data: np.ndarray, end_idx: int, window_size: int,
    ) -> np.ndarray:
        """
        Extract a fixed-size window ending at end_idx (inclusive).
        Zero-pads on the left if not enough bars exist.
        """
        start_idx = max(0, end_idx - window_size + 1)
        window = data[start_idx:end_idx + 1]

        # Zero-pad if window is too small
        if len(window) < window_size:
            pad_size = window_size - len(window)
            padding = np.zeros((pad_size, data.shape[1]), dtype=np.float32)
            window = np.vstack([padding, window])

        return window.copy()

    # ─────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────

    def _get_current_price(self) -> float:
        """Get current close price from M5 data."""
        idx = min(self.current_m5_step, self.n_m5_bars - 1)
        return float(self.data_m5[idx, self.close_idx])

    def _get_simulated_hour(self) -> int:
        """Simulate hour of day from M5 step index (M5 = 5 min bars)."""
        minutes = (self.current_m5_step * 5) % (24 * 60)
        return minutes // 60

    def _estimate_atr_pips(self, price: float) -> float:
        """Estimate ATR in pips from recent M5 price moves."""
        lookback = min(14, self.current_m5_step - self.start_m5_step)
        if lookback < 2:
            return 20.0  # Default 20 pips

        start = max(0, self.current_m5_step - lookback)
        end = self.current_m5_step + 1
        prices = self.data_m5[start:end, self.close_idx]
        returns = np.abs(np.diff(prices))
        atr = float(np.mean(returns))
        atr_pips = atr / self.pip_value

        return max(atr_pips, 5.0)

    def _total_unrealized_pnl(self, current_price: float) -> float:
        """Calculate total unrealized PnL across all open positions."""
        total = 0.0
        for pos in self.positions:
            total += self._calc_position_pnl(pos, current_price)
        return total

    def _calc_position_pnl(self, pos: Position, current_price: float) -> float:
        """Calculate PnL for a single position."""
        price_diff = (current_price - pos.entry_price) * pos.direction
        return price_diff * pos.lots * self.lot_value

    def _check_sl_tp(
        self, pos: Position, current_price: float,
    ) -> tuple[bool, float, float]:
        """
        Check if SL or TP hit for a position.
        Returns: (hit, pnl, risk_reward_ratio)
        """
        if pos.direction > 0:  # LONG
            if current_price <= pos.sl_price:
                pnl = (pos.sl_price - pos.entry_price) * pos.lots * self.lot_value
                return True, pnl, 0.0

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
            "step": self.current_m5_step,
            "total_trades": len(self.trade_history),
        }


# ── Legacy alias for backward compatibility ──
PropFirmTradingEnv = MultiTFTradingEnv
