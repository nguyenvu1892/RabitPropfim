"""
PropFirm Trading Environment -- Custom Gymnasium environment for DRL training.

V3.9 -- "Giao Thoa" Architecture:
    - Discrete action space: BUY=0, SELL=1, HOLD=2, CLOSE=3
    - 4-TF Frame Stacking: H1 (56) + M15 (56) + M5 (56) + M1×5 (280) = 448-dim
    - +6 features per bar: OB proximity + Volume spike + Spread + Session + OB Touch Count + Choppiness
    - Trailing SL: Move SL to breakeven when unrealized PnL >= 1R
    - Auto SL from M5 Swing Points, NO fixed TP
    - Fixed lot (0.01), action_mode: "discrete" or "continuous" (legacy)
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


# --- Position Tracking ---

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


# --- Multi-TF Trading Environment ---

class MultiTFTradingEnv(gym.Env):
    """
    Multi-Timeframe Trading Environment.

    V3.4 supports two action modes:
        - "discrete": Discrete(4) BUY/SELL/HOLD/CLOSE
        - "continuous": Box(5,) (legacy)

    Observation (V3.9 discrete mode):
        Flat Box(448,) = 1 H1 bar (56) + 1 M15 bar (56) + 1 M5 bar (56) + 5 M1 bars (280)
        +6 features per bar: OB prox + Vol spike + Spread + Session + OB Touch + Choppiness
    """

    metadata = {"render_modes": ["human"]}

    # Legacy lookback (for continuous mode compatibility)
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
        ohlcv_m5: np.ndarray | None = None,
        action_mode: str = "discrete",  # V3.3: "discrete" or "continuous"
    ) -> None:
        super().__init__()

        # --- Data (4 timeframes) ---
        self.data_m1 = data_m1.astype(np.float32)
        self.data_m5 = data_m5.astype(np.float32)
        self.data_m15 = data_m15.astype(np.float32)
        self.data_h1 = data_h1.astype(np.float32)

        self.n_features = n_features
        self.close_idx = close_idx
        self.render_mode = render_mode
        self.action_mode = action_mode

        # --- V3: Real OHLCV for price calculations ---
        if ohlcv_m5 is not None:
            self.ohlcv_m5 = ohlcv_m5.astype(np.float32)
            self._use_real_ohlcv = True
            logger.info("V3: Using REAL OHLCV prices (shape=%s)", self.ohlcv_m5.shape)
        else:
            self._use_real_ohlcv = False
            log_ret_col = 27
            log_returns = data_m5[:, log_ret_col].astype(np.float64)
            cum_log_ret = np.cumsum(np.nan_to_num(log_returns, nan=0.0))
            cum_log_ret = np.clip(cum_log_ret, -20.0, 20.0)
            synthetic_close = 1000.0 * np.exp(cum_log_ret)
            synthetic_close = np.nan_to_num(synthetic_close, nan=1000.0, posinf=1e8, neginf=100.0)
            self.ohlcv_m5 = np.column_stack([
                synthetic_close,
                synthetic_close * 1.001,
                synthetic_close * 0.999,
                synthetic_close,
                np.ones(len(synthetic_close)),
            ]).astype(np.float32)
            logger.warning("V3: No OHLCV provided -- using SYNTHETIC prices")

        # Time alignment ratios
        self.m1_per_m5 = 5
        self.m5_per_m15 = 3
        self.m15_per_h1 = 4
        self.n_m5_bars = len(self.data_m5)

        # --- V3.3: Precompute rolling ATR for normalization ---
        self.atr_array = self._precompute_atr(period=14)

        # --- Config ---
        self.config = config
        self.initial_balance = initial_balance
        self.episode_length = min(episode_length, self.n_m5_bars - 70 - 1)
        self.pip_value = pip_value
        self.lot_value = lot_value
        self.commission_per_lot = commission_per_lot

        # Prop Firm rules
        self.max_daily_dd = config.get("max_daily_drawdown", 0.05)
        self.max_total_dd = config.get("max_total_drawdown", 0.10)
        self.max_lots = config.get("max_lots_per_trade", 10.0)
        self.max_positions = config.get("max_open_positions", 5)
        self.trading_start = config.get("trading_start_utc", 1)
        self.trading_end = config.get("trading_end_utc", 22)

        # V3.4: Fixed lot only. SL from swing points, no fixed TP.
        self.fixed_lot = config.get("stage1_fixed_lot", 0.01)
        self.swing_lookback = config.get("swing_lookback", 20)
        self.sl_buffer_mult = config.get("sl_buffer_mult", 0.1)  # Buffer beyond swing
        self.sl_fallback_mult = config.get("sl_fallback_mult", 2.0)  # Fallback if no swing

        # Legacy thresholds (continuous mode only)
        self.confidence_threshold = config.get("confidence_threshold", 0.15)
        self.m5_threshold = config.get("m5_normal_threshold", 0.50)
        self.m1_threshold = config.get("m1_sniper_threshold", 0.85)

        # --- Components ---
        self.physics = MarketPhysics(config)
        self.reward_engine = RewardEngine(config)

        # --- Spaces ---
        if self.action_mode == "discrete":
            # V3.9: BUY=0, SELL=1, HOLD=2, CLOSE=3
            self.action_space = spaces.Discrete(4)
            # Flat obs: 1 H1 (56) + 1 M15 (56) + 1 M5 (56) + 5 M1 (280) = 448-dim
            self.observation_space = spaces.Box(
                low=-10.0, high=10.0,
                shape=(448,),
                dtype=np.float32,
            )
        else:
            # Legacy continuous mode
            self.observation_space = spaces.Dict({
                "m1": spaces.Box(low=-10.0, high=10.0, shape=(self.LOOKBACK_M1, n_features), dtype=np.float32),
                "m5": spaces.Box(low=-10.0, high=10.0, shape=(self.LOOKBACK_M5, n_features), dtype=np.float32),
                "m15": spaces.Box(low=-10.0, high=10.0, shape=(self.LOOKBACK_M15, n_features), dtype=np.float32),
                "h1": spaces.Box(low=-10.0, high=10.0, shape=(self.LOOKBACK_H1, n_features), dtype=np.float32),
            })
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, 0.0, 0.5, 0.5], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 3.0, 5.0], dtype=np.float32),
                dtype=np.float32,
            )

        # --- State ---
        self._reset_state()

    def _precompute_atr(self, period: int = 14) -> np.ndarray:
        """Precompute rolling ATR from OHLCV for normalization."""
        ohlcv = self.ohlcv_m5
        highs = ohlcv[:, 1]
        lows = ohlcv[:, 2]
        closes = ohlcv[:, 3]

        # True Range
        tr = np.zeros(len(ohlcv), dtype=np.float32)
        tr[0] = highs[0] - lows[0]
        for i in range(1, len(ohlcv)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i - 1])
            tr3 = abs(lows[i] - closes[i - 1])
            tr[i] = max(tr1, tr2, tr3)

        # Rolling mean (SMA of True Range)
        atr = np.zeros(len(ohlcv), dtype=np.float32)
        for i in range(len(ohlcv)):
            start = max(0, i - period + 1)
            atr[i] = np.mean(tr[start:i + 1])

        # Prevent division by zero
        atr = np.maximum(atr, 1e-8)
        return atr

    def _reset_state(self) -> None:
        """Initialize or reset all episode state."""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.daily_peak = self.initial_balance

        self.positions: list[Position] = []
        self.trade_history: list[dict] = []
        self.current_m5_step = 0
        self.start_m5_step = 0
        self.trades_today = 0
        self.steps_since_last_trade = 0
        self.prev_unrealized = 0.0
        self._next_ticket = 1

        # V3.9: OB Touch Count state
        self._active_ob_zone = None  # (ob_high, ob_low) or None
        self._ob_touch_count = 0

    # --- Gymnasium API ---

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple:
        """Reset environment for a new episode."""
        super().reset(seed=seed)
        self._reset_state()

        min_start_m5 = max(self.LOOKBACK_M5, 15)  # Need ATR warmup
        max_start_m5 = self.n_m5_bars - self.episode_length - 1

        if max_start_m5 > min_start_m5:
            self.start_m5_step = self.np_random.integers(min_start_m5, max_start_m5)
        else:
            self.start_m5_step = min_start_m5

        self.current_m5_step = self.start_m5_step

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action) -> tuple:
        """
        Execute one M5 time step.

        Args:
            action: int (discrete mode: 0=BUY, 1=SELL, 2=HOLD, 3=CLOSE)
                    or np.ndarray (continuous mode)
        """
        self.current_m5_step += 1
        self.steps_since_last_trade += 1

        price = self._get_current_price()
        hour_utc = self._get_simulated_hour()

        if self.action_mode == "discrete":
            return self._step_discrete(int(action), price, hour_utc)
        else:
            return self._step_continuous(action, price, hour_utc)

    def _find_swing_sl(self, direction: int, m5_idx: int, entry_price: float) -> float:
        """
        V3.4: Find SL from M5 swing points.

        BUY  -> SL = recent Swing Low  - buffer
        SELL -> SL = recent Swing High + buffer

        Swing Low  = bar where low < low of bars on both sides
        Swing High = bar where high > high of bars on both sides
        """
        lookback = self.swing_lookback
        start_idx = max(1, m5_idx - lookback)
        atr = float(self.atr_array[min(m5_idx, len(self.atr_array) - 1)])
        buffer = atr * self.sl_buffer_mult

        if direction > 0:  # BUY → find Swing Low
            best_swing = None
            for i in range(m5_idx - 1, start_idx, -1):
                if i < 1 or i >= len(self.ohlcv_m5) - 1:
                    continue
                low_i = float(self.ohlcv_m5[i, 2])
                low_prev = float(self.ohlcv_m5[i - 1, 2])
                low_next = float(self.ohlcv_m5[i + 1, 2])
                if low_i < low_prev and low_i < low_next:
                    # Valid swing low, must be below entry
                    if low_i < entry_price:
                        best_swing = low_i
                        break
            if best_swing is not None:
                return best_swing - buffer
            else:
                # Fallback: ATR-based
                return entry_price - atr * self.sl_fallback_mult

        else:  # SELL → find Swing High
            best_swing = None
            for i in range(m5_idx - 1, start_idx, -1):
                if i < 1 or i >= len(self.ohlcv_m5) - 1:
                    continue
                high_i = float(self.ohlcv_m5[i, 1])
                high_prev = float(self.ohlcv_m5[i - 1, 1])
                high_next = float(self.ohlcv_m5[i + 1, 1])
                if high_i > high_prev and high_i > high_next:
                    if high_i > entry_price:
                        best_swing = high_i
                        break
            if best_swing is not None:
                return best_swing + buffer
            else:
                return entry_price + atr * self.sl_fallback_mult

    def _step_discrete(self, action: int, price: float, hour_utc: int) -> tuple:
        """V3.4: Discrete BUY/SELL/HOLD/CLOSE step."""
        # action: 0=BUY, 1=SELL, 2=HOLD, 3=CLOSE

        # --- V3.9: Trailing SL (breakeven at 1R) ---
        for pos in self.positions:
            sl_distance = abs(pos.entry_price - pos.sl_price)
            unrealized = self._calc_position_pnl(pos, price)
            # When profit >= 1R, move SL to entry (breakeven)
            r_value = sl_distance * pos.lots * self.lot_value * self.pip_value
            if r_value > 0 and unrealized >= r_value:
                if pos.direction > 0 and pos.sl_price < pos.entry_price:
                    pos.sl_price = pos.entry_price  # Trail up for LONG
                elif pos.direction < 0 and pos.sl_price > pos.entry_price:
                    pos.sl_price = pos.entry_price  # Trail down for SHORT

        # --- Check existing positions (SL only, no TP) ---
        realized_pnl = 0.0
        trade_closed = False
        manual_close = False
        manual_close_pnl = 0.0

        for pos in list(self.positions):
            hit, pnl = self._check_sl_only(pos, price)
            if hit:
                realized_pnl += pnl
                trade_closed = True
                self.balance += pnl
                self.positions.remove(pos)
                self.trade_history.append({
                    "ticket": pos.ticket, "direction": pos.direction,
                    "entry": pos.entry_price, "exit": price,
                    "pnl": pnl, "lots": pos.lots,
                    "duration": self.current_m5_step - pos.entry_step,
                    "close_reason": "SL_HIT" if pnl < 0 else "TRAILING_BE",
                })

        # --- Session close ---
        if hour_utc >= self.trading_end and self.positions:
            for pos in list(self.positions):
                pnl = self._calc_position_pnl(pos, price)
                realized_pnl += pnl
                trade_closed = True
                self.balance += pnl
                self.trade_history.append({
                    "ticket": pos.ticket, "direction": pos.direction,
                    "entry": pos.entry_price, "exit": price,
                    "pnl": pnl, "lots": pos.lots,
                    "duration": self.current_m5_step - pos.entry_step,
                    "close_reason": "SESSION_END",
                })
            self.positions.clear()

        # --- CLOSE action (V3.4) ---
        if action == 3 and self.positions:
            for pos in list(self.positions):
                pnl = self._calc_position_pnl(pos, price)
                manual_close_pnl += pnl
                realized_pnl += pnl
                trade_closed = True
                manual_close = True
                self.balance += pnl
                self.trade_history.append({
                    "ticket": pos.ticket, "direction": pos.direction,
                    "entry": pos.entry_price, "exit": price,
                    "pnl": pnl, "lots": pos.lots,
                    "duration": self.current_m5_step - pos.entry_step,
                    "close_reason": "MANUAL_CLOSE",
                })
            self.positions.clear()

        # --- Execute BUY/SELL ---
        trade_opened = False
        spread_cost = 0.0

        if action in (0, 1):  # BUY or SELL
            direction = 1 if action == 0 else -1

            if (
                len(self.positions) < self.max_positions
                and self.trading_start <= hour_utc < self.trading_end
            ):
                lot_size = self.fixed_lot

                exec_result = self.physics.execute_order(
                    price=price,
                    direction=direction,
                    lot_size=lot_size,
                    hour_utc=hour_utc,
                    pip_value=self.pip_value,
                )

                if exec_result.filled:
                    # V3.4: Auto SL from swing points, NO TP
                    sl_price = self._find_swing_sl(
                        direction, self.current_m5_step, exec_result.fill_price
                    )
                    tp_price = 0.0  # No fixed TP — bot uses CLOSE

                    commission = self.commission_per_lot * exec_result.fill_lots
                    self.balance -= commission

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

        # --- Update equity ---
        unrealized = self._total_unrealized_pnl(price)
        self.equity = self.balance + unrealized
        self.peak_equity = max(self.peak_equity, self.equity)
        self.daily_peak = max(self.daily_peak, self.equity)

        # --- Calculate reward ---
        breakdown = self.reward_engine.calculate(
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized,
            prev_unrealized_pnl=self.prev_unrealized,
            has_open_positions=len(self.positions) > 0,
            spread_cost=spread_cost,
            commission=self.commission_per_lot * (self.fixed_lot if trade_opened else 0),
            trades_today=self.trades_today,
            steps_since_last_trade=self.steps_since_last_trade,
            account_balance=self.balance,
            trade_just_opened=trade_opened,
            trade_just_closed=trade_closed,
            hour_utc=hour_utc,
            manual_close=manual_close,
            manual_close_pnl=manual_close_pnl,
        )

        self.prev_unrealized = unrealized
        reward = float(breakdown.total)
        reward = reward / 100.0
        reward = max(-10.0, min(10.0, reward))

        # --- Termination ---
        terminated = False
        truncated = False

        done, reason = self.reward_engine.is_episode_done(
            max(0, (self.daily_peak - self.equity) / max(self.daily_peak, 1)),
            max(0, (self.peak_equity - self.equity) / max(self.peak_equity, 1)),
            self.max_total_dd,
        )
        if done:
            terminated = True

        if self.current_m5_step - self.start_m5_step >= self.episode_length:
            truncated = True
        if self.current_m5_step >= self.n_m5_bars - 1:
            truncated = True

        obs = self._get_observation()
        info = self._get_info()
        info["reward_breakdown"] = breakdown.to_dict()

        return obs, reward, terminated, truncated, info

    def _step_continuous(self, action: np.ndarray, price: float, hour_utc: int) -> tuple:
        """Legacy continuous action step (V3.2 compatible). For Stage 2+."""
        # Parse 5-dim action
        confidence = float(np.clip(action[0], -1.0, 1.0))
        entry_type_raw = float(np.clip(action[1], -1.0, 1.0))
        risk_fraction = float(np.clip(action[2], 0.0, 1.0))
        sl_mult = float(np.clip(action[3], 0.5, 3.0))
        tp_mult = float(np.clip(action[4], 0.5, 5.0))

        entry_type_str = "standby"
        abs_conf = abs(confidence)
        if entry_type_raw > 0 and abs_conf >= self.m1_threshold:
            entry_type_str = "m1_sniper"
        elif entry_type_raw <= 0 and abs_conf >= self.m5_threshold:
            entry_type_str = "m5_normal"

        # --- Check SL/TP ---
        realized_pnl = 0.0
        rr_ratio = 0.0
        trade_closed = False

        for pos in list(self.positions):
            hit, pnl, rr = self._check_sl_tp(pos, price)
            if hit:
                realized_pnl += pnl
                rr_ratio = max(rr_ratio, rr)
                trade_closed = True
                self.balance += pnl
                self.positions.remove(pos)
                self.trade_history.append({
                    "ticket": pos.ticket, "direction": pos.direction,
                    "entry": pos.entry_price, "exit": price,
                    "pnl": pnl, "lots": pos.lots,
                    "duration": self.current_m5_step - pos.entry_step,
                })

        # --- Session close ---
        if hour_utc >= self.trading_end and self.positions:
            for pos in list(self.positions):
                pnl = self._calc_position_pnl(pos, price)
                realized_pnl += pnl
                trade_closed = True
                self.balance += pnl
                self.trade_history.append({
                    "ticket": pos.ticket, "direction": pos.direction,
                    "entry": pos.entry_price, "exit": price,
                    "pnl": pnl, "lots": pos.lots,
                    "duration": self.current_m5_step - pos.entry_step,
                    "close_reason": "SESSION_END",
                })
            self.positions.clear()

        # --- Action Gating ---
        trade_opened = False
        spread_cost = 0.0
        lot_size = 0.0

        if entry_type_str in ("m5_normal", "m1_sniper"):
            direction = 1 if confidence > 0 else -1

            if (
                len(self.positions) < self.max_positions
                and self.trading_start <= hour_utc < self.trading_end
            ):
                scaled_confidence = (abs_conf - self.confidence_threshold) / (
                    1.0 - self.confidence_threshold
                )
                lot_size = risk_fraction * self.max_lots * scaled_confidence
                lot_size = max(0.01, min(lot_size, self.max_lots))

                atr_pips = self._estimate_atr_pips(price)
                sl_pips = atr_pips * sl_mult
                tp_pips = atr_pips * tp_mult

                exec_result = self.physics.execute_order(
                    price=price, direction=direction,
                    lot_size=lot_size, hour_utc=hour_utc,
                    pip_value=self.pip_value,
                )

                if exec_result.filled:
                    if direction > 0:
                        sl_price = exec_result.fill_price - sl_pips * self.pip_value
                        tp_price = exec_result.fill_price + tp_pips * self.pip_value
                    else:
                        sl_price = exec_result.fill_price + sl_pips * self.pip_value
                        tp_price = exec_result.fill_price - tp_pips * self.pip_value

                    commission = self.commission_per_lot * exec_result.fill_lots
                    self.balance -= commission

                    pos = Position(
                        ticket=self._next_ticket, direction=direction,
                        entry_price=exec_result.fill_price,
                        lots=exec_result.fill_lots,
                        sl_price=sl_price, tp_price=tp_price,
                        entry_step=self.current_m5_step,
                        spread_cost=exec_result.spread_pips * self.pip_value * exec_result.fill_lots * self.lot_value,
                    )
                    self._next_ticket += 1
                    self.positions.append(pos)
                    self.trades_today += 1
                    self.steps_since_last_trade = 0
                    trade_opened = True
                    spread_cost = pos.spread_cost

        # --- Equity ---
        unrealized = self._total_unrealized_pnl(price)
        self.equity = self.balance + unrealized
        self.peak_equity = max(self.peak_equity, self.equity)
        self.daily_peak = max(self.daily_peak, self.equity)

        daily_dd = max(0, (self.daily_peak - self.equity) / self.daily_peak)
        total_dd = max(0, (self.peak_equity - self.equity) / self.peak_equity)

        # --- Reward ---
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
            abs_confidence=abs_conf,
        )

        self.prev_unrealized = unrealized
        reward = float(breakdown.total)
        reward = reward / 100.0
        reward = max(-10.0, min(10.0, reward))

        # --- Termination ---
        terminated = False
        truncated = False

        done, reason = self.reward_engine.is_episode_done(daily_dd, total_dd, self.max_total_dd)
        if done:
            terminated = True

        if self.current_m5_step - self.start_m5_step >= self.episode_length:
            truncated = True
        if self.current_m5_step >= self.n_m5_bars - 1:
            truncated = True

        obs = self._get_observation()
        info = self._get_info()
        info["reward_breakdown"] = breakdown.to_dict()

        return obs, reward, terminated, truncated, info

    # --- Observation Construction ---

    def _get_observation(self):
        """Build observation based on action_mode."""
        if self.action_mode == "discrete":
            return self._get_obs_discrete()
        else:
            return self._get_obs_continuous()

    def _compute_ob_proximity(self, data: np.ndarray, bar_idx: int, lookback: int = 20) -> float:
        """
        Compute distance (in bars, 0-1 scaled) from current bar to nearest Order Block.
        OB columns: order_block_bull=35, order_block_bear=36 in 50-dim features.
        Returns 0.0 if OB is at current bar, 1.0 if no OB found within lookback.
        """
        ob_bull_col = min(35, data.shape[1] - 1)
        ob_bear_col = min(36, data.shape[1] - 1)
        start = max(0, bar_idx - lookback)
        for dist in range(0, bar_idx - start + 1):
            idx = bar_idx - dist
            if idx < 0 or idx >= len(data):
                continue
            if data[idx, ob_bull_col] > 0.5 or data[idx, ob_bear_col] > 0.5:
                return float(dist) / max(lookback, 1)  # 0.0 = at OB, 1.0 = far
        return 1.0  # No OB found

    def _compute_volume_spike(self, data: np.ndarray, bar_idx: int, threshold: float = 2.0) -> float:
        """
        Detect volume spike: volume_ratio (col 20) > threshold.
        Returns spike magnitude (0.0 if no spike, capped at 1.0).
        """
        vol_ratio_col = min(20, data.shape[1] - 1)
        if bar_idx < 0 or bar_idx >= len(data):
            return 0.0
        vol_ratio = float(data[bar_idx, vol_ratio_col])
        if vol_ratio > threshold:
            return min((vol_ratio - threshold) / threshold, 1.0)
        return 0.0

    def _compute_synthetic_spread(self, m5_idx: int) -> float:
        """
        V3.7: Synthetic spread based on trading session.
        Asia(low vol) = wide spread, London/NY(high vol) = tight spread.
        Returns normalized spread: 0.0 (tight) to 1.0 (wide).
        """
        # M1 bars per day ≈ 1440, M5 bars per day ≈ 288
        bar_in_day = m5_idx % 288  # Position within trading day
        hour = (bar_in_day * 5) / 60  # Convert to hours (0-24)
        # Session breakdown (UTC):
        # Asia:   0-8  (wide spread)
        # London: 8-16 (tight spread)
        # NY:     13-21 (tight spread)
        # Off:    21-24 (wide spread)
        if 8 <= hour < 21:  # London + NY overlap
            return 0.1 + 0.1 * abs(hour - 14.5) / 6.5  # Tightest at London/NY overlap
        else:
            return 0.7 + 0.3 * min(abs(hour - 4), abs(24 - hour + 4)) / 4  # Wide in Asia

    def _compute_session_phase(self, m5_idx: int) -> float:
        """
        V3.7: Session phase encoding.
        0.0 = Asia (low vol), 0.5 = London, 1.0 = NY, 0.25 = overlap.
        """
        bar_in_day = m5_idx % 288
        hour = (bar_in_day * 5) / 60
        if 0 <= hour < 8:
            return 0.0   # Asia
        elif 8 <= hour < 13:
            return 0.5   # London
        elif 13 <= hour < 17:
            return 0.75  # London/NY overlap (best liquidity)
        elif 17 <= hour < 21:
            return 1.0   # NY solo
        else:
            return 0.1   # Off-hours

    def _compute_ob_touch_count(self, m5_idx: int) -> float:
        """
        V3.9: Track how many times M5 candles have tested the active Order Block.
        Returns touch_count / 5.0 (normalized 0-1, clipped).
        """
        if not self._use_real_ohlcv or m5_idx >= len(self.ohlcv_m5):
            return 0.0

        ob_bull_col = min(35, self.data_m5.shape[1] - 1)
        ob_bear_col = min(36, self.data_m5.shape[1] - 1)

        # Check if a new OB has formed in recent bars
        lookback = 20
        new_ob_zone = None
        for dist in range(0, min(lookback, m5_idx)):
            idx = m5_idx - dist
            if idx < 0 or idx >= len(self.data_m5):
                continue
            if self.data_m5[idx, ob_bull_col] > 0.5 or self.data_m5[idx, ob_bear_col] > 0.5:
                # Found the nearest OB — use that bar's OHLCV as OB zone
                ob_high = float(self.ohlcv_m5[idx, 1])  # High
                ob_low = float(self.ohlcv_m5[idx, 2])    # Low
                new_ob_zone = (ob_high, ob_low)
                break

        if new_ob_zone is None:
            self._active_ob_zone = None
            self._ob_touch_count = 0
            return 0.0

        # If OB changed, reset counter
        if self._active_ob_zone != new_ob_zone:
            self._active_ob_zone = new_ob_zone
            self._ob_touch_count = 0

        # Check if current M5 bar touches the OB zone (wick or close)
        ob_high, ob_low = self._active_ob_zone
        cur_high = float(self.ohlcv_m5[m5_idx, 1])
        cur_low = float(self.ohlcv_m5[m5_idx, 2])
        cur_close = float(self.ohlcv_m5[m5_idx, 3])

        # Check if OB is broken (close through the entire zone)
        if cur_close > ob_high + (ob_high - ob_low) * 0.5:
            self._active_ob_zone = None
            self._ob_touch_count = 0
            return 0.0
        if cur_close < ob_low - (ob_high - ob_low) * 0.5:
            self._active_ob_zone = None
            self._ob_touch_count = 0
            return 0.0

        # Touch = wick into zone
        if cur_low <= ob_high and cur_high >= ob_low:
            self._ob_touch_count += 1

        return min(self._ob_touch_count / 5.0, 1.0)

    def _compute_m15_choppiness(self, m5_idx: int, period: int = 14) -> float:
        """
        V3.9: M15 Choppiness Index (CHOP-14).
        CHOP = 100 * log10(sum(TR_14) / (MaxHigh_14 - MinLow_14)) / log10(14)
        Returns normalized value in [0, 1] (raw CHOP / 100).
        """
        if not self._use_real_ohlcv:
            return 0.5  # Neutral default

        # Convert M5 index to approximate M15 index
        m15_idx = m5_idx // self.m5_per_m15
        # We need M15 OHLCV — derive from M5 OHLCV by grouping 3 bars
        m15_start = max(0, (m15_idx - period) * self.m5_per_m15)
        m15_end = min(len(self.ohlcv_m5), (m15_idx + 1) * self.m5_per_m15)

        if m15_end - m15_start < self.m5_per_m15 * period:
            return 0.5  # Not enough data

        # Build M15 OHLCV from M5 groups
        m15_highs = []
        m15_lows = []
        m15_closes = []
        m15_prev_closes = []

        for g in range(period):
            g_start = m15_start + g * self.m5_per_m15
            g_end = min(g_start + self.m5_per_m15, len(self.ohlcv_m5))
            if g_end <= g_start:
                return 0.5
            group = self.ohlcv_m5[g_start:g_end]
            m15_highs.append(float(np.max(group[:, 1])))
            m15_lows.append(float(np.min(group[:, 2])))
            m15_closes.append(float(group[-1, 3]))
            # Previous close for True Range
            if g > 0:
                prev_g_end = g_start
                prev_g_start = max(0, prev_g_end - self.m5_per_m15)
                m15_prev_closes.append(float(self.ohlcv_m5[prev_g_end - 1, 3]))
            else:
                m15_prev_closes.append(m15_closes[0])

        # Compute True Range for each M15 bar
        total_tr = 0.0
        for i in range(period):
            tr1 = m15_highs[i] - m15_lows[i]
            tr2 = abs(m15_highs[i] - m15_prev_closes[i])
            tr3 = abs(m15_lows[i] - m15_prev_closes[i])
            total_tr += max(tr1, tr2, tr3)

        max_high = max(m15_highs)
        min_low = min(m15_lows)
        range_hl = max_high - min_low

        if range_hl < 1e-10 or total_tr < 1e-10:
            return 1.0  # Perfectly choppy

        chop = 100.0 * np.log10(total_tr / range_hl) / np.log10(period)
        chop = np.clip(chop, 0.0, 100.0)
        return float(chop / 100.0)  # Normalize to [0, 1]

    def _enrich_bar(self, bar: np.ndarray, data: np.ndarray, bar_idx: int, m5_idx: int = 0) -> np.ndarray:
        """V3.9: Append 6 features to a bar (50→56)."""
        ob_prox = self._compute_ob_proximity(data, bar_idx)
        vol_spike = self._compute_volume_spike(data, bar_idx)
        spread = self._compute_synthetic_spread(m5_idx)
        session = self._compute_session_phase(m5_idx)
        ob_touch = self._compute_ob_touch_count(m5_idx)
        choppiness = self._compute_m15_choppiness(m5_idx)
        return np.append(bar, [ob_prox, vol_spike, spread, session, ob_touch, choppiness]).astype(np.float32)

    def _get_obs_discrete(self) -> np.ndarray:
        """
        V3.9: Flat 448-dim observation.
        = 1 H1 bar (56) + 1 M15 bar (56) + 1 M5 bar (56) + 5 M1 bars (280)

        H1: macro trend | M15: structure context | M5: entry zone | M1: sniper precision
        +6 per bar: OB proximity + Volume spike + Spread + Session + OB Touch + Choppiness
        All features normalized by ATR.
        """
        m5_idx = min(self.current_m5_step, self.n_m5_bars - 1)
        atr = self.atr_array[m5_idx]

        # H1: current bar (50+4=54-dim) — macro trend (BOS/CHoCH)
        h1_idx = min(m5_idx // (self.m5_per_m15 * self.m15_per_h1), len(self.data_h1) - 1)
        h1_bar = self.data_h1[h1_idx].copy()
        h1_bar[:10] = h1_bar[:10] / atr
        h1_bar = self._enrich_bar(h1_bar, self.data_h1, h1_idx, m5_idx)

        # M15: current bar (50+4=54-dim) — structure context (OB/FVG)
        m15_idx = min(m5_idx // self.m5_per_m15, len(self.data_m15) - 1)
        m15_bar = self.data_m15[m15_idx].copy()
        m15_bar[:10] = m15_bar[:10] / atr
        m15_bar = self._enrich_bar(m15_bar, self.data_m15, m15_idx, m5_idx)

        # M5: current bar (50+4=54-dim) — entry zone
        m5_bar = self.data_m5[m5_idx].copy()
        m5_bar[:10] = m5_bar[:10] / atr
        m5_bar = self._enrich_bar(m5_bar, self.data_m5, m5_idx, m5_idx)

        # M1: last 5 bars (5×54=270-dim) — sniper precision
        m1_end_idx = min(m5_idx * self.m1_per_m5 + self.m1_per_m5 - 1, len(self.data_m1) - 1)
        m1_start_idx = max(0, m1_end_idx - 4)  # 5 bars
        m1_enriched = []
        for i in range(m1_start_idx, m1_end_idx + 1):
            bar = self.data_m1[i].copy()
            bar[:10] = bar[:10] / atr
            bar = self._enrich_bar(bar, self.data_m1, i, m5_idx)
            m1_enriched.append(bar)

        # Zero-pad if fewer than 5 M1 bars
        while len(m1_enriched) < 5:
            m1_enriched.insert(0, np.zeros(56, dtype=np.float32))

        m1_flat = np.concatenate(m1_enriched)

        # Flatten: H1 (56) + M15 (56) + M5 (56) + M1 (280) = 448
        obs = np.concatenate([h1_bar, m15_bar, m5_bar, m1_flat]).astype(np.float32)

        # Clip to prevent extreme values
        obs = np.clip(obs, -10.0, 10.0)
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

        return obs

    def _get_obs_continuous(self) -> dict[str, np.ndarray]:
        """Legacy dict observation for continuous mode."""
        m5_idx = min(self.current_m5_step, self.n_m5_bars - 1)

        m5_obs = self._get_window(self.data_m5, m5_idx, self.LOOKBACK_M5)
        m1_idx = min(m5_idx * self.m1_per_m5, len(self.data_m1) - 1)
        m1_obs = self._get_window(self.data_m1, m1_idx, self.LOOKBACK_M1)
        m15_idx = min(m5_idx // self.m5_per_m15, len(self.data_m15) - 1)
        m15_obs = self._get_window(self.data_m15, m15_idx, self.LOOKBACK_M15)
        h1_idx = min(m5_idx // (self.m5_per_m15 * self.m15_per_h1), len(self.data_h1) - 1)
        h1_obs = self._get_window(self.data_h1, h1_idx, self.LOOKBACK_H1)

        return {"m1": m1_obs, "m5": m5_obs, "m15": m15_obs, "h1": h1_obs}

    def _get_window(
        self, data: np.ndarray, end_idx: int, window_size: int,
    ) -> np.ndarray:
        """Extract a fixed-size window ending at end_idx."""
        start_idx = max(0, end_idx - window_size + 1)
        window = data[start_idx:end_idx + 1]
        if len(window) < window_size:
            pad_size = window_size - len(window)
            padding = np.zeros((pad_size, data.shape[1]), dtype=np.float32)
            window = np.vstack([padding, window])
        return window.copy()

    # --- Internal Helpers ---

    def _get_current_price(self) -> float:
        idx = min(self.current_m5_step, len(self.ohlcv_m5) - 1)
        return float(self.ohlcv_m5[idx, 3])

    def _get_current_high(self) -> float:
        idx = min(self.current_m5_step, len(self.ohlcv_m5) - 1)
        return float(self.ohlcv_m5[idx, 1])

    def _get_current_low(self) -> float:
        idx = min(self.current_m5_step, len(self.ohlcv_m5) - 1)
        return float(self.ohlcv_m5[idx, 2])

    def _get_simulated_hour(self) -> int:
        minutes = (self.current_m5_step * 5) % (24 * 60)
        return minutes // 60

    def _estimate_atr_pips(self, price: float) -> float:
        """Get ATR in pips from precomputed array."""
        idx = min(self.current_m5_step, len(self.atr_array) - 1)
        atr_price = float(self.atr_array[idx])
        atr_pips = atr_price / self.pip_value
        return max(atr_pips, 5.0)

    def _total_unrealized_pnl(self, current_price: float) -> float:
        total = 0.0
        for pos in self.positions:
            total += self._calc_position_pnl(pos, current_price)
        return total

    def _calc_position_pnl(self, pos: Position, current_price: float) -> float:
        price_diff = (current_price - pos.entry_price) * pos.direction
        return price_diff * pos.lots * self.lot_value

    def _check_sl_only(
        self, pos: Position, current_price: float,
    ) -> tuple[bool, float]:
        """V3.4: Check SL only. No TP — bot uses CLOSE action."""
        if pos.direction > 0:  # LONG
            if current_price <= pos.sl_price:
                pnl = (pos.sl_price - pos.entry_price) * pos.lots * self.lot_value
                return True, pnl
        else:  # SHORT
            if current_price >= pos.sl_price:
                pnl = (pos.entry_price - pos.sl_price) * pos.lots * self.lot_value
                return True, pnl
        return False, 0.0

    def _get_info(self) -> dict[str, Any]:
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


# Legacy alias
PropFirmTradingEnv = MultiTFTradingEnv
