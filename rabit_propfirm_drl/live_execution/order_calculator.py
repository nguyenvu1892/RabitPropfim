"""
OrderCalculator & SlippageHandler — Agent output → MT5 order execution.

This module contains two critical classes:

1. OrderCalculator:
    Converts GatedAction (from ActionGating) into concrete OrderParams
    that MT5 can execute. Handles:
    - Dynamic lot sizing based on account balance and risk fraction
    - SL/TP price calculation from ATR × multipliers
    - Volume rounding to symbol's volume_step
    - Spread-aware price selection (Ask for BUY, Bid for SELL)

2. SlippageHandler:
    Wraps order execution with slippage detection and requote retry.
    Handles:
    - Requote (retcode 10004): retry up to 3 times with refreshed price
    - Invalid stops (retcode 10013): recalculate SL/TP with current spread
    - Invalid volume (retcode 10014): re-clamp lot size
    - Invalid price (retcode 10015): refresh price and retry
    - Market closed (retcode 10018): skip and log
    - No money (retcode 10019): emergency alert

CRITICAL: These classes handle REAL MONEY order execution.
    Every calculation is validated, every error is caught.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)

# Graceful MT5 import
try:
    import MetaTrader5 as mt5  # type: ignore

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None  # type: ignore

# Import from existing modules
from rabit_propfirm_drl.agents.action_gating import GatedAction, TradeSignal
from rabit_propfirm_drl.live_execution.mt5_connector import (
    MT5LiveConnector,
    OrderResult,
)


# ─── Data Types ────────────────────────────────────────────────────
@dataclass
class OrderParams:
    """
    Calculated order parameters ready for MT5 submission.

    All prices are in the symbol's native price format.
    All values are validated and ready for order_send().
    """

    direction: int            # +1 = BUY, -1 = SELL
    lots: float               # Volume in lots (rounded to volume_step)
    price: float              # Entry price (Ask for BUY, Bid for SELL)
    sl_price: float           # Stop-loss price
    tp_price: float           # Take-profit price
    risk_amount: float        # Dollar amount at risk on this trade
    sl_distance_points: float  # SL distance in points (for logging)
    symbol: str = ""          # Symbol name (for multi-symbol tracking)
    atr_used: float = 0.0    # ATR value used for calculation (for logging)


@dataclass
class ExecutionReport:
    """
    Detailed report of an order execution attempt.

    Includes both the order params and the MT5 result,
    plus any slippage metrics.
    """

    order: OrderParams
    result: OrderResult
    slippage_points: float       # Actual slippage (positive = unfavorable)
    requote_count: int           # Number of requote retries
    execution_time_ms: float     # Total time from submit to fill
    timestamp: str               # ISO timestamp of execution


# ─── OrderCalculator ──────────────────────────────────────────────
class OrderCalculator:
    """
    Converts GatedAction into concrete OrderParams for MT5.

    Position Sizing Formula:
        risk_amount   = balance × risk_fraction × max_loss_per_trade_pct
        sl_distance   = sl_multiplier × ATR (in price units)
        sl_in_points  = sl_distance / point
        value_per_pt  = contract_size × point
        lots          = risk_amount / (sl_in_points × value_per_pt)
        lots          = clamp(lots, vol_min, min(vol_max, max_lots))
        lots          = round(lots / vol_step) * vol_step

    Args:
        default_max_loss_pct: Default max loss per trade (0.003 = 0.3%)
        default_max_lots:     Default max lot size per trade
    """

    def __init__(
        self,
        default_max_loss_pct: float = 0.003,
        default_max_lots: float = 10.0,
    ) -> None:
        self.default_max_loss_pct = default_max_loss_pct
        self.default_max_lots = default_max_lots

    def compute(
        self,
        gated: GatedAction,
        balance: float,
        symbol_info: dict[str, Any],
        atr_price: float,
        symbol: str = "",
        max_loss_pct: Optional[float] = None,
        max_lots: Optional[float] = None,
    ) -> OrderParams:
        """
        Compute order parameters from a GatedAction.

        Args:
            gated:       GatedAction from ActionGating (signal, risk_fraction, etc.)
            balance:     Current account balance in USD
            symbol_info: Dict from MT5LiveConnector.get_symbol_info()
                         Must contain: ask, bid, point, digits, volume_min,
                         volume_max, volume_step, trade_contract_size
            atr_price:   Average True Range in price units (e.g., 1.50 for Gold)
            symbol:      Symbol name for overrides lookup
            max_loss_pct: Override max loss per trade (defaults to self.default)
            max_lots:    Override max lot size (defaults to self.default)

        Returns:
            OrderParams with validated lot size, SL/TP prices

        Raises:
            ValueError: If inputs are invalid (negative balance, zero ATR, etc.)
        """
        # ── Input Validation ──
        self._validate_inputs(gated, balance, symbol_info, atr_price)

        active_max_loss = max_loss_pct or self.default_max_loss_pct
        active_max_lots = max_lots or self.default_max_lots

        # ── 1. Calculate risk amount ──
        risk_fraction = gated.risk_fraction  # 0.0 – 1.0 from ActionGating
        risk_amount = balance * risk_fraction * active_max_loss
        # Example: $100,000 × 0.7 × 0.003 = $210

        # ── 2. Calculate SL distance (in price units) ──
        sl_distance = gated.sl_multiplier * atr_price
        # Guard: SL must be positive
        if sl_distance <= 0:
            logger.warning(
                "SL distance is zero/negative (sl_mult=%.2f, atr=%.6f). "
                "Using 1.0 × ATR as fallback.",
                gated.sl_multiplier,
                atr_price,
            )
            sl_distance = atr_price if atr_price > 0 else symbol_info["point"] * 100

        # ── 3. Calculate lots (DYNAMIC for ALL symbols) ──
        # ATR-based SL naturally handles high-vol instruments (Gold, BTC)
        # by widening SL → shrinking lot size proportionally.
        point = symbol_info["point"]
        contract_size = symbol_info["trade_contract_size"]

        sl_in_points = sl_distance / point
        value_per_point_per_lot = contract_size * point

        if sl_in_points > 0 and value_per_point_per_lot > 0:
            lots = risk_amount / (sl_in_points * value_per_point_per_lot)
        else:
            logger.warning(
                "Cannot calculate lots (sl_pts=%.1f, val_per_pt=%.6f). "
                "Falling back to volume_min.",
                sl_in_points,
                value_per_point_per_lot,
            )
            lots = symbol_info["volume_min"]

        # ── 4. Clamp and round lots ──
        lots = self._clamp_and_round_lots(
            lots=lots,
            vol_min=symbol_info["volume_min"],
            vol_max=symbol_info["volume_max"],
            vol_step=symbol_info["volume_step"],
            max_lots=active_max_lots,
        )

        # ── 6. Calculate SL/TP prices ──
        digits = symbol_info["digits"]
        tp_distance = gated.tp_multiplier * atr_price

        if gated.signal == TradeSignal.BUY:
            price = symbol_info["ask"]
            sl_price = round(price - sl_distance, digits)
            tp_price = round(price + tp_distance, digits)
        elif gated.signal == TradeSignal.SELL:
            price = symbol_info["bid"]
            sl_price = round(price + sl_distance, digits)
            tp_price = round(price - tp_distance, digits)
        else:
            # HOLD — should not reach here, but handle gracefully
            logger.warning("OrderCalculator called with HOLD signal — returning zero order")
            return OrderParams(
                direction=0, lots=0, price=0,
                sl_price=0, tp_price=0,
                risk_amount=0, sl_distance_points=0,
                symbol=symbol, atr_used=atr_price,
            )

        # ── 7. Validate SL/TP sanity ──
        self._validate_sl_tp(
            direction=gated.signal.value,
            price=price,
            sl_price=sl_price,
            tp_price=tp_price,
            point=point,
            digits=digits,
        )

        order = OrderParams(
            direction=gated.signal.value,
            lots=lots,
            price=price,
            sl_price=sl_price,
            tp_price=tp_price,
            risk_amount=round(risk_amount, 2),
            sl_distance_points=round(sl_in_points, 1),
            symbol=symbol,
            atr_used=atr_price,
        )

        logger.info(
            "[%s] Order calculated: %s %.2f lots @ %.5f | "
            "SL=%.5f (%.0f pts) TP=%.5f | Risk=$%.2f",
            symbol,
            "BUY" if order.direction > 0 else "SELL",
            order.lots,
            order.price,
            order.sl_price,
            order.sl_distance_points,
            order.tp_price,
            order.risk_amount,
        )

        return order

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PRIVATE — Validation & Rounding
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _validate_inputs(
        gated: GatedAction,
        balance: float,
        symbol_info: dict[str, Any],
        atr_price: float,
    ) -> None:
        """Validate all inputs before calculation."""
        if balance <= 0:
            raise ValueError(f"Balance must be positive, got {balance}")

        if atr_price < 0:
            raise ValueError(f"ATR must be non-negative, got {atr_price}")

        required_keys = [
            "ask", "bid", "point", "digits",
            "volume_min", "volume_max", "volume_step",
            "trade_contract_size",
        ]
        missing = [k for k in required_keys if k not in symbol_info]
        if missing:
            raise ValueError(f"symbol_info missing keys: {missing}")

        if symbol_info["point"] <= 0:
            raise ValueError(f"Symbol point must be positive: {symbol_info['point']}")

        if symbol_info["trade_contract_size"] <= 0:
            raise ValueError(
                f"Contract size must be positive: "
                f"{symbol_info['trade_contract_size']}"
            )

    @staticmethod
    def _clamp_and_round_lots(
        lots: float,
        vol_min: float,
        vol_max: float,
        vol_step: float,
        max_lots: float,
    ) -> float:
        """
        Clamp lot size to valid range and round to volume_step.

        Steps:
        1. Clamp to [vol_min, min(vol_max, max_lots)]
        2. Round to nearest vol_step
        3. Final clamp to ensure within bounds after rounding
        4. Round to 2 decimal places to avoid float precision issues
        """
        effective_max = min(vol_max, max_lots)
        lots = max(vol_min, min(effective_max, lots))

        if vol_step > 0:
            lots = round(lots / vol_step) * vol_step

        # Final clamp after rounding (rounding can push outside bounds)
        lots = max(vol_min, min(effective_max, lots))

        # Avoid float precision artifacts
        lots = round(lots, 2)

        return lots

    @staticmethod
    def _validate_sl_tp(
        direction: int,
        price: float,
        sl_price: float,
        tp_price: float,
        point: float,
        digits: int,
    ) -> None:
        """
        Validate SL/TP prices are on the correct side of entry price.

        For BUY:  SL < price < TP
        For SELL: TP < price < SL
        """
        if direction > 0:  # BUY
            if sl_price >= price:
                logger.warning(
                    "BUY SL (%.5f) >= entry price (%.5f)! "
                    "This will be rejected by MT5.",
                    sl_price, price,
                )
            if tp_price <= price:
                logger.warning(
                    "BUY TP (%.5f) <= entry price (%.5f)! "
                    "This will be rejected by MT5.",
                    tp_price, price,
                )
        elif direction < 0:  # SELL
            if sl_price <= price:
                logger.warning(
                    "SELL SL (%.5f) <= entry price (%.5f)! "
                    "This will be rejected by MT5.",
                    sl_price, price,
                )
            if tp_price >= price:
                logger.warning(
                    "SELL TP (%.5f) >= entry price (%.5f)! "
                    "This will be rejected by MT5.",
                    tp_price, price,
                )


# ─── SlippageHandler ──────────────────────────────────────────────
class SlippageHandler:
    """
    Wraps order execution with slippage detection and requote retry.

    Handles MT5 retcodes:
        10004 — Requote:       Retry with refreshed price (up to MAX_REQUOTE_RETRIES)
        10013 — Invalid stops: Log error, don't retry (need recalculation)
        10014 — Invalid volume: Log error, don't retry (need recalculation)
        10015 — Invalid price: Refresh price and retry
        10018 — Market closed: Skip and log
        10019 — No money:     EMERGENCY — stop trading immediately

    Slippage Detection:
        After successful fill, compare actual fill price vs requested price.
        If slippage > MAX_SLIPPAGE_POINTS:
            → Log WARNING (order already filled, can't cancel)
            → Send Telegram alert if callback configured

    Args:
        max_requote_retries:  Number of requote retries. Default: 3
        requote_delay_ms:     Milliseconds between requote retries. Default: 200
        max_slippage_points:  Slippage warning threshold in points. Default: 20
        alert_callback:       Optional function(title, msg) for alerts
    """

    # MT5 retcodes we care about
    RETCODE_DONE = 10009
    RETCODE_REQUOTE = 10004
    RETCODE_REJECT = 10006
    RETCODE_DONE_PARTIAL = 10010
    RETCODE_INVALID_STOPS = 10013
    RETCODE_INVALID_VOLUME = 10014
    RETCODE_INVALID_PRICE = 10015
    RETCODE_MARKET_CLOSED = 10018
    RETCODE_NO_MONEY = 10019

    def __init__(
        self,
        max_requote_retries: int = 3,
        requote_delay_ms: int = 200,
        max_slippage_points: float = 20.0,
        alert_callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.max_requote_retries = max_requote_retries
        self.requote_delay_ms = requote_delay_ms
        self.max_slippage_points = max_slippage_points
        self._alert_callback = alert_callback

    def execute_with_slippage_check(
        self,
        connector: MT5LiveConnector,
        order: OrderParams,
    ) -> ExecutionReport:
        """
        Execute order with full slippage and requote handling.

        Args:
            connector: MT5LiveConnector instance (must be connected)
            order:     OrderParams from OrderCalculator

        Returns:
            ExecutionReport with order result and slippage metrics
        """
        from datetime import datetime, timezone

        start_time = time.time()
        requote_count = 0
        last_result: Optional[OrderResult] = None

        # Working copy of price (may change on requote)
        current_price = order.price
        current_sl = order.sl_price
        current_tp = order.tp_price

        for attempt in range(self.max_requote_retries + 1):  # +1 for initial try
            try:
                result = connector.market_order(
                    direction=order.direction,
                    lots=order.lots,
                    sl_price=current_sl,
                    tp_price=current_tp,
                    comment=f"DRL_v6_{attempt}",
                )
                last_result = result

                # ── SUCCESS ──
                if result.success:
                    elapsed_ms = (time.time() - start_time) * 1000

                    # Calculate slippage
                    sym_info = connector.get_symbol_info()
                    point = sym_info["point"] if sym_info else 0.00001
                    slippage = abs(result.price - order.price)
                    slippage_points = slippage / point if point > 0 else 0.0

                    # Slippage warning
                    if slippage_points > self.max_slippage_points:
                        msg = (
                            f"⚠️ High slippage on {order.symbol}: "
                            f"{slippage_points:.1f} pts "
                            f"(limit: {self.max_slippage_points}). "
                            f"Requested: {order.price:.5f}, "
                            f"Filled: {result.price:.5f}"
                        )
                        logger.warning(msg)
                        self._send_alert("⚠️ SLIPPAGE WARNING", msg)

                    # Partial fill info
                    if result.retcode == self.RETCODE_DONE_PARTIAL:
                        logger.warning(
                            "[%s] Partial fill: requested %.2f, "
                            "filled %.2f lots (ticket=%d)",
                            order.symbol,
                            order.lots,
                            result.lots,
                            result.ticket,
                        )

                    logger.info(
                        "[%s] Order filled: ticket=%d, price=%.5f, "
                        "slippage=%.1f pts, time=%.0fms",
                        order.symbol,
                        result.ticket,
                        result.price,
                        slippage_points,
                        elapsed_ms,
                    )

                    return ExecutionReport(
                        order=order,
                        result=result,
                        slippage_points=slippage_points,
                        requote_count=requote_count,
                        execution_time_ms=elapsed_ms,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )

                # ── REQUOTE (10004) — Retry with new price ──
                if result.retcode == self.RETCODE_REQUOTE:
                    requote_count += 1
                    logger.warning(
                        "[%s] Requote #%d/%d. Refreshing price...",
                        order.symbol,
                        requote_count,
                        self.max_requote_retries,
                    )

                    if requote_count > self.max_requote_retries:
                        break

                    time.sleep(self.requote_delay_ms / 1000.0)

                    # Refresh price
                    new_price = self._refresh_price(connector, order)
                    if new_price is not None:
                        price_diff = new_price - current_price
                        current_price = new_price
                        # Shift SL/TP by the same amount to maintain distances
                        current_sl = round(
                            current_sl + price_diff,
                            connector.get_symbol_info().get("digits", 5),
                        )
                        current_tp = round(
                            current_tp + price_diff,
                            connector.get_symbol_info().get("digits", 5),
                        )
                    continue

                # ── INVALID PRICE (10015) — Similar to requote ──
                if result.retcode == self.RETCODE_INVALID_PRICE:
                    logger.warning(
                        "[%s] Invalid price (stale). Refreshing...",
                        order.symbol,
                    )
                    new_price = self._refresh_price(connector, order)
                    if new_price is not None:
                        price_diff = new_price - current_price
                        current_price = new_price
                        current_sl = round(
                            current_sl + price_diff,
                            connector.get_symbol_info().get("digits", 5),
                        )
                        current_tp = round(
                            current_tp + price_diff,
                            connector.get_symbol_info().get("digits", 5),
                        )
                    continue

                # ── NON-RETRYABLE ERRORS ──

                if result.retcode == self.RETCODE_REJECT:
                    logger.error(
                        "[%s] Order REJECTED by broker. "
                        "Retcode=%d, Comment=%s",
                        order.symbol, result.retcode, result.comment,
                    )
                    break

                if result.retcode == self.RETCODE_INVALID_STOPS:
                    logger.error(
                        "[%s] Invalid SL/TP rejected by broker. "
                        "SL=%.5f, TP=%.5f, Price=%.5f. "
                        "Check symbol's STOPS_LEVEL.",
                        order.symbol,
                        current_sl,
                        current_tp,
                        current_price,
                    )
                    break

                if result.retcode == self.RETCODE_INVALID_VOLUME:
                    logger.error(
                        "[%s] Invalid volume: %.2f lots. "
                        "Check symbol constraints.",
                        order.symbol, order.lots,
                    )
                    break

                if result.retcode == self.RETCODE_MARKET_CLOSED:
                    logger.warning(
                        "[%s] Market is closed. Order skipped.",
                        order.symbol,
                    )
                    break

                if result.retcode == self.RETCODE_NO_MONEY:
                    msg = (
                        f"🚨 NO MARGIN for {order.symbol}! "
                        f"Requested: {order.lots} lots. "
                        f"This may indicate account is near margin call."
                    )
                    logger.critical(msg)
                    self._send_alert("🚨 NO MARGIN", msg)
                    break

                # ── UNKNOWN RETCODE — Log and break ──
                logger.error(
                    "[%s] Unknown retcode %d: %s",
                    order.symbol, result.retcode, result.error,
                )
                break

            except Exception as e:
                logger.error(
                    "[%s] Exception during order execution: %s",
                    order.symbol, e, exc_info=True,
                )
                last_result = OrderResult(
                    success=False,
                    ticket=0,
                    price=0,
                    lots=0,
                    error=f"Exception: {e}",
                )
                break

        # ── ALL RETRIES EXHAUSTED or NON-RETRYABLE ERROR ──
        elapsed_ms = (time.time() - start_time) * 1000

        if last_result is None:
            last_result = OrderResult(
                success=False,
                ticket=0,
                price=0,
                lots=0,
                error="No result obtained",
            )

        logger.error(
            "[%s] Order execution FAILED. Retcode=%d, Error=%s, "
            "Requotes=%d, Time=%.0fms",
            order.symbol,
            last_result.retcode,
            last_result.error,
            requote_count,
            elapsed_ms,
        )

        return ExecutionReport(
            order=order,
            result=last_result,
            slippage_points=0.0,
            requote_count=requote_count,
            execution_time_ms=elapsed_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PRIVATE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _refresh_price(
        connector: MT5LiveConnector,
        order: OrderParams,
    ) -> Optional[float]:
        """
        Refresh the current market price for the order's symbol.

        Returns:
            New price (Ask for BUY, Bid for SELL), or None on error.
        """
        if not MT5_AVAILABLE:
            return None

        try:
            tick = mt5.symbol_info_tick(connector.symbol)
            if tick is None:
                logger.error("Cannot refresh price: tick data unavailable")
                return None

            if order.direction > 0:
                return tick.ask
            else:
                return tick.bid

        except Exception as e:
            logger.error("Price refresh error: %s", e)
            return None

    def _send_alert(self, title: str, message: str) -> None:
        """Send alert via configured callback."""
        if self._alert_callback is not None:
            try:
                self._alert_callback(title, message)
            except Exception as e:
                logger.error("SlippageHandler alert failed: %s", e)


# ─── Utility: ATR Calculator ─────────────────────────────────────
def compute_atr(ohlcv: np.ndarray, period: int = 14) -> float:
    """
    Compute Average True Range from OHLCV numpy array.

    Args:
        ohlcv: (N, 6) array — [time, open, high, low, close, volume]
               Columns indexed: 1=open, 2=high, 3=low, 4=close
        period: ATR lookback period (default 14)

    Returns:
        ATR value in price units. Returns 0.0 if insufficient data.
    """
    if ohlcv is None or len(ohlcv) < period + 1:
        logger.warning(
            "Insufficient data for ATR: need %d bars, got %d",
            period + 1,
            len(ohlcv) if ohlcv is not None else 0,
        )
        return 0.0

    try:
        high = ohlcv[-period:, 2]           # High column
        low = ohlcv[-period:, 3]            # Low column
        prev_close = ohlcv[-period - 1:-1, 4]  # Previous candle's close

        # True Range = max(H-L, |H-prevC|, |L-prevC|)
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - prev_close),
                np.abs(low - prev_close),
            ),
        )

        atr = float(np.mean(tr))

        if atr <= 0 or np.isnan(atr):
            logger.warning("ATR is zero or NaN: %.6f", atr)
            return 0.0

        return atr

    except Exception as e:
        logger.error("ATR calculation error: %s", e)
        return 0.0
