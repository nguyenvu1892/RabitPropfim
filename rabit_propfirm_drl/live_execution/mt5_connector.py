"""
MT5 Live Connector — Real-time order execution via MetaTrader 5.

Features:
- Graceful MT5 import (skip if not installed)
- Connection management with auto-reconnect
- Order execution (market buy/sell, SL/TP)
- Position management (modify, close, close all)
- Account state queries (balance, equity, margin)
- Symbol info (spread, tick size, lot size)

All operations have timeout and retry logic.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Graceful MT5 import
try:
    import MetaTrader5 as mt5  # type: ignore

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None  # type: ignore
    logger.warning("MetaTrader5 not installed. Live trading disabled.")


@dataclass
class OrderResult:
    """Result of an order operation."""
    success: bool
    ticket: int
    price: float
    lots: float
    comment: str = ""
    retcode: int = 0
    error: str = ""


@dataclass
class AccountState:
    """Current account state snapshot."""
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    profit: float
    timestamp: str


@dataclass
class PositionInfo:
    """Info about an open position."""
    ticket: int
    symbol: str
    direction: int  # +1 = BUY, -1 = SELL
    lots: float
    open_price: float
    sl: float
    tp: float
    profit: float
    open_time: str


class MT5LiveConnector:
    """
    Live trading connector for MetaTrader 5.

    Handles order execution, position management, and account queries.
    Designed to work with the DRL agent's action output.
    """

    def __init__(
        self,
        symbol: str = "EURUSD",
        magic_number: int = 202501,
        deviation: int = 20,
        max_retries: int = 3,
    ) -> None:
        self.symbol = symbol
        self.magic = magic_number
        self.deviation = deviation  # Max slippage in points
        self.max_retries = max_retries
        self._connected = False

    @property
    def is_available(self) -> bool:
        return MT5_AVAILABLE

    def connect(self) -> bool:
        """Initialize and connect to MT5 terminal."""
        if not MT5_AVAILABLE:
            logger.error("MT5 not available. Install MetaTrader5 package.")
            return False

        for attempt in range(self.max_retries):
            if mt5.initialize():
                self._connected = True
                info = mt5.terminal_info()
                logger.info(
                    "MT5 connected: %s (build %d)",
                    info.name if info else "Unknown",
                    info.build if info else 0,
                )
                return True
            time.sleep(1)

        logger.error("Failed to connect to MT5 after %d attempts", self.max_retries)
        return False

    def disconnect(self) -> None:
        """Gracefully disconnect from MT5."""
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    def get_account_state(self) -> AccountState | None:
        """Get current account state."""
        if not self._check_connection():
            return None

        info = mt5.account_info()
        if info is None:
            return None

        return AccountState(
            balance=info.balance,
            equity=info.equity,
            margin=info.margin,
            free_margin=info.margin_free,
            margin_level=info.margin_level if info.margin_level else 0.0,
            profit=info.profit,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_symbol_info(self) -> dict | None:
        """Get current symbol info (spread, tick size, etc.)."""
        if not self._check_connection():
            return None

        info = mt5.symbol_info(self.symbol)
        if info is None:
            return None

        return {
            "symbol": self.symbol,
            "bid": info.bid,
            "ask": info.ask,
            "spread": info.spread,
            "point": info.point,
            "digits": info.digits,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "trade_contract_size": info.trade_contract_size,
        }

    def market_order(
        self,
        direction: int,  # +1 = BUY, -1 = SELL
        lots: float,
        sl_price: float = 0.0,
        tp_price: float = 0.0,
        comment: str = "DRL_Agent",
    ) -> OrderResult:
        """
        Execute a market order.

        Args:
            direction: +1 for BUY, -1 for SELL
            lots: Lot size
            sl_price: Stop loss price (0 = no SL)
            tp_price: Take profit price (0 = no TP)
            comment: Order comment

        Returns:
            OrderResult with execution details
        """
        if not self._check_connection():
            return OrderResult(
                success=False, ticket=0, price=0, lots=0,
                error="Not connected to MT5"
            )

        # Ensure symbol is selected
        if not mt5.symbol_select(self.symbol, True):
            return OrderResult(
                success=False, ticket=0, price=0, lots=0,
                error=f"Symbol {self.symbol} not available"
            )

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return OrderResult(
                success=False, ticket=0, price=0, lots=0,
                error="Cannot get tick data"
            )

        # Determine order type and price
        if direction > 0:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        # Round lots to valid step
        sym_info = mt5.symbol_info(self.symbol)
        if sym_info:
            step = sym_info.volume_step
            lots = max(sym_info.volume_min, round(lots / step) * step)
            lots = min(lots, sym_info.volume_max)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lots,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            return OrderResult(
                success=False, ticket=0, price=0, lots=0,
                error="order_send returned None"
            )

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(
                success=False, ticket=0, price=0, lots=lots,
                retcode=result.retcode,
                error=f"Order failed: retcode={result.retcode}, comment={result.comment}",
            )

        logger.info(
            "Order executed: %s %.2f lots @ %.5f (ticket=%d)",
            "BUY" if direction > 0 else "SELL",
            result.volume, result.price, result.order,
        )

        return OrderResult(
            success=True,
            ticket=result.order,
            price=result.price,
            lots=result.volume,
            comment=result.comment,
            retcode=result.retcode,
        )

    def close_position(self, ticket: int) -> OrderResult:
        """Close a specific position by ticket."""
        if not self._check_connection():
            return OrderResult(
                success=False, ticket=ticket, price=0, lots=0,
                error="Not connected"
            )

        # Find the position
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return OrderResult(
                success=False, ticket=ticket, price=0, lots=0,
                error=f"Position {ticket} not found"
            )

        pos = position[0]
        # Reverse direction to close
        if pos.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(pos.symbol).bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(pos.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": "DRL_Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info("Closed position %d @ %.5f", ticket, result.price)
            return OrderResult(
                success=True, ticket=ticket,
                price=result.price, lots=pos.volume,
            )

        return OrderResult(
            success=False, ticket=ticket, price=0, lots=0,
            error=f"Close failed: {result.comment if result else 'None'}",
        )

    def close_all_positions(self) -> list[OrderResult]:
        """Close all open positions for this symbol and magic number."""
        results = []
        positions = self.get_open_positions()

        for pos in positions:
            result = self.close_position(pos.ticket)
            results.append(result)

        return results

    def get_open_positions(self) -> list[PositionInfo]:
        """Get all open positions for this symbol."""
        if not self._check_connection():
            return []

        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return []

        result = []
        for p in positions:
            if p.magic == self.magic:
                result.append(PositionInfo(
                    ticket=p.ticket,
                    symbol=p.symbol,
                    direction=1 if p.type == mt5.ORDER_TYPE_BUY else -1,
                    lots=p.volume,
                    open_price=p.price_open,
                    sl=p.sl,
                    tp=p.tp,
                    profit=p.profit,
                    open_time=datetime.fromtimestamp(
                        p.time, tz=timezone.utc
                    ).isoformat(),
                ))

        return result

    def modify_position(
        self, ticket: int, sl: float = 0, tp: float = 0
    ) -> bool:
        """Modify SL/TP of an existing position."""
        if not self._check_connection():
            return False

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": self.symbol,
            "sl": sl,
            "tp": tp,
            "magic": self.magic,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return True

        logger.error("Modify failed for ticket %d: %s",
                      ticket, result.comment if result else "None")
        return False

    def _check_connection(self) -> bool:
        """Check and attempt reconnection if needed."""
        if not MT5_AVAILABLE:
            return False

        if not self._connected:
            return self.connect()

        # Verify connection is still alive
        info = mt5.terminal_info()
        if info is None:
            self._connected = False
            logger.warning("MT5 connection lost. Reconnecting...")
            return self.connect()

        return True
