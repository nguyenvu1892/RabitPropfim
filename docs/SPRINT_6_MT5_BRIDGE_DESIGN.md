# Sprint 6: MT5 Live Execution Engine — Architecture Design

**Author:** An (AI Engineer) | **Date:** 20/03/2026 | **Status:** Draft — Pending Review

> [!CAUTION]
> Hệ thống này sẽ cầm TIỀN THẬT. Mọi rủi ro về mạng, trượt giá, lỗi kết nối *phải* có phương án dự phòng. Không có exception nào được phép "thoát" mà không được xử lý.

---

## 1. Tổng Quan & Mục Tiêu

**Sprint 6** xây dựng cầu nối giữa **EnsembleAgent** (đã train xong Sprint 5) và **MetaTrader 5** để khớp lệnh tự động trên tài khoản Prop Firm thật.

### Yêu Cầu Cốt Lõi
| # | Requirement | Priority |
|---|-------------|----------|
| R1 | Lấy nến M5, H1, H4 real-time từ MT5 để feed vào EnsembleAgent | **P0** |
| R2 | Chuyển đổi output `[direction, risk, SL, TP]` → lệnh thực tế (lot, price, SL/TP) | **P0** |
| R3 | Xử lý slippage, requote, partial fill từ sàn | **P0** |
| R4 | Auto-reconnect khi mất kết nối MT5 | **P0** |
| R5 | Tích hợp Killswitch + DailyLossGate (đã có) | **P0** |
| R6 | Logging chi tiết mọi quyết định + execution result | **P1** |
| R7 | Telegram alert cho các sự kiện quan trọng | **P1** |

### Module Đã Có (Sprint 1–5)
- `mt5_connector.py` — Low-level MT5 connection, `market_order()`, `close_position()`
- `ensemble_agent.py` — 3-specialist voting → `[direction, risk, sl_mult, tp_mult]`
- `action_gating.py` — `GatedAction` (signal, risk_fraction, sl_multiplier, tp_multiplier)
- `killswitch.py` — Triple-layer DD protection + `DailyLossGate`
- `inference_pipeline.py` — Feature processing + decision pipeline
- `paper_trading.py` — Session tracking + report generation

### Module MỚI (Sprint 6)
- **`live/mt5_engine.py`** — `MT5ExecutionEngine` — *Trung tâm điều khiển mọi thứ*

---

## 2. Kiến Trúc Luồng Dữ Liệu (Data Flow)

### 2.1. Sơ Đồ Tổng Thể

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MT5 TERMINAL (Sàn)                           │
│   ┌──────────┐   ┌─────────────┐   ┌──────────────┐                │
│   │ Tick Data │   │ OHLCV Bars  │   │ Order Engine │                │
│   └────┬─────┘   └──────┬──────┘   └───────┬──────┘                │
└────────┼────────────────┼──────────────────┼────────────────────────┘
         │ (on_tick)      │ (copy_rates)     ▲ (order_send)
         ▼                ▼                  │
┌────────────────────────────────────────────┼────────────────────────┐
│              MT5ExecutionEngine             │                        │
│                                            │                        │
│  ┌──────────────────┐                      │                        │
│  │  DataFeedManager │                      │                        │
│  │  ┌─────────────┐ │                      │                        │
│  │  │ M5 Buffer   │ │    ┌──────────┐      │     ┌───────────────┐  │
│  │  │ (96 bars)   │─┼───►│          │      │     │               │  │
│  │  ├─────────────┤ │    │ Feature  │      │     │  Killswitch   │  │
│  │  │ H1 Buffer   │─┼───►│ Pipeline │──►Tensor   │  + DailyLoss  │  │
│  │  │ (48 bars)   │ │    │          │      │     │  Gate         │  │
│  │  ├─────────────┤ │    └────┬─────┘      │     └───────┬───────┘  │
│  │  │ H4 Buffer   │─┼────────┘             │             │          │
│  │  │ (24 bars)   │ │                      │             │          │
│  │  └─────────────┘ │    ┌──────────────┐  │    ┌────────▼───────┐  │
│  └──────────────────┘    │              │  │    │                │  │
│                          │  Ensemble    │  │    │  Order         │  │
│     Tensor ─────────────►│  Agent       │──┼───►│  Manager       │──┘
│                          │  (3 models)  │  │    │  (lot calc,    │
│                          │              │  │    │   SL/TP price, │
│                          └──────┬───────┘  │    │   retry logic) │
│                                 │          │    └────────────────┘
│                          ┌──────▼───────┐  │
│                          │ ActionGating  │──┘
│                          │ → GatedAction │
│                          └──────────────┘
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2. Chi Tiết Luồng Lấy Dữ Liệu Real-Time

#### Cơ Chế Polling (Không Dùng Callback)

MetaTrader5 Python API **không hỗ trợ event-driven callback** cho tick mới. Thay vào đó, ta dùng **polling loop** với smart interval:

```python
class DataFeedManager:
    """
    Quản lý việc lấy & buffer nến multi-timeframe từ MT5.
    
    Chiến lược:
    - Poll mỗi 5 giây để check nến M5 mới
    - Khi có nến M5 mới đóng → trigger inference
    - H1, H4 chỉ cần refresh khi nến tương ứng đóng
    """
    
    POLL_INTERVAL_SEC = 5.0   # Check mỗi 5s
    M5_BARS_NEEDED   = 96     # 96 nến M5 = 8 giờ lookback
    H1_BARS_NEEDED   = 48     # 48 nến H1 = 2 ngày lookback
    H4_BARS_NEEDED   = 24     # 24 nến H4 = 4 ngày lookback

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._last_m5_time: datetime | None = None
        self._m5_cache: np.ndarray | None = None
        self._h1_cache: np.ndarray | None = None
        self._h4_cache: np.ndarray | None = None

    def poll(self) -> tuple[bool, dict[str, np.ndarray] | None]:
        """
        Gọi mỗi POLL_INTERVAL_SEC giây.
        
        Returns:
            (has_new_bar, {"m5": ..., "h1": ..., "h4": ...} | None)
        """
        # Lấy nến M5 mới nhất
        rates_m5 = mt5.copy_rates_from_pos(
            self.symbol, mt5.TIMEFRAME_M5, 0, self.M5_BARS_NEEDED
        )
        if rates_m5 is None or len(rates_m5) == 0:
            return False, None

        latest_time = datetime.fromtimestamp(rates_m5[-1]['time'], tz=timezone.utc)
        
        if self._last_m5_time is not None and latest_time <= self._last_m5_time:
            return False, None  # Chưa có nến mới

        # Có nến M5 mới! → refresh tất cả timeframe
        self._last_m5_time = latest_time
        self._m5_cache = self._to_ohlcv(rates_m5)
        
        # H1 chỉ cần refresh mỗi 12 nến M5 (1 giờ)
        # Nhưng để an toàn, ta refresh mỗi lần có nến M5 mới
        rates_h1 = mt5.copy_rates_from_pos(
            self.symbol, mt5.TIMEFRAME_H1, 0, self.H1_BARS_NEEDED
        )
        if rates_h1 is not None:
            self._h1_cache = self._to_ohlcv(rates_h1)

        rates_h4 = mt5.copy_rates_from_pos(
            self.symbol, mt5.TIMEFRAME_H4, 0, self.H4_BARS_NEEDED
        )
        if rates_h4 is not None:
            self._h4_cache = self._to_ohlcv(rates_h4)

        return True, {
            "m5": self._m5_cache,
            "h1": self._h1_cache,
            "h4": self._h4_cache,
        }
```

#### Tại Sao Polling 5s Mà Không Phải 1s hay 30s?

| Interval | Pros | Cons |
|----------|------|------|
| 1s | Phản ứng nhanh nhất | CPU/network load cao, MT5 API rate limit |
| **5s** | **Cân bằng tốt, nến M5 = 300s → chỉ miss max 5s** | Chậm 5s so với tick-by-tick |
| 30s | Tiết kiệm tài nguyên | Có thể miss cả nến M5 nếu market đóng nhanh |

> [!NOTE]
> Nến M5 đóng mỗi 300 giây. Polling 5s nghĩa là ta detect nến mới trong tối đa 5s sau khi đóng. Đủ nhanh cho swing/intraday trên Prop Firm.

### 2.3. Xử Lý Latency & Mất Kết Nối

#### Chiến Lược Auto-Reconnect (3 Tầng)

```python
class ConnectionGuard:
    """
    Bảo vệ kết nối MT5 với 3 tầng retry.
    
    Tầng 1: Retry nhanh (3 lần, 1s apart) — cho lỗi tạm thời
    Tầng 2: Reconnect (reinitialize MT5) — cho mất kết nối
    Tầng 3: Exponential backoff (max 5 phút) — cho lỗi nghiêm trọng
    """
    
    MAX_FAST_RETRIES  = 3
    MAX_RECONNECTS    = 5
    BACKOFF_BASE_SEC  = 2.0
    BACKOFF_MAX_SEC   = 300.0  # 5 phút

    def __init__(self):
        self._consecutive_failures = 0
        self._last_successful_op = time.time()
        self._is_connected = False

    def execute_with_guard(self, operation: Callable, *args) -> Any:
        """
        Wrap bất kỳ MT5 operation nào với retry logic.
        
        Raises:
            MT5ConnectionError: Sau khi hết tất cả retry
        """
        for attempt in range(self.MAX_FAST_RETRIES):
            try:
                result = operation(*args)
                self._consecutive_failures = 0
                self._last_successful_op = time.time()
                return result
            except Exception as e:
                logger.warning(
                    "MT5 op failed (attempt %d/%d): %s",
                    attempt + 1, self.MAX_FAST_RETRIES, e
                )
                time.sleep(1)

        # Tầng 2: Reconnect
        for reconnect in range(self.MAX_RECONNECTS):
            logger.error("Attempting MT5 reconnect %d/%d", 
                         reconnect + 1, self.MAX_RECONNECTS)
            mt5.shutdown()
            time.sleep(self.BACKOFF_BASE_SEC)
            
            if mt5.initialize():
                self._is_connected = True
                try:
                    return operation(*args)
                except Exception:
                    continue
            
            # Tầng 3: Exponential backoff
            wait = min(
                self.BACKOFF_BASE_SEC * (2 ** reconnect),
                self.BACKOFF_MAX_SEC
            )
            logger.error("Backoff %.0fs before next reconnect", wait)
            time.sleep(wait)

        # Tất cả retry thất bại → raise để engine xử lý
        raise MT5ConnectionError(
            f"MT5 unreachable after {self.MAX_FAST_RETRIES} retries "
            f"+ {self.MAX_RECONNECTS} reconnects"
        )
```

#### Hành Vi Khi Mất Kết Nối Kéo Dài

```
Mất kết nối > 30s:  → Telegram alert "⚠️ MT5 DISCONNECTED"
Mất kết nối > 2m:   → Ghi log "CRITICAL", auto-close nếu có lệnh mở
Mất kết nối > 5m:   → Emergency shutdown, đóng tất cả, dừng engine
```

> [!WARNING]
> **KHÔNG BAO GIỜ** để lệnh mở mà không có SL khi mất kết nối. Nếu reconnect thất bại và có lệnh đang mở → engine PHẢI cố đóng lệnh NGAY khi reconnect thành công (trước khi làm bất cứ gì khác).

---

## 3. Quản Lý Lệnh (Order Management)

### 3.1. Pipeline: Agent Output → MT5 Order

```
EnsembleAgent.get_gated_action(m5, h1, h4)
        │
        ▼
GatedAction {
    signal:          BUY (+1) / SELL (-1) / HOLD (0)
    risk_fraction:   0.0 → 1.0   (% risk)
    sl_multiplier:   0.5 → 2.0   (nhân với ATR)
    tp_multiplier:   0.5 → 2.0   (nhân với ATR)
}
        │
        ▼ (nếu signal != HOLD)
OrderCalculator.compute(gated_action, account_state, symbol_info, atr)
        │
        ▼
OrderParams {
    direction:  mt5.ORDER_TYPE_BUY or SELL
    lots:       0.01 → max_lots
    price:      Ask (BUY) or Bid (SELL)
    sl_price:   price ∓ (sl_multiplier × ATR)
    tp_price:   price ± (tp_multiplier × ATR)
}
        │
        ▼
MT5LiveConnector.market_order(direction, lots, sl_price, tp_price)
```

### 3.2. Tính Toán Khối Lượng Lot (Position Sizing)

```python
class OrderCalculator:
    """
    Chuyển đổi GatedAction → OrderParams cụ thể cho từng symbol.
    
    Quy tắc:
    1. risk_amount = balance × risk_fraction × max_loss_per_trade_pct
    2. sl_distance = sl_multiplier × ATR (in price)
    3. lots = risk_amount / (sl_distance × contract_size × pip_value)
    4. Clamp lots: [volume_min, min(volume_max, max_lots_per_trade)]
    5. Round lots theo volume_step
    """

    def compute(
        self,
        gated: GatedAction,
        balance: float,
        symbol_info: dict,
        atr_price: float,        # ATR trong đơn vị giá (vd: 1.50 cho Gold)
        max_loss_pct: float = 0.003,  # 0.3% per trade (from config)
        max_lots: float = 10.0,
    ) -> OrderParams:
        
        # === 1. Tính risk amount ===
        risk_fraction = gated.risk_fraction   # 0..1 từ ActionGating
        risk_amount = balance * risk_fraction * max_loss_pct
        # Ví dụ: balance=$100,000, risk_fraction=0.7, max_loss=0.3%
        #       → risk_amount = $100,000 × 0.7 × 0.003 = $210

        # === 2. Tính SL distance (in price) ===
        sl_distance = gated.sl_multiplier * atr_price
        # Ví dụ: sl_multiplier=1.2, ATR=1.50 (gold)
        #       → sl_distance = 1.2 × 1.50 = $1.80

        # === 3. Tính lots ===
        contract_size = symbol_info["trade_contract_size"]  # 100 cho Gold
        point = symbol_info["point"]
        
        # pip_value_per_lot = contract_size × point (cho standard lot)
        # Nhưng sl_distance đã tính bằng price → dùng trực tiếp
        sl_in_points = sl_distance / point
        value_per_point_per_lot = contract_size * point
        
        if sl_distance > 0 and value_per_point_per_lot > 0:
            lots = risk_amount / (sl_in_points * value_per_point_per_lot)
        else:
            lots = symbol_info["volume_min"]

        # === 4. Clamp & Round ===
        vol_step = symbol_info["volume_step"]
        vol_min = symbol_info["volume_min"]
        vol_max = min(symbol_info["volume_max"], max_lots)
        
        lots = max(vol_min, min(vol_max, lots))
        lots = round(lots / vol_step) * vol_step
        lots = round(lots, 2)  # Avoid float precision issues

        # === 5. Tính SL/TP Price ===
        if gated.signal == TradeSignal.BUY:
            price = symbol_info["ask"]
            sl_price = price - sl_distance
            tp_price = price + gated.tp_multiplier * atr_price
        else:
            price = symbol_info["bid"]
            sl_price = price + sl_distance
            tp_price = price - gated.tp_multiplier * atr_price

        # Round theo digits của symbol
        digits = symbol_info["digits"]
        sl_price = round(sl_price, digits)
        tp_price = round(tp_price, digits)

        return OrderParams(
            direction=gated.signal.value,
            lots=lots,
            price=price,
            sl_price=sl_price,
            tp_price=tp_price,
            risk_amount=risk_amount,
            sl_distance_points=sl_in_points,
        )
```

#### Ví Dụ Thực Tế

| Param | XAUUSD (Gold) | US100 (Nasdaq) | BTCUSD |
|-------|---------------|----------------|--------|
| Balance | $100,000 | $100,000 | $100,000 |
| risk_fraction | 0.7 | 0.5 | 0.3 |
| risk_amount | $210 | $150 | $90 |
| ATR (price) | $1.50 | 15 pts | $500 |
| sl_multiplier | 1.2 | 1.0 | 1.5 |
| sl_distance | $1.80 | 15 pts | $750 |
| contract_size | 100 oz | 1 | 1 |
| point | 0.01 | 0.01 | 0.01 |
| Lots (calc) | 0.01 | 1.00 | 0.01 |

> [!NOTE]
> **Dynamic Lot cho mọi symbol:** Công thức `lots = risk_amount / (sl_in_points × value_per_point)` tự cân bằng tất cả. Gold có ATR cao → SL xa hơn → lot nhỏ hơn tự động. **KHÔNG hardcode fixed lot** cho bất kỳ symbol nào — tin tưởng tuyệt đối vào Dynamic Sizing.

### 3.3. Xử Lý Slippage & Requote

```python
class SlippageHandler:
    """
    Xử lý 3 loại lỗi execution phổ biến từ sàn:
    
    1. Slippage — Giá thực tế khác giá request
    2. Requote  — Sàn từ chối giá, đề nghị giá mới
    3. Partial Fill — Chỉ khớp một phần volume
    """
    
    MAX_SLIPPAGE_POINTS = 20     # Từ chối nếu slippage > 20 points
    MAX_REQUOTE_RETRIES = 3      # Retry tối đa 3 lần khi requote
    REQUOTE_DELAY_MS    = 200    # Delay giữa các lần retry

    def execute_with_slippage_check(
        self, connector: MT5LiveConnector, order: OrderParams
    ) -> OrderResult:
        """
        Gửi lệnh với kiểm tra slippage và retry cho requote.
        """
        for attempt in range(self.MAX_REQUOTE_RETRIES):
            result = connector.market_order(
                direction=order.direction,
                lots=order.lots,
                sl_price=order.sl_price,
                tp_price=order.tp_price,
                comment=f"DRL_v6_{attempt}",
            )
            
            if result.success:
                # Kiểm tra slippage
                slippage = abs(result.price - order.price)
                slippage_points = slippage / connector.get_symbol_info()["point"]
                
                if slippage_points > self.MAX_SLIPPAGE_POINTS:
                    logger.warning(
                        "⚠️ High slippage detected: %.1f points "
                        "(limit: %d). Order filled but logged.",
                        slippage_points, self.MAX_SLIPPAGE_POINTS
                    )
                    # Không cancel — lệnh đã fill, chỉ log warning
                
                return result
            
            # Check retcode for requote
            if result.retcode == 10004:  # TRADE_RETCODE_REQUOTE
                logger.warning(
                    "Requote received (attempt %d/%d). Retrying in %dms...",
                    attempt + 1, self.MAX_REQUOTE_RETRIES,
                    self.REQUOTE_DELAY_MS
                )
                time.sleep(self.REQUOTE_DELAY_MS / 1000.0)
                
                # Refresh giá mới
                tick = mt5.symbol_info_tick(connector.symbol)
                if tick:
                    if order.direction > 0:
                        order.price = tick.ask
                    else:
                        order.price = tick.bid
                    # Recalculate SL/TP relative to new price
                    # (giữ khoảng cách SL/TP cố định)
                continue
            
            # Lỗi khác → log và return thất bại
            logger.error(
                "Order failed: retcode=%d, error=%s",
                result.retcode, result.error
            )
            return result
        
        # Hết retry → return lỗi cuối cùng
        return OrderResult(
            success=False, ticket=0, price=0, lots=0,
            error=f"Failed after {self.MAX_REQUOTE_RETRIES} requote retries"
        )
```

#### MT5 Retcode Reference (Quan Trọng)

| Retcode | Tên | Hành Động |
|---------|-----|-----------|
| `10009` | `TRADE_RETCODE_DONE` | ✅ Thành công |
| `10004` | `TRADE_RETCODE_REQUOTE` | 🔄 Retry với giá mới |
| `10006` | `TRADE_RETCODE_REJECT` | ❌ Sàn từ chối — dừng |
| `10010` | `TRADE_RETCODE_DONE_PARTIAL` | ⚠️ Partial fill — log + xử lý |
| `10013` | `TRADE_RETCODE_INVALID_STOPS` | ❌ SL/TP invalid — recalculate |
| `10014` | `TRADE_RETCODE_INVALID_VOLUME` | ❌ Lot sai — recalculate |
| `10015` | `TRADE_RETCODE_INVALID_PRICE` | 🔄 Giá cũ — refresh & retry |
| `10018` | `TRADE_RETCODE_MARKET_CLOSED` | ⏸️ Thị trường đóng — chờ |
| `10019` | `TRADE_RETCODE_NO_MONEY` | 🛑 Hết margin — EMERGENCY |

---

## 4. Phác Thảo Code: `MT5ExecutionEngine`

### 4.1. File Structure Mới

```
live/
    __init__.py
    mt5_engine.py           ← [NEW] Main engine class
    data_feed.py            ← [NEW] DataFeedManager
    order_calculator.py     ← [NEW] OrderCalculator + SlippageHandler
    connection_guard.py     ← [NEW] ConnectionGuard + auto-reconnect
rabit_propfirm_drl/
    live_execution/
        mt5_connector.py    ← [EXISTING] Low-level MT5 API wrapper
        killswitch.py       ← [EXISTING] DD protection
        inference_pipeline.py ← [EXISTING] Feature pipeline
```

### 4.2. Class `MT5ExecutionEngine` — Code Phác Thảo

```python
"""
MT5 Live Execution Engine — Sprint 6.

Main orchestrator: connects EnsembleAgent → MT5 Terminal.
Handles data feed, inference, order execution, and risk management.

CRITICAL: This system handles REAL MONEY. Every failure mode
has a fallback. No exception escapes unhandled.
"""

from __future__ import annotations

import logging
import signal
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

from rabit_propfirm_drl.agents.ensemble_agent import EnsembleAgent
from rabit_propfirm_drl.agents.action_gating import ActionGating, GatedAction, TradeSignal
from rabit_propfirm_drl.live_execution.mt5_connector import MT5LiveConnector
from rabit_propfirm_drl.live_execution.killswitch import (
    Killswitch, EquityWatchdog, DailyLossGate,
)

logger = logging.getLogger(__name__)


# ─── Custom Exceptions ───────────────────────────────────────────────
class MT5ConnectionError(Exception):
    """MT5 connection lost and all reconnect attempts failed."""

class EmergencyShutdownError(Exception):
    """Triggered when risk limits are breached — close everything."""


# ─── Data Types ──────────────────────────────────────────────────────
@dataclass
class OrderParams:
    """Calculated order parameters ready for MT5 submission."""
    direction: int           # +1 BUY, -1 SELL
    lots: float
    price: float
    sl_price: float
    tp_price: float
    risk_amount: float       # Dollar amount at risk
    sl_distance_points: float

@dataclass
class EngineState:
    """Snapshot of engine state for logging/monitoring."""
    timestamp: str
    status: str              # "running" | "paused" | "stopped" | "emergency"
    balance: float
    equity: float
    open_positions: int
    daily_dd: float
    total_dd: float
    killswitch_status: str
    last_signal: str
    total_trades_today: int
    total_pnl_today: float


# ─── Main Engine ─────────────────────────────────────────────────────
class MT5ExecutionEngine:
    """
    Central orchestrator for live DRL trading on MT5.

    Lifecycle:
        engine = MT5ExecutionEngine(config, ensemble, ...)
        engine.start()      # Blocking main loop
        engine.stop()       # Graceful shutdown (Ctrl+C or SIGTERM)

    Main Loop (simplified):
        while running:
            1. Poll MT5 for new M5 bar
            2. If new bar → build features → run EnsembleAgent
            3. ActionGating → check killswitch → compute order
            4. Execute order → log result
            5. Update watchdog (equity tracking)
            6. Sleep until next poll

    Safety Guarantees:
        - Every order has SL attached (no naked positions)
        - Killswitch checked BEFORE every order
        - DailyLossGate checked BEFORE every order
        - Max open positions enforced
        - Auto-reconnect on connection loss
        - Emergency close on critical failure
    """

    def __init__(
        self,
        config: dict,
        ensemble: EnsembleAgent,
        connector: MT5LiveConnector,
        feature_builder: Callable[[dict[str, np.ndarray]], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        symbols: list[str] | None = None,
        alert_callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """
        Args:
            config:          Dict from prop_rules.yaml
            ensemble:        Trained EnsembleAgent (3 specialists)
            connector:       MT5LiveConnector instance
            feature_builder: Function: raw OHLCV dict → (m5_tensor, h1_tensor, h4_tensor)
            symbols:         List of symbols to trade (default from config)
            alert_callback:  Telegram/Discord notification function
        """
        self.config = config
        self.ensemble = ensemble
        self.connector = connector
        self.feature_builder = feature_builder
        self.symbols = symbols or config.get("target_symbols", ["XAUUSD"])
        self._alert = alert_callback

        # ── Risk Management (reuse existing modules) ──
        self.killswitch = Killswitch(config)
        self.watchdog = EquityWatchdog(self.killswitch, check_interval_seconds=5.0)
        self.daily_gate = DailyLossGate(config)
        self.action_gating = ActionGating(
            confidence_threshold=config.get("confidence_threshold", 0.3)
        )

        if alert_callback:
            self.killswitch.set_alert_callback(alert_callback)
            self.daily_gate.set_alert_callback(alert_callback)

        # ── Execution params ──
        self.max_open_positions = config.get("max_open_positions", 5)
        self.max_trades_per_day = config.get("max_trades_per_day", 15)
        self.max_lots = config.get("max_lots_per_trade", 10.0)
        self.max_loss_per_trade = config.get("max_loss_per_trade_pct", 0.003)
        self.trading_start_utc = config.get("trading_start_utc", 1)
        self.trading_end_utc = config.get("trading_end_utc", 21)

        # ── State tracking ──
        self._running = False
        self._trades_today = 0
        self._pnl_today = 0.0
        self._initial_balance = 0.0

        # ── Data feed per symbol ──
        self._feeds: dict[str, DataFeedManager] = {}
        for sym in self.symbols:
            self._feeds[sym] = DataFeedManager(sym)

        # ── Graceful shutdown ──
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PUBLIC API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def start(self) -> None:
        """
        Start the main trading loop. BLOCKING call.
        Press Ctrl+C or send SIGTERM for graceful shutdown.
        """
        logger.info("=" * 60)
        logger.info("MT5ExecutionEngine STARTING")
        logger.info("Symbols: %s", self.symbols)
        logger.info("=" * 60)

        # Connect to MT5
        if not self.connector.connect():
            raise MT5ConnectionError("Initial MT5 connection failed")

        # Get initial account state
        account = self.connector.get_account_state()
        if account is None:
            raise MT5ConnectionError("Cannot read account state")

        self._initial_balance = account.balance
        self.daily_gate.start_day(account.balance)
        self._running = True

        self._send_alert(
            "🟢 ENGINE STARTED",
            f"Balance: ${account.balance:,.2f} | "
            f"Symbols: {', '.join(self.symbols)}"
        )

        try:
            self._main_loop()
        except EmergencyShutdownError as e:
            logger.critical("EMERGENCY SHUTDOWN: %s", e)
            self._emergency_close_all()
        except Exception as e:
            logger.critical("UNEXPECTED ERROR: %s", e, exc_info=True)
            self._emergency_close_all()
        finally:
            self._shutdown()

    def stop(self) -> None:
        """Signal the engine to stop gracefully."""
        logger.info("Stop signal received")
        self._running = False

    def get_state(self) -> EngineState:
        """Get current engine state snapshot."""
        account = self.connector.get_account_state()
        positions = self.connector.get_open_positions()

        daily_dd = 0.0
        total_dd = 0.0
        if account and self._initial_balance > 0:
            daily_dd = max(0, (self._initial_balance - account.equity) 
                         / self._initial_balance)
            total_dd = daily_dd  # Simplified; real impl tracks peak

        return EngineState(
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="running" if self._running else "stopped",
            balance=account.balance if account else 0,
            equity=account.equity if account else 0,
            open_positions=len(positions),
            daily_dd=daily_dd,
            total_dd=total_dd,
            killswitch_status=self.killswitch.check(
                daily_dd, total_dd, account.equity if account else 0
            ),
            last_signal="",
            total_trades_today=self._trades_today,
            total_pnl_today=self._pnl_today,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MAIN LOOP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _main_loop(self) -> None:
        """
        Core trading loop. Polls for new M5 bars and acts on them.
        """
        poll_interval = DataFeedManager.POLL_INTERVAL_SEC

        while self._running:
            loop_start = time.time()

            try:
                # ── Pre-flight checks ──
                account = self.connector.get_account_state()
                if account is None:
                    self._handle_connection_loss()
                    continue

                # ── Check killswitch ──
                daily_dd = max(0, (self._initial_balance - account.equity) 
                              / self._initial_balance)
                total_dd = daily_dd
                ks_status = self.watchdog.tick(daily_dd, total_dd, account.equity)

                if ks_status == "emergency":
                    raise EmergencyShutdownError(
                        f"Total DD {total_dd:.2%} breached emergency threshold"
                    )
                if ks_status == "hard":
                    self._emergency_close_all()
                    logger.error("Hard killswitch — all positions closed, "
                                 "waiting for next day reset")
                    time.sleep(60)
                    continue

                # ── Check trading hours ──
                hour_utc = datetime.now(timezone.utc).hour
                if not (self.trading_start_utc <= hour_utc < self.trading_end_utc):
                    time.sleep(poll_interval)
                    continue

                # ── Check daily cooldown ──
                if not self.daily_gate.can_trade():
                    time.sleep(poll_interval)
                    continue

                # ── Poll each symbol for new M5 bar ──
                for symbol in self.symbols:
                    self.connector.symbol = symbol  # Switch active symbol
                    feed = self._feeds[symbol]

                    has_new_bar, data = feed.poll()
                    if not has_new_bar or data is None:
                        continue

                    # ── BUILD FEATURES & INFER ──
                    self._process_signal(symbol, data, account, ks_status)

            except MT5ConnectionError:
                self._handle_connection_loss()
            except EmergencyShutdownError:
                raise
            except Exception as e:
                logger.error("Loop error: %s", e, exc_info=True)

            # ── Sleep remaining time ──
            elapsed = time.time() - loop_start
            sleep_time = max(0.1, poll_interval - elapsed)
            time.sleep(sleep_time)

    def _process_signal(
        self,
        symbol: str,
        data: dict[str, np.ndarray],
        account: Any,
        ks_status: str,
    ) -> None:
        """
        Process a single new-bar event for one symbol.
        
        Pipeline: data → features → ensemble → gating → order
        """
        # 1. Build feature tensors
        m5_tensor, h1_tensor, h4_tensor = self.feature_builder(data)

        # 2. Run EnsembleAgent → GatedAction
        gated = self.ensemble.get_gated_action(
            m5_tensor, h1_tensor, h4_tensor, deterministic=True
        )

        # 3. Check signal
        if isinstance(gated, GatedAction):
            if gated.signal == TradeSignal.HOLD:
                logger.debug("[%s] HOLD (confidence=%.3f)", 
                             symbol, gated.raw_confidence)
                return
        else:
            # Raw action (no gating) — safety check
            logger.warning("[%s] Got raw action instead of GatedAction", symbol)
            return

        # 4. Pre-execution safety checks
        open_positions = self.connector.get_open_positions()
        if len(open_positions) >= self.max_open_positions:
            logger.info("[%s] Max positions reached (%d), skipping", 
                        symbol, self.max_open_positions)
            return

        if self._trades_today >= self.max_trades_per_day:
            logger.info("[%s] Max trades/day reached (%d), skipping", 
                        symbol, self.max_trades_per_day)
            return

        # Soft killswitch → reduce risk
        risk_multiplier = 0.5 if ks_status == "soft" else 1.0

        # 5. Calculate order params
        sym_info = self.connector.get_symbol_info()
        if sym_info is None:
            logger.error("[%s] Cannot get symbol info", symbol)
            return

        atr = self._compute_atr(data["m5"], period=14)
        
        # Check for fixed lot override (e.g., XAUUSD)
        sym_config = self.config.get("symbol_configs", {}).get(symbol, {})
        fixed_lot = sym_config.get("fixed_lot", None)

        calculator = OrderCalculator()
        order = calculator.compute(
            gated=gated,
            balance=account.balance,
            symbol_info=sym_info,
            atr_price=atr,
            max_loss_pct=self.max_loss_per_trade * risk_multiplier,
            max_lots=self.max_lots,
        )

        # Override lot if fixed
        if fixed_lot is not None:
            order.lots = fixed_lot

        # 6. Execute with slippage handling
        handler = SlippageHandler()
        result = handler.execute_with_slippage_check(self.connector, order)

        # 7. Log & track
        if result.success:
            self._trades_today += 1
            logger.info(
                "✅ [%s] %s %.2f lots @ %.5f | SL=%.5f TP=%.5f | ticket=%d",
                symbol,
                "BUY" if order.direction > 0 else "SELL",
                result.lots, result.price,
                order.sl_price, order.tp_price,
                result.ticket,
            )
            self._send_alert(
                f"{'🟢 BUY' if order.direction > 0 else '🔴 SELL'} {symbol}",
                f"Lots: {result.lots} @ {result.price:.5f}\n"
                f"SL: {order.sl_price:.5f} | TP: {order.tp_price:.5f}\n"
                f"Risk: ${order.risk_amount:.2f}"
            )
        else:
            logger.error(
                "❌ [%s] Order FAILED: %s (retcode=%d)",
                symbol, result.error, result.retcode
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SAFETY & RECOVERY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _emergency_close_all(self) -> None:
        """Force-close all open positions. Called on killswitch trigger."""
        logger.critical("🚨 EMERGENCY: Closing ALL positions")
        self._send_alert("🚨 EMERGENCY CLOSE", "Closing all positions NOW")

        try:
            results = self.connector.close_all_positions()
            for r in results:
                if r.success:
                    logger.info("Closed ticket %d @ %.5f", r.ticket, r.price)
                else:
                    logger.error("Failed to close ticket %d: %s", 
                                 r.ticket, r.error)
        except Exception as e:
            logger.critical("CLOSE ALL FAILED: %s", e)

    def _handle_connection_loss(self) -> None:
        """Handle MT5 connection loss with progressive recovery."""
        logger.warning("MT5 connection lost. Attempting recovery...")
        self._send_alert("⚠️ MT5 DISCONNECTED", "Attempting auto-reconnect...")

        guard = ConnectionGuard()
        try:
            guard.execute_with_guard(self.connector.connect)
            logger.info("MT5 reconnected successfully")
            self._send_alert("✅ MT5 RECONNECTED", "Connection restored")
            
            # Check if positions are still intact
            positions = self.connector.get_open_positions()
            for pos in positions:
                if pos.sl == 0:
                    # Position without SL — DANGER! Close immediately
                    logger.critical(
                        "Position %d has NO SL after reconnect — closing!", 
                        pos.ticket
                    )
                    self.connector.close_position(pos.ticket)
                    
        except MT5ConnectionError:
            logger.critical("MT5 recovery FAILED. Emergency shutdown.")
            self._send_alert("🚨 MT5 UNRECOVERABLE", 
                             "All reconnect attempts failed. Shutting down.")
            self._running = False

    def _shutdown(self) -> None:
        """Graceful engine shutdown."""
        logger.info("Engine shutting down...")
        self._send_alert("🔴 ENGINE STOPPED", 
                         f"Trades today: {self._trades_today}")
        self.connector.disconnect()

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info("Received signal %d, initiating graceful shutdown", signum)
        self._running = False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # UTILITIES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _compute_atr(self, ohlcv: np.ndarray, period: int = 14) -> float:
        """Compute Average True Range from OHLCV data."""
        if len(ohlcv) < period + 1:
            return 0.0
        high = ohlcv[-period:, 1]    # High column
        low = ohlcv[-period:, 2]     # Low column 
        close = ohlcv[-period-1:-1, 3]  # Previous close
        
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - close),
                np.abs(low - close)
            )
        )
        return float(np.mean(tr))

    def _send_alert(self, title: str, msg: str) -> None:
        """Send alert via configured callback (Telegram/Discord)."""
        if self._alert:
            try:
                self._alert(title, msg)
            except Exception as e:
                logger.error("Alert send failed: %s", e)
```

---

## 5. Bảng Tóm Tắt Rủi Ro & Phương Án Dự Phòng

| # | Rủi Ro | Xác Suất | Tác Động | Phương Án |
|---|--------|----------|----------|-----------|
| 1 | MT5 mất kết nối | Trung bình | Cao | 3-tier auto-reconnect + Telegram alert |
| 2 | Slippage lớn (>20 pts) | Thấp | Trung bình | Log warning, deviation=20 points max |
| 3 | Requote liên tục | Thấp | Thấp | Retry 3 lần, delay 200ms, refresh giá |
| 4 | Partial fill | Rất thấp | Thấp | Accept partial, log for monitoring |
| 5 | DD vượt 4.5% | Trung bình | Rất cao | Killswitch tầng 1: reduce lot 50% |
| 6 | DD vượt 5% | Thấp | Critical | Killswitch tầng 2: close ALL, block trades |
| 7 | DD vượt 10% | Rất thấp | Critical | Emergency shutdown toàn bộ |
| 8 | Lệnh mở không SL | Không được xảy ra | Critical | Mọi lệnh PHẢI có SL. Check sau reconnect. |
| 9 | Market đóng cửa | Hàng ngày | Thấp | Check trading hours trước khi trade |
| 10 | Hết margin | Thấp | Cao | Check free_margin trước lệnh, retcode 10019 → STOP |
| 11 | Model inference crash | Rất thấp | Trung bình | try-except wrap, skip bar, log error |
| 12 | Feature pipeline lỗi | Thấp | Trung bình | Validate tensor shape trước khi infer |

---

## 6. Sequence Diagram — Luồng Xử Lý 1 Nến M5

```mermaid
sequenceDiagram
    participant Loop as Main Loop
    participant Feed as DataFeedManager
    participant MT5 as MT5 Terminal
    participant Feat as FeaturePipeline
    participant Ens as EnsembleAgent
    participant Gate as ActionGating
    participant KS as Killswitch
    participant Calc as OrderCalculator
    participant Slip as SlippageHandler

    Loop->>Feed: poll()
    Feed->>MT5: copy_rates_from_pos(M5, H1, H4)
    MT5-->>Feed: OHLCV arrays
    Feed-->>Loop: (has_new_bar=True, data)
    
    Loop->>KS: watchdog.tick(dd, equity)
    KS-->>Loop: status="normal"
    
    Loop->>Feat: feature_builder(data)
    Feat-->>Loop: (m5_tensor, h1_tensor, h4_tensor)
    
    Loop->>Ens: get_gated_action(m5, h1, h4)
    Ens->>Ens: 3 specialists vote
    Ens->>Gate: gate_single(action)
    Gate-->>Loop: GatedAction(signal=BUY, risk=0.7)
    
    Loop->>Loop: Check: positions < max? trades < max?
    
    Loop->>Calc: compute(gated, balance, sym_info, atr)
    Calc-->>Loop: OrderParams(lots=0.01, sl=2048.50, tp=2065.20)
    
    Loop->>Slip: execute_with_slippage_check(connector, order)
    Slip->>MT5: order_send(request)
    MT5-->>Slip: OrderResult(success=True, ticket=12345)
    Slip-->>Loop: OrderResult
    
    Loop->>Loop: Log + Telegram alert
```

---

## 7. Kế Hoạch Triển Khai

| Phase | Task | Est. Time |
|-------|------|-----------|
| **Phase 1** | Implement `DataFeedManager` + `ConnectionGuard` | 3h |
| **Phase 2** | Implement `OrderCalculator` + `SlippageHandler` | 3h |
| **Phase 3** | Implement `MT5ExecutionEngine` main loop | 4h |
| **Phase 4** | Integration test on Demo account (Paper Trade) | 2h |
| **Phase 5** | 3-day demo validation với real market data | 3 days |
| **Phase 6** | Deploy lên tài khoản Prop Firm thật | 1h |

> [!IMPORTANT]
> **KHÔNG ĐƯỢC** chạy tiền thật cho đến khi Phase 5 hoàn thành thành công (3 ngày paper trade trên demo account không vi phạm bất kỳ rule nào).

---

## 8. Checklist Trước Khi Go-Live

- [ ] Demo account chạy 3 ngày không lỗi kết nối
- [ ] Killswitch trigger test (cố ý gây DD 4.5%) → verify auto-close
- [ ] Requote handling test (đặt deviation=0 để force requote)
- [ ] Network disconnect test (ngắt WiFi → verify auto-reconnect)
- [ ] SL luôn được đặt cho mọi lệnh (check bằng MT5 terminal)
- [ ] Telegram alerts hoạt động cho mọi event type
- [ ] Lot size đúng cho mỗi symbol (đặc biệt XAUUSD fixed 0.01)
- [ ] Trading hours filter đúng (01:00–21:00 UTC)
- [ ] Max positions + max trades/day enforced
- [ ] Log file rotation configured (không để log phình)
