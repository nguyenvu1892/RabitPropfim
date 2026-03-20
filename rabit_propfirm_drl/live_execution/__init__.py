"""
Live Execution — MT5 bridge, killswitch, watchdog, monitoring.

Modules:
    mt5_connector       — Low-level MT5 API wrapper (Sprint 1)
    killswitch          — Triple-layer DD protection + DailyLossGate (Sprint 3)
    inference_pipeline  — Feature processing + decision pipeline (Sprint 4)
    paper_trading       — Paper trading orchestrator + session reports (Sprint 4)
    data_feed           — Real-time multi-TF OHLCV polling (Sprint 6)
    connection_guard    — 3-tier auto-reconnect protection (Sprint 6)
    order_calculator    — Lot sizing, SL/TP calc, slippage handling (Sprint 6)
"""
