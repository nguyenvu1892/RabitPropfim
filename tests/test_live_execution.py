"""
Tests for Sprint 6 — Live Execution Engine.

Validates:
- MT5 connector graceful degradation (no MT5 installed)
- Inference pipeline decision logic (BUY/SELL/HOLD/BLOCKED)
- Paper trading session management
- Report generation and Prop Firm pass/fail logic
- Session stats tracking
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from live_execution.inference_pipeline import LiveInferencePipeline, InferenceResult
from live_execution.mt5_connector import MT5LiveConnector, OrderResult
from live_execution.paper_trading import (
    PaperTradingOrchestrator,
    PaperTradingReport,
    TradingSession,
)


# ─────────────────────────────────────────────
# Config Fixture
# ─────────────────────────────────────────────

def _config() -> dict:
    return {
        "confidence_threshold": 0.3,
        "trading_start_utc": 1,
        "trading_end_utc": 21,
        "max_daily_drawdown": 0.05,
        "max_total_drawdown": 0.10,
        "profit_target": 0.08,
        "min_trading_days": 5,
    }


def _dummy_model(state: np.ndarray) -> np.ndarray:
    """Dummy model that returns fixed action."""
    return np.array([0.5, 0.3, 1.0, 2.0], dtype=np.float32)


# ─────────────────────────────────────────────
# MT5 Connector Tests (graceful degradation)
# ─────────────────────────────────────────────

class TestMT5Connector:

    def test_graceful_without_mt5(self) -> None:
        """Should not crash when MT5 is not installed."""
        connector = MT5LiveConnector(symbol="EURUSD")
        # If MT5 not installed, connect returns False
        if not connector.is_available:
            assert not connector.connect()

    def test_order_result_dataclass(self) -> None:
        result = OrderResult(
            success=True, ticket=12345, price=1.1050, lots=0.1
        )
        assert result.success
        assert result.ticket == 12345

    def test_get_open_positions_without_connection(self) -> None:
        connector = MT5LiveConnector()
        positions = connector.get_open_positions()
        assert positions == []

    def test_market_order_without_connection(self) -> None:
        connector = MT5LiveConnector()
        connector._connected = False
        result = connector.market_order(direction=1, lots=0.01)
        assert not result.success


# ─────────────────────────────────────────────
# Inference Pipeline Tests
# ─────────────────────────────────────────────

class TestInferencePipeline:

    def test_buy_decision(self) -> None:
        """High positive confidence → BUY."""
        model_fn = lambda s: np.array([0.8, 0.5, 1.0, 2.0])
        pipeline = LiveInferencePipeline(model_fn, _config())
        result = pipeline.infer(np.zeros(14), hour_utc=12)
        assert result.decision == "BUY"

    def test_sell_decision(self) -> None:
        """High negative confidence → SELL."""
        model_fn = lambda s: np.array([-0.8, 0.5, 1.0, 2.0])
        pipeline = LiveInferencePipeline(model_fn, _config())
        result = pipeline.infer(np.zeros(14), hour_utc=12)
        assert result.decision == "SELL"

    def test_hold_decision(self) -> None:
        """Low confidence → HOLD."""
        model_fn = lambda s: np.array([0.1, 0.5, 1.0, 2.0])
        pipeline = LiveInferencePipeline(model_fn, _config())
        result = pipeline.infer(np.zeros(14), hour_utc=12)
        assert result.decision == "HOLD"

    def test_blocked_by_hard_killswitch(self) -> None:
        """Hard killswitch → BLOCKED."""
        pipeline = LiveInferencePipeline(_dummy_model, _config())
        result = pipeline.infer(np.zeros(14), hour_utc=12, killswitch_status="hard")
        assert result.decision == "BLOCKED"

    def test_outside_trading_hours(self) -> None:
        """Outside trading hours → OUTSIDE_HOURS."""
        pipeline = LiveInferencePipeline(_dummy_model, _config())
        result = pipeline.infer(np.zeros(14), hour_utc=23)
        assert result.decision == "OUTSIDE_HOURS"

    def test_soft_killswitch_reduces_exposure(self) -> None:
        """Soft killswitch: only high confidence passes, risk reduced."""
        # Very high confidence should still trade
        high_conf = lambda s: np.array([0.9, 0.5, 1.0, 2.0])
        pipeline = LiveInferencePipeline(high_conf, _config())
        result = pipeline.infer(np.zeros(14), hour_utc=12, killswitch_status="soft")
        # 0.9 >= 0.3 * 1.5 = 0.45, so should trade
        assert result.decision == "BUY"

    def test_session_stats(self) -> None:
        pipeline = LiveInferencePipeline(_dummy_model, _config())
        pipeline.infer(np.zeros(14), hour_utc=12)
        pipeline.infer(np.zeros(14), hour_utc=12)
        pipeline.infer(np.zeros(14), hour_utc=23)  # Outside hours
        stats = pipeline.get_session_stats()
        assert stats["total_inferences"] == 3
        assert stats["buys"] == 2
        assert stats["outside_hours"] == 1

    def test_inference_result_fields(self) -> None:
        pipeline = LiveInferencePipeline(_dummy_model, _config())
        result = pipeline.infer(np.zeros(14), hour_utc=12)
        assert isinstance(result, InferenceResult)
        assert "confidence" in result.action
        assert result.killswitch_status == "normal"


# ─────────────────────────────────────────────
# Paper Trading Tests
# ─────────────────────────────────────────────

class TestPaperTrading:

    def test_session_lifecycle(self) -> None:
        orch = PaperTradingOrchestrator(_config(), _dummy_model)
        session = orch.start_session(10000.0)
        assert session.start_balance == 10000.0

        orch.record_trade(pnl=50.0, is_winner=True)
        orch.record_trade(pnl=-20.0, is_winner=False)
        orch.end_session(10030.0)

        assert len(orch.sessions) == 1
        assert orch.sessions[0].total_trades == 2
        assert orch.sessions[0].winning_trades == 1

    def test_report_generation(self) -> None:
        orch = PaperTradingOrchestrator(_config(), _dummy_model)
        # Simulate 5 trading days
        balance = 10000.0
        for day in range(5):
            orch.start_session(balance)
            orch.record_trade(pnl=200.0, is_winner=True)
            orch.record_trade(pnl=-50.0, is_winner=False)
            balance += 150.0
            orch.update_equity(balance)
            orch.end_session(balance)

        report = orch.generate_report()
        assert report.total_days == 5
        assert report.total_trades == 10
        assert report.final_balance > report.initial_balance

    def test_prop_firm_pass_criteria(self) -> None:
        """Should pass when all Prop Firm rules are met."""
        orch = PaperTradingOrchestrator(_config(), _dummy_model)
        balance = 10000.0
        for day in range(6):
            orch.start_session(balance)
            orch.record_trade(pnl=200.0, is_winner=True)
            balance += 200.0
            orch.update_equity(balance)
            orch.end_session(balance)

        report = orch.generate_report()
        # 6 days, 12% return, should pass
        assert report.prop_firm_pass

    def test_prop_firm_fail_insufficient_days(self) -> None:
        orch = PaperTradingOrchestrator(_config(), _dummy_model)
        orch.start_session(10000.0)
        orch.end_session(10500.0)
        report = orch.generate_report()
        assert not report.prop_firm_pass
        assert "trading days" in report.failure_reason

    def test_report_save_load(self, tmp_path: Path) -> None:
        orch = PaperTradingOrchestrator(_config(), _dummy_model)
        orch.start_session(10000.0)
        orch.end_session(10200.0)
        report = orch.generate_report()

        path = tmp_path / "report.json"
        report.save(path)
        assert path.exists()

        with open(path) as f:
            data = json.load(f)
        assert "total_trades" in data

    def test_win_rate_calculation(self) -> None:
        session = TradingSession(
            date="2025-01-01", start_balance=10000,
            total_trades=10, winning_trades=6,
        )
        assert session.win_rate == 0.6

    def test_daily_return_pct(self) -> None:
        session = TradingSession(
            date="2025-01-01", start_balance=10000,
            end_balance=10200,
        )
        assert abs(session.daily_return_pct - 2.0) < 0.01

    def test_alert_callback(self) -> None:
        orch = PaperTradingOrchestrator(_config(), _dummy_model)
        alerts = []
        orch.set_alert_callback(lambda t, d: alerts.append(t))
        orch.start_session(10000.0)
        orch.end_session(10100.0)
        assert len(alerts) == 1  # Session end alert
