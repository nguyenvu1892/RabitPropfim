"""
Tests for Action Gating — Sprint 3.5.

Validates:
- HOLD when |confidence| < threshold
- BUY when confidence > threshold
- SELL when confidence < -threshold
- Edge cases at boundary (c=0.29, c=0.31)
- Risk scaling proportional to confidence
"""

from __future__ import annotations

import torch
import pytest

from agents.action_gating import ActionGating, TradeSignal, GatedAction


class TestActionGating:
    """Tests for ActionGating confidence-based trade filter."""

    def test_hold_low_positive_confidence(self) -> None:
        """c=0.1 (< 0.3 threshold) → HOLD, risk=0."""
        gating = ActionGating(confidence_threshold=0.3)
        # action: [confidence=0.1, risk=0.5, sl=0.0, tp=0.0]
        action = torch.tensor([[0.1, 0.5, 0.0, 0.0]])
        result = gating.gate_single(action)
        assert result.signal == TradeSignal.HOLD
        assert result.risk_fraction == 0.0
        assert result.confidence_scale == 0.0

    def test_hold_low_negative_confidence(self) -> None:
        """c=-0.2 (|c|=0.2 < 0.3 threshold) → HOLD."""
        gating = ActionGating(confidence_threshold=0.3)
        action = torch.tensor([[-0.2, 0.8, 0.5, 0.5]])
        result = gating.gate_single(action)
        assert result.signal == TradeSignal.HOLD
        assert result.risk_fraction == 0.0

    def test_hold_zero_confidence(self) -> None:
        """c=0.0 → always HOLD regardless of other values."""
        gating = ActionGating(confidence_threshold=0.3)
        action = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
        result = gating.gate_single(action)
        assert result.signal == TradeSignal.HOLD

    def test_buy_high_confidence(self) -> None:
        """c=0.8 (> 0.3 threshold) → BUY, risk > 0."""
        gating = ActionGating(confidence_threshold=0.3)
        action = torch.tensor([[0.8, 0.5, 0.0, 0.0]])
        result = gating.gate_single(action)
        assert result.signal == TradeSignal.BUY
        assert result.risk_fraction > 0.0
        assert result.confidence_scale > 0.0

    def test_sell_negative_confidence(self) -> None:
        """c=-0.8 → SELL, risk > 0."""
        gating = ActionGating(confidence_threshold=0.3)
        action = torch.tensor([[-0.8, 0.5, 0.0, 0.0]])
        result = gating.gate_single(action)
        assert result.signal == TradeSignal.SELL
        assert result.risk_fraction > 0.0

    def test_edge_case_below_threshold(self) -> None:
        """c=0.29 (just below 0.3) → HOLD."""
        gating = ActionGating(confidence_threshold=0.3)
        action = torch.tensor([[0.29, 0.5, 0.0, 0.0]])
        result = gating.gate_single(action)
        assert result.signal == TradeSignal.HOLD
        assert result.risk_fraction == 0.0

    def test_edge_case_above_threshold(self) -> None:
        """c=0.31 (just above 0.3) → BUY with small risk."""
        gating = ActionGating(confidence_threshold=0.3)
        action = torch.tensor([[0.31, 0.5, 0.0, 0.0]])
        result = gating.gate_single(action)
        assert result.signal == TradeSignal.BUY
        assert result.risk_fraction > 0.0
        # Confidence scale should be very small (~0.014)
        assert result.confidence_scale < 0.05

    def test_edge_case_negative_boundary(self) -> None:
        """c=-0.31 → SELL."""
        gating = ActionGating(confidence_threshold=0.3)
        action = torch.tensor([[-0.31, 0.5, 0.0, 0.0]])
        result = gating.gate_single(action)
        assert result.signal == TradeSignal.SELL

    def test_risk_scales_with_confidence(self) -> None:
        """Higher confidence → higher risk fraction."""
        gating = ActionGating(confidence_threshold=0.3)
        # Same raw_risk, different confidence
        low_conf = torch.tensor([[0.4, 0.5, 0.0, 0.0]])
        high_conf = torch.tensor([[0.9, 0.5, 0.0, 0.0]])
        result_low = gating.gate_single(low_conf)
        result_high = gating.gate_single(high_conf)
        assert result_high.risk_fraction > result_low.risk_fraction, (
            f"high conf risk {result_high.risk_fraction} should > "
            f"low conf risk {result_low.risk_fraction}"
        )

    def test_max_confidence_gives_max_scale(self) -> None:
        """c=1.0 → confidence_scale = 1.0 (maximum)."""
        gating = ActionGating(confidence_threshold=0.3)
        action = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        result = gating.gate_single(action)
        assert result.signal == TradeSignal.BUY
        assert abs(result.confidence_scale - 1.0) < 1e-5

    def test_batch_gating(self) -> None:
        """Batch of actions: mix of HOLD, BUY, SELL."""
        gating = ActionGating(confidence_threshold=0.3)
        actions = torch.tensor([
            [0.1, 0.5, 0.0, 0.0],   # HOLD
            [0.8, 0.5, 0.0, 0.0],   # BUY
            [-0.6, 0.5, 0.0, 0.0],  # SELL
            [0.0, 0.5, 0.0, 0.0],   # HOLD
        ])
        results = gating.gate(actions)
        assert len(results) == 4
        assert results[0].signal == TradeSignal.HOLD
        assert results[1].signal == TradeSignal.BUY
        assert results[2].signal == TradeSignal.SELL
        assert results[3].signal == TradeSignal.HOLD

    def test_sl_tp_multiplier_range(self) -> None:
        """SL/TP multipliers should be in [0.5, 2.0]."""
        gating = ActionGating(confidence_threshold=0.3)
        # Extreme SL/TP values
        for sl_raw, tp_raw in [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)]:
            action = torch.tensor([[0.8, 0.5, sl_raw, tp_raw]])
            result = gating.gate_single(action)
            assert 0.45 <= result.sl_multiplier <= 2.05  # small float tolerance
            assert 0.45 <= result.tp_multiplier <= 2.05

    def test_custom_threshold(self) -> None:
        """Different threshold: c=0.4 HOLD at threshold=0.5, BUY at threshold=0.3."""
        gating_strict = ActionGating(confidence_threshold=0.5)
        gating_loose = ActionGating(confidence_threshold=0.3)
        action = torch.tensor([[0.4, 0.5, 0.0, 0.0]])
        assert gating_strict.gate_single(action).signal == TradeSignal.HOLD
        assert gating_loose.gate_single(action).signal == TradeSignal.BUY

    def test_raw_confidence_preserved(self) -> None:
        """Original confidence value should be preserved in GatedAction."""
        gating = ActionGating(confidence_threshold=0.3)
        action = torch.tensor([[0.75, 0.5, 0.0, 0.0]])
        result = gating.gate_single(action)
        assert abs(result.raw_confidence - 0.75) < 1e-5
