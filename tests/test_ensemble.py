"""
Tests for Sprint 5.1: EnsembleAgent.

Uses mock agents to verify regime-aware weighted voting,
top-agent SL/TP selection, and score computation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from agents.ensemble_agent import (
    EnsembleAgent,
    REGIME_BOOST_TABLE,
    MAX_WEIGHT,
    MIN_WEIGHT,
)


# =============================================
# Mock Agent (mimics SACTransformerActor)
# =============================================

class MockAgent(nn.Module):
    """
    Returns a fixed action when called.
    action: [confidence, risk, sl, tp] in [-1, 1]
    """

    def __init__(self, action: list[float]) -> None:
        super().__init__()
        self._action = torch.tensor([action], dtype=torch.float32)
        # Dummy param so it's a valid nn.Module
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self, m5, h1, h4, deterministic=True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = m5.shape[0]
        action = self._action.expand(batch, -1)
        log_prob = torch.zeros(batch, 1)
        return action, log_prob


# =============================================
# Tests
# =============================================

class TestEnsembleAgent:
    """Core EnsembleAgent tests."""

    @pytest.fixture
    def dummy_inputs(self):
        """Dummy M5/H1/H4 tensors, batch=1."""
        m5 = torch.randn(1, 64, 28)
        h1 = torch.randn(1, 48, 28)
        h4 = torch.randn(1, 24, 28)
        return m5, h1, h4

    def test_trending_boosts_trend_agent(self, dummy_inputs) -> None:
        """
        In a trending scenario, TrendAgent (idx=0) should dominate.
        Its SL/TP should be selected as the final action's SL/TP.
        """
        # TrendAgent: BUY 0.9, risk=0.5, SL=0.3, TP=0.8
        trend_agent = MockAgent([0.9, 0.5, 0.3, 0.8])
        # RangeAgent: BUY 0.3, risk=0.2, SL=0.1, TP=0.4
        range_agent = MockAgent([0.3, 0.2, 0.1, 0.4])
        # VolAgent: SELL -0.2, risk=0.7, SL=0.6, TP=0.9
        vol_agent = MockAgent([-0.2, 0.7, 0.6, 0.9])

        ensemble = EnsembleAgent(
            agents=[trend_agent, range_agent, vol_agent],
            regime_detector=None,  # No regime -> equal base weights
            action_gating=None,
            base_weights=[0.4, 0.3, 0.3],
        )

        m5, h1, h4 = dummy_inputs
        action = ensemble.get_action(m5, h1, h4)

        # Check shape
        assert action.shape == (4,)

        # TrendAgent has highest confidence (0.9) and highest base_weight (0.4)
        # -> score_trend = 0.9 * 0.4 = 0.36
        # -> score_range = 0.3 * 0.3 = 0.09
        # -> score_vol   = 0.2 * 0.3 = 0.06
        # top_idx = 0 (TrendAgent) -> SL/TP from TrendAgent
        assert action[1] == pytest.approx(0.5, abs=1e-5)  # risk from Trend
        assert action[2] == pytest.approx(0.3, abs=1e-5)  # SL from Trend
        assert action[3] == pytest.approx(0.8, abs=1e-5)  # TP from Trend

    def test_direction_is_weighted_sum(self, dummy_inputs) -> None:
        """
        Direction should be sum(a[0] * score_i), not majority vote.
        """
        # Agent 0: BUY  0.8 (strong)
        # Agent 1: SELL -0.6
        # Agent 2: BUY  0.4
        agents = [
            MockAgent([0.8, 0.1, 0.1, 0.1]),
            MockAgent([-0.6, 0.2, 0.2, 0.2]),
            MockAgent([0.4, 0.3, 0.3, 0.3]),
        ]

        ensemble = EnsembleAgent(
            agents=agents,
            regime_detector=None,
            action_gating=None,
            base_weights=[0.4, 0.3, 0.3],
        )

        m5, h1, h4 = dummy_inputs
        action = ensemble.get_action(m5, h1, h4)

        # Compute expected:
        # No regime -> normalized base_weights = [0.4, 0.3, 0.3]
        w = np.array([0.4, 0.3, 0.3])
        confs = [0.8, 0.6, 0.4]
        dirs = [0.8, -0.6, 0.4]
        scores = [c * wi for c, wi in zip(confs, w)]
        expected_dir = sum(d * s for d, s in zip(dirs, scores))

        assert action[0] == pytest.approx(expected_dir, abs=1e-4)

    def test_top_agent_sl_tp_not_averaged(self, dummy_inputs) -> None:
        """
        SL/TP must come from the single top-scoring agent, NOT averaged.
        """
        # Agent 0: low confidence, unique SL/TP
        # Agent 1: HIGH confidence, unique SL/TP
        # Agent 2: medium confidence, unique SL/TP
        agents = [
            MockAgent([0.1, 0.11, 0.12, 0.13]),
            MockAgent([0.9, 0.21, 0.22, 0.23]),  # Top scorer
            MockAgent([0.5, 0.31, 0.32, 0.33]),
        ]

        ensemble = EnsembleAgent(
            agents=agents,
            regime_detector=None,
            action_gating=None,
            base_weights=[0.3, 0.4, 0.3],  # Agent 1 has highest base
        )

        m5, h1, h4 = dummy_inputs
        action = ensemble.get_action(m5, h1, h4)

        # Agent 1 score = 0.9 * 0.4 = 0.36 (highest)
        # So SL/TP should be from Agent 1
        assert action[1] == pytest.approx(0.21, abs=1e-5)
        assert action[2] == pytest.approx(0.22, abs=1e-5)
        assert action[3] == pytest.approx(0.23, abs=1e-5)

    def test_all_hold_returns_near_zero_direction(self, dummy_inputs) -> None:
        """When all agents have near-zero confidence, direction stays near zero."""
        agents = [
            MockAgent([0.01, 0.1, 0.1, 0.1]),
            MockAgent([-0.02, 0.2, 0.2, 0.2]),
            MockAgent([0.01, 0.3, 0.3, 0.3]),
        ]

        ensemble = EnsembleAgent(
            agents=agents,
            regime_detector=None,
            action_gating=None,
        )

        m5, h1, h4 = dummy_inputs
        action = ensemble.get_action(m5, h1, h4)

        # Very small confidences -> direction near 0
        assert abs(action[0]) < 0.01

    def test_two_agents_minimum(self) -> None:
        """Ensemble works with just 2 agents."""
        agents = [MockAgent([0.5, 0.1, 0.2, 0.3]), MockAgent([0.6, 0.4, 0.5, 0.6])]

        ensemble = EnsembleAgent(
            agents=agents,
            regime_detector=None,
            action_gating=None,
            base_weights=[0.5, 0.5],
        )

        m5 = torch.randn(1, 64, 28)
        h1 = torch.randn(1, 48, 28)
        h4 = torch.randn(1, 24, 28)
        action = ensemble.get_action(m5, h1, h4)
        assert action.shape == (4,)

    def test_single_agent_raises(self) -> None:
        """Need at least 2 agents."""
        with pytest.raises(AssertionError, match="at least 2"):
            EnsembleAgent(agents=[MockAgent([0.5, 0.1, 0.2, 0.3])])

    def test_diagnostics_output(self, dummy_inputs) -> None:
        """get_agent_diagnostics returns all expected keys."""
        agents = [
            MockAgent([0.8, 0.1, 0.2, 0.3]),
            MockAgent([0.4, 0.4, 0.5, 0.6]),
            MockAgent([0.6, 0.7, 0.8, 0.9]),
        ]

        ensemble = EnsembleAgent(
            agents=agents, regime_detector=None, action_gating=None,
        )

        m5, h1, h4 = dummy_inputs
        diag = ensemble.get_agent_diagnostics(m5, h1, h4)

        assert "weights" in diag
        assert "actions" in diag
        assert "scores" in diag
        assert "top_agent_idx" in diag
        assert "weighted_direction" in diag
        assert "final_action" in diag
        assert len(diag["actions"]) == 3
        assert len(diag["scores"]) == 3

    def test_weight_normalization(self, dummy_inputs) -> None:
        """Weights should sum to 1.0."""
        agents = [
            MockAgent([0.5, 0.1, 0.1, 0.1]),
            MockAgent([0.5, 0.2, 0.2, 0.2]),
            MockAgent([0.5, 0.3, 0.3, 0.3]),
        ]

        ensemble = EnsembleAgent(
            agents=agents, regime_detector=None, action_gating=None,
            base_weights=[0.5, 0.3, 0.2],
        )

        m5, _, _ = dummy_inputs
        weights = ensemble._compute_weights(m5)

        assert weights.sum() == pytest.approx(1.0, abs=1e-5)

    def test_inference_speed(self, dummy_inputs) -> None:
        """get_action should complete in under 5ms (no GPU, mock agents)."""
        import time

        agents = [
            MockAgent([0.8, 0.1, 0.2, 0.3]),
            MockAgent([0.4, 0.4, 0.5, 0.6]),
            MockAgent([0.6, 0.7, 0.8, 0.9]),
        ]

        ensemble = EnsembleAgent(
            agents=agents, regime_detector=None, action_gating=None,
        )

        m5, h1, h4 = dummy_inputs

        # Warmup
        ensemble.get_action(m5, h1, h4)

        # Time 100 calls
        start = time.perf_counter()
        for _ in range(100):
            ensemble.get_action(m5, h1, h4)
        elapsed = (time.perf_counter() - start) / 100 * 1000  # ms

        assert elapsed < 5.0, f"Too slow: {elapsed:.2f}ms per call"


class TestRegimeBoostTable:
    """Tests for the precomputed boost table."""

    def test_table_shape(self) -> None:
        assert REGIME_BOOST_TABLE.shape == (4, 3)

    def test_trending_boosts_trend_agent(self) -> None:
        """Regime 0 (trend_up) should boost column 0 (TrendAgent)."""
        assert REGIME_BOOST_TABLE[0, 0] > REGIME_BOOST_TABLE[0, 1]  # Trend > Range

    def test_ranging_boosts_range_agent(self) -> None:
        """Regime 2 (ranging) should boost column 1 (RangeAgent)."""
        assert REGIME_BOOST_TABLE[2, 1] > REGIME_BOOST_TABLE[2, 0]  # Range > Trend

    def test_volatile_boosts_vol_agent(self) -> None:
        """Regime 3 (volatile) should boost column 2 (VolAgent)."""
        assert REGIME_BOOST_TABLE[3, 2] > REGIME_BOOST_TABLE[3, 0]  # Vol > Trend
