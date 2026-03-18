"""
Tests for Sprint 3 — Neural Network Models.

Validates:
- Transformer forward pass shape correctness
- Cross-Attention fusion produces correct output dims
- Actor outputs valid actions and log probs
- Twin Critic outputs valid Q-values
- Regime detector produces valid probabilities
- Multi-TF encoder end-to-end pipeline
"""

from __future__ import annotations

import torch
import pytest

from models.transformer import TimeSeriesTransformer, LearnablePositionalEncoding
from models.cross_attention import CrossAttentionFusion, MultiTimeframeEncoder
from models.actor_critic import Actor, TwinQCritic
from models.regime_detector import RegimeDetector


# ─────────────────────────────────────────────
# Transformer Tests
# ─────────────────────────────────────────────

class TestTransformer:

    def test_output_shape(self) -> None:
        model = TimeSeriesTransformer(n_features=14, d_model=64, n_heads=4, n_layers=2)
        x = torch.randn(8, 96, 14)  # batch=8, seq=96, features=14
        out = model(x)
        assert out.shape == (8, 64), f"Expected (8, 64), got {out.shape}"

    def test_single_sample(self) -> None:
        model = TimeSeriesTransformer(n_features=14, d_model=64, n_heads=4, n_layers=2)
        x = torch.randn(1, 50, 14)
        out = model(x)
        assert out.shape == (1, 64)

    def test_gradient_flow(self) -> None:
        model = TimeSeriesTransformer(n_features=14, d_model=64, n_heads=4, n_layers=2)
        x = torch.randn(4, 96, 14, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should flow back to input"

    def test_different_seq_lengths(self) -> None:
        model = TimeSeriesTransformer(n_features=14, d_model=64, n_heads=4, max_seq_len=200)
        for seq_len in [10, 50, 96, 150]:
            x = torch.randn(2, seq_len, 14)
            out = model(x)
            assert out.shape == (2, 64)


# ─────────────────────────────────────────────
# Cross-Attention Tests
# ─────────────────────────────────────────────

class TestCrossAttention:

    def test_fusion_output_shape(self) -> None:
        fusion = CrossAttentionFusion(d_model=64, n_heads=4)
        query = torch.randn(8, 64)    # M15 representation
        context = torch.randn(8, 2, 64)  # [H1, H4] stacked
        out = fusion(query, context)
        assert out.shape == (8, 64)

    def test_multi_tf_encoder(self) -> None:
        encoder = MultiTimeframeEncoder(
            n_features_m15=14, n_features_h1=14, n_features_h4=14,
            d_model=64, n_heads=4, n_layers=2,
        )
        m15 = torch.randn(4, 96, 14)
        h1 = torch.randn(4, 48, 14)
        h4 = torch.randn(4, 30, 14)
        out = encoder(m15, h1, h4)
        assert out.shape == (4, 64)

    def test_multi_tf_gradient_flow(self) -> None:
        encoder = MultiTimeframeEncoder(
            n_features_m15=14, n_features_h1=14, n_features_h4=14,
            d_model=64, n_heads=4, n_layers=2,
        )
        m15 = torch.randn(2, 96, 14, requires_grad=True)
        h1 = torch.randn(2, 48, 14, requires_grad=True)
        h4 = torch.randn(2, 30, 14, requires_grad=True)
        out = encoder(m15, h1, h4)
        loss = out.sum()
        loss.backward()
        assert m15.grad is not None
        assert h1.grad is not None
        assert h4.grad is not None


# ─────────────────────────────────────────────
# Actor Tests
# ─────────────────────────────────────────────

class TestActor:

    def test_output_shape(self) -> None:
        actor = Actor(state_dim=64, action_dim=4)
        state = torch.randn(8, 64)
        action, log_prob = actor(state)
        assert action.shape == (8, 4)
        assert log_prob.shape == (8, 1)

    def test_action_bounded(self) -> None:
        """Actions should be bounded in [-1, 1] due to tanh."""
        actor = Actor(state_dim=64, action_dim=4)
        state = torch.randn(32, 64)
        action, _ = actor(state)
        assert torch.all(action >= -1.0), "Actions should be >= -1"
        assert torch.all(action <= 1.0), "Actions should be <= 1"

    def test_deterministic_mode(self) -> None:
        actor = Actor(state_dim=64, action_dim=4)
        state = torch.randn(4, 64)
        a1, _ = actor(state, deterministic=True)
        a2, _ = actor(state, deterministic=True)
        assert torch.allclose(a1, a2), "Deterministic mode should be reproducible"

    def test_stochastic_mode_varies(self) -> None:
        actor = Actor(state_dim=64, action_dim=4)
        state = torch.randn(4, 64)
        a1, _ = actor(state, deterministic=False)
        a2, _ = actor(state, deterministic=False)
        # With different random samples, actions should differ
        # (extremely unlikely to be identical)
        assert not torch.allclose(a1, a2), "Stochastic mode should produce different actions"

    def test_log_prob_finite(self) -> None:
        actor = Actor(state_dim=64, action_dim=4)
        state = torch.randn(16, 64)
        _, log_prob = actor(state)
        assert torch.all(torch.isfinite(log_prob)), "Log probs should be finite"


# ─────────────────────────────────────────────
# Critic Tests
# ─────────────────────────────────────────────

class TestCritic:

    def test_twin_q_output_shape(self) -> None:
        critic = TwinQCritic(state_dim=64, action_dim=4)
        state = torch.randn(8, 64)
        action = torch.randn(8, 4)
        q1, q2 = critic(state, action)
        assert q1.shape == (8, 1)
        assert q2.shape == (8, 1)

    def test_min_q(self) -> None:
        critic = TwinQCritic(state_dim=64, action_dim=4)
        state = torch.randn(8, 64)
        action = torch.randn(8, 4)
        min_q = critic.min_q(state, action)
        q1, q2 = critic(state, action)
        expected_min = torch.min(q1, q2)
        assert torch.allclose(min_q, expected_min)


# ─────────────────────────────────────────────
# Regime Detector Tests
# ─────────────────────────────────────────────

class TestRegimeDetector:

    def test_output_shapes(self) -> None:
        detector = RegimeDetector(input_dim=64, n_regimes=4)
        x = torch.randn(8, 64)
        probs, emb = detector(x)
        assert probs.shape == (8, 4)
        assert emb.shape == (8, 64)

    def test_probabilities_sum_to_one(self) -> None:
        detector = RegimeDetector(input_dim=64, n_regimes=4)
        x = torch.randn(16, 64)
        probs, _ = detector(x)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(16), atol=1e-5)

    def test_predict_regime_returns_indices(self) -> None:
        detector = RegimeDetector(input_dim=64, n_regimes=4)
        x = torch.randn(8, 64)
        regime_idx = detector.predict_regime(x)
        assert regime_idx.shape == (8,)
        assert torch.all(regime_idx >= 0)
        assert torch.all(regime_idx < 4)

    def test_regime_names(self) -> None:
        detector = RegimeDetector(input_dim=64, n_regimes=4)
        assert len(detector.regime_names) == 4
        assert "trend_up" in detector.regime_names
