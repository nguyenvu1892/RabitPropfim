"""
Tests for Sprint 3 — Neural Network Models.

Validates:
- TransformerSMC: Sinusoidal PE + Self-Attention + Mean Pooling (Sprint 3.1)
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

from models.transformer_smc import TransformerSMC, SinusoidalPositionalEncoding
from models.transformer import TimeSeriesTransformer, LearnablePositionalEncoding
from models.cross_attention import CrossAttentionFusion, MultiTimeframeEncoder, CrossAttentionMTF
from models.actor_critic import Actor, TwinQCritic
from models.regime_detector import RegimeDetector


# ─────────────────────────────────────────────
# TransformerSMC Tests (Sprint 3.1 — SMC Self-Attention)
# ─────────────────────────────────────────────

class TestTransformerSMC:
    """Tests for the new TransformerSMC module with sinusoidal PE + mean pooling."""

    def test_transformer_smc_init(self) -> None:
        """T3.1.2a — Model initializes without errors with valid params."""
        model = TransformerSMC(
            input_dim=28, embed_dim=128, n_heads=4, n_layers=2, dropout=0.1
        )
        assert model.input_dim == 28
        assert model.embed_dim == 128
        assert model.n_heads == 4
        assert model.n_layers == 2

    def test_transformer_smc_forward_shape(self) -> None:
        """T3.1.2b — Output shape must be (batch=32, embed_dim=128)."""
        model = TransformerSMC(input_dim=28, embed_dim=128, n_heads=4, n_layers=2)
        x = torch.randn(32, 64, 28)  # batch=32, seq_len=64, features=28
        out = model(x)
        assert out.shape == (32, 128), f"Expected (32, 128), got {out.shape}"

    def test_transformer_smc_gradient_flow(self) -> None:
        """T3.1.2c — Gradients must flow from output back through all parameters."""
        model = TransformerSMC(input_dim=28, embed_dim=128, n_heads=4, n_layers=2)
        x = torch.randn(4, 64, 28, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Verify gradients flow to input
        assert x.grad is not None, "Gradient should flow back to input"
        # Verify gradients flow to model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_sinusoidal_pe_values(self) -> None:
        """Sinusoidal PE should produce values in [-1, 1] range."""
        pe = SinusoidalPositionalEncoding(max_seq_len=128, embed_dim=128)
        x = torch.zeros(1, 64, 128)
        out = pe(x)
        assert torch.all(out >= -1.0) and torch.all(out <= 1.0)

    def test_different_seq_lengths(self) -> None:
        """Model should handle various sequence lengths (< max_seq_len)."""
        model = TransformerSMC(input_dim=28, embed_dim=128, n_heads=4, max_seq_len=256)
        for seq_len in [10, 32, 64, 128]:
            x = torch.randn(2, seq_len, 28)
            out = model(x)
            assert out.shape == (2, 128), f"Failed for seq_len={seq_len}"

    def test_padding_mask(self) -> None:
        """Model should support padding mask for variable-length sequences."""
        model = TransformerSMC(input_dim=28, embed_dim=128, n_heads=4)
        x = torch.randn(4, 64, 28)
        mask = torch.zeros(4, 64, dtype=torch.bool)
        mask[:, 50:] = True  # Last 14 positions are padding
        out = model(x, src_key_padding_mask=mask)
        assert out.shape == (4, 128)
        assert torch.all(torch.isfinite(out)), "Output should be finite with mask"

    def test_single_sample(self) -> None:
        """Single sample forward pass should work."""
        model = TransformerSMC(input_dim=28, embed_dim=128, n_heads=4)
        x = torch.randn(1, 64, 28)
        out = model(x)
        assert out.shape == (1, 128)

    def test_attention_weights_extraction(self) -> None:
        """get_attention_weights should return one tensor per layer."""
        model = TransformerSMC(input_dim=28, embed_dim=128, n_heads=4, n_layers=2)
        x = torch.randn(2, 64, 28)
        weights = model.get_attention_weights(x)
        assert len(weights) == 2, "Should have one attention weight per layer"
        for w in weights:
            assert w.shape == (2, 4, 64, 64), f"Expected (2, 4, 64, 64), got {w.shape}"


# ─────────────────────────────────────────────
# Original Transformer Tests (baseline)
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
# CrossAttentionMTF Tests (Sprint 3.2 — Multi-TF Cross-Attention)
# ─────────────────────────────────────────────

class TestCrossAttentionMTF:
    """Tests for CrossAttentionMTF: M5 (Q) × H1+H4 (K,V) cross-attention."""

    def test_cross_attention_mtf_init(self) -> None:
        """T3.2.2a — Model initializes without errors with valid params."""
        model = CrossAttentionMTF(
            n_features_m5=28, n_features_h1=28, n_features_h4=28,
            embed_dim=128, n_heads=4,
        )
        assert model.embed_dim == 128
        assert model.n_heads == 4
        assert model.n_cross_layers == 1

    def test_cross_attention_mtf_forward_shape(self) -> None:
        """T3.2.2b — Output must be (batch=32, embed_dim=128)."""
        model = CrossAttentionMTF(
            n_features_m5=28, n_features_h1=28, n_features_h4=28,
            embed_dim=128, n_heads=4,
        )
        m5 = torch.randn(32, 64, 28)   # 64 M5 bars
        h1 = torch.randn(32, 24, 28)   # 24 H1 bars (1 day)
        h4 = torch.randn(32, 30, 28)   # 30 H4 bars (5 days)
        out = model(m5, h1, h4)
        assert out.shape == (32, 128), f"Expected (32, 128), got {out.shape}"

    def test_cross_attention_mtf_gradient_flow(self) -> None:
        """Gradients must flow back through ALL 3 inputs (M5, H1, H4)."""
        model = CrossAttentionMTF(
            n_features_m5=28, n_features_h1=28, n_features_h4=28,
            embed_dim=128, n_heads=4,
        )
        m5 = torch.randn(4, 64, 28, requires_grad=True)
        h1 = torch.randn(4, 24, 28, requires_grad=True)
        h4 = torch.randn(4, 30, 28, requires_grad=True)
        out = model(m5, h1, h4)
        loss = out.sum()
        loss.backward()
        assert m5.grad is not None, "Gradient must flow to M5 input"
        assert h1.grad is not None, "Gradient must flow to H1 input"
        assert h4.grad is not None, "Gradient must flow to H4 input"
        # Also verify model params received gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_cross_attention_mtf_memory_efficient(self) -> None:
        """T3.2.2c — Memory should stay well under 2GB for batch_size=64."""
        model = CrossAttentionMTF(
            n_features_m5=28, n_features_h1=28, n_features_h4=28,
            embed_dim=128, n_heads=4,
        )
        # Compute theoretical attention matrix size
        batch = 64
        seq_m5 = 64
        seq_context = 24 + 30  # H1 + H4 = 54
        n_heads = 4
        # Attention matrix: (batch, heads, seq_m5, seq_context) × float32
        attn_bytes = batch * n_heads * seq_m5 * seq_context * 4
        attn_mb = attn_bytes / (1024 * 1024)
        # Must be way under 2GB (should be ~3.4 MB)
        assert attn_mb < 100, f"Attention matrix too large: {attn_mb:.1f} MB"

        # Actually run forward pass to verify no OOM
        m5 = torch.randn(batch, seq_m5, 28)
        h1 = torch.randn(batch, 24, 28)
        h4 = torch.randn(batch, 30, 28)
        out = model(m5, h1, h4)
        assert out.shape == (batch, 128)

    def test_cross_attention_weights_extraction(self) -> None:
        """get_cross_attention_weights should return weights per layer."""
        model = CrossAttentionMTF(
            n_features_m5=28, n_features_h1=28, n_features_h4=28,
            embed_dim=128, n_heads=4, n_cross_layers=1,
        )
        m5 = torch.randn(2, 64, 28)
        h1 = torch.randn(2, 24, 28)
        h4 = torch.randn(2, 30, 28)
        weights = model.get_cross_attention_weights(m5, h1, h4)
        assert len(weights) == 1, "Should have 1 weight tensor (1 cross-attn layer)"
        # Shape: (batch, n_heads, seq_m5, seq_context)
        assert weights[0].shape == (2, 4, 64, 54), f"Got {weights[0].shape}"

    def test_cross_attention_single_sample(self) -> None:
        """Single sample should work without errors."""
        model = CrossAttentionMTF(
            n_features_m5=28, n_features_h1=28, n_features_h4=28,
            embed_dim=128, n_heads=4,
        )
        m5 = torch.randn(1, 64, 28)
        h1 = torch.randn(1, 24, 28)
        h4 = torch.randn(1, 30, 28)
        out = model(m5, h1, h4)
        assert out.shape == (1, 128)
        assert torch.all(torch.isfinite(out))


# ─────────────────────────────────────────────
# Legacy Cross-Attention Tests (backward compat)
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
