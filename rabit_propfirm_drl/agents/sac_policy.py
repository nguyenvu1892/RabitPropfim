"""
SAC Policy with Hierarchical Cross-Attention — The "Brain" of the AI Trader.

v2.0 — Cognitive Architecture Integration:
    - 4-TF inputs: M1, M5, M15, H1 (no more H4)
    - 50-dim feature vector per TF (28 raw + 22 knowledge)
    - HierarchicalCrossAttentionMTF: Entry (M1×M5) + Structure (M15×H1)
    - EpisodicMemory: auxiliary confidence modifier (OUTSIDE gradient graph)

Architecture Overview:
    ┌────────────────────────────────────────────────────────────────────┐
    │                   Hierarchical Feature Extractor                   │
    │                                                                    │
    │  M1 (128, 50) ──┐                                                 │
    │                  ├─ HierarchicalCrossAttentionMTF ──┐              │
    │  M5 (64, 50)  ──┘   Entry: M1(Q)×M5(KV)            │              │
    │                                                      │              │
    │  M15 (48, 50) ──┐                                   │              │
    │                  ├─ Structure: M15(Q)×H1(KV) ───────┤              │
    │  H1 (24, 50)  ──┘                                   │              │
    │                                                      ▼              │
    │  Outputs: m1_latent(128) + entry_latent(128) +                    │
    │           structure_latent(128)                                     │
    │                                                                    │
    │  m1_latent ──► RegimeDetector ──► regime_emb(128) + probs(4)     │
    │                                                                    │
    │  global_state = cat[m1, entry, structure, regime_emb, probs]      │
    │              = (128 + 128 + 128 + 128 + 4) = 516                  │
    └────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
            ┌──────────────┐       ┌──────────────┐
            │    Actor     │       │  Twin Critic  │
            │  MLP Head    │       │  Q1, Q2 Heads │
            │ → (μ, log σ) │       │ → Q(s,a)      │
            │ → tanh(N)    │       │ min(Q1,Q2)    │
            │ → action[4]  │       │               │
            └──────┬───────┘       └──────────────┘
                   │
                   ▼
            ┌──────────────┐
            │ Action Gating│  ← EpisodicMemory bonus (NO gradient!)
            │ confidence ×  │     query(knowledge_vec) → ±0.3
            │ (1 + bonus)  │
            └──────────────┘

    Action space: [confidence, risk_frac, sl_mult, tp_mult] ∈ [-1, 1]^4

All parameters configurable. Zero hardcoding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from models.cross_attention import HierarchicalCrossAttentionMTF
from models.regime_detector import RegimeDetector

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6


class HierarchicalFeatureExtractor(nn.Module):
    """
    Shared feature extractor for the Cognitive Architecture.

    Replaces the old TransformerFeatureExtractor (3-TF, flat attention) with:
    1. HierarchicalCrossAttentionMTF → m1_latent + entry_latent + structure_latent
    2. RegimeDetector → regime_emb + regime_probs

    Output: global_state = cat[m1_latent, entry_latent, structure_latent,
                               regime_emb, regime_probs]
            Shape: (batch, embed_dim * 4 + n_regimes)
            Default: (batch, 128 * 4 + 4) = (batch, 516)
    """

    def __init__(
        self,
        n_features: int = 50,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_cross_layers: int = 1,
        n_regimes: int = 4,
        dropout: float = 0.1,
        max_seq_m1: int = 192,
        max_seq_m5: int = 128,
        max_seq_m15: int = 64,
        max_seq_h1: int = 48,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_regimes = n_regimes

        # Module 1: HierarchicalCrossAttentionMTF — 4-TF fusion
        self.hierarchical_attn = HierarchicalCrossAttentionMTF(
            n_features_m1=n_features,
            n_features_m5=n_features,
            n_features_m15=n_features,
            n_features_h1=n_features,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_cross_layers=n_cross_layers,
            dropout=dropout,
            max_len_m1=max_seq_m1,
            max_len_m5=max_seq_m5,
            max_len_m15=max_seq_m15,
            max_len_h1=max_seq_h1,
        )

        # Module 2: RegimeDetector — market state classifier
        # Uses m1_latent (the deepest-encoded signal) as input
        self.regime_detector = RegimeDetector(
            input_dim=embed_dim,
            n_regimes=n_regimes,
            hidden_dim=64,
            dropout=dropout,
        )

        # Output dimension:
        # m1_latent + entry_latent + structure_latent + regime_emb + regime_probs
        self.output_dim = embed_dim * 4 + n_regimes  # 128*4 + 4 = 516

    def forward(
        self,
        m1_features: torch.Tensor,
        m5_features: torch.Tensor,
        m15_features: torch.Tensor,
        h1_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract global state from 4-TF features.

        Args:
            m1_features:  (batch, seq_m1, n_features)
            m5_features:  (batch, seq_m5, n_features)
            m15_features: (batch, seq_m15, n_features)
            h1_features:  (batch, seq_h1, n_features)

        Returns:
            (batch, output_dim=516) — global state vector
        """
        # Step 1: Hierarchical Cross-Attention
        # Entry cluster: M1(Q) × M5(KV) → precise entry timing
        # Structure cluster: M15(Q) × H1(KV) → market context
        m1_latent, entry_latent, structure_latent = self.hierarchical_attn(
            m1_features, m5_features, m15_features, h1_features,
        )  # each (B, 128)

        # Step 2: RegimeDetector — from M1 deep representation
        # "Is the market trending, ranging, or volatile?"
        regime_probs, regime_emb = self.regime_detector(m1_latent)
        # regime_probs: (B, 4), regime_emb: (B, 128)

        # Step 3: Concatenate all into global state
        global_state = torch.cat(
            [m1_latent, entry_latent, structure_latent, regime_emb, regime_probs],
            dim=-1,
        )  # (B, 516)

        return global_state


class SACTransformerActor(nn.Module):
    """
    SAC Actor with Hierarchical Cross-Attention backbone.

    v2.0: 4-TF inputs, 50-dim features, 516-dim global state.

    Output: tanh-squashed actions in [-1, 1]^4
    """

    def __init__(
        self,
        n_features: int = 50,
        action_dim: int = 4,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_cross_layers: int = 1,
        n_regimes: int = 4,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Shared feature extractor
        self.feature_extractor = HierarchicalFeatureExtractor(
            n_features=n_features,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_cross_layers=n_cross_layers,
            n_regimes=n_regimes,
            dropout=dropout,
        )

        # MLP head on top of extracted features
        state_dim = self.feature_extractor.output_dim
        hidden_dims = hidden_dims or [256, 256]

        layers: list[nn.Module] = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)

    def forward(
        self,
        m1: torch.Tensor,
        m5: torch.Tensor,
        m15: torch.Tensor,
        h1: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: 4-TF raw features → action + log_prob.

        Args:
            m1:  (batch, seq_m1, n_features)  — M1 candle features (50-dim)
            m5:  (batch, seq_m5, n_features)  — M5 candle features
            m15: (batch, seq_m15, n_features) — M15 context features
            h1:  (batch, seq_h1, n_features)  — H1 context features
            deterministic: If True, return mean action (no sampling)

        Returns:
            (action, log_prob) — action: (batch, 4), log_prob: (batch, 1)
        """
        # Extract features through Hierarchical pipeline
        global_state = self.feature_extractor(m1, m5, m15, h1)  # (B, 516)

        # MLP head
        features = self.trunk(global_state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = log_std.exp()

        # Squashed Gaussian sampling
        normal = Normal(mean, std)

        if deterministic:
            action_pre = mean
        else:
            action_pre = normal.rsample()  # Reparameterization trick

        # Tanh squash → actions in [-1, 1]
        action = torch.tanh(action_pre)

        # Log probability with tanh correction
        # log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u))
        log_prob = normal.log_prob(action_pre)
        log_prob -= torch.log(1.0 - action.pow(2) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_global_state(
        self,
        m1: torch.Tensor,
        m5: torch.Tensor,
        m15: torch.Tensor,
        h1: torch.Tensor,
    ) -> torch.Tensor:
        """Extract global state (for critic input sharing, if needed)."""
        return self.feature_extractor(m1, m5, m15, h1)


class SACTransformerCritic(nn.Module):
    """
    Twin Q-Critic with Hierarchical Cross-Attention backbone.

    Shares the same architecture as the Actor, but maintains
    separate weights (no parameter sharing).

    Q(s, a) = MLP(cat[global_state, action]) → scalar
    """

    def __init__(
        self,
        n_features: int = 50,
        action_dim: int = 4,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_cross_layers: int = 1,
        n_regimes: int = 4,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Separate feature extractor (NOT shared with actor)
        self.feature_extractor = HierarchicalFeatureExtractor(
            n_features=n_features,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_cross_layers=n_cross_layers,
            n_regimes=n_regimes,
            dropout=dropout,
        )

        state_dim = self.feature_extractor.output_dim
        hidden_dims = hidden_dims or [256, 256]

        # Twin Q-networks
        self.q1 = self._build_q_net(state_dim, action_dim, hidden_dims)
        self.q2 = self._build_q_net(state_dim, action_dim, hidden_dims)

    @staticmethod
    def _build_q_net(
        state_dim: int, action_dim: int, hidden_dims: list[int],
    ) -> nn.Sequential:
        """Build a single Q-network: (state + action) → Q-value."""
        layers: list[nn.Module] = []
        prev_dim = state_dim + action_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.GELU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def forward(
        self,
        m1: torch.Tensor,
        m5: torch.Tensor,
        m15: torch.Tensor,
        h1: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Twin Q-value forward pass.

        Args:
            m1, m5, m15, h1: 4-TF feature sequences
            action: (batch, action_dim=4)

        Returns:
            (Q1, Q2) — both (batch, 1)
        """
        global_state = self.feature_extractor(m1, m5, m15, h1)
        sa = torch.cat([global_state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def min_q(
        self,
        m1: torch.Tensor,
        m5: torch.Tensor,
        m15: torch.Tensor,
        h1: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """min(Q1, Q2) — conservative Q-estimate for SAC."""
        q1, q2 = self.forward(m1, m5, m15, h1, action)
        return torch.min(q1, q2)


def apply_episodic_memory_bonus(
    raw_confidence: float,
    memory_bonus: float,
) -> float:
    """
    Apply EpisodicMemory bonus to raw confidence score.

    CRITICAL: This function operates OUTSIDE the gradient graph.
    It must ONLY be called during inference/action-gating, NEVER
    inside a loss function or backprop path.

    The bonus adjusts the agent's confidence by up to ±30%, making
    it more/less likely to cross the action gating threshold.

    Args:
        raw_confidence: Agent's raw confidence output from Actor [-1, 1]
        memory_bonus: Float from EpisodicMemory.query() in [-0.3, +0.3]

    Returns:
        Adjusted confidence in [-1, 1]

    Example:
        Agent outputs confidence = 0.25 (below 0.3 threshold → HOLD)
        Memory bonus = +0.1 (similar setups were profitable)
        Adjusted = 0.25 * (1 + 0.1) = 0.275 → still HOLD

        Agent outputs confidence = 0.35 (above threshold → BUY)
        Memory bonus = -0.2 (similar setups lost money)
        Adjusted = 0.35 * (1 - 0.2) = 0.28 → HOLD (memory prevented bad trade!)
    """
    # Scale confidence by memory bonus
    adjusted = raw_confidence * (1.0 + memory_bonus)

    # Clamp back to [-1, 1]
    return max(-1.0, min(1.0, adjusted))


# ── Legacy backward compatibility ──
# Old code may reference TransformerFeatureExtractor from this module.
# Map to the new class with the same interface shape.
TransformerFeatureExtractor = HierarchicalFeatureExtractor
