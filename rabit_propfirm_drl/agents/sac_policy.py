"""
SAC Policy with Transformer Backbone — The "Brain" of the AI Trader.

Purpose:
    Replaces the simple MLP prototype with a full neural architecture that
    can THINK about market context before making decisions.

Architecture Overview:
    ┌────────────────────────────────────────────────────────────────────┐
    │                    Shared Feature Extractor                       │
    │                                                                    │
    │  M5 (64, 28) ──► TransformerSMC ──────────► smc_latent (128)     │
    │                                                                    │
    │  M5 + H1 + H4 ──► CrossAttentionMTF ──────► mtf_latent (128)     │
    │                                                                    │
    │  smc_latent ──► RegimeDetector ───► regime_emb (128) + probs (4) │
    │                                                                    │
    │  global_state = cat[smc_latent, mtf_latent, regime_emb] = (388)  │
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
            └──────────────┘       └──────────────┘

    Action space: [confidence, risk_frac, sl_mult, tp_mult] ∈ [-1, 1]^4
    Then ActionGating converts confidence to BUY/SELL/HOLD decisions.

All parameters configurable. Zero hardcoding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from models.transformer_smc import TransformerSMC
from models.cross_attention import CrossAttentionMTF
from models.regime_detector import RegimeDetector

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6


class TransformerFeatureExtractor(nn.Module):
    """
    Shared feature extractor combining all Sprint 3 modules.

    This is the "perception system" — it processes raw market data from
    multiple timeframes and produces a single state vector that captures:
    1. M5 pattern recognition (TransformerSMC)
    2. Multi-TF context awareness (CrossAttentionMTF)
    3. Market regime sensitivity (RegimeDetector)

    Output: global_state = cat[smc_latent, mtf_latent, regime_emb, regime_probs]
            Shape: (batch, embed_dim * 3 + n_regimes)
            Default: (batch, 128*3 + 4) = (batch, 388)
    """

    def __init__(
        self,
        n_features: int = 28,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        n_cross_layers: int = 1,
        n_regimes: int = 4,
        dropout: float = 0.1,
        max_seq_m5: int = 128,
        max_seq_h1: int = 48,
        max_seq_h4: int = 64,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_regimes = n_regimes

        # Module 1: TransformerSMC — M5 self-attention
        self.transformer_smc = TransformerSMC(
            input_dim=n_features,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_transformer_layers,
            dropout=dropout,
            max_seq_len=max_seq_m5,
        )

        # Module 2: CrossAttentionMTF — M5 × H1/H4
        self.cross_attention = CrossAttentionMTF(
            n_features_m5=n_features,
            n_features_h1=n_features,
            n_features_h4=n_features,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_cross_layers=n_cross_layers,
            dropout=dropout,
            max_len_m5=max_seq_m5,
            max_len_h1=max_seq_h1,
            max_len_h4=max_seq_h4,
        )

        # Module 3: RegimeDetector — market state classifier
        self.regime_detector = RegimeDetector(
            input_dim=embed_dim,
            n_regimes=n_regimes,
            hidden_dim=64,
            dropout=dropout,
        )

        # Output dimension: smc_latent + mtf_latent + regime_emb + regime_probs
        self.output_dim = embed_dim * 3 + n_regimes  # 128 + 128 + 128 + 4 = 388

    def forward(
        self,
        m5_features: torch.Tensor,
        h1_features: torch.Tensor,
        h4_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract global state from multi-timeframe features.

        Args:
            m5_features: (batch, seq_m5, n_features) — M5 candle features
            h1_features: (batch, seq_h1, n_features) — H1 context features
            h4_features: (batch, seq_h4, n_features) — H4 context features

        Returns:
            (batch, output_dim=388) — global state vector
        """
        # Step 1: TransformerSMC — M5 pattern recognition
        # "Which FVG/OB/BOS patterns in this M5 window are important?"
        smc_latent = self.transformer_smc(m5_features)  # (B, 128)

        # Step 2: CrossAttentionMTF — Multi-TF awareness
        # "How does the H4/H1 structure align with M5 patterns?"
        mtf_latent = self.cross_attention(
            m5_features, h1_features, h4_features
        )  # (B, 128)

        # Step 3: RegimeDetector — Market regime
        # "Is the market trending, ranging, or volatile right now?"
        regime_probs, regime_emb = self.regime_detector(smc_latent)  # (B, 4), (B, 128)

        # Step 4: Concatenate into global state
        # [smc_latent | mtf_latent | regime_emb | regime_probs]
        global_state = torch.cat(
            [smc_latent, mtf_latent, regime_emb, regime_probs], dim=-1
        )  # (B, 388)

        return global_state


class SACTransformerActor(nn.Module):
    """
    SAC Actor with Transformer backbone.

    Replaces the old MLP Actor with:
    1. TransformerFeatureExtractor → global_state (388-dim)
    2. MLP Head → (mean, log_std) → Squashed Gaussian

    Output: tanh-squashed actions in [-1, 1]^4
    """

    def __init__(
        self,
        n_features: int = 28,
        action_dim: int = 4,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        n_cross_layers: int = 1,
        n_regimes: int = 4,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Shared feature extractor
        self.feature_extractor = TransformerFeatureExtractor(
            n_features=n_features,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
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
        m5: torch.Tensor,
        h1: torch.Tensor,
        h4: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: raw features → action + log_prob.

        Args:
            m5: (batch, seq_m5, n_features)
            h1: (batch, seq_h1, n_features)
            h4: (batch, seq_h4, n_features)
            deterministic: If True, return mean action (no sampling)

        Returns:
            (action, log_prob) — action: (batch, 4), log_prob: (batch, 1)
        """
        # Extract features through Transformer pipeline
        global_state = self.feature_extractor(m5, h1, h4)  # (B, 388)

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
        m5: torch.Tensor,
        h1: torch.Tensor,
        h4: torch.Tensor,
    ) -> torch.Tensor:
        """Extract global state (for critic input)."""
        return self.feature_extractor(m5, h1, h4)


class SACTransformerCritic(nn.Module):
    """
    Twin Q-Critic with Transformer backbone.

    Shares the same feature extractor architecture as the Actor,
    but maintains separate weights (no parameter sharing).

    Q(s, a) = MLP(cat[global_state, action]) → scalar
    """

    def __init__(
        self,
        n_features: int = 28,
        action_dim: int = 4,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        n_cross_layers: int = 1,
        n_regimes: int = 4,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Separate feature extractor (NOT shared with actor)
        self.feature_extractor = TransformerFeatureExtractor(
            n_features=n_features,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
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
        state_dim: int, action_dim: int, hidden_dims: list[int]
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
        m5: torch.Tensor,
        h1: torch.Tensor,
        h4: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Twin Q-value forward pass.

        Args:
            m5, h1, h4: Multi-TF feature sequences
            action: (batch, action_dim=4)

        Returns:
            (Q1, Q2) — both (batch, 1)
        """
        global_state = self.feature_extractor(m5, h1, h4)
        sa = torch.cat([global_state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def min_q(
        self,
        m5: torch.Tensor,
        h1: torch.Tensor,
        h4: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """min(Q1, Q2) — conservative Q-estimate for SAC."""
        q1, q2 = self.forward(m5, h1, h4, action)
        return torch.min(q1, q2)
