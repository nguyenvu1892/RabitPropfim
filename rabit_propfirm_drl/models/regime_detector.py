"""
Regime Detector — HMM-inspired regime classification for market state detection.

Detects regimes:
- Trend Up: sustained bullish momentum
- Trend Down: sustained bearish momentum
- Ranging: low volatility, mean-reverting
- Volatile: high volatility, erratic moves

Implemented as a learnable neural network module (not traditional HMM)
to allow end-to-end training with the DRL agent.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegimeDetector(nn.Module):
    """
    Neural regime classifier.

    Input: time-series features (from Transformer output or raw features)
    Output: regime probabilities (soft assignment) and regime embedding

    Can be used standalone or integrated into the actor-critic pipeline.
    """

    def __init__(
        self,
        input_dim: int,
        n_regimes: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        """
        Args:
            input_dim: Dimension of input features
            n_regimes: Number of market regimes
            hidden_dim: Hidden layer size
        """
        super().__init__()
        self.n_regimes = n_regimes

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_regimes),
        )

        # Regime embeddings (learnable)
        self.regime_embeddings = nn.Embedding(n_regimes, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, input_dim) — encoded state features

        Returns:
            (regime_probs, regime_embedding)
            - regime_probs: (batch, n_regimes) — soft regime probabilities
            - regime_embedding: (batch, input_dim) — weighted sum of regime embeddings
        """
        # Get regime logits and probabilities
        logits = self.classifier(x)            # (B, n_regimes)
        probs = F.softmax(logits, dim=-1)      # (B, n_regimes)

        # Weighted sum of regime embeddings
        # probs: (B, n_regimes) → (B, n_regimes, 1)
        # embeddings: (n_regimes, input_dim) → (1, n_regimes, input_dim)
        all_embeddings = self.regime_embeddings.weight.unsqueeze(0)  # (1, K, D)
        weighted = probs.unsqueeze(-1) * all_embeddings              # (B, K, D)
        regime_emb = weighted.sum(dim=1)                             # (B, D)

        return probs, regime_emb

    def predict_regime(self, x: torch.Tensor) -> torch.Tensor:
        """Get hard regime prediction. Returns: (batch,) — regime indices."""
        logits = self.classifier(x)
        return logits.argmax(dim=-1)

    @property
    def regime_names(self) -> list[str]:
        """Human-readable regime names."""
        names = ["trend_up", "trend_down", "ranging", "volatile"]
        return names[: self.n_regimes]
