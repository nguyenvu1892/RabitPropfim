"""
Cross-Attention Module — Fuses M15 query features with H1/H4 context.

Architecture:
- Query: M15 Transformer output (primary decision timeframe)
- Key/Value: H4/H1 Transformer outputs (higher-TF structure)
- Multi-head cross-attention → fused representation

This captures macro market structure while making fine-grained M15 decisions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.transformer import TimeSeriesTransformer


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention module that fuses query (M15) with context (H1/H4).

    Query attends to context keys/values to extract relevant
    higher-timeframe information for decision-making.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-attention: query attends to context.

        Args:
            query: (batch, d_model) — M15 query representation
            context: (batch, n_contexts, d_model) — H1/H4 context representations

        Returns:
            (batch, d_model) — fused representation
        """
        # Reshape query for attention: (B, 1, d_model)
        q = query.unsqueeze(1)

        # Cross-attention
        normed_q = self.norm_q(q)
        normed_ctx = self.norm_kv(context)
        attn_out, _ = self.cross_attn(normed_q, normed_ctx, normed_ctx)
        q = q + attn_out

        # Feed-forward
        normed = self.norm_ff(q)
        ff_out = self.ff(normed)
        q = q + ff_out

        return q.squeeze(1)  # (B, d_model)


class MultiTimeframeEncoder(nn.Module):
    """
    Full multi-timeframe encoder with Cross-Attention fusion.

    Architecture:
    1. M15 Transformer → query representation
    2. H1 Transformer → context representation
    3. H4 Transformer → context representation
    4. Cross-Attention: M15 query attends to [H1, H4] contexts
    5. Output: fused representation (batch, d_model)
    """

    def __init__(
        self,
        n_features_m15: int,
        n_features_h1: int,
        n_features_h4: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_len_m15: int = 96,
        max_len_h1: int = 48,
        max_len_h4: int = 30,
    ) -> None:
        super().__init__()

        # Individual timeframe encoders
        self.m15_encoder = TimeSeriesTransformer(
            n_features=n_features_m15,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_seq_len=max_len_m15,
        )

        self.h1_encoder = TimeSeriesTransformer(
            n_features=n_features_h1,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=max(1, n_layers - 1),  # Lighter for context
            dropout=dropout,
            max_seq_len=max_len_h1,
        )

        self.h4_encoder = TimeSeriesTransformer(
            n_features=n_features_h4,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=max(1, n_layers - 1),
            dropout=dropout,
            max_seq_len=max_len_h4,
        )

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.d_model = d_model

    def forward(
        self,
        m15_seq: torch.Tensor,
        h1_seq: torch.Tensor,
        h4_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with multi-timeframe fusion.

        Args:
            m15_seq: (batch, seq_m15, n_features_m15)
            h1_seq: (batch, seq_h1, n_features_h1)
            h4_seq: (batch, seq_h4, n_features_h4)

        Returns:
            (batch, d_model) — fused multi-TF representation
        """
        # Encode each timeframe
        m15_repr = self.m15_encoder(m15_seq)   # (B, d_model)
        h1_repr = self.h1_encoder(h1_seq)      # (B, d_model)
        h4_repr = self.h4_encoder(h4_seq)      # (B, d_model)

        # Stack context for cross-attention: (B, 2, d_model)
        context = torch.stack([h1_repr, h4_repr], dim=1)

        # Fuse query (M15) with context (H1 + H4)
        fused = self.fusion(m15_repr, context)  # (B, d_model)

        return fused
