"""
Time-Series Transformer Encoder — Processes sequential market data.

Architecture:
- Positional encoding (learnable)
- Multi-head self-attention layers
- Feed-forward network with GELU activation
- Layer normalization (pre-norm style for stability)

Designed for M15 time-series features as QUERY input.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for time-series data."""

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: (batch, seq_len, d_model)"""
        return x + self.pe[:, : x.size(1), :]


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block with pre-norm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass. x: (batch, seq_len, d_model)

        Uses pre-norm: LayerNorm → Attention/FF → Residual
        (More stable training than post-norm)
        """
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=src_mask)
        x = x + attn_out

        # Feed-forward with residual
        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + ff_out

        return x


class TimeSeriesTransformer(nn.Module):
    """
    Transformer encoder for time-series market data.

    Input: (batch, seq_len, n_features)
    Output: (batch, d_model) — aggregated representation

    Uses CLS token aggregation for sequence-level output.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 96,
    ) -> None:
        """
        Args:
            n_features: Number of input features per timestep
            d_model: Transformer embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder blocks
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        # Input projection: n_features → d_model
        self.input_proj = nn.Linear(n_features, d_model)

        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding (max_seq_len + 1 for CLS)
        self.pos_enc = LearnablePositionalEncoding(max_seq_len + 1, d_model)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, n_features) — raw feature sequences
            mask: Optional attention mask

        Returns:
            (batch, d_model) — CLS token output (sequence-level representation)
        """
        batch_size = x.size(0)

        # Project features to d_model
        x = self.input_proj(x)  # (B, T, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)

        # Add positional encoding
        x = self.pos_enc(x)

        # Pass through encoder blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final norm
        x = self.final_norm(x)

        # Return CLS token output
        return x[:, 0, :]  # (B, d_model)
