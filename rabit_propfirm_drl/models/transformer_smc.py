"""
TransformerSMC — Self-Attention encoder for SMC pattern recognition.

Purpose:
    Processes a sequence of M5 candle features (SMC + Volume + Price Action)
    and learns WHICH bars and patterns are most important for trading decisions.

    Unlike a simple MLP that flattens all bars equally, Self-Attention allows
    the model to dynamically weight: "this FVG bar 20 steps ago matters more
    than the noise bar 3 steps ago."

Architecture:
    Input:  (batch, seq_len=64, n_features=28)  — M5 SMC feature windows
                                    │
                          Linear Projection
                          n_features → embed_dim
                                    │
                        Sinusoidal Positional Encoding
                        (preserves temporal ordering)
                                    │
                        ┌───────────────────────┐
                        │  TransformerEncoder    │
                        │  × n_layers (2)        │
                        │  Multi-Head Self-Attn  │
                        │  (n_heads=4)           │
                        │  + Feed-Forward + Norm │
                        └───────────────────────┘
                                    │
                           Mean Pooling
                    (aggregate all timestep outputs)
                                    │
    Output: (batch, embed_dim=128)  — latent representation

Key design decisions:
    1. Sinusoidal PE (not learnable): Better generalization to unseen
       sequence positions. Captures relative temporal distance naturally.
    2. Mean Pooling (not CLS token): More stable for financial time-series
       where every bar carries information, not just a summary token.
    3. Pre-norm style: More stable training, especially with limited data.

All parameters configurable via __init__ (zero hardcoding).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (Vaswani et al., "Attention Is All You Need").

    Encodes position information using sin/cos functions at different frequencies.
    This allows the model to: (a) know the ORDER of bars in the sequence,
    and (b) infer RELATIVE distances between bars (sin/cos difference is a
    function of distance, not absolute position).

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Why sinusoidal (not learnable)?
        - Generalizes to unseen sequence lengths
        - Captures relative position naturally
        - More robust with limited financial data
    """

    def __init__(self, max_seq_len: int, embed_dim: int) -> None:
        """
        Args:
            max_seq_len: Maximum sequence length to encode.
            embed_dim: Embedding dimension (must be even for sin/cos pairs).
        """
        super().__init__()

        # Build the PE matrix: (max_seq_len, embed_dim)
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)

        # Compute the div_term: 10000^(2i/d_model) using log-space for numerical stability
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / embed_dim)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter — no gradients, but moves with model device)
        # Shape: (1, max_seq_len, embed_dim) for broadcasting
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)

        Returns:
            Tensor of same shape with positional information added.
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerSMC(nn.Module):
    """
    Transformer Self-Attention encoder for SMC pattern recognition.

    Takes a sequence of SMC feature vectors and produces a single latent
    representation that captures which bars/patterns are most relevant.

    Self-Attention mechanism:
        For each bar in the sequence, the model computes attention scores
        against ALL other bars. This means if bar #15 has a bullish FVG
        and bar #45 has a BOS confirmation, the model can learn to
        attend to both and relate them — something an MLP cannot do.

    Args:
        input_dim: Number of input features per timestep (e.g., 28 for SMC features).
        embed_dim: Transformer embedding dimension (default: 128).
        n_heads: Number of attention heads (default: 4). Each head attends to
                 different aspects (e.g., one head for structure, another for volume).
        n_layers: Number of Transformer encoder layers (default: 2).
        dropout: Dropout rate for regularization (default: 0.1).
        max_seq_len: Maximum supported sequence length (default: 256).
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()

        # Validate: embed_dim must be divisible by n_heads
        assert embed_dim % n_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"
        )

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        # ── Step 1: Linear Projection ──
        # Project raw features (28-dim) into the Transformer's embedding space (128-dim).
        # This is analogous to "tokenization" in NLP — each candle bar becomes a token.
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ── Step 2: Positional Encoding ──
        # Add position information so the model knows the ORDER of bars.
        # Without this, Self-Attention is permutation-invariant (order-agnostic).
        self.positional_encoding = SinusoidalPositionalEncoding(max_seq_len, embed_dim)

        # ── Step 3: Transformer Encoder ──
        # Stack of Self-Attention layers.
        # Each layer: Multi-Head Attention → Add & Norm → Feed-Forward → Add & Norm
        #
        # Why PyTorch's nn.TransformerEncoder?
        # - Production-ready, optimized implementation
        # - Supports batch_first=True for (batch, seq, features) convention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,  # Standard 4x expansion in FFN
            dropout=dropout,
            activation="gelu",             # GELU > ReLU for financial data
            batch_first=True,              # (batch, seq, features) convention
            norm_first=True,               # Pre-norm: more stable training
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(embed_dim),   # Final layer norm
        )

        # ── Step 4: Output dropout ──
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass: raw features → latent vector.

        This is where the "thinking" happens:
        1. Each bar gets projected to embed_dim and receives position info.
        2. Self-Attention lets every bar "look at" every other bar.
           → Model learns: "FVG at bar 20 + BOS at bar 45 = strong signal"
        3. Mean Pooling aggregates all bar representations into one vector.

        Args:
            x: Input features, shape (batch_size, seq_len, input_dim).
               Example: (32, 64, 28) — 32 samples, 64 M5 bars, 28 SMC features.
            src_key_padding_mask: Optional mask for padded positions,
               shape (batch_size, seq_len). True = ignore this position.

        Returns:
            Latent vector, shape (batch_size, embed_dim).
            Example: (32, 128) — one 128-dim representation per sample.
        """
        # Step 1: Project features — (B, T, 28) → (B, T, 128)
        x = self.input_projection(x)

        # Step 2: Add positional encoding — (B, T, 128) → (B, T, 128)
        x = self.positional_encoding(x)

        # Step 3: Self-Attention — (B, T, 128) → (B, T, 128)
        # Each bar now attends to all other bars, learning which patterns
        # across the sequence are most relevant for the trading decision.
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Step 4: Mean Pooling — (B, T, 128) → (B, 128)
        # Aggregate all timestep outputs into a single vector.
        # If padding mask is provided, exclude padded positions from mean.
        if src_key_padding_mask is not None:
            # Invert mask: True = valid position
            valid_mask = ~src_key_padding_mask  # (B, T)
            valid_mask = valid_mask.unsqueeze(-1).float()  # (B, T, 1)
            x = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        else:
            x = x.mean(dim=1)  # Simple mean over sequence dimension

        # Step 5: Apply dropout
        x = self.output_dropout(x)

        return x

    def get_attention_weights(
        self, x: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Extract attention weights from all layers (for SHAP/interpretability).

        Useful for understanding: "Which bars did the model focus on?"

        Args:
            x: Input features, shape (batch_size, seq_len, input_dim).

        Returns:
            List of attention weight tensors, one per layer.
            Each tensor shape: (batch_size, n_heads, seq_len, seq_len).
        """
        # Project and encode
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        # Manually pass through encoder layers to capture attention weights
        attention_weights: list[torch.Tensor] = []
        for layer in self.transformer_encoder.layers:
            # Pre-norm self-attention
            x_norm = layer.norm1(x)
            _, attn_w = layer.self_attn(
                x_norm, x_norm, x_norm, need_weights=True, average_attn_weights=False
            )
            attention_weights.append(attn_w)

            # Complete the layer forward pass
            x = x + layer.self_attn(x_norm, x_norm, x_norm)[0]
            x = x + layer._ff_block(layer.norm2(x))

        return attention_weights
