"""
CrossAttentionMTF — Multi-Timeframe Cross-Attention for SMC trading.

Purpose:
    Lets the M5 (execution TF) "ask questions" to H1/H4 (context TFs).
    This mimics how a real trader works:
        1. First check H4 → "Is the macro trend bullish or bearish?"
        2. Then check H1 → "Is there a BOS/CHoCH confirming the H4 bias?"
        3. Finally check M5 → "Is there a FVG/OB entry on M5 that aligns?"

    Cross-Attention allows each M5 bar to attend to ALL H1+H4 bars,
    learning which higher-TF structures are most relevant for NOW.

Architecture:
    H4 features (30 bars) ──► H4 Encoder ──► context_h4 (30, embed_dim)
    H1 features (24 bars) ──► H1 Encoder ──► context_h1 (24, embed_dim)
                                              │
                               Concat ────────┤ context (54, embed_dim)
                                              │
    M5 features (64 bars) ──► M5 Projection ──► query (64, embed_dim)
                                              │
                            ┌─────────────────▼──────────────────┐
                            │    Cross-Attention Layer            │
                            │    Q = M5 query (64 tokens)        │
                            │    K = H1+H4 context (54 tokens)   │
                            │    V = H1+H4 context (54 tokens)   │
                            │                                    │
                            │    Each M5 bar "looks at" all H1   │
                            │    and H4 bars to find relevant    │
                            │    structure/context information.   │
                            └────────────────────────────────────┘
                                              │
                                         Mean Pooling
                                              │
                               Output: (batch, embed_dim=128)

Key design decisions:
    1. FULL SEQUENCE cross-attention (not CLS-to-CLS): Each M5 bar can attend
       to specific H1/H4 bars, not just a compressed summary. This preserves
       spatial/temporal information from higher TFs.
    2. Lightweight H1/H4 encoders (1 layer): Higher TFs are "context" — they
       don't need deep processing, just enough to project into shared space.
    3. Shared embed_dim: All TFs project to same dimension so cross-attention works.
    4. Memory efficient: H1 (24) + H4 (30) = 54 context tokens is small.
       Attention matrix = 64 × 54 = 3,456 entries (tiny vs NLP's 512×512).

All parameters configurable via __init__ (zero hardcoding).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.transformer_smc import SinusoidalPositionalEncoding


class ContextEncoder(nn.Module):
    """
    Lightweight encoder for context timeframes (H1, H4).

    Uses a single Transformer encoder layer to process higher-TF features.
    Kept light because context TFs are "background" — they inform but don't
    directly execute trades. Deep processing is reserved for M5 (query TF).

    Input:  (batch, seq_len, n_features)
    Output: (batch, seq_len, embed_dim) — full sequence, NOT pooled
            (we need the full sequence for cross-attention K/V)
    """

    def __init__(
        self,
        n_features: int,
        embed_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()

        # Project raw features into shared embedding space
        self.input_projection = nn.Sequential(
            nn.Linear(n_features, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Positional encoding (sinusoidal — consistent with TransformerSMC)
        self.positional_encoding = SinusoidalPositionalEncoding(max_seq_len, embed_dim)

        # Single lightweight Transformer layer for context processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,  # 2x (not 4x) — lighter than query encoder
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=1,  # Single layer — enough for context summarization
            norm=nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode context timeframe features.

        Args:
            x: (batch, seq_len, n_features) — H1 or H4 raw features

        Returns:
            (batch, seq_len, embed_dim) — encoded context sequence
            NOTE: Returns FULL sequence (not pooled) for cross-attention K/V
        """
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        return x


class CrossAttentionMTF(nn.Module):
    """
    Multi-Timeframe Cross-Attention module.

    Core concept — Q, K, V separation:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  In nn.MultiheadAttention(query, key, value):                      │
    │                                                                     │
    │  Q (Query) = "What am I looking for?"                              │
    │    → M5 bars: "I'm bar #42 with a pin bar. Is there H4 support?"  │
    │                                                                     │
    │  K (Key) = "What information is available?"                        │
    │    → H1+H4 bars: "I'm an H4 bullish BOS" / "I'm an H1 FVG zone"  │
    │                                                                     │
    │  V (Value) = "What information should I extract?"                  │
    │    → H1+H4 bars: the actual encoded features to attend to         │
    │                                                                     │
    │  Attention Score = softmax(Q × K^T / √d) × V                      │
    │    → Each M5 bar gets a weighted combination of H1/H4 information  │
    │    → Weights determined by relevance (Q·K similarity)              │
    └─────────────────────────────────────────────────────────────────────┘

    Memory analysis (batch_size=64):
        Query:   (64, 64, 128)   = 512K elements
        Context: (64, 54, 128)   = 442K elements
        Attn:    (64, 4, 64, 54) = 884K elements  (4 heads)
        Total:   ~7.4 MB — well under 2GB limit ✅

    Args:
        n_features_m5: Number of M5 features per timestep (e.g., 28).
        n_features_h1: Number of H1 features per timestep (e.g., 28).
        n_features_h4: Number of H4 features per timestep (e.g., 28).
        embed_dim: Shared embedding dimension (default: 128).
        n_heads: Number of attention heads (default: 4).
        n_cross_layers: Number of cross-attention layers (default: 1).
        dropout: Dropout rate (default: 0.1).
        max_len_m5: Max M5 sequence length (default: 128).
        max_len_h1: Max H1 sequence length (default: 48).
        max_len_h4: Max H4 sequence length (default: 64).
    """

    def __init__(
        self,
        n_features_m5: int,
        n_features_h1: int,
        n_features_h4: int,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_cross_layers: int = 1,
        dropout: float = 0.1,
        max_len_m5: int = 128,
        max_len_h1: int = 48,
        max_len_h4: int = 64,
    ) -> None:
        super().__init__()

        assert embed_dim % n_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"
        )

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_cross_layers = n_cross_layers

        # ── M5 Query Projection ──
        # Projects M5 features into shared embedding space for Q vectors.
        # No Transformer encoding here — TransformerSMC handles that separately.
        # This module focuses purely on cross-TF attention.
        self.m5_projection = nn.Sequential(
            nn.Linear(n_features_m5, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.m5_pos_enc = SinusoidalPositionalEncoding(max_len_m5, embed_dim)

        # ── H1 / H4 Context Encoders ──
        # Encode higher-TF sequences to serve as Key/Value in cross-attention.
        self.h1_encoder = ContextEncoder(
            n_features=n_features_h1,
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_len_h1,
        )
        self.h4_encoder = ContextEncoder(
            n_features=n_features_h4,
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_len_h4,
        )

        # ── Cross-Attention Layers ──
        # M5 (Q) attends to concatenated H1+H4 (K, V).
        # Can stack multiple layers for deeper cross-TF reasoning.
        self.cross_attention_layers = nn.ModuleList()
        self.cross_norms_q = nn.ModuleList()
        self.cross_norms_kv = nn.ModuleList()
        self.cross_ff = nn.ModuleList()
        self.cross_norms_ff = nn.ModuleList()

        for _ in range(n_cross_layers):
            self.cross_attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=n_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.cross_norms_q.append(nn.LayerNorm(embed_dim))
            self.cross_norms_kv.append(nn.LayerNorm(embed_dim))

            # Feed-forward block after cross-attention (processes fused info)
            self.cross_ff.append(nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            ))
            self.cross_norms_ff.append(nn.LayerNorm(embed_dim))

        # ── Final Layer Norm + Dropout ──
        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        m5_features: torch.Tensor,
        h1_features: torch.Tensor,
        h4_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multi-timeframe cross-attention forward pass.

        Flow:
        1. Encode H1 + H4 → full context sequences
        2. Concat H1 + H4 → unified context (K, V)
        3. Project M5 → query sequence (Q)
        4. Cross-Attention: each M5 bar attends to all H1+H4 bars
        5. Mean Pool → single output vector

        Args:
            m5_features: (batch, seq_m5, n_features_m5)
                         Example: (32, 64, 28) — 64 M5 bars
            h1_features: (batch, seq_h1, n_features_h1)
                         Example: (32, 24, 28) — 24 H1 bars (1 day)
            h4_features: (batch, seq_h4, n_features_h4)
                         Example: (32, 30, 28) — 30 H4 bars (5 days)

        Returns:
            (batch, embed_dim) — MTF-aware representation.
            Example: (32, 128)
        """
        # ── Step 1: Encode Context (H1 + H4) ──
        # Each encoder returns the FULL sequence (not pooled)
        # because we need individual bar representations for K/V.
        h1_encoded = self.h1_encoder(h1_features)   # (B, 24, embed_dim)
        h4_encoded = self.h4_encoder(h4_features)   # (B, 30, embed_dim)

        # ── Step 2: Concatenate Context ──
        # Stack H1 and H4 sequences along the time dimension.
        # context = [h4_bar_0, ..., h4_bar_29, h1_bar_0, ..., h1_bar_23]
        # Total context length = 30 + 24 = 54 tokens
        context = torch.cat([h4_encoded, h1_encoded], dim=1)  # (B, 54, embed_dim)

        # ── Step 3: Project M5 Query ──
        # Project M5 raw features into embedding space, add position info.
        query = self.m5_projection(m5_features)  # (B, 64, embed_dim)
        query = self.m5_pos_enc(query)            # (B, 64, embed_dim)

        # ── Step 4: Cross-Attention ──
        # For each cross-attention layer:
        #   Q = M5 bars (what we want to enrich with context)
        #   K = H1+H4 bars (what information is available)
        #   V = H1+H4 bars (what values to extract)
        #
        # Attention mechanism (per head):
        #   score(i,j) = Q[i] · K[j]^T / √(embed_dim/n_heads)
        #   attn_weights = softmax(scores)  — shape: (64, 54)
        #   output[i] = Σ_j attn_weights[i,j] × V[j]
        #
        # Meaning: M5 bar #i gets a weighted mix of all H1+H4 bars,
        # where weights reflect how relevant each context bar is.
        for i in range(self.n_cross_layers):
            # Pre-norm (more stable than post-norm)
            q_normed = self.cross_norms_q[i](query)
            kv_normed = self.cross_norms_kv[i](context)

            # Cross-attention: Q=M5, K=V=H1+H4
            attn_output, _ = self.cross_attention_layers[i](
                query=q_normed,    # Q: What is M5 looking for?
                key=kv_normed,     # K: What context features exist?
                value=kv_normed,   # V: What context values to extract?
            )

            # Residual connection: preserve original M5 info + add context
            query = query + attn_output

            # Feed-forward: process the fused M5+context information
            ff_normed = self.cross_norms_ff[i](query)
            query = query + self.cross_ff[i](ff_normed)

        # ── Step 5: Final Norm + Mean Pool ──
        query = self.final_norm(query)   # (B, 64, embed_dim)
        output = query.mean(dim=1)       # (B, embed_dim) — aggregate all M5 bars
        output = self.output_dropout(output)

        return output

    def get_cross_attention_weights(
        self,
        m5_features: torch.Tensor,
        h1_features: torch.Tensor,
        h4_features: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Extract cross-attention weights for interpretability.

        Shows: which H1/H4 bars does each M5 bar attend to most?
        Useful for understanding the model's multi-TF reasoning.

        Returns:
            List of attention weight tensors, one per cross-attention layer.
            Each tensor shape: (batch, n_heads, seq_m5, seq_h1+seq_h4).
        """
        h1_encoded = self.h1_encoder(h1_features)
        h4_encoded = self.h4_encoder(h4_features)
        context = torch.cat([h4_encoded, h1_encoded], dim=1)

        query = self.m5_projection(m5_features)
        query = self.m5_pos_enc(query)

        weights: list[torch.Tensor] = []
        for i in range(self.n_cross_layers):
            q_normed = self.cross_norms_q[i](query)
            kv_normed = self.cross_norms_kv[i](context)

            attn_output, attn_w = self.cross_attention_layers[i](
                query=q_normed, key=kv_normed, value=kv_normed,
                need_weights=True, average_attn_weights=False,
            )
            weights.append(attn_w)

            query = query + attn_output
            ff_normed = self.cross_norms_ff[i](query)
            query = query + self.cross_ff[i](ff_normed)

        return weights


# ── Legacy aliases for backward compatibility ──
# Old tests import CrossAttentionFusion and MultiTimeframeEncoder.
# These wrappers preserve the old API while using new internals.

class CrossAttentionFusion(nn.Module):
    """Legacy wrapper — redirects to new CrossAttentionMTF internal logic."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = query.unsqueeze(1)
        normed_q = self.norm_q(q)
        normed_ctx = self.norm_kv(context)
        attn_out, _ = self.cross_attn(normed_q, normed_ctx, normed_ctx)
        q = q + attn_out
        normed = self.norm_ff(q)
        q = q + self.ff(normed)
        return q.squeeze(1)


class MultiTimeframeEncoder(nn.Module):
    """Legacy wrapper — uses old TimeSeriesTransformer for backward compat."""

    def __init__(
        self,
        n_features_m15: int, n_features_h1: int, n_features_h4: int,
        d_model: int = 128, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1,
        max_len_m15: int = 96, max_len_h1: int = 48, max_len_h4: int = 30,
    ) -> None:
        super().__init__()
        from models.transformer import TimeSeriesTransformer
        self.m15_encoder = TimeSeriesTransformer(
            n_features=n_features_m15, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, dropout=dropout, max_seq_len=max_len_m15,
        )
        self.h1_encoder = TimeSeriesTransformer(
            n_features=n_features_h1, d_model=d_model, n_heads=n_heads,
            n_layers=max(1, n_layers - 1), dropout=dropout, max_seq_len=max_len_h1,
        )
        self.h4_encoder = TimeSeriesTransformer(
            n_features=n_features_h4, d_model=d_model, n_heads=n_heads,
            n_layers=max(1, n_layers - 1), dropout=dropout, max_seq_len=max_len_h4,
        )
        self.fusion = CrossAttentionFusion(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.d_model = d_model

    def forward(self, m15_seq: torch.Tensor, h1_seq: torch.Tensor, h4_seq: torch.Tensor) -> torch.Tensor:
        m15_repr = self.m15_encoder(m15_seq)
        h1_repr = self.h1_encoder(h1_seq)
        h4_repr = self.h4_encoder(h4_seq)
        context = torch.stack([h1_repr, h4_repr], dim=1)
        return self.fusion(m15_repr, context)
