"""
V3.8 AttentionPPO -- Cross-Attention (Micro=Q, Macro=K/V) with Fake Setup Mining.

Architecture:
    432-dim flat obs → 8 tokens × 54-dim
    
    Stage 1: Independent Feature Extraction
        Macro: H1 + M15 (2 tokens) → Self-Attention (1L, 4H) → macro_ctx
        Micro: M5 + M1×5 (6 tokens) → Self-Attention (1L, 4H) → micro_ctx
    
    Stage 2: Gated Cross-Attention
        Q = micro_ctx, K = V = macro_ctx
        Micro tokens look up Macro context to validate their setups.
        Output → Average pooled → Actor/Critic/Contrastive Head

Tokens (54-dim = 50 raw + OB_proximity + Volume_spike + Spread + Session):
    [H1] [M15] [M5] [M1_bar1] [M1_bar2] [M1_bar3] [M1_bar4] [M1_bar5]
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

TOKEN_NAMES = ["H1", "M15", "M5", "M1_b1", "M1_b2", "M1_b3", "M1_b4", "M1_b5"]
MACRO_TOKENS = [0, 1]      # H1, M15
MICRO_TOKENS = [2, 3, 4, 5, 6, 7]  # M5, M1_b1-b5


class AttentionPPO(nn.Module):
    """
    V3.8 Cross-Attention PPO.
    """

    def __init__(
        self,
        obs_dim: int = 432,
        n_actions: int = 4,
        n_tokens: int = 8,
        token_dim: int = 54,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        contrastive_dim: int = 128,
        confidence_threshold: float = 0.70,
        confidence_mode: str = "relative",
        confidence_ratio: float = 2.0,
        token_dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.confidence_threshold = confidence_threshold
        self.confidence_mode = confidence_mode
        self.confidence_ratio = confidence_ratio
        self.token_dropout_rate = token_dropout_rate
        self.token_dropout_enabled = True

        # --- Shared Token Embedding ---
        self.token_proj = nn.Linear(token_dim, d_model)

        # --- Stage 1: Self-Attention Encoders ---
        self.macro_pos = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)
        macro_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.macro_encoder = nn.TransformerEncoder(macro_layer, num_layers=1)

        self.micro_pos = nn.Parameter(torch.randn(1, 6, d_model) * 0.02)
        micro_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.micro_encoder = nn.TransformerEncoder(micro_layer, num_layers=1)

        # --- Stage 2: Cross-Attention (Micro looks at Macro) ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.cross_ff_norm = nn.LayerNorm(d_model)

        # --- Output Heads ---
        self.actor_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, n_actions),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1),
        )
        self.contrastive_head = nn.Sequential(
            nn.Linear(d_model, contrastive_dim),
        )

        self._attn_weights = None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _tokenize(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        return obs.view(B, self.n_tokens, self.token_dim)

    def _apply_token_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.token_dropout_enabled:
            return tokens
        B, N, D = tokens.shape
        mask = torch.bernoulli(
            torch.full((B, N, 1), 1.0 - self.token_dropout_rate, device=tokens.device)
        )
        tokens = tokens * mask / (1.0 - self.token_dropout_rate)
        return tokens

    def _encode_self_attn(self, x: torch.Tensor, encoder, pos_embed):
        """Run self-attention encoder and extract weights."""
        x_norm = encoder.layers[0].norm1(x)
        attn_out, attn_w = encoder.layers[0].self_attn(
            x_norm, x_norm, x_norm,
            need_weights=True, average_attn_weights=False,
        )
        x = x + encoder.layers[0].dropout1(attn_out)
        x_norm2 = encoder.layers[0].norm2(x)
        ff_out = encoder.layers[0].linear2(
            encoder.layers[0].dropout(
                encoder.layers[0].activation(encoder.layers[0].linear1(x_norm2))
            )
        )
        x = x + encoder.layers[0].dropout2(ff_out)
        return x, attn_w  # x: (B, N, d_model), attn_w: (B, heads, N, N)

    def _encode(self, obs: torch.Tensor):
        """
        V3.8 Cross-Attention Encoding.
        Returns: pooled (B, d_model), cross_attn_w (B, 1, 8, 8)
        """
        B = obs.shape[0]
        tokens = self._tokenize(obs)
        tokens = self._apply_token_dropout(tokens)
        x = self.token_proj(tokens)  # (B, 8, d_model)

        # 1. Macro Context (H1, M15)
        macro_x = x[:, MACRO_TOKENS, :] + self.macro_pos
        macro_ctx, macro_attn = self._encode_self_attn(macro_x, self.macro_encoder, self.macro_pos)

        # 2. Micro Context (M5, M1x5)
        micro_x = x[:, MICRO_TOKENS, :] + self.micro_pos
        micro_ctx, micro_attn = self._encode_self_attn(micro_x, self.micro_encoder, self.micro_pos)

        # 3. Cross-Attention: Micro (Q) queries Macro (K, V)
        micro_ctx_norm = self.cross_norm(micro_ctx)
        macro_ctx_norm = self.cross_norm(macro_ctx)
        
        # attn_w: (B, seq_Q, seq_K) -> (B, 6, 2)
        attn_out, cross_attn_w = self.cross_attn(
            query=micro_ctx_norm,
            key=macro_ctx_norm,
            value=macro_ctx_norm,
            need_weights=True,
            average_attn_weights=True,
        )
        
        # Add & Norm, then FF
        micro_fused = micro_ctx + attn_out
        micro_fused = micro_fused + self.cross_ff(self.cross_ff_norm(micro_fused))

        # Output is pooled micro tokens (since they now contain macro context)
        pooled = micro_fused.mean(dim=1)  # (B, d_model)

        # --- Fake 8x8 matrix for backward compatibility with explain_trade.py ---
        # The cross_attn_w gives us (B, 6, 2) -> How much each Micro token looks at each Macro token.
        # We will pad this to (B, 8, 8) format so backtest_v36.py doesn't crash.
        # We place cross_attn_w in the [2:8, 0:2] block.
        attn_compat = torch.zeros(B, 8, 8, device=obs.device)
        attn_compat[:, MICRO_TOKENS[0]:MICRO_TOKENS[-1]+1, MACRO_TOKENS[0]:MACRO_TOKENS[-1]+1] = cross_attn_w
        
        # Add self-attention for completeness in analysis
        attn_compat[:, MACRO_TOKENS[0]:MACRO_TOKENS[-1]+1, MACRO_TOKENS[0]:MACRO_TOKENS[-1]+1] = macro_attn.mean(dim=1)
        # Normalize rows to 1 for display
        attn_compat = attn_compat / (attn_compat.sum(dim=-1, keepdim=True) + 1e-8)
        
        attn_compat = attn_compat.unsqueeze(1)  # (B, 1, 8, 8)

        return pooled, attn_compat

    def forward(self, obs: torch.Tensor):
        pooled, attn_weights = self._encode(obs)
        logits = self.actor_head(pooled)
        value = self.critic_head(pooled)
        self._attn_weights = attn_weights
        return logits, value, attn_weights

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        pooled, _ = self._encode(obs)
        return self.critic_head(pooled).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        pooled, attn_weights = self._encode(obs)
        logits = self.actor_head(pooled)
        value = self.critic_head(pooled).squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
            if not self.training:
                probs = torch.softmax(logits, dim=-1)
                is_entry = (action == 0) | (action == 1)

                if self.confidence_mode == "relative":
                    action_prob = probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
                    hold_prob = probs[:, 2]
                    low_confidence = action_prob < (self.confidence_ratio * hold_prob)
                else:
                    max_prob = probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
                    low_confidence = max_prob < self.confidence_threshold

                gate_mask = is_entry & low_confidence
                action = torch.where(gate_mask, torch.tensor(2, device=obs.device), action)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def get_embedding(self, obs: torch.Tensor) -> torch.Tensor:
        pooled, _ = self._encode(obs)
        embed = self.contrastive_head(pooled)
        return F.normalize(embed, p=2, dim=-1)

    def get_attention_weights(self, obs: torch.Tensor) -> torch.Tensor:
        _, attn_weights = self._encode(obs)
        return attn_weights
