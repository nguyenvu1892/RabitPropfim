"""
V3.7.1 AttentionPPO -- 2-Stage Hierarchical Attention with Token Dropout.

Architecture:
    432-dim flat obs → 8 tokens × 54-dim
    
    Stage 1 (Specialist):
        MacroAttn: H1 + M15 (2 tokens) → Self-Attention (1L, 4H) → macro_ctx
        MicroAttn: M5 + M1×5 (6 tokens) → Self-Attention (1L, 4H) → micro_ctx
    
    Stage 2 (Fusion):
        FusionAttn: [macro_ctx, micro_ctx] → Cross-Attention (1L, 4H) → decision
    
    Token Dropout: 15% random masking during S1/S3 training (OFF during S2/eval)
    
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
    2-Stage Hierarchical Attention PPO.
    
    Input:  (B, 432) flat obs
    Output: logits (B, 4), value (B, 1), attn_weights (B, 1, 8, 8)
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
        self.token_dropout_enabled = True  # Toggle per stage

        # --- Shared Token Embedding ---
        self.token_proj = nn.Linear(token_dim, d_model)

        # --- Stage 1: Specialist Encoders ---
        # Macro: H1 + M15 (2 tokens)
        self.macro_pos = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)
        macro_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.macro_encoder = nn.TransformerEncoder(macro_layer, num_layers=1)

        # Micro: M5 + M1×5 (6 tokens)
        self.micro_pos = nn.Parameter(torch.randn(1, 6, d_model) * 0.02)
        micro_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.micro_encoder = nn.TransformerEncoder(micro_layer, num_layers=1)

        # --- Stage 2: Fusion Encoder ---
        self.fusion_pos = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(fusion_layer, num_layers=1)

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
        """Reshape (B, 432) → (B, 8, 54) tokens."""
        B = obs.shape[0]
        return obs.view(B, self.n_tokens, self.token_dim)

    def _apply_token_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Token Dropout: randomly zero out entire tokens during training.
        Each token has token_dropout_rate probability of being masked.
        """
        if not self.training or not self.token_dropout_enabled:
            return tokens
        B, N, D = tokens.shape
        # Create mask: (B, N, 1) — each token independently masked
        mask = torch.bernoulli(
            torch.full((B, N, 1), 1.0 - self.token_dropout_rate, device=tokens.device)
        )
        # Scale by 1/(1-p) to maintain expected value (like standard dropout)
        tokens = tokens * mask / (1.0 - self.token_dropout_rate)
        return tokens

    def _encode_specialist(self, x: torch.Tensor, encoder, pos_embed):
        """Run one specialist encoder with manual attention extraction."""
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
        2-Stage Hierarchical Encoding.
        Returns: pooled (B, d_model), token_importance (B, 8)
        """
        B = obs.shape[0]
        tokens = self._tokenize(obs)
        tokens = self._apply_token_dropout(tokens)
        x = self.token_proj(tokens)  # (B, 8, d_model)

        # --- Stage 1: Specialist Attention ---
        # Macro: tokens 0,1 (H1, M15)
        macro_x = x[:, MACRO_TOKENS, :] + self.macro_pos  # (B, 2, d_model)
        macro_out, macro_attn = self._encode_specialist(macro_x, self.macro_encoder, self.macro_pos)
        macro_ctx = macro_out.mean(dim=1)  # (B, d_model)

        # Micro: tokens 2-7 (M5, M1×5)
        micro_x = x[:, MICRO_TOKENS, :] + self.micro_pos  # (B, 6, d_model)
        micro_out, micro_attn = self._encode_specialist(micro_x, self.micro_encoder, self.micro_pos)
        micro_ctx = micro_out.mean(dim=1)  # (B, d_model)

        # --- Stage 2: Fusion ---
        fusion_input = torch.stack([macro_ctx, micro_ctx], dim=1)  # (B, 2, d_model)
        fusion_input = fusion_input + self.fusion_pos
        fusion_out, fusion_attn = self._encode_specialist(fusion_input, self.fusion_encoder, self.fusion_pos)
        pooled = fusion_out.mean(dim=1)  # (B, d_model)

        # --- Compute per-token importance (B, 8) for backward compatibility ---
        # fusion_attn: (B, heads, 2, 2) — [macro_weight, micro_weight]
        fusion_weights = fusion_attn.mean(dim=1)  # (B, 2, 2) → average heads
        # How much the model relies on macro vs micro (row-wise sum = 1)
        macro_importance = fusion_weights[:, :, 0].mean(dim=1)  # (B,) — macro column
        micro_importance = fusion_weights[:, :, 1].mean(dim=1)  # (B,) — micro column

        # Distribute importance within each specialist
        macro_self = macro_attn.mean(dim=1).mean(dim=1)  # (B, 2) — per-token within macro
        micro_self = micro_attn.mean(dim=1).mean(dim=1)  # (B, 6) — per-token within micro

        # Normalize within each group
        macro_self = macro_self / (macro_self.sum(dim=1, keepdim=True) + 1e-8)
        micro_self = micro_self / (micro_self.sum(dim=1, keepdim=True) + 1e-8)

        # Final per-token importance: group_importance × within_group_importance
        token_imp = torch.zeros(B, 8, device=obs.device)
        token_imp[:, 0] = macro_importance * macro_self[:, 0]  # H1
        token_imp[:, 1] = macro_importance * macro_self[:, 1]  # M15
        token_imp[:, 2] = micro_importance * micro_self[:, 0]  # M5
        token_imp[:, 3] = micro_importance * micro_self[:, 1]  # M1_b1
        token_imp[:, 4] = micro_importance * micro_self[:, 2]  # M1_b2
        token_imp[:, 5] = micro_importance * micro_self[:, 3]  # M1_b3
        token_imp[:, 6] = micro_importance * micro_self[:, 4]  # M1_b4
        token_imp[:, 7] = micro_importance * micro_self[:, 5]  # M1_b5

        # Normalize to sum=1
        token_imp = token_imp / (token_imp.sum(dim=1, keepdim=True) + 1e-8)

        # Construct fake (B, 1, 8, 8) attention matrix for backward compatibility
        # Each row is a copy of token_importance
        attn_compat = token_imp.unsqueeze(1).unsqueeze(1).expand(-1, 1, 8, -1)  # (B, 1, 8, 8)

        return pooled, attn_compat

    def forward(self, obs: torch.Tensor):
        """Returns: logits (B, 4), value (B, 1), attn_weights (B, 1, 8, 8)"""
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
            # --- Confidence Gate (INFERENCE ONLY) ---
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
        """Get L2-normalized embedding for contrastive learning."""
        pooled, _ = self._encode(obs)
        embed = self.contrastive_head(pooled)
        return F.normalize(embed, p=2, dim=-1)

    def get_attention_weights(self, obs: torch.Tensor) -> torch.Tensor:
        """Get attention weights for analysis. Returns (B, 1, 8, 8)."""
        _, attn_weights = self._encode(obs)
        return attn_weights
