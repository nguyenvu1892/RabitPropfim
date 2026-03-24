"""
V3.6 AttentionPPO -- Self-Attention Actor-Critic with Contrastive Head.

Architecture:
    400-dim flat obs → 8 tokens × 50-dim → Self-Attention (2L, 4H) → Actor/Critic/Contrastive heads
    
Tokens:
    [H1] [M15] [M5] [M1_bar1] [M1_bar2] [M1_bar3] [M1_bar4] [M1_bar5]
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

TOKEN_NAMES = ["H1", "M15", "M5", "M1_b1", "M1_b2", "M1_b3", "M1_b4", "M1_b5"]


class AttentionPPO(nn.Module):
    """
    Self-Attention PPO with Contrastive Head.
    
    Input:  (B, 400) flat obs
    Output: logits (B, 4), value (B, 1), attn_weights (B, n_heads, 8, 8)
    """

    def __init__(
        self,
        obs_dim: int = 400,
        n_actions: int = 4,
        n_tokens: int = 8,
        token_dim: int = 50,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        contrastive_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.d_model = d_model
        self.n_heads = n_heads

        # --- Token Embedding ---
        self.token_proj = nn.Linear(token_dim, d_model)

        # --- Positional Encoding (learnable) ---
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Output Heads ---
        self.actor_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_actions),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.contrastive_head = nn.Sequential(
            nn.Linear(d_model, contrastive_dim),
        )

        # --- Store attention weights ---
        self._attn_weights = None
        self._register_hooks()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _register_hooks(self):
        """Hook to capture attention weights from TransformerEncoder."""
        def hook_fn(module, input, output):
            # nn.MultiheadAttention stores attn_weights when need_weights=True
            pass  # We'll use manual attention instead

        # We need a custom approach since PyTorch TransformerEncoder 
        # doesn't expose attention weights easily. We'll compute them manually.
        pass

    def _tokenize(self, obs: torch.Tensor) -> torch.Tensor:
        """Reshape (B, 400) → (B, 8, 50) tokens."""
        B = obs.shape[0]
        return obs.view(B, self.n_tokens, self.token_dim)

    def _encode(self, obs: torch.Tensor):
        """
        Full forward through attention layers.
        Returns: pooled (B, d_model), attn_weights (B, n_heads, 8, 8)
        """
        B = obs.shape[0]
        tokens = self._tokenize(obs)                # (B, 8, 50)
        x = self.token_proj(tokens)                  # (B, 8, 64)
        x = x + self.pos_embed                       # (B, 8, 64)

        # Manual attention to capture weights
        attn_weights_all = []
        for layer in self.transformer.layers:
            # Self-attention sublayer
            x_norm = layer.norm1(x)
            # Manually call multihead attention
            attn_out, attn_w = layer.self_attn(
                x_norm, x_norm, x_norm,
                need_weights=True,
                average_attn_weights=False,  # Get per-head weights
            )
            x = x + layer.dropout1(attn_out)

            # Feedforward sublayer
            x_norm2 = layer.norm2(x)
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_norm2))))
            x = x + layer.dropout2(ff_out)

            attn_weights_all.append(attn_w)  # (B, n_heads, 8, 8)

        # Average across layers
        attn_weights = torch.stack(attn_weights_all).mean(dim=0)  # (B, n_heads, 8, 8)

        # Mean pool across tokens
        pooled = x.mean(dim=1)  # (B, d_model)

        return pooled, attn_weights

    def forward(self, obs: torch.Tensor):
        """
        Returns: logits (B, 4), value (B, 1), attn_weights (B, n_heads, 8, 8)
        """
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
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def get_embedding(self, obs: torch.Tensor) -> torch.Tensor:
        """Get L2-normalized embedding for contrastive learning."""
        pooled, _ = self._encode(obs)
        embed = self.contrastive_head(pooled)
        return F.normalize(embed, p=2, dim=-1)

    def get_attention_weights(self, obs: torch.Tensor) -> torch.Tensor:
        """Get attention weights for analysis. Returns (B, n_heads, 8, 8)."""
        _, attn_weights = self._encode(obs)
        return attn_weights
