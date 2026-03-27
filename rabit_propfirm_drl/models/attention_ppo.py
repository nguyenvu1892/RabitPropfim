"""
V4.2 AttentionPPO -- Dual Memory Banks (Song Mã Ký Ức).

Architecture:
    448-dim flat obs → 8 tokens × 56-dim → d_model=64
    
    Stage 1: Independent Feature Extraction
        Macro: H1 + M15 (2 tokens) → Self-Attention (1L, 4H) → macro_ctx
        Micro: M5 + M1×5 (6 tokens) → Self-Attention (1L, 4H) → micro_ctx
    
    Stage 2: Memory-Augmented Cross-Attention
        Q = micro_ctx (6 tokens)
        K = V = [macro_ctx (2) + win_memory (8) + loss_memory (8)] = 18 tokens
        Micro tokens query against: Macro context + Win patterns + Loss patterns
        
    Output → Average pooled → Actor/Critic/Contrastive Head

Memory Banks (16 × 64-dim):
    Sổ Tay Vàng (Win):  [4 Frozen Core] + [4 Adaptive EMA]
    Sổ Tay Đen (Loss):  [4 Frozen Core] + [4 Adaptive EMA]
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
    V4.2 Cross-Attention PPO with Dual Memory Banks.
    """

    def __init__(
        self,
        obs_dim: int = 464,
        n_actions: int = 4,
        n_tokens: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        contrastive_dim: int = 128,
        confidence_threshold: float = 0.70,
        confidence_mode: str = "relative",
        confidence_ratio: float = 2.0,
        token_dropout_rate: float = 0.15,
        n_memory_per_bank: int = 8,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_tokens = n_tokens
        self.token_dim = obs_dim // n_tokens
        self.d_model = d_model
        self.n_heads = n_heads
        self.confidence_threshold = confidence_threshold
        self.confidence_mode = confidence_mode
        self.confidence_ratio = confidence_ratio
        self.token_dropout_rate = token_dropout_rate
        self.token_dropout_enabled = True
        self.n_memory_per_bank = n_memory_per_bank

        # --- Shared Token Embedding ---
        self.token_proj = nn.Linear(self.token_dim, d_model)

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

        # --- V4.2: Dual Memory Banks ---
        # Learnable memory tokens (initialized from K-Means prototypes)
        self.win_memory = nn.Parameter(torch.randn(n_memory_per_bank, d_model) * 0.02)
        self.loss_memory = nn.Parameter(torch.randn(n_memory_per_bank, d_model) * 0.02)
        # Frozen masks (set after loading prototypes)
        self.register_buffer("win_frozen_mask", torch.zeros(n_memory_per_bank, dtype=torch.bool))
        self.register_buffer("loss_frozen_mask", torch.zeros(n_memory_per_bank, dtype=torch.bool))

        # --- Stage 2: Memory-Augmented Cross-Attention ---
        # Q = Micro (6 tokens), K/V = Macro (2) + Win Memory (8) + Loss Memory (8) = 18 tokens
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
        # V4.3: Auxiliary R:R Head
        self.rr_head = nn.Sequential(
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

    def load_memory_prototypes(self, proto_path: str):
        """Load K-Means prototypes into memory banks."""
        data = torch.load(proto_path, map_location=self.win_memory.device, weights_only=False)
        with torch.no_grad():
            self.win_memory.copy_(data["win_prototypes"][:self.n_memory_per_bank])
            self.loss_memory.copy_(data["loss_prototypes"][:self.n_memory_per_bank])
            self.win_frozen_mask.copy_(data["win_frozen_mask"][:self.n_memory_per_bank])
            self.loss_frozen_mask.copy_(data["loss_frozen_mask"][:self.n_memory_per_bank])
        print(f"V4.2: Loaded memory prototypes from {proto_path}")
        print(f"  Win: {self.win_frozen_mask.sum()} frozen, {(~self.win_frozen_mask).sum()} adaptive")
        print(f"  Loss: {self.loss_frozen_mask.sum()} frozen, {(~self.loss_frozen_mask).sum()} adaptive")

    def freeze_core_memory_grads(self):
        """Zero out gradients for frozen core prototypes after backward."""
        if self.win_memory.grad is not None:
            self.win_memory.grad[self.win_frozen_mask] = 0.0
        if self.loss_memory.grad is not None:
            self.loss_memory.grad[self.loss_frozen_mask] = 0.0

    def freeze_price_layers(self):
        """V4.4: Freeze raw Price Action processing layers.
        
        Frozen: token_proj, macro_encoder, micro_encoder, macro_pos, micro_pos
        Unfrozen: cross_attn, cross_norm, cross_ff, rr_head, actor_head, 
                  critic_head, contrastive_head, win_memory, loss_memory
        """
        frozen_modules = [self.token_proj, self.macro_encoder, self.micro_encoder]
        frozen_params = [self.macro_pos, self.micro_pos]
        
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False
        for param in frozen_params:
            param.requires_grad = False
        
        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"V4.4: Frozen {n_frozen:,} params | Trainable {n_trainable:,} params")

    def warm_start_from_v42(self, ckpt_path: str, device="cpu"):
        """V4.4: Load V4.2 checkpoint (448-dim) into 488-dim model.
        
        Handles token_dim mismatch (56 → 61) by copying compatible weights
        into the first 56 columns of the new token_proj, leaving the last 5
        (for Futures features) randomly initialized.
        """
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        old_state = ckpt["model_state_dict"]
        new_state = self.state_dict()
        
        loaded = 0
        skipped = []
        for key in old_state:
            if key not in new_state:
                skipped.append(key)
                continue
            if old_state[key].shape == new_state[key].shape:
                new_state[key] = old_state[key]
                loaded += 1
            elif key == "token_proj.weight":
                # V4.2: (64, 56), V4.4: (64, 61) — copy first 56 cols
                old_w = old_state[key]  # (64, 56)
                new_state[key][:, :old_w.shape[1]] = old_w
                loaded += 1
                print(f"  token_proj.weight: ({old_w.shape[1]}) → ({new_state[key].shape[1]}) [partial load]")
            else:
                skipped.append(f"{key} shape {old_state[key].shape}→{new_state[key].shape}")
        
        self.load_state_dict(new_state)
        print(f"V4.4 Warm-start: {loaded} params loaded, {len(skipped)} skipped")
        if skipped:
            print(f"  Skipped: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
        return ckpt.get("step", 0)

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
        V4.2 Memory-Augmented Cross-Attention Encoding.
        Q = Micro (6), K/V = [Macro (2) + Win Memory (8) + Loss Memory (8)] = 18
        Returns: pooled (B, d_model), attn_compat (B, 1, 8, 8)
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

        # 3. V4.2: Build Memory-Augmented K/V
        # Memory tokens: (n_mem, d_model) → expand to (B, n_mem, d_model)
        win_mem = self.win_memory.unsqueeze(0).expand(B, -1, -1)   # (B, 8, 64)
        loss_mem = self.loss_memory.unsqueeze(0).expand(B, -1, -1) # (B, 8, 64)

        # K/V = [Macro (2) + Win Memory (8) + Loss Memory (8)] = 18 tokens
        kv_tokens = torch.cat([macro_ctx, win_mem, loss_mem], dim=1)  # (B, 18, 64)

        # 4. Cross-Attention: Micro (Q) queries [Macro + Memory] (K, V)
        micro_ctx_norm = self.cross_norm(micro_ctx)
        kv_norm = self.cross_norm(kv_tokens)

        # attn_w: (B, 6, 18)
        attn_out, cross_attn_w = self.cross_attn(
            query=micro_ctx_norm,
            key=kv_norm,
            value=kv_norm,
            need_weights=True,
            average_attn_weights=True,
        )

        # Add & Norm, then FF
        micro_fused = micro_ctx + attn_out
        micro_fused = micro_fused + self.cross_ff(self.cross_ff_norm(micro_fused))

        # Output is pooled micro tokens
        pooled = micro_fused.mean(dim=1)  # (B, d_model)

        # --- Backward-compatible 8x8 attention matrix ---
        attn_compat = torch.zeros(B, 8, 8, device=obs.device)
        # Extract Macro attention (first 2 cols of cross_attn_w)
        macro_attn_slice = cross_attn_w[:, :, :2]  # (B, 6, 2) - micro→macro
        attn_compat[:, MICRO_TOKENS[0]:MICRO_TOKENS[-1]+1, MACRO_TOKENS[0]:MACRO_TOKENS[-1]+1] = macro_attn_slice
        # Macro self-attention
        attn_compat[:, MACRO_TOKENS[0]:MACRO_TOKENS[-1]+1, MACRO_TOKENS[0]:MACRO_TOKENS[-1]+1] = macro_attn.mean(dim=1)
        attn_compat = attn_compat / (attn_compat.sum(dim=-1, keepdim=True) + 1e-8)
        attn_compat = attn_compat.unsqueeze(1)  # (B, 1, 8, 8)

        # Store full cross-attention for analysis (optional)
        self._cross_attn_full = cross_attn_w  # (B, 6, 18)

        return pooled, attn_compat

    def forward(self, obs: torch.Tensor):
        pooled, attn_weights = self._encode(obs)
        logits = self.actor_head(pooled)
        value = self.critic_head(pooled)
        rr_pred = self.rr_head(pooled)
        self._attn_weights = attn_weights
        return logits, value, rr_pred, attn_weights

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        pooled, _ = self._encode(obs)
        return self.critic_head(pooled).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        pooled, attn_weights = self._encode(obs)
        logits = self.actor_head(pooled)
        value = self.critic_head(pooled).squeeze(-1)
        rr_pred = self.rr_head(pooled).squeeze(-1)
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
        return action, log_prob, entropy, value, rr_pred

    def get_embedding(self, obs: torch.Tensor) -> torch.Tensor:
        pooled, _ = self._encode(obs)
        embed = self.contrastive_head(pooled)
        return F.normalize(embed, p=2, dim=-1)

    def get_attention_weights(self, obs: torch.Tensor) -> torch.Tensor:
        _, attn_weights = self._encode(obs)
        return attn_weights

    def get_memory_attention(self) -> dict:
        """Return memory attention breakdown for analysis."""
        if self._cross_attn_full is None:
            return {}
        # cross_attn_full: (B, 6, 18) → cols: [0:2]=Macro, [2:10]=Win, [10:18]=Loss
        w = self._cross_attn_full.mean(dim=0).mean(dim=0)  # (18,)
        return {
            "macro_attn": w[:2].sum().item(),
            "win_memory_attn": w[2:2+self.n_memory_per_bank].sum().item(),
            "loss_memory_attn": w[2+self.n_memory_per_bank:].sum().item(),
        }
