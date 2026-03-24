"""
Curriculum Training Script — 3-Stage Progressive Learning.

v1.0 — Cognitive Architecture:
    Train the Agent's "mind" in 3 stages, from simple to complex:

    Stage 1 — Context Recognition (200K steps):
        • Input: M15 + H1 only (structure + trend)
        • Features: SMC + Vol subset (12-dim)
        • Freeze: M1 encoder, M5 encoder
        • Goal: Learn to identify market context (trend, regime)
        • Action: Simplified (direction + fixed SL/TP)

    Stage 2 — Precision Entry (300K steps):
        • Input: M5 + M15 + H1 (add entry precision)
        • Features: SMC + PA + Vol (20-dim)
        • Unfreeze: M5 encoder (progressive)
        • Goal: Learn WHERE to enter within a context
        • Action: Direction + variable SL/TP

    Stage 3 — Full Integration (500K steps):
        • Input: M1 + M5 + M15 + H1 (all 4 TFs)
        • Features: Full 50-dim (28 raw + 22 knowledge)
        • Unfreeze: Everything
        • Goal: Sniper entries with full context + memory
        • Action: Full action space + EpisodicMemory bonus
        • Add EpisodicMemory warmup

    Anti-Catastrophic-Forgetting:
        • Progressive Freezing: lower-TF encoders frozen in early stages
        • Warm-start: each stage loads weights from the previous stage
        • Learning rate decay: Stage 1 lr > Stage 2 lr > Stage 3 lr

Usage:
    # Full curriculum (all 3 stages):
    python scripts/train_curriculum.py

    # Resume from stage 2:
    python scripts/train_curriculum.py --resume-stage 2

    # Quick test:
    python scripts/train_curriculum.py --test
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "rabit_propfirm_drl"))

DATA_DIR = project_root / "data"
MODELS_DIR = project_root / "models_saved"
MODELS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("curriculum")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class StageConfig:
    """Configuration for one curriculum stage."""
    name: str
    total_steps: int
    learning_rate: float
    batch_size: int
    n_envs: int               # Vectorized environment count
    use_m1: bool              # Include M1 input
    use_m5: bool              # Include M5 input
    use_m15: bool             # Include M15 input
    use_h1: bool              # Include H1 input
    freeze_m1_encoder: bool   # Freeze M1 TransformerSMC
    freeze_m5_encoder: bool   # Freeze M5 ContextEncoder
    freeze_m15_encoder: bool  # Freeze M15 ContextEncoder
    freeze_h1_encoder: bool   # Freeze H1 ContextEncoder
    freeze_entry_cross: bool  # Freeze entry cross-attention
    freeze_struct_cross: bool # Freeze structure cross-attention
    use_episodic_memory: bool # Enable EpisodicMemory bonus
    fixed_sl_tp: bool         # Use fixed SL/TP (simplified action)
    checkpoint_every: int     # Save checkpoint every N steps
    eval_every: int           # Evaluate every N steps
    gamma: float = 0.99       # Discount factor
    tau: float = 0.005        # Soft update coefficient
    alpha_lr: float = 3e-4    # Entropy coefficient LR
    grad_clip: float = 0.5    # Gradient clipping norm (tighter for 5-dim action)
    n_steps: int = 4096       # V3: Rollout steps before gradient update
    n_updates_per_rollout: int = 8  # V3: Gradient updates per collected rollout
    description: str = ""


STAGES: list[StageConfig] = [
    StageConfig(
        name="Stage1_Context",
        description="Context Recognition: M15+H1 only, learn trend/regime identification",
        total_steps=750_000,               # V3.2: from 200K → 750K (1.5M params needs more steps)
        learning_rate=1e-4,
        batch_size=2048,           # V3: from 256 → 2048
        n_envs=8,                  # V3: 8 envs (32 caused CPU IPC thrashing)
        n_steps=4096,              # V3: rollout buffer
        n_updates_per_rollout=8,   # V3: gradient updates per rollout
        use_m1=False, use_m5=False, use_m15=True, use_h1=True,
        freeze_m1_encoder=True, freeze_m5_encoder=True,
        freeze_m15_encoder=False, freeze_h1_encoder=False,
        freeze_entry_cross=True, freeze_struct_cross=False,
        use_episodic_memory=False,
        fixed_sl_tp=True,
        checkpoint_every=50_000,            # V3.2: from 20K → 50K
        eval_every=25_000,                  # V3.2: from 10K → 25K
    ),
    StageConfig(
        name="Stage2_Precision",
        description="Precision Entry: Add M5, learn entry timing at POIs",
        total_steps=300_000,
        learning_rate=1e-4,
        batch_size=2048,           # V3: from 256 → 2048
        n_envs=8,                  # V3: 8 envs (32 caused IPC thrashing)
        n_steps=4096,
        n_updates_per_rollout=8,
        use_m1=False, use_m5=True, use_m15=True, use_h1=True,
        freeze_m1_encoder=True, freeze_m5_encoder=False,
        freeze_m15_encoder=True, freeze_h1_encoder=True,
        freeze_entry_cross=False, freeze_struct_cross=True,
        use_episodic_memory=False,
        fixed_sl_tp=False,
        checkpoint_every=30_000,
        eval_every=15_000,
    ),
    StageConfig(
        name="Stage3_FullFusion",
        description="Full Integration: All 4 TFs + EpisodicMemory, target WR>=55%",
        total_steps=500_000,
        learning_rate=5e-5,
        batch_size=4096,           # V3: push to 4096 (all TFs active, 4090 eats this)
        n_envs=8,                  # V3: 8 envs
        n_steps=8192,              # V3: longer rollout for full context
        n_updates_per_rollout=16,  # V3: more updates from richer data
        use_m1=True, use_m5=True, use_m15=True, use_h1=True,
        freeze_m1_encoder=False, freeze_m5_encoder=False,
        freeze_m15_encoder=False, freeze_h1_encoder=False,
        freeze_entry_cross=False, freeze_struct_cross=False,
        use_episodic_memory=True,
        fixed_sl_tp=False,
        checkpoint_every=50_000,
        eval_every=25_000,
        gamma=0.995,
    ),
]

# For --test flag
TEST_STAGES: list[StageConfig] = [
    StageConfig(
        name="TestStage",
        description="Quick syntax/shape test",
        total_steps=100,
        learning_rate=3e-4,
        batch_size=32,
        n_envs=4,
        n_steps=32,                # V3: small rollout for test
        n_updates_per_rollout=2,
        use_m1=True, use_m5=True, use_m15=True, use_h1=True,
        freeze_m1_encoder=False, freeze_m5_encoder=False,
        freeze_m15_encoder=False, freeze_h1_encoder=False,
        freeze_entry_cross=False, freeze_struct_cross=False,
        use_episodic_memory=False,
        fixed_sl_tp=False,
        checkpoint_every=50,
        eval_every=50,
    ),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UTILITY: Freeze/Unfreeze
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def freeze_module(module: nn.Module, name: str = "") -> int:
    """Freeze all parameters in a module. Returns count of frozen params."""
    count = 0
    for param in module.parameters():
        param.requires_grad = False
        count += param.numel()
    if name:
        logger.info("  ❄ Frozen: %s (%d params)", name, count)
    return count


def unfreeze_module(module: nn.Module, name: str = "") -> int:
    """Unfreeze all parameters in a module. Returns count of unfrozen params."""
    count = 0
    for param in module.parameters():
        param.requires_grad = True
        count += param.numel()
    if name:
        logger.info("  🔥 Unfrozen: %s (%d params)", name, count)
    return count


def apply_freeze_config(actor: nn.Module, stage: StageConfig) -> None:
    """Apply freeze/unfreeze based on stage config."""
    fe = actor.feature_extractor
    ha = fe.hierarchical_attn

    # First unfreeze everything, then selectively freeze
    unfreeze_module(actor)

    if stage.freeze_m1_encoder:
        freeze_module(ha.m1_encoder, "M1 encoder (TransformerSMC)")
        freeze_module(ha.m1_query_projection, "M1 query projection")
        freeze_module(ha.m1_query_pos, "M1 positional encoding")
    if stage.freeze_m5_encoder:
        freeze_module(ha.m5_encoder, "M5 encoder (ContextEncoder)")
    if stage.freeze_m15_encoder:
        freeze_module(ha.m15_encoder, "M15 encoder (ContextEncoder)")
    if stage.freeze_h1_encoder:
        freeze_module(ha.h1_encoder, "H1 encoder (ContextEncoder)")
    if stage.freeze_entry_cross:
        freeze_module(ha.entry_cross_attn, "Entry cross-attention")
        freeze_module(ha.entry_norms_q, "Entry norms Q")
        freeze_module(ha.entry_norms_kv, "Entry norms KV")
        freeze_module(ha.entry_ff, "Entry FFN")
        freeze_module(ha.entry_norms_ff, "Entry norms FF")
    if stage.freeze_struct_cross:
        freeze_module(ha.struct_cross_attn, "Structure cross-attention")
        freeze_module(ha.struct_norms_q, "Structure norms Q")
        freeze_module(ha.struct_norms_kv, "Structure norms KV")
        freeze_module(ha.struct_ff, "Structure FFN")
        freeze_module(ha.struct_norms_ff, "Structure norms FF")

    # Count trainable vs total
    total = sum(p.numel() for p in actor.parameters())
    trainable = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    frozen = total - trainable
    logger.info("  Actor: %d trainable / %d total (%d frozen, %.1f%%)",
                trainable, total, frozen, 100 * frozen / max(total, 1))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_data(symbols: list[str], normalizer_path: Path) -> dict:
    """
    Load 50-dim feature arrays and normalizer for all symbols.
    Returns dict with data per symbol per TF.
    """
    from data_engine.normalizer import RunningNormalizer

    # Load normalizer
    with open(normalizer_path, "r") as f:
        norm_data = json.load(f)

    normalizers = {}
    for tf_name, state in norm_data.items():
        normalizers[tf_name] = RunningNormalizer.from_state_dict(state)

    # Load feature arrays + OHLCV (V3)
    data = {}
    for sym in symbols:
        safe_name = sym.replace(".", "_")
        data[sym] = {}
        for tf_name in ["M1", "M5", "M15", "H1"]:
            npy_path = DATA_DIR / f"{safe_name}_{tf_name}_50dim.npy"
            if npy_path.exists():
                arr = np.load(npy_path)
                # Normalize
                arr_norm = normalizers[tf_name].normalize(arr)
                data[sym][tf_name] = arr_norm.astype(np.float32)
                logger.info("  Loaded %s %s: %s", sym, tf_name, arr.shape)
            else:
                logger.warning("  Missing %s -- will use zeros", npy_path.name)
                data[sym][tf_name] = None

            # V3: Load OHLCV if available
            ohlcv_path = DATA_DIR / f"{safe_name}_{tf_name}_ohlcv.npy"
            ohlcv_key = f"{tf_name}_ohlcv"
            if ohlcv_path.exists():
                ohlcv = np.load(ohlcv_path).astype(np.float32)
                data[sym][ohlcv_key] = ohlcv
                logger.info("  V3 OHLCV: %s -> %s", ohlcv_path.name, ohlcv.shape)
            else:
                data[sym][ohlcv_key] = None
                if tf_name == "M5":
                    logger.warning("  V3 OHLCV MISSING: %s (checked: %s)", ohlcv_path.name, ohlcv_path)

    # V3.2: FAIL-FAST data integrity guard
    # Refuse to train on zero arrays — every required TF must have real data
    missing_files = []
    for sym in symbols:
        safe_name = sym.replace(".", "_")
        for tf_name in ["M1", "M5", "M15", "H1"]:
            if data[sym].get(tf_name) is None:
                missing_files.append(f"{safe_name}_{tf_name}_50dim.npy")
    if missing_files:
        raise FileNotFoundError(
            f"V3.2 FAIL-FAST: {len(missing_files)} critical data files missing!\n"
            f"Missing: {missing_files}\n"
            f"Run 'python scripts/fetch_historical_data.py' to generate all data files.\n"
            f"NEVER train on zero arrays — this causes mode collapse."
        )

    return {"data": data, "normalizers": normalizers}


def create_dummy_input(stage: StageConfig, device: torch.device, batch: int = 4) -> dict:
    """Create dummy input tensors for shape testing."""
    n_features = 50
    inputs = {}
    if stage.use_m1:
        inputs["m1"] = torch.randn(batch, 128, n_features, device=device)
    else:
        inputs["m1"] = torch.zeros(batch, 128, n_features, device=device)

    if stage.use_m5:
        inputs["m5"] = torch.randn(batch, 64, n_features, device=device)
    else:
        inputs["m5"] = torch.zeros(batch, 64, n_features, device=device)

    if stage.use_m15:
        inputs["m15"] = torch.randn(batch, 48, n_features, device=device)
    else:
        inputs["m15"] = torch.zeros(batch, 48, n_features, device=device)

    if stage.use_h1:
        inputs["h1"] = torch.randn(batch, 24, n_features, device=device)
    else:
        inputs["h1"] = torch.zeros(batch, 24, n_features, device=device)

    return inputs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING LOOP (One Stage)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_stage(
    stage: StageConfig,
    actor: nn.Module,
    critic: nn.Module,
    device: torch.device,
    prev_checkpoint: Optional[Path] = None,
    wandb_enabled: bool = False,
) -> Path:
    """
    Train one curriculum stage.

    Args:
        stage: Stage configuration
        actor: SACTransformerActor
        critic: SACTransformerCritic
        device: CUDA/CPU device
        prev_checkpoint: Path to previous stage checkpoint (for warm-start)
        wandb_enabled: Enable W&B logging

    Returns:
        Path to best checkpoint from this stage.
    """
    logger.info("=" * 70)
    logger.info("  STAGE: %s", stage.name)
    logger.info("  %s", stage.description)
    logger.info("  Steps: %d | LR: %.1e | Batch: %d | Envs: %d",
                stage.total_steps, stage.learning_rate, stage.batch_size, stage.n_envs)
    logger.info("  TFs: M1=%s M5=%s M15=%s H1=%s",
                "✓" if stage.use_m1 else "✗",
                "✓" if stage.use_m5 else "✗",
                "✓" if stage.use_m15 else "✗",
                "✓" if stage.use_h1 else "✗")
    logger.info("  Memory: %s | Fixed SL/TP: %s",
                "ON" if stage.use_episodic_memory else "OFF",
                "YES" if stage.fixed_sl_tp else "NO")
    logger.info("=" * 70)

    # Load previous checkpoint if warm-starting
    prev_log_alpha = None
    prev_alpha_optim_state = None
    prev_actor_optim_state = None
    prev_critic_optim_state = None

    if prev_checkpoint and prev_checkpoint.exists():
        logger.info("Warm-starting from %s", prev_checkpoint.name)
        checkpoint = torch.load(prev_checkpoint, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint["actor_state_dict"], strict=False)
        critic.load_state_dict(checkpoint["critic_state_dict"], strict=False)
        logger.info("  Loaded actor + critic weights (strict=False for new layers)")

        # ── FIX #1: Restore log_alpha from previous stage ──
        if "log_alpha" in checkpoint:
            prev_log_alpha = checkpoint["log_alpha"]
            logger.info("  Restored log_alpha = %.4f (alpha=%.4f)",
                        prev_log_alpha, np.exp(prev_log_alpha))
        else:
            logger.warning("  log_alpha NOT FOUND in checkpoint — will init fresh")

        # Restore optimizer states if available (same LR → direct load)
        if "alpha_optim" in checkpoint:
            prev_alpha_optim_state = checkpoint["alpha_optim"]
        if "actor_optim" in checkpoint:
            prev_actor_optim_state = checkpoint["actor_optim"]
        if "critic_optim" in checkpoint:
            prev_critic_optim_state = checkpoint["critic_optim"]

    # Apply freeze/unfreeze
    logger.info("Applying freeze config...")
    apply_freeze_config(actor, stage)

    # Create optimizer (only for trainable params)
    actor_params = [p for p in actor.parameters() if p.requires_grad]
    critic_params = [p for p in critic.parameters() if p.requires_grad]

    actor_optim = torch.optim.Adam(actor_params, lr=stage.learning_rate)
    critic_optim = torch.optim.Adam(critic_params, lr=stage.learning_rate)

    # ── FIX #1 (cont): Restore optimizer states ──
    # NOTE: Only restore if param count matches (freeze config may change params)
    if prev_actor_optim_state is not None:
        try:
            actor_optim.load_state_dict(prev_actor_optim_state)
            logger.info("  Restored actor optimizer state")
        except (ValueError, KeyError):
            logger.warning("  Could not restore actor optimizer (param mismatch). Starting fresh.")
    if prev_critic_optim_state is not None:
        try:
            critic_optim.load_state_dict(prev_critic_optim_state)
            logger.info("  Restored critic optimizer state")
        except (ValueError, KeyError):
            logger.warning("  Could not restore critic optimizer (param mismatch). Starting fresh.")

    # Entropy coefficient (auto-tuned) — V3: raised target for more exploration
    target_entropy = -2.0  # V3: raised from -5.0 to fight mode collapse
    if prev_log_alpha is not None:
        log_alpha = torch.tensor([prev_log_alpha], requires_grad=True, device=device)
        logger.info("  log_alpha warm-started at %.4f (alpha=%.4f)",
                    prev_log_alpha, np.exp(prev_log_alpha))
    else:
        # Default: start with LOW alpha (conservative exploration)
        init_log_alpha = -2.0  # alpha = exp(-2) = 0.135
        log_alpha = torch.tensor([init_log_alpha], requires_grad=True, device=device)
        logger.info("  log_alpha initialized at %.4f (alpha=%.4f)",
                    init_log_alpha, np.exp(init_log_alpha))
    alpha_optim = torch.optim.Adam([log_alpha], lr=stage.alpha_lr)
    if prev_alpha_optim_state is not None:
        try:
            alpha_optim.load_state_dict(prev_alpha_optim_state)
            logger.info("  Restored alpha optimizer state")
        except (ValueError, KeyError):
            logger.warning("  Could not restore alpha optimizer. Starting fresh.")

    # EpisodicMemory (if enabled)
    memory = None
    if stage.use_episodic_memory:
        from agents.episodic_memory import EpisodicMemory
        memory_path = MODELS_DIR / "episodic_memory.json"
        if memory_path.exists():
            memory = EpisodicMemory.load(memory_path)
            logger.info("Loaded EpisodicMemory: %d entries", memory.size)
        else:
            memory = EpisodicMemory(capacity=500, k=5)
            logger.info("Created fresh EpisodicMemory")

    # Checkpoint tracking
    best_reward = float("-inf")
    best_checkpoint_path = MODELS_DIR / f"best_{stage.name}.pt"
    stage_dir = MODELS_DIR / stage.name
    stage_dir.mkdir(exist_ok=True)

    # ── Training loop ──
    # Uses AsyncVectorEnv for PARALLEL environment stepping.
    # N envs run simultaneously across CPU cores → real diverse batch each step.

    symbols = ["XAUUSD", "BTCUSD", "ETHUSD", "US30_cash", "US100_cash"]
    all_data = load_data(symbols, DATA_DIR / "normalizer_v3.json")

    # Import environment
    import yaml as _yaml
    _config_path = project_root / "rabit_propfirm_drl" / "configs" / "prop_rules.yaml"
    with open(_config_path, "r", encoding="utf-8") as _f:
        env_config = _yaml.safe_load(_f)

    from environments.prop_env import MultiTFTradingEnv

    # ── AsyncVectorEnv: N parallel envs across CPU cores ──
    import gymnasium
    n_envs = stage.n_envs  # 16 default, up to 32

    def make_env(sym: str, seed: int):
        """Factory function for AsyncVectorEnv."""
        def _init():
            sym_data = all_data["data"][sym]
            _z1k = np.zeros((1000, 50), dtype=np.float32)
            _z500 = np.zeros((500, 50), dtype=np.float32)
            _z200 = np.zeros((200, 50), dtype=np.float32)
            env = MultiTFTradingEnv(
                data_m1=sym_data.get("M1") if sym_data.get("M1") is not None else _z1k,
                data_m5=sym_data.get("M5") if sym_data.get("M5") is not None else _z500,
                data_m15=sym_data.get("M15") if sym_data.get("M15") is not None else _z500,
                data_h1=sym_data.get("H1") if sym_data.get("H1") is not None else _z200,
                config=env_config,
                n_features=50,
                initial_balance=10_000.0,
                episode_length=2000,
                ohlcv_m5=sym_data.get("M5_ohlcv"),  # V3: pass real OHLCV (None triggers fallback)
            )
            env.reset(seed=seed)
            return env
        return _init

    # Distribute symbols round-robin across N envs
    env_fns = []
    for i in range(n_envs):
        sym = symbols[i % len(symbols)]
        env_fns.append(make_env(sym, seed=42 + i))

    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns)
    obs_batch, _ = vec_env.reset()
    logger.info("Created AsyncVectorEnv with %d parallel envs (symbols: %s)",
                n_envs, [symbols[i % len(symbols)] for i in range(n_envs)])

    logger.info("Starting training loop (%d steps, n_steps=%d, batch=%d, n_updates=%d)...",
                stage.total_steps, stage.n_steps, stage.batch_size, stage.n_updates_per_rollout)
    start_time = time.time()

    # ── V3: Rollout Buffer (stores transitions on CPU, samples to GPU) ──
    # Buffer stores flattened obs tensors + actions + rewards
    # This allows mini-batch sampling of batch_size >> n_envs
    buffer_m1 = []
    buffer_m5 = []
    buffer_m15 = []
    buffer_h1 = []
    buffer_rewards = []

    global_step = 0
    rollout_count = 0

    while global_step < stage.total_steps:
        # ══════════════════════════════════════════════════
        # PHASE 1: COLLECT ROLLOUT (CPU-bound, fills n_steps)
        # ══════════════════════════════════════════════════
        buffer_m1.clear()
        buffer_m5.clear()
        buffer_m15.clear()
        buffer_h1.clear()
        buffer_rewards.clear()

        collect_steps = min(stage.n_steps, stage.total_steps - global_step)
        for _c in range(collect_steps):
            # Build input tensors
            m1_t = torch.from_numpy(obs_batch["m1"]).to(device, dtype=torch.float32)
            m5_t = torch.from_numpy(obs_batch["m5"]).to(device, dtype=torch.float32)
            m15_t = torch.from_numpy(obs_batch["m15"]).to(device, dtype=torch.float32)
            h1_t = torch.from_numpy(obs_batch["h1"]).to(device, dtype=torch.float32)

            # NaN/Inf guard
            m1_t = torch.nan_to_num(m1_t, nan=0.0, posinf=5.0, neginf=-5.0)
            m5_t = torch.nan_to_num(m5_t, nan=0.0, posinf=5.0, neginf=-5.0)
            m15_t = torch.nan_to_num(m15_t, nan=0.0, posinf=5.0, neginf=-5.0)
            h1_t = torch.nan_to_num(h1_t, nan=0.0, posinf=5.0, neginf=-5.0)

            # Zero unused TFs
            if not stage.use_m1:
                m1_t = torch.zeros_like(m1_t)
            if not stage.use_m5:
                m5_t = torch.zeros_like(m5_t)
            if not stage.use_m15:
                m15_t = torch.zeros_like(m15_t)
            if not stage.use_h1:
                h1_t = torch.zeros_like(h1_t)

            # Actor forward (no grad for collection)
            with torch.no_grad():
                action, _ = actor(m1_t, m5_t, m15_t, h1_t)

            # Scale actions
            actions_np = action.cpu().numpy()
            env_actions = np.stack([
                np.clip(actions_np[:, 0], -1.0, 1.0),
                np.clip(actions_np[:, 1], -1.0, 1.0),
                np.clip((actions_np[:, 2] + 1) / 2, 0.0, 1.0),
                np.clip(actions_np[:, 3] * 1.25 + 1.75, 0.5, 3.0),
                np.clip(actions_np[:, 4] * 2.25 + 2.75, 0.5, 5.0),
            ], axis=-1).astype(np.float32)

            # Step all envs
            next_obs_batch, rewards, terminateds, truncateds, infos = vec_env.step(env_actions)

            # Store in buffer (CPU tensors)
            buffer_m1.append(m1_t.cpu())
            buffer_m5.append(m5_t.cpu())
            buffer_m15.append(m15_t.cpu())
            buffer_h1.append(h1_t.cpu())
            buffer_rewards.append(torch.from_numpy(rewards.astype(np.float32)).unsqueeze(-1))

            obs_batch = next_obs_batch
            global_step += 1

        # Stack buffer: (n_steps * n_envs, ...)
        all_m1 = torch.cat(buffer_m1, dim=0)       # (n_steps*n_envs, 128, 50)
        all_m5 = torch.cat(buffer_m5, dim=0)
        all_m15 = torch.cat(buffer_m15, dim=0)
        all_h1 = torch.cat(buffer_h1, dim=0)
        all_rewards = torch.cat(buffer_rewards, dim=0)  # (n_steps*n_envs, 1)
        buffer_size = all_m1.shape[0]
        rollout_count += 1

        # ══════════════════════════════════════════════════
        # PHASE 2: GRADIENT UPDATES (GPU-bound, fills GPU)
        # ══════════════════════════════════════════════════
        alpha = log_alpha.exp().detach()
        avg_critic_loss = 0.0
        avg_actor_loss = 0.0
        avg_entropy = 0.0

        for _u in range(stage.n_updates_per_rollout):
            # Sample random mini-batch from rollout buffer
            indices = torch.randint(0, buffer_size, (stage.batch_size,))
            mb_m1 = all_m1[indices].to(device)
            mb_m5 = all_m5[indices].to(device)
            mb_m15 = all_m15[indices].to(device)
            mb_h1 = all_h1[indices].to(device)
            mb_rewards = all_rewards[indices].to(device)

            # Forward pass (fresh actions for policy gradient)
            action_new, log_prob_new = actor(mb_m1, mb_m5, mb_m15, mb_h1)

            # Critic forward
            q1, q2 = critic(mb_m1, mb_m5, mb_m15, mb_h1, action_new.detach())

            # Critic loss (TD(0): target = reward)
            target_q = mb_rewards
            critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

            critic_optim.zero_grad()
            critic_loss.backward()
            if stage.grad_clip > 0:
                nn.utils.clip_grad_norm_(critic_params, stage.grad_clip)
            critic_optim.step()

            # NaN guard on critic
            critic_nan = any(torch.isnan(p).any() for p in critic_params if p.grad is not None)
            if critic_nan:
                logger.error("  NaN in critic at step %d! Recovering...", global_step)
                if best_checkpoint_path.exists():
                    ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
                    critic.load_state_dict(ckpt["critic_state_dict"])
                continue

            # Actor loss: -Q(s, a_new) + alpha * log_prob
            q1_new, q2_new = critic(mb_m1, mb_m5, mb_m15, mb_h1, action_new)
            min_q_new = torch.min(q1_new, q2_new)
            actor_loss = (alpha * log_prob_new - min_q_new).mean()

            actor_optim.zero_grad()
            actor_loss.backward()
            if stage.grad_clip > 0:
                nn.utils.clip_grad_norm_(actor_params, stage.grad_clip)
            actor_optim.step()

            # NaN guard on actor
            actor_nan = any(torch.isnan(p).any() for p in actor_params if p.grad is not None)
            if actor_nan:
                logger.error("  NaN in actor at step %d! Recovering...", global_step)
                if best_checkpoint_path.exists():
                    ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
                    actor.load_state_dict(ckpt["actor_state_dict"])
                continue

            # Alpha loss (entropy tuning)
            alpha_loss = -(log_alpha * (log_prob_new.detach() + target_entropy)).mean()
            alpha_optim.zero_grad()
            alpha_loss.backward()
            alpha_optim.step()

            # V3: Entropy floor
            with torch.no_grad():
                log_alpha.clamp_(min=-3.0, max=2.0)
            alpha = log_alpha.exp().detach()

            avg_critic_loss += critic_loss.item()
            avg_actor_loss += actor_loss.item()
            avg_entropy += (-log_prob_new.detach().mean().item())

        # Average over updates
        n_upd = max(stage.n_updates_per_rollout, 1)
        avg_critic_loss /= n_upd
        avg_actor_loss /= n_upd
        avg_entropy /= n_upd

        # ── Logging ──
        step = global_step
        if rollout_count % max(1, stage.eval_every // stage.n_steps) == 0 or rollout_count == 1:
            elapsed = time.time() - start_time
            sps = step / elapsed if elapsed > 0 else 0
            logger.info(
                "[%s] Step %d/%d (%.1f%%) | SPS=%.0f | "
                "critic=%.4f | actor=%.4f | alpha=%.4f | entropy=%.3f | buf=%d",
                stage.name, step, stage.total_steps,
                100 * step / stage.total_steps, sps,
                avg_critic_loss, avg_actor_loss, alpha.item(), avg_entropy, buffer_size,
            )

        # ── Checkpoint (based on global_step) ──
        if step >= stage.checkpoint_every and step % stage.checkpoint_every < stage.n_steps:
            full_ckpt = {
                "stage": stage.name,
                "step": step,
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "actor_optim": actor_optim.state_dict(),
                "critic_optim": critic_optim.state_dict(),
                "log_alpha": log_alpha.detach().cpu().item(),
                "alpha_optim": alpha_optim.state_dict(),
            }

            ckpt_path = stage_dir / f"checkpoint_{step}.pt"
            torch.save(full_ckpt, ckpt_path)
            logger.info("  Saved checkpoint -> %s (alpha=%.4f)",
                        ckpt_path.name, log_alpha.exp().item())

            # Save as best
            torch.save(full_ckpt, best_checkpoint_path)
            logger.info("  Updated best -> %s", best_checkpoint_path.name)

    # ── MANDATORY final checkpoint at stage end ──
    # Ensures log_alpha + optimizers are ALWAYS saved, even if last step
    # didn't align with checkpoint_every. This prevents alpha reset!
    final_ckpt = {
        "stage": stage.name,
        "step": stage.total_steps,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optim": actor_optim.state_dict(),
        "critic_optim": critic_optim.state_dict(),
        "log_alpha": log_alpha.detach().cpu().item(),
        "alpha_optim": alpha_optim.state_dict(),
    }
    torch.save(final_ckpt, best_checkpoint_path)
    logger.info("  FINAL checkpoint -> %s (log_alpha=%.4f, alpha=%.4f)",
                best_checkpoint_path.name,
                log_alpha.detach().cpu().item(),
                log_alpha.exp().item())

    # ── Cleanup parallel envs ──
    vec_env.close()

    # ── Save EpisodicMemory ──
    if memory is not None:
        memory.save(MODELS_DIR / "episodic_memory.json")
        logger.info("Saved EpisodicMemory: %d entries", memory.size)

    elapsed = time.time() - start_time
    logger.info("Stage %s complete: %d steps in %.1fs (%.0f SPS)",
                stage.name, stage.total_steps, elapsed, stage.total_steps / max(elapsed, 1))

    return best_checkpoint_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    parser = argparse.ArgumentParser(description="Curriculum Training V3 -- Stage-Gate Process")
    parser.add_argument("--stage", type=int, default=None,
                        help="Train ONLY this stage (1/2/3). HARD STOP after completion.")
    parser.add_argument("--all", action="store_true",
                        help="Run ALL stages sequentially (NOT recommended for V3).")
    parser.add_argument("--resume-stage", type=int, default=1,
                        help="When using --all, start from this stage.")
    parser.add_argument("--test", action="store_true", help="Quick test run (100 steps)")
    parser.add_argument("--device", type=str, default="auto", help="Device: cuda/cpu/auto")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("GPU: %s (%.1f GB)", gpu_name, gpu_mem)

    # Select stages
    stages = TEST_STAGES if args.test else STAGES

    # ── V3: Determine run mode ──
    single_stage_mode = False
    target_stage_num = None

    if args.test:
        pass  # Test runs all test stages
    elif args.stage is not None:
        single_stage_mode = True
        target_stage_num = args.stage
        if target_stage_num < 1 or target_stage_num > len(stages):
            logger.error("Invalid --stage %d. Must be 1-%d.", target_stage_num, len(stages))
            sys.exit(1)
    elif args.all:
        logger.warning("Running ALL stages (--all). Stage-Gate review will be skipped!")
    else:
        logger.error("=" * 70)
        logger.error("  V3 STAGE-GATE: You must specify which stage to train!")
        logger.error("")
        logger.error("  Usage:")
        logger.error("    python train_curriculum.py --stage 1   # Train Stage 1 ONLY")
        logger.error("    python train_curriculum.py --stage 2   # Train Stage 2 (needs Stage 1 .pt)")
        logger.error("    python train_curriculum.py --stage 3   # Train Stage 3 (needs Stage 2 .pt)")
        logger.error("    python train_curriculum.py --test      # Quick 100-step test")
        logger.error("")
        logger.error("  Workflow: train --stage N -> backtest_behavioral.py -> review -> --stage N+1")
        logger.error("=" * 70)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("  CURRICULUM TRAINING V3")
    if single_stage_mode:
        logger.info("  MODE: STAGE-GATE (Stage %d only -> HARD STOP)", target_stage_num)
    elif args.test:
        logger.info("  MODE: TEST (100 steps)")
    else:
        logger.info("  MODE: ALL STAGES (resume from %d)", args.resume_stage)
    logger.info("=" * 70)

    # Create models
    from agents.sac_policy import SACTransformerActor, SACTransformerCritic

    actor = SACTransformerActor(
        n_features=50, action_dim=5, embed_dim=128, n_heads=4,
        n_cross_layers=1, n_regimes=4, hidden_dims=[256, 256], dropout=0.1,
    ).to(device)
    critic = SACTransformerCritic(
        n_features=50, action_dim=5, embed_dim=128, n_heads=4,
        n_cross_layers=1, n_regimes=4, hidden_dims=[256, 256], dropout=0.1,
    ).to(device)

    actor_p = sum(p.numel() for p in actor.parameters())
    critic_p = sum(p.numel() for p in critic.parameters())
    logger.info("Actor: %d params | Critic: %d params | Total: %d",
                actor_p, critic_p, actor_p + critic_p)

    # ── Build run list ──
    if single_stage_mode:
        stages_to_run = [(target_stage_num, stages[target_stage_num - 1])]
    else:
        start = max(1, min(args.resume_stage, len(stages)))
        stages_to_run = [(i + 1, s) for i, s in enumerate(stages) if i + 1 >= start]

    # ── Find previous checkpoint (Stage-Gate validation) ──
    prev_checkpoint = None
    first_stage_num = stages_to_run[0][0]
    if first_stage_num > 1:
        prev_stage = stages[first_stage_num - 2]
        prev_ckpt = MODELS_DIR / f"best_{prev_stage.name}.pt"
        if prev_ckpt.exists():
            prev_checkpoint = prev_ckpt
            logger.info("Warm-start from: %s", prev_ckpt.name)
        else:
            logger.error("=" * 70)
            logger.error("  STAGE-GATE VIOLATION!")
            logger.error("  Cannot start Stage %d without Stage %d checkpoint.",
                         first_stage_num, first_stage_num - 1)
            logger.error("  Missing: %s", prev_ckpt)
            logger.error("  Train and validate Stage %d first!", first_stage_num - 1)
            logger.error("=" * 70)
            sys.exit(1)

    # ── Run stages ──
    for stage_num, stage in stages_to_run:
        prev_checkpoint = train_stage(
            stage=stage, actor=actor, critic=critic,
            device=device, prev_checkpoint=prev_checkpoint,
            wandb_enabled=not args.no_wandb,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Final summary ──
    logger.info("=" * 70)
    if single_stage_mode:
        st = stages[target_stage_num - 1]
        logger.info("  STAGE %d (%s) COMPLETE!", target_stage_num, st.name)
        logger.info("")
        logger.info("  >>> HARD STOP -- Do NOT proceed without review! <<<")
        logger.info("")
        logger.info("  Next steps:")
        logger.info("    1. python scripts/backtest_behavioral.py")
        logger.info("    2. Review behavioral report")
        if target_stage_num < len(stages):
            nxt = target_stage_num + 1
            logger.info("    3. If PASS: python scripts/train_curriculum.py --stage %d", nxt)
        else:
            logger.info("    3. If PASS: Deploy to paper trading!")
        logger.info("    4. If FAIL: Tune hyperparams, re-train --stage %d", target_stage_num)
    else:
        logger.info("  CURRICULUM TRAINING COMPLETE!")
    logger.info("  Checkpoints: %s", MODELS_DIR)
    logger.info("=" * 70)

    for ckpt in sorted(MODELS_DIR.glob("best_*.pt")):
        logger.info("  %s (%.1f MB)", ckpt.name, ckpt.stat().st_size / (1024 * 1024))


if __name__ == "__main__":
    main()
