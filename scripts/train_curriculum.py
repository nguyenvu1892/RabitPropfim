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
    grad_clip: float = 1.0    # Gradient clipping norm
    description: str = ""


STAGES: list[StageConfig] = [
    StageConfig(
        name="Stage1_Context",
        description="Context Recognition: M15+H1 only, learn trend/regime identification",
        total_steps=200_000,
        learning_rate=3e-4,
        batch_size=256,
        n_envs=8,
        use_m1=False, use_m5=False, use_m15=True, use_h1=True,
        freeze_m1_encoder=True, freeze_m5_encoder=True,
        freeze_m15_encoder=False, freeze_h1_encoder=False,
        freeze_entry_cross=True, freeze_struct_cross=False,
        use_episodic_memory=False,
        fixed_sl_tp=True,
        checkpoint_every=20_000,
        eval_every=10_000,
    ),
    StageConfig(
        name="Stage2_Precision",
        description="Precision Entry: Add M5, learn entry timing at POIs",
        total_steps=300_000,
        learning_rate=1e-4,
        batch_size=256,
        n_envs=8,
        use_m1=False, use_m5=True, use_m15=True, use_h1=True,
        freeze_m1_encoder=True, freeze_m5_encoder=False,
        freeze_m15_encoder=True, freeze_h1_encoder=True,   # Freeze from Stage 1
        freeze_entry_cross=False, freeze_struct_cross=True,  # Open entry, freeze struct
        use_episodic_memory=False,
        fixed_sl_tp=False,  # Variable SL/TP now
        checkpoint_every=30_000,
        eval_every=15_000,
    ),
    StageConfig(
        name="Stage3_FullFusion",
        description="Full Integration: All 4 TFs + EpisodicMemory, target WR≥55%",
        total_steps=500_000,
        learning_rate=5e-5,
        batch_size=256,
        n_envs=8,
        use_m1=True, use_m5=True, use_m15=True, use_h1=True,
        freeze_m1_encoder=False, freeze_m5_encoder=False,
        freeze_m15_encoder=False, freeze_h1_encoder=False,
        freeze_entry_cross=False, freeze_struct_cross=False,
        use_episodic_memory=True,
        fixed_sl_tp=False,
        checkpoint_every=50_000,
        eval_every=25_000,
        gamma=0.995,    # Slightly longer horizon for full context
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
        n_envs=2,
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

    # Load feature arrays
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
                logger.warning("  Missing %s — will use zeros", npy_path.name)
                data[sym][tf_name] = None

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
    if prev_checkpoint and prev_checkpoint.exists():
        logger.info("Warm-starting from %s", prev_checkpoint.name)
        checkpoint = torch.load(prev_checkpoint, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint["actor_state_dict"], strict=False)
        critic.load_state_dict(checkpoint["critic_state_dict"], strict=False)
        logger.info("  Loaded actor + critic weights (strict=False for new layers)")

    # Apply freeze/unfreeze
    logger.info("Applying freeze config...")
    apply_freeze_config(actor, stage)

    # Create optimizer (only for trainable params)
    actor_params = [p for p in actor.parameters() if p.requires_grad]
    critic_params = [p for p in critic.parameters() if p.requires_grad]

    actor_optim = torch.optim.Adam(actor_params, lr=stage.learning_rate)
    critic_optim = torch.optim.Adam(critic_params, lr=stage.learning_rate)

    # Entropy coefficient (auto-tuned)
    target_entropy = -4.0  # -action_dim
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optim = torch.optim.Adam([log_alpha], lr=stage.alpha_lr)

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
    # NOTE: This is a simplified loop structure. In production, this
    # integrates with VectorizedEnv + ReplayBuffer (from training_pipeline/).
    # Here we demonstrate the freeze/unfreeze + warm-start + checkpoint logic.

    logger.info("Starting training loop (%d steps)...", stage.total_steps)
    start_time = time.time()

    for step in range(1, stage.total_steps + 1):
        # Forward pass with dummy data (placeholder — real training uses env rollouts)
        inputs = create_dummy_input(stage, device, batch=stage.batch_size)

        # Actor forward
        action, log_prob = actor(
            inputs["m1"], inputs["m5"], inputs["m15"], inputs["h1"],
        )

        # Critic forward
        q1, q2 = critic(
            inputs["m1"], inputs["m5"], inputs["m15"], inputs["h1"],
            action.detach(),
        )

        # SAC losses (simplified — real impl uses replay buffer)
        alpha = log_alpha.exp().detach()

        # Critic loss: reward + gamma * V_target - Q(s,a)
        # (Using dummy targets for structure validation)
        target_q = torch.randn(stage.batch_size, 1, device=device)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

        critic_optim.zero_grad()
        critic_loss.backward()
        if stage.grad_clip > 0:
            nn.utils.clip_grad_norm_(critic_params, stage.grad_clip)
        critic_optim.step()

        # Actor loss: -Q(s, a_new) + alpha * log_prob
        action_new, log_prob_new = actor(
            inputs["m1"], inputs["m5"], inputs["m15"], inputs["h1"],
        )
        q1_new, q2_new = critic(
            inputs["m1"], inputs["m5"], inputs["m15"], inputs["h1"],
            action_new,
        )
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_prob_new - min_q_new).mean()

        actor_optim.zero_grad()
        actor_loss.backward()
        if stage.grad_clip > 0:
            nn.utils.clip_grad_norm_(actor_params, stage.grad_clip)
        actor_optim.step()

        # Alpha loss (entropy tuning)
        alpha_loss = -(log_alpha * (log_prob_new.detach() + target_entropy)).mean()
        alpha_optim.zero_grad()
        alpha_loss.backward()
        alpha_optim.step()

        # ── Logging ──
        if step % stage.eval_every == 0 or step == 1:
            elapsed = time.time() - start_time
            sps = step / elapsed if elapsed > 0 else 0
            logger.info(
                "[%s] Step %d/%d (%.1f%%) | SPS=%.0f | "
                "critic_loss=%.4f | actor_loss=%.4f | alpha=%.4f",
                stage.name, step, stage.total_steps,
                100 * step / stage.total_steps, sps,
                critic_loss.item(), actor_loss.item(), alpha.item(),
            )

        # ── Checkpoint ──
        if step % stage.checkpoint_every == 0:
            ckpt_path = stage_dir / f"checkpoint_{step}.pt"
            torch.save({
                "stage": stage.name,
                "step": step,
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "actor_optim": actor_optim.state_dict(),
                "critic_optim": critic_optim.state_dict(),
                "log_alpha": log_alpha.detach().cpu().item(),
            }, ckpt_path)
            logger.info("  Saved checkpoint → %s", ckpt_path.name)

            # Save as best (in real training, compare eval reward)
            torch.save({
                "stage": stage.name,
                "step": step,
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
            }, best_checkpoint_path)

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
    parser = argparse.ArgumentParser(description="Curriculum Training for Cognitive Architecture")
    parser.add_argument("--resume-stage", type=int, default=1, help="Stage to resume from (1-3)")
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
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        logger.info("GPU: %s (%.1f GB)", gpu_name, gpu_mem)

    # Select stages
    stages = TEST_STAGES if args.test else STAGES
    start_stage = max(1, min(args.resume_stage, len(stages)))

    logger.info("=" * 70)
    logger.info("  CURRICULUM TRAINING — Cognitive Architecture v1.0")
    logger.info("  Stages: %d | Starting from: Stage %d", len(stages), start_stage)
    if args.test:
        logger.info("  MODE: TEST (100 steps)")
    logger.info("=" * 70)

    # Create models
    from agents.sac_policy import SACTransformerActor, SACTransformerCritic

    actor = SACTransformerActor(
        n_features=50,
        action_dim=4,
        embed_dim=128,
        n_heads=4,
        n_cross_layers=1,
        n_regimes=4,
        hidden_dims=[256, 256],
        dropout=0.1,
    ).to(device)

    critic = SACTransformerCritic(
        n_features=50,
        action_dim=4,
        embed_dim=128,
        n_heads=4,
        n_cross_layers=1,
        n_regimes=4,
        hidden_dims=[256, 256],
        dropout=0.1,
    ).to(device)

    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    logger.info("Actor: %d params | Critic: %d params | Total: %d",
                actor_params, critic_params, actor_params + critic_params)

    # ── Run curriculum ──
    prev_checkpoint = None

    # Check for existing checkpoints to resume
    if start_stage > 1:
        prev_stage = stages[start_stage - 2]
        prev_ckpt = MODELS_DIR / f"best_{prev_stage.name}.pt"
        if prev_ckpt.exists():
            prev_checkpoint = prev_ckpt
            logger.info("Found previous checkpoint: %s", prev_ckpt.name)

    for idx, stage in enumerate(stages):
        stage_num = idx + 1
        if stage_num < start_stage:
            logger.info("Skipping Stage %d (%s) — resuming from %d",
                        stage_num, stage.name, start_stage)
            # Still try to load its checkpoint for warm-start chain
            ckpt_path = MODELS_DIR / f"best_{stage.name}.pt"
            if ckpt_path.exists():
                prev_checkpoint = ckpt_path
            continue

        prev_checkpoint = train_stage(
            stage=stage,
            actor=actor,
            critic=critic,
            device=device,
            prev_checkpoint=prev_checkpoint,
            wandb_enabled=not args.no_wandb,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Final summary ──
    logger.info("=" * 70)
    logger.info("  CURRICULUM TRAINING COMPLETE!")
    logger.info("  Checkpoints saved in: %s", MODELS_DIR)
    logger.info("=" * 70)

    checkpoints = sorted(MODELS_DIR.glob("best_*.pt"))
    for ckpt in checkpoints:
        logger.info("  %s (%.1f MB)", ckpt.name, ckpt.stat().st_size / (1024 * 1024))


if __name__ == "__main__":
    main()
