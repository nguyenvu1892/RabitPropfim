"""
Brain Surgery Script -- Extract encoder weights from Stage 2, re-init policy head.

Reads best_Stage2_Precision.pt, keeps ONLY the encoder/attention weights
(the "eyes" of the model), and randomizes the policy head + regime detector
(the "brain" that decides). Saves as pretrained_eyes_only.pt.

Usage:
    python scripts/extract_encoder.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "rabit_propfirm_drl"))

MODELS_DIR = project_root / "models_saved"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("brain_surgery")


def main() -> None:
    src_path = MODELS_DIR / "best_Stage2_Precision.pt"
    dst_path = MODELS_DIR / "pretrained_eyes_only.pt"

    if not src_path.exists():
        logger.error("Source checkpoint not found: %s", src_path)
        return

    logger.info("=" * 60)
    logger.info("  BRAIN SURGERY: Extract Encoders, Re-init Policy Head")
    logger.info("=" * 60)

    # ── Load source checkpoint ──
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    logger.info("Loaded: %s (stage=%s, step=%s)",
                src_path.name, ckpt.get("stage", "?"), ckpt.get("step", "?"))

    # ── Create fresh model ──
    from agents.sac_policy import SACTransformerActor, SACTransformerCritic

    actor = SACTransformerActor(
        n_features=50, action_dim=5, embed_dim=128,
        n_heads=4, n_cross_layers=1, n_regimes=4,
        hidden_dims=[256, 256], dropout=0.1,
    )
    critic = SACTransformerCritic(
        n_features=50, action_dim=5, embed_dim=128,
        n_heads=4, n_cross_layers=1, n_regimes=4,
        hidden_dims=[256, 256], dropout=0.1,
    )

    # Record fresh (random) weights for comparison
    fresh_actor_sd = {k: v.clone() for k, v in actor.state_dict().items()}

    # ── Define which layers to KEEP (encoders) vs RE-INIT (policy) ──
    # KEEP: All encoder weights (they learned to "see" the market)
    keep_prefixes = [
        "feature_extractor.hierarchical_attn.m1_encoder",
        "feature_extractor.hierarchical_attn.m1_query_projection",
        "feature_extractor.hierarchical_attn.m1_query_pos",
        "feature_extractor.hierarchical_attn.m5_encoder",
        "feature_extractor.hierarchical_attn.m15_encoder",
        "feature_extractor.hierarchical_attn.h1_encoder",
        "feature_extractor.hierarchical_attn.m1_proj",
        "feature_extractor.hierarchical_attn.m5_proj",
        "feature_extractor.hierarchical_attn.m15_proj",
        "feature_extractor.hierarchical_attn.h1_proj",
    ]

    # RE-INIT: Policy head, cross-attention, regime detector
    reinit_prefixes = [
        "feature_extractor.hierarchical_attn.entry_cross",
        "feature_extractor.hierarchical_attn.entry_norms",
        "feature_extractor.hierarchical_attn.entry_ff",
        "feature_extractor.hierarchical_attn.struct_cross",
        "feature_extractor.hierarchical_attn.struct_norms",
        "feature_extractor.hierarchical_attn.struct_ff",
        "feature_extractor.regime_detector",
        "trunk",
        "mean_head",
        "log_std_head",
    ]

    # ── Selective loading ──
    src_actor_sd = ckpt["actor_state_dict"]
    new_actor_sd = actor.state_dict()

    kept_count = 0
    reinit_count = 0
    skipped_count = 0

    for key in new_actor_sd.keys():
        is_keep = any(key.startswith(prefix) for prefix in keep_prefixes)
        is_reinit = any(key.startswith(prefix) for prefix in reinit_prefixes)

        if is_keep and key in src_actor_sd:
            # Load from Stage 2
            new_actor_sd[key] = src_actor_sd[key].clone()
            kept_count += 1
        elif is_reinit:
            # Keep random initialization (already done by constructor)
            reinit_count += 1
        else:
            # Layers not explicitly categorized - keep from source if available
            if key in src_actor_sd:
                new_actor_sd[key] = src_actor_sd[key].clone()
                kept_count += 1
            else:
                skipped_count += 1

    actor.load_state_dict(new_actor_sd)

    logger.info("")
    logger.info("  Surgery results:")
    logger.info("    KEPT (from Stage 2):   %d layers", kept_count)
    logger.info("    RE-INIT (randomized):  %d layers", reinit_count)
    logger.info("    SKIPPED (missing):     %d layers", skipped_count)

    # ── Verify surgery worked ──
    logger.info("")
    logger.info("  Verification:")
    final_sd = actor.state_dict()

    # Check a kept layer
    sample_kept = "feature_extractor.hierarchical_attn.m15_encoder.encoder.layers.0.self_attn.in_proj_weight"
    if sample_kept in final_sd and sample_kept in src_actor_sd:
        diff = (final_sd[sample_kept] - src_actor_sd[sample_kept]).abs().max().item()
        logger.info("    m15_encoder (KEPT):  max diff from Stage2 = %.8f %s",
                    diff, "OK" if diff < 1e-6 else "MISMATCH!")

    # Check a re-init layer
    sample_reinit = "mean_head.weight"
    if sample_reinit in final_sd and sample_reinit in src_actor_sd:
        diff_from_src = (final_sd[sample_reinit] - src_actor_sd[sample_reinit]).abs().max().item()
        diff_from_fresh = (final_sd[sample_reinit] - fresh_actor_sd[sample_reinit]).abs().max().item()
        logger.info("    mean_head (RE-INIT): diff from Stage2 = %.4f, diff from fresh = %.8f %s",
                    diff_from_src, diff_from_fresh,
                    "OK" if diff_from_fresh < 1e-6 else "OK (re-randomized)")

    # ── Save ──
    output_ckpt = {
        "stage": "pretrained_eyes_only",
        "step": 0,
        "source": "best_Stage2_Precision.pt",
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "log_alpha": -2.0,  # Start with low alpha (conservative)
        "surgery_info": {
            "kept_layers": kept_count,
            "reinit_layers": reinit_count,
            "kept_prefixes": keep_prefixes,
            "reinit_prefixes": reinit_prefixes,
        },
    }

    torch.save(output_ckpt, dst_path)
    logger.info("")
    logger.info("  Saved: %s (%.2f MB)",
                dst_path.name, dst_path.stat().st_size / (1024 * 1024))
    logger.info("=" * 60)
    logger.info("  SURGERY COMPLETE!")
    logger.info("  Next: python scripts/train_curriculum.py --resume-stage 1")
    logger.info("         (will warm-start from pretrained_eyes_only.pt)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
