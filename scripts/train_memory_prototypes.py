#!/usr/bin/env python3
"""
V4.5 Stage 2: K-Means Crucible — Dual Memory Bank Prototypes.

Loads the Master Vault (JSONL), passes observations through AttentionPPO's
internal _encode() to get d_model=64 pooled representations, then clusters
WIN and LOSS observations SEPARATELY into 8 centroids each using MiniBatchKMeans.

Output: memory_prototypes_v45.pt containing:
  - win_prototypes:  [8, 64] tensor
  - loss_prototypes: [8, 64] tensor
  - win_frozen_mask:  [8] bool (first 4 = frozen core)
  - loss_frozen_mask: [8] bool (first 4 = frozen core)
"""
import sys, json, logging, numpy as np, torch
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rabit_propfirm_drl"))

from models.attention_ppo import AttentionPPO
from training_pipeline.contrastive_memory import ContrastiveMemory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_memory_prototypes")

PROJECT = Path(__file__).resolve().parent.parent
MODELS = PROJECT / "models_saved"
MEMORY_DIR = MODELS / "contrastive_memory_v36"


def extract_pooled(model, obs_np, device, batch_size=2048):
    """Extract d_model=64 pooled representations via model._encode()."""
    pooled_list = []
    with torch.no_grad():
        for i in range(0, len(obs_np), batch_size):
            batch = torch.from_numpy(obs_np[i:i+batch_size]).float().to(device)
            batch = torch.nan_to_num(batch, nan=0.0)
            pooled, _ = model._encode(batch)  # (B, d_model=64)
            pooled_list.append(pooled.cpu().numpy())
    return np.concatenate(pooled_list, axis=0)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-per-bank", type=int, default=8, help="Clusters per bank (win/loss)")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--n-frozen-core", type=int, default=4, help="Number of frozen core prototypes per bank")
    parser.add_argument("--obs-dim", type=int, default=488, help="Obs dim of the target model (488 for unified)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Find best checkpoint
    ckpt_path = None
    for candidate in ["best_v43_stage3_A.pt", "best_v43_stage2_A.pt", "best_v43_stage1_A.pt",
                       "best_v43_stage3_B.pt", "best_v43_stage2_B.pt", "best_v43_stage1_B.pt"]:
        p = MODELS / candidate
        if p.exists():
            ckpt_path = p
            break

    if ckpt_path is None:
        logger.error("No checkpoint found in %s. Cannot extract embeddings.", MODELS)
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    old_obs_dim = ckpt.get("obs_dim", 464)
    
    # Build model with target obs_dim
    model = AttentionPPO(obs_dim=args.obs_dim, n_actions=4).to(device)
    
    # Warm-start if dimensions differ
    if old_obs_dim != args.obs_dim:
        model.warm_start_from_v42(str(ckpt_path), device=device)
    else:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    logger.info("Loaded AttentionPPO from %s (old_obs=%d, target_obs=%d)", ckpt_path.name, old_obs_dim, args.obs_dim)

    # Load Master Vault
    logger.info("Loading Master Vault from %s", MEMORY_DIR)
    mem = ContrastiveMemory.load(MEMORY_DIR)
    
    total_wins = sum(len(mem.wins[s]) for s in mem.wins)
    total_losses = sum(len(mem.losses[s]) for s in mem.losses)
    logger.info("Master Vault: %d wins, %d losses", total_wins, total_losses)

    if total_wins < args.k_per_bank or total_losses < args.k_per_bank:
        logger.error("Not enough data! Need at least %d wins and %d losses.", args.k_per_bank, args.k_per_bank)
        return

    # Collect observations and zero-pad shorter ones to obs_dim
    def collect_and_pad(entries_dict):
        all_obs = []
        for sym in entries_dict:
            for e in entries_dict[sym]:
                obs = e["obs"]
                if len(obs) < args.obs_dim:
                    obs = np.pad(obs, (0, args.obs_dim - len(obs)), mode="constant", constant_values=0.0)
                all_obs.append(obs)
        return np.stack(all_obs) if all_obs else np.zeros((0, args.obs_dim), dtype=np.float32)

    win_obs = collect_and_pad(mem.wins)
    loss_obs = collect_and_pad(mem.losses)
    logger.info("Win obs: %s | Loss obs: %s", win_obs.shape, loss_obs.shape)

    # Extract d_model=64 pooled representations
    logger.info("Extracting pooled representations (d_model=64)...")
    win_pooled = extract_pooled(model, win_obs, device, args.batch_size)   # (N_win, 64)
    loss_pooled = extract_pooled(model, loss_obs, device, args.batch_size) # (N_loss, 64)
    logger.info("Win pooled: %s | Loss pooled: %s", win_pooled.shape, loss_pooled.shape)

    # MiniBatchKMeans — SEPARATE clustering for each bank
    logger.info("Clustering %d win entries into %d prototypes...", len(win_pooled), args.k_per_bank)
    km_win = MiniBatchKMeans(n_clusters=args.k_per_bank, batch_size=args.batch_size, n_init=10, random_state=42)
    km_win.fit(win_pooled)
    win_prototypes = torch.from_numpy(km_win.cluster_centers_).float()  # (8, 64)

    logger.info("Clustering %d loss entries into %d prototypes...", len(loss_pooled), args.k_per_bank)
    km_loss = MiniBatchKMeans(n_clusters=args.k_per_bank, batch_size=args.batch_size, n_init=10, random_state=42)
    km_loss.fit(loss_pooled)
    loss_prototypes = torch.from_numpy(km_loss.cluster_centers_).float()  # (8, 64)

    # Build frozen masks: first n_frozen_core = frozen (immutable wisdom), rest = adaptive (EMA-updateable)
    win_frozen_mask = torch.zeros(args.k_per_bank, dtype=torch.bool)
    win_frozen_mask[:args.n_frozen_core] = True
    loss_frozen_mask = torch.zeros(args.k_per_bank, dtype=torch.bool)
    loss_frozen_mask[:args.n_frozen_core] = True

    # Save
    out_path = MODELS / "memory_prototypes_v45.pt"
    torch.save({
        "win_prototypes": win_prototypes,
        "loss_prototypes": loss_prototypes,
        "win_frozen_mask": win_frozen_mask,
        "loss_frozen_mask": loss_frozen_mask,
    }, out_path)

    logger.info("=" * 70)
    logger.info("  MEMORY PROTOTYPES V4.5 COMPLETE!")
    logger.info("  Win prototypes:  %s (frozen=%d, adaptive=%d)", win_prototypes.shape, args.n_frozen_core, args.k_per_bank - args.n_frozen_core)
    logger.info("  Loss prototypes: %s (frozen=%d, adaptive=%d)", loss_prototypes.shape, args.n_frozen_core, args.k_per_bank - args.n_frozen_core)
    logger.info("  Saved → %s", out_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
