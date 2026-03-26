#!/usr/bin/env python3
"""
V4.2: Harvest Memory Prototypes (Dual Memory Banks).
1. Run V4.0 model through all symbols, collect Win/Loss observations
2. Encode through contrastive head → 64-dim embeddings
3. K-Means(8) per group → 8 Win + 8 Loss prototypes
4. Top-4 dense = Frozen Core, Bottom-4 = Adaptive
5. Save as memory_prototypes_v42.pt
"""
import sys, json, numpy as np, torch
from pathlib import Path
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rabit_propfirm_drl"))

import yaml
from models.attention_ppo import AttentionPPO
from environments.prop_env import MultiTFTradingEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = Path(__file__).resolve().parent.parent / "data"
MODELS = Path(__file__).resolve().parent.parent / "models_saved"


def harvest_obs(model, max_win=5000, max_loss=5000):
    """Run model through all symbols, collect Win/Loss obs at entry time."""
    with open(Path(__file__).resolve().parent.parent / "rabit_propfirm_drl" / "configs" / "prop_rules.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["stage1_mode"] = True

    syms = ["XAUUSD", "BTCUSD", "ETHUSD", "US30_cash", "US100_cash"]
    win_obs_list = []
    loss_obs_list = []

    for sym in syms:
        safe = sym.replace(".", "_")
        sd = {}
        for tf in ["M1", "M5", "M15", "H1"]:
            sd[tf] = np.load(DATA / f"{safe}_{tf}_50dim.npy")
        ohlcv = np.load(DATA / f"{safe}_M5_ohlcv.npy")
        env = MultiTFTradingEnv(
            data_m1=sd["M1"], data_m5=sd["M5"], data_m15=sd["M15"], data_h1=sd["H1"],
            config={**cfg, "symbol": sym}, n_features=50, initial_balance=10000,
            episode_length=2000, ohlcv_m5=ohlcv, action_mode="discrete",
        )

        for ep in range(20):
            obs, _ = env.reset(seed=42 + ep)
            entry_obs = {}  # ticket -> obs at entry

            for step in range(2000):
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                obs_t = torch.nan_to_num(obs_t, nan=0.0)
                with torch.no_grad():
                    logits, _, _ = model(obs_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = int(dist.sample().item())

                # Track new positions at entry
                prev_tickets = {p.ticket for p in env.positions}
                obs_next, _, term, trunc, info = env.step(action)
                new_tickets = {p.ticket for p in env.positions} - prev_tickets
                for t in new_tickets:
                    entry_obs[t] = obs.copy()

                # Check closed trades
                for trade in env.trade_history:
                    ticket = trade.get("ticket")
                    if ticket in entry_obs:
                        pnl = trade.get("pnl", 0)
                        e_obs = entry_obs.pop(ticket)
                        if pnl > 0 and len(win_obs_list) < max_win:
                            win_obs_list.append(e_obs)
                        elif pnl < 0 and len(loss_obs_list) < max_loss:
                            loss_obs_list.append(e_obs)

                obs = obs_next
                if term or trunc:
                    break

            # Collect remaining from trade_history
            for trade in env.trade_history:
                ticket = trade.get("ticket")
                if ticket in entry_obs:
                    pnl = trade.get("pnl", 0)
                    e_obs = entry_obs.pop(ticket)
                    if pnl > 0 and len(win_obs_list) < max_win:
                        win_obs_list.append(e_obs)
                    elif pnl < 0 and len(loss_obs_list) < max_loss:
                        loss_obs_list.append(e_obs)

        print(f"  {sym}: Win={len(win_obs_list)}, Loss={len(loss_obs_list)}")
        if len(win_obs_list) >= max_win and len(loss_obs_list) >= max_loss:
            break

    return np.array(win_obs_list[:max_win]), np.array(loss_obs_list[:max_loss])


def encode_to_embeddings(model, obs_array):
    """Encode obs through contrastive head → 64-dim embeddings."""
    embeddings = []
    batch_size = 256
    for i in range(0, len(obs_array), batch_size):
        batch = torch.from_numpy(obs_array[i:i+batch_size]).float().to(device)
        batch = torch.nan_to_num(batch, nan=0.0)
        with torch.no_grad():
            emb = model.get_embedding(batch)  # (B, 128) from contrastive head
            # Use pooled representation instead (64-dim d_model)
            pooled, _ = model._encode(batch)
            embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def cluster_prototypes(embeddings, n_clusters=8):
    """K-Means clustering → prototypes + density ranking."""
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(embeddings)

    # Count samples per cluster (density)
    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    # Sort by density (densest first = most representative = Frozen Core)
    density_order = np.argsort(-cluster_sizes)

    prototypes = km.cluster_centers_[density_order]
    densities = cluster_sizes[density_order]

    return prototypes, densities


def main():
    print("=" * 60)
    print("  V4.2 MEMORY HARVEST")
    print("=" * 60)

    # Load V4.0 model
    ckpt = torch.load(MODELS / "best_v36_stage3.pt", map_location=device, weights_only=False)
    model = AttentionPPO(obs_dim=448, n_actions=4).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)  # V4.0 ckpt doesn't have memory params
    model.eval()
    print(f"Loaded V4.0 S3 (step={ckpt.get('step', 0)})")

    # 1. Harvest observations
    print("\n[1/4] Harvesting Win/Loss observations...")
    win_obs, loss_obs = harvest_obs(model, max_win=5000, max_loss=5000)
    print(f"  Collected: {len(win_obs)} Win, {len(loss_obs)} Loss")

    # 2. Encode to embeddings
    print("\n[2/4] Encoding to 64-dim embeddings...")
    win_emb = encode_to_embeddings(model, win_obs)
    loss_emb = encode_to_embeddings(model, loss_obs)
    print(f"  Win embeddings: {win_emb.shape}, Loss embeddings: {loss_emb.shape}")

    # 3. K-Means clustering
    print("\n[3/4] K-Means clustering (8 prototypes each)...")
    win_proto, win_density = cluster_prototypes(win_emb, n_clusters=8)
    loss_proto, loss_density = cluster_prototypes(loss_emb, n_clusters=8)

    print(f"  Win prototypes: {win_proto.shape}")
    print(f"    Densities: {win_density}")
    print(f"    Frozen Core (top-4): clusters with {win_density[:4]} samples")
    print(f"    Adaptive (bottom-4): clusters with {win_density[4:]} samples")

    print(f"  Loss prototypes: {loss_proto.shape}")
    print(f"    Densities: {loss_density}")
    print(f"    Frozen Core (top-4): clusters with {loss_density[:4]} samples")
    print(f"    Adaptive (bottom-4): clusters with {loss_density[4:]} samples")

    # 4. Save
    print("\n[4/4] Saving memory prototypes...")
    save_path = MODELS / "memory_prototypes_v42.pt"
    torch.save({
        "win_prototypes": torch.from_numpy(win_proto).float(),   # (8, 64)
        "loss_prototypes": torch.from_numpy(loss_proto).float(), # (8, 64)
        "win_frozen_mask": torch.tensor([True]*4 + [False]*4),   # Top-4 frozen
        "loss_frozen_mask": torch.tensor([True]*4 + [False]*4),
        "win_density": win_density,
        "loss_density": loss_density,
        "n_win_obs": len(win_obs),
        "n_loss_obs": len(loss_obs),
    }, save_path)
    print(f"  Saved → {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
