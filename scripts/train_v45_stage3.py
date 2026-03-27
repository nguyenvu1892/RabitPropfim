#!/usr/bin/env python3
"""
V4.5 Stage 3 — "Tích Lũy Tuyệt Đối" (Absolute Accumulation)

Key changes from V4.4 Stage 3:
  1. Memory Injection: Loads memory_prototypes_v45.pt (Master Vault KMeans) 
     instead of random init. Bot starts with IMMORTAL MEMORIES.
  2. Warm-start + Freeze: Loads best V4.4 weights, FREEZES all Base Price 
     Action layers (token_proj, macro/micro encoder). Only Cross-Attention, 
     R:R Head, Actor, Critic, Contrastive, and Memory Banks are trainable.
  3. Unified Battlefield: Trains on ALL 5 symbols simultaneously. 
     TradFi (464-dim) is zero-padded to 488-dim to match Crypto.

Usage:
    python scripts/train_v45_stage3.py --total-steps 6500000 --n-envs 12
"""
from __future__ import annotations
import argparse, json, logging, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "rabit_propfirm_drl"))

DATA_DIR = project_root / "data"
MODELS_DIR = project_root / "models_saved"
MODELS_DIR.mkdir(exist_ok=True)
MEMORY_DIR = MODELS_DIR / "contrastive_memory_v36"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("v45_train")

ALL_SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD", "US30_cash", "US100_cash"]
CRYPTO_SYMBOLS = {"BTCUSD", "ETHUSD"}
TARGET_OBS_DIM = 488  # Unified dimension (Crypto native, TradFi zero-padded)


class RolloutBuffer:
    def __init__(self, n_steps, n_envs, obs_dim, device):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device
        self.obs = torch.zeros((n_steps, n_envs, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((n_steps, n_envs), dtype=torch.long)
        self.rewards = torch.zeros((n_steps, n_envs), dtype=torch.float32)
        self.log_probs = torch.zeros((n_steps, n_envs), dtype=torch.float32)
        self.values = torch.zeros((n_steps, n_envs), dtype=torch.float32)
        self.dones = torch.zeros((n_steps, n_envs), dtype=torch.float32)
        self.advantages = torch.zeros((n_steps, n_envs), dtype=torch.float32)
        self.returns = torch.zeros((n_steps, n_envs), dtype=torch.float32)
        self.rr_preds = torch.zeros((n_steps, n_envs), dtype=torch.float32)
        self.undiscounted_returns = torch.zeros((n_steps, n_envs), dtype=torch.float32)
        self.ptr = 0

    def reset(self):
        self.ptr = 0

    def add(self, obs, actions, rewards, log_probs, values, rr_preds, dones):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr] = values
        self.rr_preds[self.ptr] = rr_preds
        self.dones[self.ptr] = dones
        self.ptr += 1

    def compute_gae(self, next_value, gamma=0.99, gae_lambda=0.95):
        last_gae = 0
        last_ret = 0
        for t in reversed(range(self.ptr)):
            next_val = next_value if t == self.ptr - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
            last_ret = self.rewards[t] + last_ret * next_non_terminal
            self.undiscounted_returns[t] = last_ret
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def flatten(self):
        n = self.ptr * self.n_envs
        return {
            "obs": self.obs[:self.ptr].reshape(n, -1).to(self.device),
            "actions": self.actions[:self.ptr].reshape(n).to(self.device),
            "log_probs": self.log_probs[:self.ptr].reshape(n).to(self.device),
            "advantages": self.advantages[:self.ptr].reshape(n).to(self.device),
            "returns": self.returns[:self.ptr].reshape(n).to(self.device),
            "rr_preds": self.rr_preds[:self.ptr].reshape(n).to(self.device),
            "undiscounted_returns": self.undiscounted_returns[:self.ptr].reshape(n).to(self.device),
        }


class ZeroPadWrapper:
    """Wrapper that zero-pads TradFi observations (464-dim) to 488-dim."""
    def __init__(self, env, target_dim=488):
        self.env = env
        self.target_dim = target_dim
        self._obs_dim = None
        # Override observation_space so AsyncVectorEnv sees uniform spaces
        import gymnasium
        low = np.full(target_dim, -np.inf, dtype=np.float32)
        high = np.full(target_dim, np.inf, dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = env.action_space

    def _pad(self, obs):
        if obs.shape[-1] < self.target_dim:
            pad_width = self.target_dim - obs.shape[-1]
            obs = np.pad(obs, (0, pad_width), mode="constant", constant_values=0.0)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_dim = obs.shape[-1]
        return self._pad(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self._pad(obs), reward, term, trunc, info

    def __getattr__(self, name):
        return getattr(self.env, name)


def load_data(symbols):
    from data_engine.normalizer import RunningNormalizer
    with open(DATA_DIR / "normalizer_v3.json") as f:
        nd = json.load(f)
    norms = {k: RunningNormalizer.from_state_dict(v) for k, v in nd.items()}
    all_data = {}
    for sym in symbols:
        safe = sym.replace(".", "_")
        sd = {}
        for tf in ["M1", "M5", "M15", "H1"]:
            raw = np.load(DATA_DIR / f"{safe}_{tf}_50dim.npy").astype(np.float32)
            norm_dim = norms[tf]._mean.shape[0]  # normalizer dimension (50)
            # Always use first 50 cols for the env (standard features)
            sd[tf] = norms[tf].normalize(raw[:, :norm_dim]).astype(np.float32)
            op = DATA_DIR / f"{safe}_{tf}_ohlcv.npy"
            sd[f"{tf}_ohlcv"] = np.load(op).astype(np.float32) if op.exists() else None

        # Crypto has extra Futures features via separate file
        if sym in CRYPTO_SYMBOLS:
            fp = DATA_DIR / f"{safe}_M5_futures.npy"
            if fp.exists():
                sd["M5_futures"] = np.load(fp).astype(np.float32)

        all_data[sym] = sd
    return all_data


def train_v45_stage3(
    total_steps=6_500_000, n_envs=12, n_steps=2048, batch_size=512,
    n_epochs=4, lr=3e-5, gamma=0.99, gae_lambda=0.95, clip_coef=0.2,
    ent_coef=0.005, vf_coef=0.5, cl_coef=0.01, cl_batch=64,
    max_grad_norm=0.5, device=None, test_mode=False,
):
    if test_mode:
        total_steps, n_envs, n_steps, batch_size = 200, 5, 32, 16

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info("  V4.5 STAGE 3 — Tích Lũy Tuyệt Đối (Absolute Accumulation)")
    logger.info("  Device: %s | Steps: %s | Envs: %d", device, f"{total_steps:,}", n_envs)
    logger.info("  Unified obs_dim: %d (TradFi zero-padded)", TARGET_OBS_DIM)
    logger.info("  Symbols: %s", ALL_SYMBOLS)
    logger.info("  lr=%.1e | ent_coef=%.3f | cl_coef=%.3f", lr, ent_coef, cl_coef)
    logger.info("=" * 70)

    all_data = load_data(ALL_SYMBOLS)

    import yaml as _yaml
    with open(project_root / "rabit_propfirm_drl" / "configs" / "prop_rules.yaml", encoding="utf-8") as f:
        env_config = _yaml.safe_load(f)
    env_config["stage1_mode"] = True

    import gymnasium
    from environments.prop_env import MultiTFTradingEnv
    from models.attention_ppo import AttentionPPO
    from training_pipeline.contrastive_memory import ContrastiveMemory, contrastive_loss

    def make_env(sym, seed):
        def _init():
            sd = all_data[sym]
            is_crypto = sym in CRYPTO_SYMBOLS
            env = MultiTFTradingEnv(
                data_m1=sd["M1"], data_m5=sd["M5"], data_m15=sd["M15"],
                data_h1=sd.get("H1", np.zeros((200, 50), dtype=np.float32)),
                config=env_config, n_features=50, initial_balance=10_000.0,
                episode_length=2000, ohlcv_m5=sd.get("M5_ohlcv"),
                futures_m5=sd.get("M5_futures") if is_crypto else None,
                action_mode="discrete",
            )
            # Wrap ALL envs with zero-padding to guarantee uniform 488-dim
            return ZeroPadWrapper(env, target_dim=TARGET_OBS_DIM)
        return _init

    env_fns = [make_env(ALL_SYMBOLS[i % len(ALL_SYMBOLS)], seed=500 + i) for i in range(n_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns)
    obs_batch, _ = vec_env.reset()
    obs_dim = obs_batch.shape[1]
    assert obs_dim == TARGET_OBS_DIM, f"Expected obs_dim={TARGET_OBS_DIM}, got {obs_dim}"
    logger.info("AsyncVectorEnv: %d envs, obs=%s", n_envs, obs_batch.shape)

    # === BUILD MODEL ===
    model = AttentionPPO(
        obs_dim=TARGET_OBS_DIM, n_actions=4,
        confidence_mode="relative", confidence_ratio=2.0,
    ).to(device)

    # === WARM-START from best V4.4 checkpoint ===
    warm_ckpt = None
    for candidate in ["best_v43_stage3_A.pt", "best_v43_stage2_A.pt", "best_v43_stage1_A.pt"]:
        p = MODELS_DIR / candidate
        if p.exists():
            warm_ckpt = p
            break

    if warm_ckpt is not None:
        v44_step = model.warm_start_from_v42(str(warm_ckpt), device=device)
        logger.info("V4.5: Warm-started from %s (step=%d)", warm_ckpt.name, v44_step)
    else:
        logger.warning("No V4.4 checkpoint found! Starting from scratch.")

    # === FREEZE Base Layers (Principle 2) ===
    model.freeze_price_layers()

    # === INJECT Master Vault Memory (Principle 1) ===
    mem_proto = MODELS_DIR / "memory_prototypes_v45.pt"
    if mem_proto.exists():
        model.load_memory_prototypes(str(mem_proto))
        logger.info("V4.5: Master Vault Memory Prototypes INJECTED!")
    else:
        logger.warning("memory_prototypes_v45.pt not found! Using random init for memory banks.")

    model.token_dropout_enabled = True
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, eps=1e-5)
    buffer = RolloutBuffer(n_steps, n_envs, obs_dim, device)

    # === LOAD Contrastive Memory (anchor) ===
    memory = ContrastiveMemory.load(MEMORY_DIR)
    logger.info("Master Vault Memory (anchor): %d entries", memory.total_entries())

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("AttentionPPO: %d params (%d trainable, %d frozen)", total_params, trainable, total_params - trainable)

    best_ckpt = MODELS_DIR / "best_v45_stage3.pt"
    global_step = 0
    start_time = time.time()
    total_buys = total_sells = total_holds = total_closes = 0

    logger.info("Starting V4.5 Stage 3 (%s steps, all symbols)...", f"{total_steps:,}")

    while global_step < total_steps:
        buffer.reset()
        for step in range(n_steps):
            if global_step >= total_steps:
                break
            obs_t = torch.from_numpy(obs_batch).float()
            obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=10.0, neginf=-10.0)
            with torch.no_grad():
                action, log_prob, _, value, rr_pred = model.get_action_and_value(obs_t.to(device))
            actions_np = action.cpu().numpy()
            for a in actions_np:
                if a == 0: total_buys += 1
                elif a == 1: total_sells += 1
                elif a == 2: total_holds += 1
                else: total_closes += 1

            next_obs, rewards, terms, truncs, infos = vec_env.step(actions_np)
            dones = np.logical_or(terms, truncs).astype(np.float32)
            buffer.add(obs_t, action.cpu(),
                       torch.from_numpy(rewards.astype(np.float32)),
                       log_prob.cpu(), value.cpu(), rr_pred.cpu(), torch.from_numpy(dones))
            obs_batch = next_obs
            global_step += n_envs

        # GAE
        with torch.no_grad():
            nv = model.get_value(torch.nan_to_num(torch.from_numpy(obs_batch).float().to(device), nan=0.0)).cpu()
        buffer.compute_gae(nv, gamma, gae_lambda)

        # PPO + R:R + Contrastive UPDATE
        flat = buffer.flatten()
        n_samples = flat["obs"].shape[0]
        adv = flat["advantages"]
        flat["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        avg_pi = avg_vf = avg_ent = avg_cl = avg_rr = 0.0
        n_batches = 0

        for epoch in range(n_epochs):
            indices = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                mb_idx = indices[start:end]
                mb_obs = flat["obs"][mb_idx]
                mb_act = flat["actions"][mb_idx]
                mb_old_lp = flat["log_probs"][mb_idx]
                mb_adv = flat["advantages"][mb_idx]
                mb_ret = flat["returns"][mb_idx]
                mb_undiscounted_ret = flat["undiscounted_returns"][mb_idx]

                _, new_lp, entropy, new_val, new_rr_pred = model.get_action_and_value(mb_obs, mb_act)
                ratio = (new_lp - mb_old_lp).exp()
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                policy_loss = torch.max(pg1, pg2).mean()
                value_loss = F.mse_loss(new_val, mb_ret)
                entropy_loss = entropy.mean()

                # R:R Auxiliary Loss
                rr_loss = F.mse_loss(new_rr_pred, mb_undiscounted_ret)

                # Contrastive anchor
                cl_loss_val = torch.tensor(0.0, device=device)
                if memory is not None and memory.can_sample(min_per_symbol=3):
                    pair = memory.sample_contrastive_pairs(cl_batch, device)
                    if pair is not None:
                        cl_loss_val = contrastive_loss(model, pair[0], pair[1])

                loss = (policy_loss + vf_coef * value_loss
                        - ent_coef * entropy_loss
                        + 0.5 * rr_loss
                        + cl_coef * cl_loss_val)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                model.freeze_core_memory_grads()
                optimizer.step()

                avg_pi += policy_loss.item()
                avg_vf += value_loss.item()
                avg_ent += entropy_loss.item()
                avg_cl += cl_loss_val.item()
                avg_rr += rr_loss.item()
                n_batches += 1

        if n_batches > 0:
            avg_pi /= n_batches; avg_vf /= n_batches
            avg_ent /= n_batches; avg_cl /= n_batches; avg_rr /= n_batches

        ta = total_buys + total_sells + total_holds + total_closes
        elapsed = time.time() - start_time
        sps = global_step / max(elapsed, 1)

        logger.info(
            "[V4.5-S3] Step %d/%d (%.1f%%) | SPS=%.0f | pi=%.4f vf=%.4f ent=%.3f CL=%.4f RR=%.4f | "
            "B=%.1f%% S=%.1f%% H=%.1f%% C=%.1f%%",
            global_step, total_steps, 100*global_step/total_steps, sps,
            avg_pi, avg_vf, avg_ent, avg_cl, avg_rr,
            100*total_buys/max(ta,1), 100*total_sells/max(ta,1),
            100*total_holds/max(ta,1), 100*total_closes/max(ta,1),
        )

        # Memory attention analysis (every 500k steps)
        if global_step % 500_000 < n_steps * n_envs:
            mem_attn = model.get_memory_attention()
            if mem_attn:
                logger.info("  Memory Attention: Macro=%.1f%% Win=%.1f%% Loss=%.1f%%",
                           100*mem_attn["macro_attn"], 100*mem_attn["win_memory_attn"], 100*mem_attn["loss_memory_attn"])

        if global_step >= 100_000 and global_step % 100_000 < n_steps * n_envs:
            torch.save({
                "step": global_step, "stage": "v45_s3",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "obs_dim": obs_dim, "n_actions": 4, "model_type": "AttentionPPO",
                "confidence_mode": "relative", "confidence_ratio": 2.0,
                "version": "V4.5",
            }, best_ckpt)
            logger.info("  Saved → %s", best_ckpt.name)

    # Final save
    torch.save({
        "step": global_step, "stage": "v45_s3",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "obs_dim": obs_dim, "n_actions": 4, "model_type": "AttentionPPO",
        "confidence_mode": "relative", "confidence_ratio": 2.0,
        "version": "V4.5",
    }, best_ckpt)
    vec_env.close()

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("  V4.5 STAGE 3 COMPLETE! (Absolute Accumulation)")
    logger.info("  Steps: %s in %.1fs (%.0f SPS)", f"{global_step:,}", elapsed, global_step/max(elapsed,1))
    logger.info("  CL anchor: %.4f | R:R: %.4f | Entropy: %.3f", avg_cl, avg_rr, avg_ent)
    logger.info("  Checkpoint: %s", best_ckpt)
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="V4.5 Stage 3 Training (All Symbols)")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--total-steps", type=int, default=6_500_000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9)

    train_v45_stage3(
        total_steps=args.total_steps,
        n_envs=args.n_envs,
        device=device,
        test_mode=args.test,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
