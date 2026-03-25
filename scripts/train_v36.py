#!/usr/bin/env python3
"""
V3.6 PPO Training -- "Tu Van" (Self-Reflection)

Stage 1: PPO + Contrastive Learning (cl_coef=0.05)
  - 4-TF obs: H1(50) + M15(50) + M5(50) + M1×5(250) = 400-dim
  - AttentionPPO: 8 tokens × Self-Attention → Actor/Critic/Contrastive
  - ContrastiveMemory accumulates WIN/LOSS from live trading
  - No Imitation Learning

Usage:
    python scripts/train_v36.py --stage 1 --n-envs 12
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
logger = logging.getLogger("v36_train")
SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD", "US30_cash", "US100_cash"]


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
        self.ptr = 0

    def reset(self):
        self.ptr = 0

    def add(self, obs, actions, rewards, log_probs, values, dones):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr] = values
        self.dones[self.ptr] = dones
        self.ptr += 1

    def compute_gae(self, next_value, gamma=0.99, gae_lambda=0.95):
        last_gae = 0
        for t in reversed(range(self.ptr)):
            next_val = next_value if t == self.ptr - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def flatten(self):
        n = self.ptr * self.n_envs
        return {
            "obs": self.obs[:self.ptr].reshape(n, -1).to(self.device),
            "actions": self.actions[:self.ptr].reshape(n).to(self.device),
            "log_probs": self.log_probs[:self.ptr].reshape(n).to(self.device),
            "advantages": self.advantages[:self.ptr].reshape(n).to(self.device),
            "returns": self.returns[:self.ptr].reshape(n).to(self.device),
        }


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
            sd[tf] = norms[tf].normalize(np.load(DATA_DIR / f"{safe}_{tf}_50dim.npy")).astype(np.float32)
            op = DATA_DIR / f"{safe}_{tf}_ohlcv.npy"
            sd[f"{tf}_ohlcv"] = np.load(op).astype(np.float32) if op.exists() else None
        all_data[sym] = sd
    return all_data


def estimate_regime(data_m15, m5_idx, m5_per_m15=3):
    """Simple regime from M15 slope."""
    m15_idx = min(m5_idx // m5_per_m15, len(data_m15) - 1)
    start = max(0, m15_idx - 4)
    window = data_m15[start:m15_idx + 1]
    if len(window) < 2:
        return "unknown"
    # Use log_return column (27) for slope
    slope = float(np.mean(window[:, min(27, window.shape[1] - 1)]))
    if slope > 0.001:
        return "trending_up"
    elif slope < -0.001:
        return "trending_down"
    return "ranging"


def train_stage1(
    total_steps=750_000, n_envs=12, n_steps=2048, batch_size=512,
    n_epochs=4, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2,
    ent_coef=0.05, vf_coef=0.5, cl_coef=0.05, cl_batch=64,
    max_grad_norm=0.5, device=None, test_mode=False,
):
    if test_mode:
        total_steps, n_envs, n_steps, batch_size = 200, 4, 32, 16

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info("  V3.6 STAGE 1 -- AttentionPPO + Contrastive Learning")
    logger.info("  Device: %s | Steps: %d | Envs: %d", device, total_steps, n_envs)
    logger.info("  cl_coef=%.3f | ent_coef=%.2f | LR=%.1e", cl_coef, ent_coef, lr)
    logger.info("=" * 70)

    all_data = load_data(SYMBOLS)

    import yaml as _yaml
    with open(project_root / "rabit_propfirm_drl" / "configs" / "prop_rules.yaml") as f:
        env_config = _yaml.safe_load(f)
    env_config["stage1_mode"] = True

    import gymnasium
    from environments.prop_env import MultiTFTradingEnv
    from models.attention_ppo import AttentionPPO
    from training_pipeline.contrastive_memory import ContrastiveMemory, contrastive_loss

    def make_env(sym, seed):
        def _init():
            sd = all_data[sym]
            env = MultiTFTradingEnv(
                data_m1=sd["M1"], data_m5=sd["M5"], data_m15=sd["M15"],
                data_h1=sd.get("H1", np.zeros((200, 50), dtype=np.float32)),
                config=env_config, n_features=50, initial_balance=10_000.0,
                episode_length=2000, ohlcv_m5=sd.get("M5_ohlcv"), action_mode="discrete",
            )
            env.reset(seed=seed)
            return env
        return _init

    env_fns = [make_env(SYMBOLS[i % len(SYMBOLS)], seed=100 + i) for i in range(n_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns)
    obs_batch, _ = vec_env.reset()
    obs_dim = obs_batch.shape[1]
    logger.info("AsyncVectorEnv: %d envs, obs=%s", n_envs, obs_batch.shape)

    model = AttentionPPO(obs_dim=obs_dim, n_actions=4).to(device)
    model.token_dropout_enabled = True  # V3.7.1: Token Dropout ON for S1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    buffer = RolloutBuffer(n_steps, n_envs, obs_dim, device)
    memory = ContrastiveMemory(max_per_symbol=500)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("AttentionPPO: %d params (obs=%d, tokens=8×50)", total_params, obs_dim)

    best_ckpt = MODELS_DIR / "best_v36_stage1.pt"
    global_step = 0; start_time = time.time()
    total_buys = total_sells = total_holds = total_closes = 0
    # Track which symbol each env corresponds to
    env_syms = [SYMBOLS[i % len(SYMBOLS)] for i in range(n_envs)]

    logger.info("Starting PPO+Contrastive (%d steps)...", total_steps)

    while global_step < total_steps:
        buffer.reset()
        for step in range(n_steps):
            if global_step >= total_steps:
                break
            obs_t = torch.from_numpy(obs_batch).float()
            obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=10.0, neginf=-10.0)
            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(obs_t.to(device))
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
                       log_prob.cpu(), value.cpu(), torch.from_numpy(dones))

            # Harvest trades for contrastive memory
            for env_i in range(n_envs):
                if dones[env_i] > 0.5:
                    # Episode ended — add all trades from this env to memory
                    # We access trade_history via the info dict
                    pass  # Trades collected at episode boundary below

            obs_batch = next_obs
            global_step += n_envs

        # After rollout: collect trades from any finished episodes
        # Simple approach: run a brief collection pass
        for env_i in range(n_envs):
            try:
                sub_env = vec_env.envs[env_i]
                if hasattr(sub_env, 'trade_history'):
                    sym = env_syms[env_i]
                    for trade in sub_env.trade_history:
                        obs_at_entry = obs_batch[env_i]  # Approximate
                        pnl = trade.get("pnl", 0)
                        action = 0 if trade.get("direction", 1) > 0 else 1
                        regime = "unknown"
                        memory.add_trade(obs_at_entry, action, pnl, sym, regime)
            except Exception:
                pass

        # GAE
        with torch.no_grad():
            nv = model.get_value(torch.nan_to_num(torch.from_numpy(obs_batch).float().to(device), nan=0.0)).cpu()
        buffer.compute_gae(nv, gamma, gae_lambda)

        # PPO + Contrastive UPDATE
        flat = buffer.flatten()
        n_samples = flat["obs"].shape[0]
        adv = flat["advantages"]
        flat["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        avg_pi = avg_vf = avg_ent = avg_cl = 0.0
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

                _, new_lp, entropy, new_val = model.get_action_and_value(mb_obs, mb_act)
                ratio = (new_lp - mb_old_lp).exp()
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                policy_loss = torch.max(pg1, pg2).mean()
                value_loss = F.mse_loss(new_val, mb_ret)
                entropy_loss = entropy.mean()

                # Contrastive loss (only if memory has enough data)
                cl_loss_val = torch.tensor(0.0, device=device)
                if memory.can_sample(min_per_symbol=3):
                    pair = memory.sample_contrastive_pairs(cl_batch, device)
                    if pair is not None:
                        cl_loss_val = contrastive_loss(model, pair[0], pair[1])

                loss = (policy_loss + vf_coef * value_loss
                        - ent_coef * entropy_loss
                        + cl_coef * cl_loss_val)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                avg_pi += policy_loss.item()
                avg_vf += value_loss.item()
                avg_ent += entropy_loss.item()
                avg_cl += cl_loss_val.item()
                n_batches += 1

        if n_batches > 0:
            avg_pi /= n_batches; avg_vf /= n_batches
            avg_ent /= n_batches; avg_cl /= n_batches

        ta = total_buys + total_sells + total_holds + total_closes
        elapsed = time.time() - start_time
        sps = global_step / max(elapsed, 1)
        mem_stats = memory.total_entries()

        logger.info(
            "[S1] Step %d/%d (%.1f%%) | SPS=%.0f | pi=%.4f vf=%.4f ent=%.3f CL=%.4f | "
            "B=%.1f%% S=%.1f%% H=%.1f%% C=%.1f%% | Mem=%d",
            global_step, total_steps, 100*global_step/total_steps, sps,
            avg_pi, avg_vf, avg_ent, avg_cl,
            100*total_buys/max(ta,1), 100*total_sells/max(ta,1),
            100*total_holds/max(ta,1), 100*total_closes/max(ta,1), mem_stats,
        )

        if global_step >= 100_000 and global_step % 100_000 < n_steps * n_envs:
            ckpt = {
                "step": global_step, "stage": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "obs_dim": obs_dim, "n_actions": 4, "model_type": "AttentionPPO",
            }
            torch.save(ckpt, best_ckpt)
            logger.info("  Saved -> %s", best_ckpt.name)

    # Final save
    torch.save({
        "step": global_step, "stage": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "obs_dim": obs_dim, "n_actions": 4, "model_type": "AttentionPPO",
    }, best_ckpt)
    memory.save(MEMORY_DIR)
    vec_env.close()

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("  V3.6 STAGE 1 COMPLETE!")
    logger.info("  Steps: %d in %.1fs (%.0f SPS)", global_step, elapsed, global_step/max(elapsed,1))
    logger.info("  Memory: %d entries", memory.total_entries())
    logger.info("  Checkpoint: %s", best_ckpt)
    logger.info("=" * 70)


def train_stage2(
    total_steps=500_000, n_envs=12, n_steps=2048, batch_size=512,
    n_epochs=4, lr=1e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2,
    ent_coef=0.03, vf_coef=0.5, cl_coef=0.10, cl_batch=64,
    max_grad_norm=0.5, device=None, test_mode=False,
):
    """Stage 2: Warm-start from S1 + load ContrastiveMemory from disk."""
    if test_mode:
        total_steps, n_envs, n_steps, batch_size = 200, 4, 32, 16

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info("  V3.6 STAGE 2 -- PPO + Contrastive Learning (warm-start)")
    logger.info("  Device: %s | Steps: %d | Envs: %d", device, total_steps, n_envs)
    logger.info("  cl_coef=%.3f | ent_coef=%.2f | LR=%.1e", cl_coef, ent_coef, lr)
    logger.info("=" * 70)

    all_data = load_data(SYMBOLS)

    import yaml as _yaml
    with open(project_root / "rabit_propfirm_drl" / "configs" / "prop_rules.yaml") as f:
        env_config = _yaml.safe_load(f)
    env_config["stage1_mode"] = True

    import gymnasium
    from environments.prop_env import MultiTFTradingEnv
    from models.attention_ppo import AttentionPPO
    from training_pipeline.contrastive_memory import ContrastiveMemory, contrastive_loss

    def make_env(sym, seed):
        def _init():
            sd = all_data[sym]
            env = MultiTFTradingEnv(
                data_m1=sd["M1"], data_m5=sd["M5"], data_m15=sd["M15"],
                data_h1=sd.get("H1", np.zeros((200, 50), dtype=np.float32)),
                config=env_config, n_features=50, initial_balance=10_000.0,
                episode_length=2000, ohlcv_m5=sd.get("M5_ohlcv"), action_mode="discrete",
            )
            env.reset(seed=seed)
            return env
        return _init

    env_fns = [make_env(SYMBOLS[i % len(SYMBOLS)], seed=200 + i) for i in range(n_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns)
    obs_batch, _ = vec_env.reset()
    obs_dim = obs_batch.shape[1]
    logger.info("AsyncVectorEnv: %d envs, obs=%s", n_envs, obs_batch.shape)

    # --- WARM START from Stage 1 ---
    s1_ckpt = MODELS_DIR / "best_v36_stage1.pt"
    model = AttentionPPO(obs_dim=obs_dim, n_actions=4).to(device)
    if s1_ckpt.exists():
        ckpt = torch.load(s1_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("WARM START from S1 (step=%d)", ckpt.get("step", 0))
    else:
        logger.warning("No S1 checkpoint found, starting from scratch!")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    model.token_dropout_enabled = False  # V3.7.1: Token Dropout OFF for S2 (CL needs full context)
    buffer = RolloutBuffer(n_steps, n_envs, obs_dim, device)

    # --- LOAD Contrastive Memory from disk ---
    memory = ContrastiveMemory.load(MEMORY_DIR, max_per_symbol=500)
    logger.info("Contrastive Memory: %d entries loaded", memory.total_entries())
    for sym in SYMBOLS:
        s = memory.stats()[sym]
        logger.info("  %-12s | WIN=%4d | LOSS=%4d", sym, s["wins"], s["losses"])

    if not memory.can_sample(min_per_symbol=3):
        logger.error("NOT ENOUGH contrastive pairs! Run harvest_contrastive_v36.py first.")
        vec_env.close()
        sys.exit(1)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("AttentionPPO: %d params", total_params)

    best_ckpt = MODELS_DIR / "best_v36_stage2.pt"
    global_step = 0; start_time = time.time()
    total_buys = total_sells = total_holds = total_closes = 0

    logger.info("Starting PPO+Contrastive S2 (%d steps, cl_coef=%.2f)...", total_steps, cl_coef)

    while global_step < total_steps:
        buffer.reset()
        for step in range(n_steps):
            if global_step >= total_steps:
                break
            obs_t = torch.from_numpy(obs_batch).float()
            obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=10.0, neginf=-10.0)
            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(obs_t.to(device))
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
                       log_prob.cpu(), value.cpu(), torch.from_numpy(dones))
            obs_batch = next_obs
            global_step += n_envs

        # GAE
        with torch.no_grad():
            nv = model.get_value(torch.nan_to_num(torch.from_numpy(obs_batch).float().to(device), nan=0.0)).cpu()
        buffer.compute_gae(nv, gamma, gae_lambda)

        # PPO + Contrastive UPDATE
        flat = buffer.flatten()
        n_samples = flat["obs"].shape[0]
        adv = flat["advantages"]
        flat["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        avg_pi = avg_vf = avg_ent = avg_cl = 0.0
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

                _, new_lp, entropy, new_val = model.get_action_and_value(mb_obs, mb_act)
                ratio = (new_lp - mb_old_lp).exp()
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                policy_loss = torch.max(pg1, pg2).mean()
                value_loss = F.mse_loss(new_val, mb_ret)
                entropy_loss = entropy.mean()

                # Contrastive loss — V3.8: Fake Setup Mining
                pair = memory.sample_fake_setup_pairs(cl_batch, device)
                cl_loss_val = contrastive_loss(model, pair[0], pair[1]) if pair else torch.tensor(0.0, device=device)

                loss = (policy_loss + vf_coef * value_loss
                        - ent_coef * entropy_loss
                        + cl_coef * cl_loss_val)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                avg_pi += policy_loss.item()
                avg_vf += value_loss.item()
                avg_ent += entropy_loss.item()
                avg_cl += cl_loss_val.item()
                n_batches += 1

        if n_batches > 0:
            avg_pi /= n_batches; avg_vf /= n_batches
            avg_ent /= n_batches; avg_cl /= n_batches

        ta = total_buys + total_sells + total_holds + total_closes
        elapsed = time.time() - start_time
        sps = global_step / max(elapsed, 1)

        logger.info(
            "[S2] Step %d/%d (%.1f%%) | SPS=%.0f | pi=%.4f vf=%.4f ent=%.3f CL=%.4f | "
            "B=%.1f%% S=%.1f%% H=%.1f%% C=%.1f%%",
            global_step, total_steps, 100*global_step/total_steps, sps,
            avg_pi, avg_vf, avg_ent, avg_cl,
            100*total_buys/max(ta,1), 100*total_sells/max(ta,1),
            100*total_holds/max(ta,1), 100*total_closes/max(ta,1),
        )

        if global_step >= 100_000 and global_step % 100_000 < n_steps * n_envs:
            torch.save({
                "step": global_step, "stage": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "obs_dim": obs_dim, "n_actions": 4, "model_type": "AttentionPPO",
            }, best_ckpt)
            logger.info("  Saved -> %s", best_ckpt.name)

    # Final save
    torch.save({
        "step": global_step, "stage": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "obs_dim": obs_dim, "n_actions": 4, "model_type": "AttentionPPO",
    }, best_ckpt)
    vec_env.close()

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("  V3.6 STAGE 2 COMPLETE!")
    logger.info("  Steps: %d in %.1fs (%.0f SPS)", global_step, elapsed, global_step/max(elapsed,1))
    logger.info("  CL final: %.4f", avg_cl)
    logger.info("  Checkpoint: %s", best_ckpt)
    logger.info("=" * 70)


def train_stage3(
    total_steps=1_500_000, n_envs=12, n_steps=2048, batch_size=512,
    n_epochs=4, lr=3e-5, gamma=0.99, gae_lambda=0.95, clip_coef=0.2,
    ent_coef=0.005, vf_coef=0.5, cl_coef=0.01, cl_batch=64,
    max_grad_norm=0.5, device=None, test_mode=False,
):
    """Stage 3: Self-Improvement — PPO Fine-tuning from S2 checkpoint.

    Key changes from S2:
      - lr=3e-5 (10x lower: fine-tune, don't destroy)
      - ent_coef=0.005 (reduce indecisiveness)
      - cl_coef=0.01 (tiny anchor to prevent catastrophic forgetting)
      - confidence_mode='relative' (P(act)>2×P(HOLD) instead of hard 70%)
    """
    if test_mode:
        total_steps, n_envs, n_steps, batch_size = 200, 4, 32, 16

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info("  V3.6.1 STAGE 3 -- Self-Improvement (PPO Fine-tune from S2)")
    logger.info("  Device: %s | Steps: %d | Envs: %d", device, total_steps, n_envs)
    logger.info("  cl_coef=%.3f | ent_coef=%.3f | LR=%.1e", cl_coef, ent_coef, lr)
    logger.info("  Confidence Gate: RELATIVE (P(act) > 2×P(HOLD))")
    logger.info("=" * 70)

    all_data = load_data(SYMBOLS)

    import yaml as _yaml
    with open(project_root / "rabit_propfirm_drl" / "configs" / "prop_rules.yaml") as f:
        env_config = _yaml.safe_load(f)
    env_config["stage1_mode"] = True

    import gymnasium
    from environments.prop_env import MultiTFTradingEnv
    from models.attention_ppo import AttentionPPO
    from training_pipeline.contrastive_memory import ContrastiveMemory, contrastive_loss

    def make_env(sym, seed):
        def _init():
            sd = all_data[sym]
            env = MultiTFTradingEnv(
                data_m1=sd["M1"], data_m5=sd["M5"], data_m15=sd["M15"],
                data_h1=sd.get("H1", np.zeros((200, 50), dtype=np.float32)),
                config=env_config, n_features=50, initial_balance=10_000.0,
                episode_length=2000, ohlcv_m5=sd.get("M5_ohlcv"), action_mode="discrete",
            )
            env.reset(seed=seed)
            return env
        return _init

    env_fns = [make_env(SYMBOLS[i % len(SYMBOLS)], seed=300 + i) for i in range(n_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns)
    obs_batch, _ = vec_env.reset()
    obs_dim = obs_batch.shape[1]
    logger.info("AsyncVectorEnv: %d envs, obs=%s", n_envs, obs_batch.shape)

    # --- WARM START from Stage 2 ---
    s2_ckpt = MODELS_DIR / "best_v36_stage2.pt"
    model = AttentionPPO(
        obs_dim=obs_dim, n_actions=4,
        confidence_mode="relative", confidence_ratio=2.0,
    ).to(device)
    if s2_ckpt.exists():
        ckpt = torch.load(s2_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("WARM START from S2 (step=%d)", ckpt.get("step", 0))
    else:
        logger.warning("No S2 checkpoint found! Falling back to S1...")
        s1_ckpt = MODELS_DIR / "best_v36_stage1.pt"
        if s1_ckpt.exists():
            ckpt = torch.load(s1_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info("FALLBACK: loaded S1 (step=%d)", ckpt.get("step", 0))

    # Lower LR for fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    model.token_dropout_enabled = True  # V3.7.1: Token Dropout ON for S3
    buffer = RolloutBuffer(n_steps, n_envs, obs_dim, device)

    # --- LOAD Contrastive Memory (tiny anchor) ---
    memory = ContrastiveMemory.load(MEMORY_DIR, max_per_symbol=500)
    logger.info("Contrastive Memory (anchor): %d entries", memory.total_entries())

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("AttentionPPO: %d params (confidence_mode=%s, ratio=%.1f)",
                total_params, model.confidence_mode, model.confidence_ratio)

    best_ckpt = MODELS_DIR / "best_v36_stage3.pt"
    global_step = 0; start_time = time.time()
    total_buys = total_sells = total_holds = total_closes = 0

    logger.info("Starting S3 PPO Fine-tune (%d steps, cl_coef=%.3f, ent_coef=%.3f, lr=%.1e)...",
                total_steps, cl_coef, ent_coef, lr)

    while global_step < total_steps:
        buffer.reset()
        for step in range(n_steps):
            if global_step >= total_steps:
                break
            obs_t = torch.from_numpy(obs_batch).float()
            obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=10.0, neginf=-10.0)
            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(obs_t.to(device))
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
                       log_prob.cpu(), value.cpu(), torch.from_numpy(dones))
            obs_batch = next_obs
            global_step += n_envs

        # GAE
        with torch.no_grad():
            nv = model.get_value(torch.nan_to_num(torch.from_numpy(obs_batch).float().to(device), nan=0.0)).cpu()
        buffer.compute_gae(nv, gamma, gae_lambda)

        # PPO + tiny Contrastive anchor UPDATE
        flat = buffer.flatten()
        n_samples = flat["obs"].shape[0]
        adv = flat["advantages"]
        flat["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        avg_pi = avg_vf = avg_ent = avg_cl = 0.0
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

                _, new_lp, entropy, new_val = model.get_action_and_value(mb_obs, mb_act)
                ratio = (new_lp - mb_old_lp).exp()
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                policy_loss = torch.max(pg1, pg2).mean()
                value_loss = F.mse_loss(new_val, mb_ret)
                entropy_loss = entropy.mean()

                # Tiny contrastive anchor (prevent catastrophic forgetting)
                cl_loss_val = torch.tensor(0.0, device=device)
                if memory.can_sample(min_per_symbol=3):
                    pair = memory.sample_contrastive_pairs(cl_batch, device)
                    if pair is not None:
                        cl_loss_val = contrastive_loss(model, pair[0], pair[1])

                loss = (policy_loss + vf_coef * value_loss
                        - ent_coef * entropy_loss
                        + cl_coef * cl_loss_val)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                avg_pi += policy_loss.item()
                avg_vf += value_loss.item()
                avg_ent += entropy_loss.item()
                avg_cl += cl_loss_val.item()
                n_batches += 1

        if n_batches > 0:
            avg_pi /= n_batches; avg_vf /= n_batches
            avg_ent /= n_batches; avg_cl /= n_batches

        ta = total_buys + total_sells + total_holds + total_closes
        elapsed = time.time() - start_time
        sps = global_step / max(elapsed, 1)

        logger.info(
            "[S3] Step %d/%d (%.1f%%) | SPS=%.0f | pi=%.4f vf=%.4f ent=%.3f CL=%.4f | "
            "B=%.1f%% S=%.1f%% H=%.1f%% C=%.1f%%",
            global_step, total_steps, 100*global_step/total_steps, sps,
            avg_pi, avg_vf, avg_ent, avg_cl,
            100*total_buys/max(ta,1), 100*total_sells/max(ta,1),
            100*total_holds/max(ta,1), 100*total_closes/max(ta,1),
        )

        if global_step >= 100_000 and global_step % 100_000 < n_steps * n_envs:
            torch.save({
                "step": global_step, "stage": 3,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "obs_dim": obs_dim, "n_actions": 4, "model_type": "AttentionPPO",
                "confidence_mode": "relative", "confidence_ratio": 2.0,
            }, best_ckpt)
            logger.info("  Saved -> %s", best_ckpt.name)

    # Final save
    torch.save({
        "step": global_step, "stage": 3,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "obs_dim": obs_dim, "n_actions": 4, "model_type": "AttentionPPO",
        "confidence_mode": "relative", "confidence_ratio": 2.0,
    }, best_ckpt)
    vec_env.close()

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("  V3.6.1 STAGE 3 COMPLETE! (Self-Improvement)")
    logger.info("  Steps: %d in %.1fs (%.0f SPS)", global_step, elapsed, global_step/max(elapsed,1))
    logger.info("  CL anchor: %.4f | Entropy: %.3f", avg_cl, avg_ent)
    logger.info("  Checkpoint: %s", best_ckpt)
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="V3.6 AttentionPPO Training")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--cl-coef", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9)

    if args.stage == 1:
        train_stage1(
            total_steps=args.total_steps or 750_000,
            n_envs=args.n_envs, device=device, test_mode=args.test,
            cl_coef=args.cl_coef if args.cl_coef is not None else 0.05,
            ent_coef=args.ent_coef if args.ent_coef is not None else 0.05,
        )
    elif args.stage == 2:
        train_stage2(
            total_steps=args.total_steps or 500_000,
            n_envs=args.n_envs, device=device, test_mode=args.test,
            cl_coef=args.cl_coef if args.cl_coef is not None else 0.10,
            ent_coef=args.ent_coef if args.ent_coef is not None else 0.03,
        )
    elif args.stage == 3:
        train_stage3(
            total_steps=args.total_steps or 1_500_000,
            n_envs=args.n_envs, device=device, test_mode=args.test,
            cl_coef=args.cl_coef if args.cl_coef is not None else 0.01,
            ent_coef=args.ent_coef if args.ent_coef is not None else 0.005,
            lr=args.lr if args.lr is not None else 3e-5,
        )
    else:
        logger.error("V3.6 Stage %d not implemented", args.stage)
        sys.exit(1)

if __name__ == "__main__":
    main()

