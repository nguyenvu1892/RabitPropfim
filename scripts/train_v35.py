#!/usr/bin/env python3
"""
V3.5 PPO Training -- "Quan Tri Rui Ro"

Stage 1: Discrete(4) BUY/SELL/HOLD/CLOSE
  - 3-TF obs: M15(50) + M5(50) + M1Ã—5(250) = 350-dim
  - Auto SL from M5 swing points
  - Reward: PnL + CLOSE profit Ã—5
  - No frequency pressure

Usage:
    python scripts/train_v35.py --stage 1 --n-envs 12
    python scripts/train_v35.py --stage 1 --test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("v35_train")

SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD", "US30_cash", "US100_cash"]


# â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
# PPO MODEL -- V3.5 (350-dim obs, 4 actions)
# â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 400, n_actions: int = 4, hidden_dims: list | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)
        self.actor_head = nn.Linear(hidden_dims[-1], n_actions)
        self.critic_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, obs):
        features = self.trunk(obs)
        return self.actor_head(features), self.critic_head(features)

    def get_value(self, obs):
        features = self.trunk(obs)
        return self.critic_head(features).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        features = self.trunk(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


# â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
# ROLLOUT BUFFER
# â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

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
            if t == self.ptr - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]
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


# â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
# DATA LOADING
# â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

def load_data(symbols: list[str]) -> dict:
    from data_engine.normalizer import RunningNormalizer
    norm_path = DATA_DIR / "normalizer_v3.json"
    with open(norm_path, "r") as f:
        norm_data = json.load(f)
    normalizers = {tf: RunningNormalizer.from_state_dict(s) for tf, s in norm_data.items()}

    all_data = {}
    for sym in symbols:
        safe = sym.replace(".", "_")
        sym_data = {}
        for tf in ["M1", "M5", "M15", "H1"]:
            raw = np.load(DATA_DIR / f"{safe}_{tf}_50dim.npy")
            sym_data[tf] = normalizers[tf].normalize(raw).astype(np.float32)
            logger.info("  %s %s: %s", sym, tf, sym_data[tf].shape)
            ohlcv_path = DATA_DIR / f"{safe}_{tf}_ohlcv.npy"
            sym_data[f"{tf}_ohlcv"] = (
                np.load(ohlcv_path).astype(np.float32) if ohlcv_path.exists() else None
            )
        all_data[sym] = sym_data
    return all_data


# â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
# STAGE 1: PURE PPO (Discrete(4))
# â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

def train_stage1(
    total_steps: int = 750_000,
    n_envs: int = 12,
    n_steps: int = 2048,
    batch_size: int = 512,
    n_epochs: int = 4,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.05,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    device: torch.device = None,
    test_mode: bool = False,
):
    if test_mode:
        total_steps = 200
        n_envs = 4
        n_steps = 32
        batch_size = 16

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info("  V3.5 STAGE 1 -- PPO Discrete(4) BUY/SELL/HOLD/CLOSE")
    logger.info("  Device: %s | Steps: %d | Envs: %d", device, total_steps, n_envs)
    logger.info("  LR: %.1e | Batch: %d | Ent: %.2f | Clip: %.2f", lr, batch_size, ent_coef, clip_coef)
    logger.info("=" * 70)

    # Load data
    logger.info("Loading data...")
    all_data = load_data(SYMBOLS)

    import yaml as _yaml
    config_path = project_root / "rabit_propfirm_drl" / "configs" / "prop_rules.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        env_config = _yaml.safe_load(f)
    env_config["stage1_mode"] = True

    import gymnasium
    from environments.prop_env import MultiTFTradingEnv

    def make_env(sym: str, seed: int):
        def _init():
            sd = all_data[sym]
            env = MultiTFTradingEnv(
                data_m1=sd["M1"], data_m5=sd["M5"],
                data_m15=sd["M15"],
                data_h1=sd.get("H1", np.zeros((200, 50), dtype=np.float32)),
                config=env_config, n_features=50, initial_balance=10_000.0,
                episode_length=2000, ohlcv_m5=sd.get("M5_ohlcv"),
                action_mode="discrete",
            )
            env.reset(seed=seed)
            return env
        return _init

    env_fns = [make_env(SYMBOLS[i % len(SYMBOLS)], seed=100 + i) for i in range(n_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns)
    obs_batch, _ = vec_env.reset()

    logger.info("AsyncVectorEnv: %d envs, obs=%s", n_envs, obs_batch.shape)

    obs_dim = obs_batch.shape[1]
    model = PPOActorCritic(obs_dim=obs_dim, n_actions=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    buffer = RolloutBuffer(n_steps, n_envs, obs_dim, device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d params (obs=%d, actions=4)", total_params, obs_dim)

    best_checkpoint = MODELS_DIR / "best_v35_stage1.pt"
    ckpt_dir = MODELS_DIR / "v35_stage1"
    ckpt_dir.mkdir(exist_ok=True)

    global_step = 0
    start_time = time.time()
    total_buys = 0
    total_sells = 0
    total_holds = 0
    total_closes = 0

    logger.info("Starting PPO loop (%d total steps)...", total_steps)

    while global_step < total_steps:
        # PHASE 1: COLLECT ROLLOUT
        buffer.reset()
        for step in range(n_steps):
            if global_step >= total_steps:
                break

            obs_t = torch.from_numpy(obs_batch).float()
            obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=10.0, neginf=-10.0)

            with torch.no_grad():
                obs_gpu = obs_t.to(device)
                action, log_prob, _, value = model.get_action_and_value(obs_gpu)

            actions_np = action.cpu().numpy()
            for a in actions_np:
                if a == 0: total_buys += 1
                elif a == 1: total_sells += 1
                elif a == 2: total_holds += 1
                else: total_closes += 1

            next_obs, rewards, terminateds, truncateds, infos = vec_env.step(actions_np)
            dones = np.logical_or(terminateds, truncateds).astype(np.float32)

            buffer.add(obs_t, action.cpu(),
                       torch.from_numpy(rewards.astype(np.float32)),
                       log_prob.cpu(), value.cpu(), torch.from_numpy(dones))

            obs_batch = next_obs
            global_step += n_envs

        # GAE
        with torch.no_grad():
            next_obs_t = torch.from_numpy(obs_batch).float().to(device)
            next_obs_t = torch.nan_to_num(next_obs_t, nan=0.0)
            next_value = model.get_value(next_obs_t).cpu()
        buffer.compute_gae(next_value, gamma, gae_lambda)

        # PHASE 2: PPO UPDATE
        flat = buffer.flatten()
        n_samples = flat["obs"].shape[0]
        adv = flat["advantages"]
        flat["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        avg_pi = 0.0; avg_vf = 0.0; avg_ent = 0.0; avg_clip = 0.0
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

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                avg_pi += policy_loss.item()
                avg_vf += value_loss.item()
                avg_ent += entropy_loss.item()
                avg_clip += clip_frac
                n_batches += 1

        if n_batches > 0:
            avg_pi /= n_batches; avg_vf /= n_batches
            avg_ent /= n_batches; avg_clip /= n_batches

        total_actions = total_buys + total_sells + total_holds + total_closes
        pct_buy = 100 * total_buys / max(total_actions, 1)
        pct_sell = 100 * total_sells / max(total_actions, 1)
        pct_hold = 100 * total_holds / max(total_actions, 1)
        pct_close = 100 * total_closes / max(total_actions, 1)
        elapsed = time.time() - start_time
        sps = global_step / max(elapsed, 1)

        logger.info(
            "[S1] Step %d/%d (%.1f%%) | SPS=%.0f | pi=%.4f vf=%.4f ent=%.3f clip=%.2f | "
            "BUY=%.1f%% SELL=%.1f%% HOLD=%.1f%% CLOSE=%.1f%%",
            global_step, total_steps, 100 * global_step / total_steps, sps,
            avg_pi, avg_vf, avg_ent, avg_clip,
            pct_buy, pct_sell, pct_hold, pct_close,
        )

        # Checkpoint every 100K steps
        if global_step >= 100_000 and global_step % 100_000 < n_steps * n_envs:
            ckpt = {
                "step": global_step, "stage": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "obs_dim": obs_dim, "n_actions": 4,
                "stats": {
                    "pct_buy": pct_buy, "pct_sell": pct_sell,
                    "pct_hold": pct_hold, "pct_close": pct_close,
                },
            }
            torch.save(ckpt, ckpt_dir / f"checkpoint_{global_step}.pt")
            torch.save(ckpt, best_checkpoint)
            logger.info("  Saved -> %s", best_checkpoint.name)

    # Final save
    final = {
        "step": global_step, "stage": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "obs_dim": obs_dim, "n_actions": 4,
        "stats": {
            "pct_buy": pct_buy, "pct_sell": pct_sell,
            "pct_hold": pct_hold, "pct_close": pct_close,
        },
    }
    torch.save(final, best_checkpoint)
    vec_env.close()

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("  V3.5 STAGE 1 COMPLETE!")
    logger.info("  Steps: %d in %.1fs (%.0f SPS)", global_step, elapsed, global_step / max(elapsed, 1))
    logger.info("  BUY=%.1f%% SELL=%.1f%% HOLD=%.1f%% CLOSE=%.1f%%", pct_buy, pct_sell, pct_hold, pct_close)
    logger.info("  Checkpoint: %s", best_checkpoint)
    logger.info("=" * 70)
    return best_checkpoint


# â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
# STAGE 2: PPO + SELF-IMITATION LEARNING (VIP)
# â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

def train_stage2(
    total_steps: int = 500_000,
    n_envs: int = 12,
    n_steps: int = 2048,
    batch_size: int = 512,
    n_epochs: int = 4,
    lr: float = 1e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.05,
    vf_coef: float = 0.5,
    il_coef: float = 0.15,
    max_grad_norm: float = 0.5,
    device: torch.device = None,
    test_mode: bool = False,
):
    """Stage 2: PPO + Imitation Learning from VIP buffer."""
    if test_mode:
        total_steps = 200
        n_envs = 4
        n_steps = 32
        batch_size = 16

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VIP buffer
    vip_dir = MODELS_DIR / "vip_buffer_v35"
    vip_obs = torch.from_numpy(np.load(vip_dir / "vip_obs.npy")).float().to(device)
    vip_actions = torch.from_numpy(np.load(vip_dir / "vip_actions.npy")).long().to(device)
    n_vip = len(vip_obs)

    logger.info("=" * 70)
    logger.info("  V3.5 STAGE 2 -- PPO + IL (Self-Imitation Learning)")
    logger.info("  VIP buffer: %d expert demos", n_vip)
    logger.info("  il_coef=%.2f | ent_coef=%.2f | LR=%.1e", il_coef, ent_coef, lr)
    logger.info("=" * 70)

    # Load data & create envs
    all_data = load_data(SYMBOLS)

    import yaml as _yaml
    config_path = project_root / "rabit_propfirm_drl" / "configs" / "prop_rules.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        env_config = _yaml.safe_load(f)
    env_config["stage1_mode"] = True

    import gymnasium
    from environments.prop_env import MultiTFTradingEnv

    def make_env(sym: str, seed: int):
        def _init():
            sd = all_data[sym]
            env = MultiTFTradingEnv(
                data_m1=sd["M1"], data_m5=sd["M5"], data_m15=sd["M15"],
                data_h1=sd.get("H1", np.zeros((200, 50), dtype=np.float32)),
                config=env_config, n_features=50, initial_balance=10_000.0,
                episode_length=2000, ohlcv_m5=sd.get("M5_ohlcv"),
                action_mode="discrete",
            )
            env.reset(seed=seed)
            return env
        return _init

    env_fns = [make_env(SYMBOLS[i % len(SYMBOLS)], seed=200 + i) for i in range(n_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns)
    obs_batch, _ = vec_env.reset()
    obs_dim = obs_batch.shape[1]

    logger.info("AsyncVectorEnv: %d envs, obs=%s", n_envs, obs_batch.shape)

    # Warm-start from Stage 1
    model = PPOActorCritic(obs_dim=obs_dim, n_actions=4).to(device)
    stage1_path = MODELS_DIR / "best_v35_stage1.pt"
    s1_ckpt = torch.load(stage1_path, map_location=device, weights_only=False)
    model.load_state_dict(s1_ckpt["model_state_dict"])
    s1_step = s1_ckpt.get("step", 0)
    logger.info("Warm-started from Stage 1 (step=%d)", s1_step)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    buffer = RolloutBuffer(n_steps, n_envs, obs_dim, device)
    best_checkpoint = MODELS_DIR / "best_v35_stage2.pt"

    global_step = 0
    start_time = time.time()
    total_buys = total_sells = total_holds = total_closes = 0

    logger.info("Starting Stage 2 PPO+IL loop (%d steps, VIP=%d)...", total_steps, n_vip)

    while global_step < total_steps:
        # PHASE 1: COLLECT ROLLOUT
        buffer.reset()
        for step in range(n_steps):
            if global_step >= total_steps:
                break
            obs_t = torch.from_numpy(obs_batch).float()
            obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=10.0, neginf=-10.0)

            with torch.no_grad():
                obs_gpu = obs_t.to(device)
                action, log_prob, _, value = model.get_action_and_value(obs_gpu)

            actions_np = action.cpu().numpy()
            for a in actions_np:
                if a == 0: total_buys += 1
                elif a == 1: total_sells += 1
                elif a == 2: total_holds += 1
                else: total_closes += 1

            next_obs, rewards, terminateds, truncateds, infos = vec_env.step(actions_np)
            dones = np.logical_or(terminateds, truncateds).astype(np.float32)
            buffer.add(obs_t, action.cpu(),
                       torch.from_numpy(rewards.astype(np.float32)),
                       log_prob.cpu(), value.cpu(), torch.from_numpy(dones))
            obs_batch = next_obs
            global_step += n_envs

        # GAE
        with torch.no_grad():
            next_obs_t = torch.from_numpy(obs_batch).float().to(device)
            next_obs_t = torch.nan_to_num(next_obs_t, nan=0.0)
            next_value = model.get_value(next_obs_t).cpu()
        buffer.compute_gae(next_value, gamma, gae_lambda)

        # PHASE 2: PPO + IL UPDATE
        flat = buffer.flatten()
        n_samples = flat["obs"].shape[0]
        adv = flat["advantages"]
        flat["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        avg_pi = avg_vf = avg_ent = avg_il = 0.0
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

                # Imitation Learning: cross-entropy on VIP batch
                vip_idx = torch.randint(0, n_vip, (min(batch_size, n_vip),), device=device)
                vip_o = vip_obs[vip_idx]
                vip_a = vip_actions[vip_idx]
                logits_vip, _ = model(vip_o)
                il_loss = F.cross_entropy(logits_vip, vip_a)

                loss = (
                    policy_loss
                    + vf_coef * value_loss
                    - ent_coef * entropy_loss
                    + il_coef * il_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                avg_pi += policy_loss.item()
                avg_vf += value_loss.item()
                avg_ent += entropy_loss.item()
                avg_il += il_loss.item()
                n_batches += 1

        if n_batches > 0:
            avg_pi /= n_batches; avg_vf /= n_batches
            avg_ent /= n_batches; avg_il /= n_batches

        total_actions = total_buys + total_sells + total_holds + total_closes
        pct_buy = 100 * total_buys / max(total_actions, 1)
        pct_sell = 100 * total_sells / max(total_actions, 1)
        pct_hold = 100 * total_holds / max(total_actions, 1)
        pct_close = 100 * total_closes / max(total_actions, 1)
        elapsed = time.time() - start_time
        sps = global_step / max(elapsed, 1)

        logger.info(
            "[S2] Step %d/%d (%.1f%%) | SPS=%.0f | pi=%.4f vf=%.4f ent=%.3f IL=%.4f | "
            "BUY=%.1f%% SELL=%.1f%% HOLD=%.1f%% CLOSE=%.1f%%",
            global_step, total_steps, 100 * global_step / total_steps, sps,
            avg_pi, avg_vf, avg_ent, avg_il,
            pct_buy, pct_sell, pct_hold, pct_close,
        )

        # Save periodically
        if global_step % 100_000 < n_steps * n_envs:
            ckpt = {
                "step": s1_step + global_step, "stage": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "obs_dim": obs_dim, "n_actions": 4,
            }
            torch.save(ckpt, best_checkpoint)
            logger.info("  Saved -> %s", best_checkpoint.name)

    # Final
    final = {
        "step": s1_step + global_step, "stage": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "obs_dim": obs_dim, "n_actions": 4,
    }
    torch.save(final, best_checkpoint)
    vec_env.close()

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("  V3.5 STAGE 2 COMPLETE!")
    logger.info("  Steps: %d in %.1fs (%.0f SPS)", global_step, elapsed, global_step / max(elapsed, 1))
    logger.info("  BUY=%.1f%% SELL=%.1f%% HOLD=%.1f%% CLOSE=%.1f%%", pct_buy, pct_sell, pct_hold, pct_close)
    logger.info("  Checkpoint: %s", best_checkpoint)
    logger.info("=" * 70)
    return best_checkpoint


# â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
# MAIN
# â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

def main():
    parser = argparse.ArgumentParser(description="V3.5 PPO Training")
    parser.add_argument("--stage", type=int, default=1, help="Stage (1 or 2)")
    parser.add_argument("--test", action="store_true", help="Quick test")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--il-coef", type=float, default=0.15)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0),
                     torch.cuda.get_device_properties(0).total_memory / 1e9)

    if args.stage == 1:
        train_stage1(
            total_steps=args.total_steps or 750_000,
            n_envs=args.n_envs, device=device, test_mode=args.test,
        )
    elif args.stage == 2:
        train_stage2(
            total_steps=args.total_steps or 500_000,
            n_envs=args.n_envs, device=device, test_mode=args.test,
            il_coef=args.il_coef, ent_coef=args.ent_coef,
        )
    else:
        logger.error("V3.5 supports --stage 1 or 2")
        sys.exit(1)


if __name__ == "__main__":
    main()


