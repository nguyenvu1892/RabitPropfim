"""
V3.3 Stage 1 Training — PPO with Discrete Actions.

Clean from-scratch training loop for "Ruong Vang" architecture:
- Discrete(3): BUY=0, SELL=1, HOLD=2
- MLP policy (not Transformer — simpler, faster for discrete)
- Flat obs: 300-dim (M5 + M1 frame stacking with ATR normalization)
- PPO with GAE advantage estimation
- n_envs=12, n_steps=2048

Usage:
    python scripts/train_v33.py --stage 1
    python scripts/train_v33.py --test
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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
logger = logging.getLogger("v33_train")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PPO POLICY NETWORK (Simple MLP for Discrete Actions)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PPOActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO with discrete action space.
    
    Architecture:
        Shared trunk: 300 -> 512 -> 256 -> 128
        Actor head:   128 -> 3 (logits for BUY/SELL/HOLD)
        Critic head:  128 -> 1 (state value)
    """
    def __init__(self, obs_dim: int = 300, n_actions: int = 3, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        # Shared trunk
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)
        
        # Actor head (policy logits)
        self.actor_head = nn.Linear(hidden_dims[-1], n_actions)
        
        # Critic head (state value)
        self.critic_head = nn.Linear(hidden_dims[-1], 1)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Policy head: small init for exploration
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        # Value head: standard init
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
    
    def forward(self, obs: torch.Tensor):
        """Forward pass returning policy logits and value."""
        features = self.trunk(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value
    
    def get_action_and_value(self, obs: torch.Tensor, action=None):
        """Get action, log_prob, entropy, and value."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)
    
    def get_value(self, obs: torch.Tensor):
        features = self.trunk(obs)
        return self.critic_head(features).squeeze(-1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROLLOUT BUFFER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RolloutBuffer:
    """PPO rollout buffer with GAE."""
    
    def __init__(self, n_steps: int, n_envs: int, obs_dim: int, device: torch.device):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.device = device
        
        self.obs = torch.zeros(n_steps, n_envs, obs_dim, dtype=torch.float32)
        self.actions = torch.zeros(n_steps, n_envs, dtype=torch.long)
        self.rewards = torch.zeros(n_steps, n_envs, dtype=torch.float32)
        self.log_probs = torch.zeros(n_steps, n_envs, dtype=torch.float32)
        self.values = torch.zeros(n_steps, n_envs, dtype=torch.float32)
        self.dones = torch.zeros(n_steps, n_envs, dtype=torch.float32)
        
        self.advantages = torch.zeros(n_steps, n_envs, dtype=torch.float32)
        self.returns = torch.zeros(n_steps, n_envs, dtype=torch.float32)
        
        self.step = 0
    
    def add(self, obs, action, reward, log_prob, value, done):
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.log_probs[self.step] = log_prob
        self.values[self.step] = value
        self.dones[self.step] = done
        self.step += 1
    
    def compute_gae(self, next_value: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute Generalized Advantage Estimation."""
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
        
        self.returns = self.advantages + self.values
    
    def flatten(self):
        """Flatten (n_steps, n_envs) -> (n_steps * n_envs)."""
        n = self.n_steps * self.n_envs
        return {
            "obs": self.obs.reshape(n, -1).to(self.device),
            "actions": self.actions.reshape(n).to(self.device),
            "log_probs": self.log_probs.reshape(n).to(self.device),
            "advantages": self.advantages.reshape(n).to(self.device),
            "returns": self.returns.reshape(n).to(self.device),
        }
    
    def reset(self):
        self.step = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD", "US30_cash", "US100_cash"]

def load_data(symbols: list[str]) -> dict:
    """Load data + normalizer for all symbols."""
    from data_engine.normalizer import RunningNormalizer
    
    norm_path = DATA_DIR / "normalizer_v3.json"
    with open(norm_path, "r") as f:
        norm_data = json.load(f)
    
    normalizers = {}
    for tf_name, state in norm_data.items():
        normalizers[tf_name] = RunningNormalizer.from_state_dict(state)
    
    data = {}
    for sym in symbols:
        safe = sym.replace(".", "_")
        data[sym] = {}
        for tf in ["M1", "M5", "M15", "H1"]:
            npy = DATA_DIR / f"{safe}_{tf}_50dim.npy"
            if npy.exists():
                arr = np.load(npy)
                data[sym][tf] = normalizers[tf].normalize(arr).astype(np.float32)
                logger.info("  %s %s: %s", sym, tf, arr.shape)
            else:
                raise FileNotFoundError(f"Missing: {npy.name}")
            
            # OHLCV
            ohlcv = DATA_DIR / f"{safe}_{tf}_ohlcv.npy"
            if ohlcv.exists():
                data[sym][f"{tf}_ohlcv"] = np.load(ohlcv).astype(np.float32)
            else:
                data[sym][f"{tf}_ohlcv"] = None
    
    return data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
    """PPO training for V3.3 Stage 1 (Discrete BUY/SELL/HOLD)."""
    
    if test_mode:
        total_steps = 200
        n_envs = 4
        n_steps = 32
        batch_size = 16
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("=" * 70)
    logger.info("  V3.3 STAGE 1 -- PPO DISCRETE (BUY/SELL/HOLD)")
    logger.info("  Device: %s | Steps: %d | Envs: %d | Rollout: %d", device, total_steps, n_envs, n_steps)
    logger.info("  LR: %.1e | Batch: %d | Epochs: %d | Clip: %.2f | Ent: %.2f",
                lr, batch_size, n_epochs, clip_coef, ent_coef)
    logger.info("=" * 70)
    
    # Load data
    logger.info("Loading data...")
    all_data = load_data(SYMBOLS)
    
    # Load env config
    import yaml as _yaml
    config_path = project_root / "rabit_propfirm_drl" / "configs" / "prop_rules.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        env_config = _yaml.safe_load(f)
    
    # Force stage1_mode in config
    env_config["stage1_mode"] = True
    
    # Create vectorized environments
    import gymnasium
    from environments.prop_env import MultiTFTradingEnv
    
    def make_env(sym: str, seed: int):
        def _init():
            sd = all_data[sym]
            env = MultiTFTradingEnv(
                data_m1=sd["M1"],
                data_m5=sd["M5"],
                data_m15=sd.get("M15", np.zeros((500, 50), dtype=np.float32)),
                data_h1=sd.get("H1", np.zeros((200, 50), dtype=np.float32)),
                config=env_config,
                n_features=50,
                initial_balance=10_000.0,
                episode_length=2000,
                ohlcv_m5=sd.get("M5_ohlcv"),
                action_mode="discrete",  # V3.3: Discrete actions
            )
            env.reset(seed=seed)
            return env
        return _init
    
    env_fns = [make_env(SYMBOLS[i % len(SYMBOLS)], seed=42 + i) for i in range(n_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns)
    obs_batch, _ = vec_env.reset()
    
    logger.info("AsyncVectorEnv: %d envs (symbols: %s)",
                n_envs, [SYMBOLS[i % len(SYMBOLS)] for i in range(n_envs)])
    logger.info("Obs shape: %s, Action space: %s", obs_batch.shape, vec_env.single_action_space)
    
    # Create model
    obs_dim = obs_batch.shape[1]  # Should be 300
    model = PPOActorCritic(obs_dim=obs_dim, n_actions=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("PPO Actor-Critic: %d params (obs_dim=%d)", n_params, obs_dim)
    
    # Rollout buffer
    buffer = RolloutBuffer(n_steps, n_envs, obs_dim, device)
    
    # Tracking
    best_checkpoint = MODELS_DIR / "best_v33_stage1.pt"
    ckpt_dir = MODELS_DIR / "v33_stage1"
    ckpt_dir.mkdir(exist_ok=True)
    
    global_step = 0
    n_updates = 0
    start_time = time.time()
    
    # Stats
    total_trades = 0
    total_buys = 0
    total_sells = 0
    total_holds = 0
    episode_rewards = []
    episode_trades = []
    
    logger.info("Starting PPO training loop (%d steps)...", total_steps)
    
    while global_step < total_steps:
        # ══════════════════════════════════════════════════
        # PHASE 1: COLLECT ROLLOUT
        # ══════════════════════════════════════════════════
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
            
            # Track action distribution
            for a in actions_np:
                if a == 0: total_buys += 1
                elif a == 1: total_sells += 1
                else: total_holds += 1
            
            # Step
            next_obs, rewards, terminateds, truncateds, infos = vec_env.step(actions_np)
            dones = np.logical_or(terminateds, truncateds).astype(np.float32)
            
            # Track trades from infos
            for info in infos:
                if isinstance(info, dict):
                    total_trades = max(total_trades, info.get("total_trades", 0))
            
            buffer.add(
                obs_t,
                action.cpu(),
                torch.from_numpy(rewards.astype(np.float32)),
                log_prob.cpu(),
                value.cpu(),
                torch.from_numpy(dones),
            )
            
            obs_batch = next_obs
            global_step += n_envs
        
        # Compute GAE
        with torch.no_grad():
            next_obs_t = torch.from_numpy(obs_batch).float().to(device)
            next_obs_t = torch.nan_to_num(next_obs_t, nan=0.0, posinf=10.0, neginf=-10.0)
            next_value = model.get_value(next_obs_t).cpu()
        
        buffer.compute_gae(next_value, gamma, gae_lambda)
        
        # ══════════════════════════════════════════════════
        # PHASE 2: PPO UPDATE
        # ══════════════════════════════════════════════════
        flat = buffer.flatten()
        n_samples = flat["obs"].shape[0]
        
        # Normalize advantages
        adv = flat["advantages"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        flat["advantages"] = adv
        
        avg_policy_loss = 0.0
        avg_value_loss = 0.0
        avg_entropy = 0.0
        avg_clip_frac = 0.0
        n_batches_total = 0
        
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
                
                _, new_log_prob, entropy, new_value = model.get_action_and_value(mb_obs, mb_act)
                
                # Policy loss (clipped surrogate)
                ratio = (new_log_prob - mb_old_lp).exp()
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_value, mb_ret)
                
                # Entropy bonus
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
                # Stats
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                
                avg_policy_loss += policy_loss.item()
                avg_value_loss += value_loss.item()
                avg_entropy += entropy_loss.item()
                avg_clip_frac += clip_frac
                n_batches_total += 1
        
        n_updates += 1
        
        # Average stats
        if n_batches_total > 0:
            avg_policy_loss /= n_batches_total
            avg_value_loss /= n_batches_total
            avg_entropy /= n_batches_total
            avg_clip_frac /= n_batches_total
        
        # ══════════════════════════════════════════════════
        # PHASE 3: LOGGING
        # ══════════════════════════════════════════════════
        total_actions = total_buys + total_sells + total_holds
        pct_buy = 100 * total_buys / max(total_actions, 1)
        pct_sell = 100 * total_sells / max(total_actions, 1)
        pct_hold = 100 * total_holds / max(total_actions, 1)
        
        elapsed = time.time() - start_time
        sps = global_step / max(elapsed, 1)
        
        logger.info(
            "[V3.3] Step %d/%d (%.1f%%) | SPS=%.0f | "
            "pi=%.4f | vf=%.4f | ent=%.3f | clip=%.2f | "
            "BUY=%.1f%% SELL=%.1f%% HOLD=%.1f%%",
            global_step, total_steps,
            100 * global_step / total_steps, sps,
            avg_policy_loss, avg_value_loss, avg_entropy, avg_clip_frac,
            pct_buy, pct_sell, pct_hold,
        )
        
        # Checkpoint
        ckpt_interval = 50_000
        if global_step >= ckpt_interval and global_step % ckpt_interval < n_steps * n_envs:
            ckpt = {
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "obs_dim": obs_dim,
                "n_actions": 3,
                "stats": {
                    "total_buys": total_buys,
                    "total_sells": total_sells,
                    "total_holds": total_holds,
                    "pct_buy": pct_buy,
                    "pct_sell": pct_sell,
                    "pct_hold": pct_hold,
                },
            }
            ckpt_path = ckpt_dir / f"checkpoint_{global_step}.pt"
            torch.save(ckpt, ckpt_path)
            torch.save(ckpt, best_checkpoint)
            logger.info("  Saved checkpoint -> %s (BUY=%.1f%% SELL=%.1f%% HOLD=%.1f%%)",
                        ckpt_path.name, pct_buy, pct_sell, pct_hold)
    
    # Final save
    final_ckpt = {
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "obs_dim": obs_dim,
        "n_actions": 3,
        "stats": {
            "total_buys": total_buys,
            "total_sells": total_sells,
            "total_holds": total_holds,
            "pct_buy": 100 * total_buys / max(total_buys + total_sells + total_holds, 1),
            "pct_sell": 100 * total_sells / max(total_buys + total_sells + total_holds, 1),
            "pct_hold": 100 * total_holds / max(total_buys + total_sells + total_holds, 1),
        },
    }
    torch.save(final_ckpt, best_checkpoint)
    
    # Cleanup
    vec_env.close()
    
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("  V3.3 STAGE 1 COMPLETE!")
    logger.info("  Steps: %d in %.1fs (%.0f SPS)", global_step, elapsed, global_step / max(elapsed, 1))
    logger.info("  Actions: BUY=%.1f%% SELL=%.1f%% HOLD=%.1f%%",
                final_ckpt["stats"]["pct_buy"],
                final_ckpt["stats"]["pct_sell"],
                final_ckpt["stats"]["pct_hold"])
    logger.info("  Checkpoint: %s", best_checkpoint)
    logger.info("=" * 70)
    
    return best_checkpoint


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 2: PPO + VIP SELF-IMITATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_stage2(
    total_steps: int = 500_000,
    n_envs: int = 12,
    n_steps: int = 2048,
    batch_size: int = 512,
    n_epochs: int = 4,
    lr: float = 1e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.15,
    ent_coef: float = 0.02,
    vf_coef: float = 0.5,
    il_coef: float = 0.3,
    il_batch_size: int = 256,
    max_grad_norm: float = 0.5,
    device: torch.device = None,
    test_mode: bool = False,
):
    """
    PPO + VIP Self-Imitation Learning for Stage 2.

    Combined loss = PPO_loss + il_coef * CrossEntropy(policy, VIP_actions)

    VIP buffer provides expert demonstrations from Stage 1 winning trades
    that passed SMC filter. The imitation loss nudges the policy toward
    these high-quality patterns while PPO continues learning from env.
    """

    if test_mode:
        total_steps = 200
        n_envs = 4
        n_steps = 32
        batch_size = 16
        il_batch_size = 8

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info("  V3.3 STAGE 2 -- PPO + VIP SELF-IMITATION")
    logger.info("  Device: %s | Steps: %d | Envs: %d | IL_coef: %.2f", device, total_steps, n_envs, il_coef)
    logger.info("  LR: %.1e | Batch: %d | IL_batch: %d | Ent: %.2f",
                lr, batch_size, il_batch_size, ent_coef)
    logger.info("=" * 70)

    # Load VIP buffer
    VIP_DIR = MODELS_DIR / "vip_buffer"
    vip_obs_path = VIP_DIR / "vip_obs.npy"
    vip_act_path = VIP_DIR / "vip_actions.npy"

    if not vip_obs_path.exists():
        logger.error("VIP buffer not found! Run harvest_vip.py first.")
        sys.exit(1)

    vip_obs = torch.from_numpy(np.load(vip_obs_path)).float().to(device)
    vip_actions = torch.from_numpy(np.load(vip_act_path)).long().to(device)
    logger.info("VIP Buffer: %d experiences (obs=%s)", len(vip_obs), vip_obs.shape)

    # Load data
    logger.info("Loading data...")
    all_data = load_data(SYMBOLS)

    import yaml as _yaml
    config_path = project_root / "rabit_propfirm_drl" / "configs" / "prop_rules.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        env_config = _yaml.safe_load(f)
    env_config["stage1_mode"] = False  # Stage 2: re-enable some penalties
    env_config["inaction_nudge"] = -0.05  # Gentler inaction (bot already trades)
    env_config["inaction_threshold_steps"] = 50

    import gymnasium
    from environments.prop_env import MultiTFTradingEnv

    def make_env(sym: str, seed: int):
        def _init():
            sd = all_data[sym]
            env = MultiTFTradingEnv(
                data_m1=sd["M1"], data_m5=sd["M5"],
                data_m15=sd.get("M15", np.zeros((500, 50), dtype=np.float32)),
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

    logger.info("AsyncVectorEnv: %d envs, obs=%s", n_envs, obs_batch.shape)

    # Load Stage 1 model (warm start)
    obs_dim = obs_batch.shape[1]
    model = PPOActorCritic(obs_dim=obs_dim, n_actions=3).to(device)

    stage1_ckpt = MODELS_DIR / "best_v33_stage1.pt"
    if stage1_ckpt.exists():
        ckpt = torch.load(stage1_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Warm-started from Stage 1 (step=%d)", ckpt.get("step", 0))
    else:
        logger.warning("No Stage 1 checkpoint — training from scratch!")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    buffer = RolloutBuffer(n_steps, n_envs, obs_dim, device)

    best_checkpoint = MODELS_DIR / "best_v33_stage2.pt"
    ckpt_dir = MODELS_DIR / "v33_stage2"
    ckpt_dir.mkdir(exist_ok=True)

    global_step = 0
    start_time = time.time()
    total_buys = 0
    total_sells = 0
    total_holds = 0

    logger.info("Starting Stage 2 PPO+IL loop (%d steps, VIP=%d)...", total_steps, len(vip_obs))

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
                else: total_holds += 1

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

        # PHASE 2: PPO + IMITATION UPDATE
        flat = buffer.flatten()
        n_samples = flat["obs"].shape[0]
        adv = flat["advantages"]
        flat["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        avg_pi = 0.0; avg_vf = 0.0; avg_ent = 0.0; avg_il = 0.0; avg_clip = 0.0
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

                # PPO policy loss
                ratio = (new_lp - mb_old_lp).exp()
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                policy_loss = torch.max(pg1, pg2).mean()

                # Value loss
                value_loss = F.mse_loss(new_val, mb_ret)

                # Entropy
                entropy_loss = entropy.mean()

                # IMITATION LOSS: Sample VIP batch, compute cross-entropy
                vip_idx = torch.randint(0, len(vip_obs), (il_batch_size,), device=device)
                vip_mb_obs = vip_obs[vip_idx]
                vip_mb_act = vip_actions[vip_idx]

                vip_logits, _ = model(vip_mb_obs)
                il_loss = F.cross_entropy(vip_logits, vip_mb_act)

                # Combined loss
                loss = (policy_loss
                        + vf_coef * value_loss
                        - ent_coef * entropy_loss
                        + il_coef * il_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

                avg_pi += policy_loss.item()
                avg_vf += value_loss.item()
                avg_ent += entropy_loss.item()
                avg_il += il_loss.item()
                avg_clip += clip_frac
                n_batches += 1

        if n_batches > 0:
            avg_pi /= n_batches; avg_vf /= n_batches
            avg_ent /= n_batches; avg_il /= n_batches; avg_clip /= n_batches

        total_actions = total_buys + total_sells + total_holds
        pct_buy = 100 * total_buys / max(total_actions, 1)
        pct_sell = 100 * total_sells / max(total_actions, 1)
        pct_hold = 100 * total_holds / max(total_actions, 1)
        elapsed = time.time() - start_time
        sps = global_step / max(elapsed, 1)

        logger.info(
            "[S2] Step %d/%d (%.1f%%) | SPS=%.0f | pi=%.4f vf=%.4f ent=%.3f IL=%.4f | "
            "BUY=%.1f%% SELL=%.1f%% HOLD=%.1f%%",
            global_step, total_steps, 100 * global_step / total_steps, sps,
            avg_pi, avg_vf, avg_ent, avg_il,
            pct_buy, pct_sell, pct_hold,
        )

        # Checkpoint
        if global_step >= 50_000 and global_step % 50_000 < n_steps * n_envs:
            ckpt = {
                "step": global_step, "stage": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "obs_dim": obs_dim, "n_actions": 3,
                "stats": {"pct_buy": pct_buy, "pct_sell": pct_sell, "pct_hold": pct_hold},
            }
            torch.save(ckpt, ckpt_dir / f"checkpoint_{global_step}.pt")
            torch.save(ckpt, best_checkpoint)
            logger.info("  Saved -> %s", best_checkpoint.name)

    # Final
    final = {
        "step": global_step, "stage": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "obs_dim": obs_dim, "n_actions": 3,
        "stats": {"pct_buy": pct_buy, "pct_sell": pct_sell, "pct_hold": pct_hold},
    }
    torch.save(final, best_checkpoint)
    vec_env.close()

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("  V3.3 STAGE 2 COMPLETE!")
    logger.info("  Steps: %d in %.1fs (%.0f SPS)", global_step, elapsed, global_step / max(elapsed, 1))
    logger.info("  BUY=%.1f%% SELL=%.1f%% HOLD=%.1f%%", pct_buy, pct_sell, pct_hold)
    logger.info("  Checkpoint: %s", best_checkpoint)
    logger.info("=" * 70)
    return best_checkpoint


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="V3.3 PPO Training")
    parser.add_argument("--stage", type=int, default=1, help="Stage 1, 2, or 3")
    parser.add_argument("--test", action="store_true", help="Quick test")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--il-coef", type=float, default=None, help="Imitation loss coef")
    parser.add_argument("--ent-coef", type=float, default=None, help="Entropy coef")
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
            il_coef=args.il_coef or 0.3,
            ent_coef=args.ent_coef or 0.02,
        )
    elif args.stage == 3:
        # Stage 3: Balanced VIP — lower IL, higher entropy, warm-start from S2
        # Override best_checkpoint path to save Stage 3 separately
        il = args.il_coef if args.il_coef is not None else 0.15
        ent = args.ent_coef if args.ent_coef is not None else 0.05
        logger.info("Stage 3: il_coef=%.2f, ent_coef=%.2f (balanced)", il, ent)

        # Temporarily rename Stage 2 checkpoint for warm-start
        s2_ckpt = MODELS_DIR / "best_v33_stage2.pt"
        s1_ckpt = MODELS_DIR / "best_v33_stage1.pt"
        warm_ckpt = s2_ckpt if s2_ckpt.exists() else s1_ckpt

        # Monkey-patch to load from Stage 2 and save to Stage 3
        import shutil
        temp_s1 = MODELS_DIR / "_temp_s1_backup.pt"
        if warm_ckpt.exists() and warm_ckpt != s1_ckpt:
            shutil.copy2(s1_ckpt, temp_s1)
            shutil.copy2(warm_ckpt, s1_ckpt)
            logger.info("Warm-starting Stage 3 from Stage 2 checkpoint")

        train_stage2(
            total_steps=args.total_steps or 500_000,
            n_envs=args.n_envs, device=device, test_mode=args.test,
            il_coef=il,
            ent_coef=ent,
        )

        # Rename output to stage3
        s2_out = MODELS_DIR / "best_v33_stage2.pt"
        s3_out = MODELS_DIR / "best_v33_stage3.pt"
        if s2_out.exists():
            shutil.copy2(s2_out, s3_out)
            logger.info("Saved Stage 3 -> %s", s3_out)

        # Restore original Stage 1
        if temp_s1.exists():
            shutil.move(str(temp_s1), str(s1_ckpt))
    else:
        logger.error("Invalid stage: %d (use 1, 2, or 3)", args.stage)
        sys.exit(1)


if __name__ == "__main__":
    main()

