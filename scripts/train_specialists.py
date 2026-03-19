"""
Train 3 Specialist Agents in Parallel — Trend / Range / Volatility.

Sprint 5.2: Designed for RunPod L40 (48GB VRAM, 32 vCPUs).

Architecture:
    - 3 parallel processes (torch.multiprocessing)
    - Each process: 1 SAC agent + 8 vectorized environments
    - CPU: torch.set_num_threads(10) per process (30/32 cores used)
    - GPU: 3 models share 48GB VRAM (~9MB each, plenty of room)
    - W&B: 3 runs in group="ensemble_sprint5", 1 dashboard

Diversity (3 layers):
    1. Seed: [42, 137, 2024]
    2. Reward Shaping: Trend=hold bonus, Range=scalp bonus, Vol=survive bonus
    3. Feature Masking: each agent emphasizes different feature columns

Usage:
    python3 -u scripts/train_specialists.py [--steps 1000000]
    python3 -u scripts/train_specialists.py --test  # 5000 steps, offline W&B
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.multiprocessing as mp

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "rabit_propfirm_drl"))

DATA_DIR = project_root / "data"
MODEL_DIR = project_root / "models_saved"
MODEL_DIR.mkdir(exist_ok=True)


# =====================================================================
# CONSTANTS
# =====================================================================

FEATURE_COLS = [
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "candle_direction",
    "pin_bar_bull", "pin_bar_bear", "engulfing_bull", "engulfing_bear", "inside_bar",
    "relative_volume", "vol_delta", "climax_vol",
    "swing_high", "swing_low", "swing_trend", "bos", "choch",
    "ob_bull_dist", "ob_bear_dist", "fvg_bull_active", "fvg_bear_active",
    "liq_above", "liq_below",
    "sin_hour", "cos_hour", "sin_dow", "cos_dow",
    "log_return",
]

LOOKBACK_M5 = 64
LOOKBACK_H1 = 24
LOOKBACK_H4 = 30
N_FEATURES = 28

# Per-process resource allocation
THREADS_PER_PROCESS = 10   # 3 x 10 = 30/32 cores
N_VEC_ENVS = 8             # 8 parallel envs per agent

# Training hyperparams
BATCH_SIZE = 128            # Larger batch for L40
LR_ACTOR = 1e-4
LR_CRITIC = 3e-4
GAMMA = 0.99
TAU = 0.005
PER_CAPACITY = 200_000     # Larger buffer for L40
PER_ALPHA = 0.6
PER_BETA_START = 0.4
WARMUP_STEPS = 3000
EVAL_EVERY = 10_000


# =====================================================================
# SPECIALIST CONFIGS  (Reward Shaping + Feature Emphasis)
# =====================================================================

@dataclass
class SpecialistConfig:
    """Configuration for a specialist agent."""
    name: str
    seed: int
    # Feature emphasis: indices of columns to boost (multiply by 2.0)
    feature_emphasis: list[int] = field(default_factory=list)
    # Reward shaping params
    trend_hold_bonus: float = 0.0       # Bonus per step holding a winner
    quick_scalp_bonus: float = 0.0      # Bonus for exits within N bars with profit
    quick_scalp_max_bars: int = 10      # Max bars for quick scalp
    volatility_survive_bonus: float = 0.0  # Bonus for surviving volatile periods
    # W&B
    wandb_name: str = ""

    def describe(self) -> str:
        traits = []
        if self.trend_hold_bonus > 0:
            traits.append(f"trend_hold={self.trend_hold_bonus}")
        if self.quick_scalp_bonus > 0:
            traits.append(f"scalp={self.quick_scalp_bonus}")
        if self.volatility_survive_bonus > 0:
            traits.append(f"vol_survive={self.volatility_survive_bonus}")
        return f"{self.name} (seed={self.seed}, {', '.join(traits)})"


# Feature indices reference (from FEATURE_COLS):
# 0-3:   body_ratio, upper_wick, lower_wick, candle_direction
# 4-8:   pin_bar_bull/bear, engulfing_bull/bear, inside_bar
# 9-11:  relative_volume, vol_delta, climax_vol
# 12-14: swing_high, swing_low, swing_trend
# 15-16: bos, choch
# 17-18: ob_bull_dist, ob_bear_dist
# 19-20: fvg_bull_active, fvg_bear_active
# 21-22: liq_above, liq_below
# 23-26: sin_hour, cos_hour, sin_dow, cos_dow
# 27:    log_return

SPECIALIST_CONFIGS = [
    SpecialistConfig(
        name="TrendAgent",
        seed=42,
        feature_emphasis=[14, 15, 16, 27],  # swing_trend, bos, choch, log_return
        trend_hold_bonus=0.02,              # +0.02 reward/step if holding a winner
        wandb_name="specialist-trend",
    ),
    SpecialistConfig(
        name="RangeAgent",
        seed=137,
        feature_emphasis=[17, 18, 19, 20],  # ob_dist, fvg_active
        quick_scalp_bonus=0.5,              # +0.5 bonus for quick profitable exit
        quick_scalp_max_bars=10,
        wandb_name="specialist-range",
    ),
    SpecialistConfig(
        name="VolatilityAgent",
        seed=2024,
        feature_emphasis=[9, 10, 11, 21, 22],  # volume features, liquidity
        volatility_survive_bonus=0.3,           # +0.3 for surviving high-vol episode
        wandb_name="specialist-vol",
    ),
]


# =====================================================================
# DATA LOADING (reused from train_v2.py)
# =====================================================================

def build_htf_features(ohlcv: np.ndarray, n_features: int = 28) -> np.ndarray:
    """Build simplified features from raw OHLCV for H1/H4."""
    import math
    n = len(ohlcv)
    features = np.zeros((n, n_features), dtype=np.float32)
    o, h, l, c = ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3]
    body = np.abs(c - o) + 1e-8
    full_range = h - l + 1e-8
    features[:, 0] = body / full_range
    features[:, 1] = np.where(c >= o, (h - c), (h - o)) / full_range
    features[:, 2] = np.where(c >= o, (o - l), (c - l)) / full_range
    features[:, 3] = np.where(c >= o, 1.0, -1.0)
    features[:, 4] = ((features[:, 2] > 0.6) & (features[:, 0] < 0.3)).astype(np.float32)
    features[:, 5] = ((features[:, 1] > 0.6) & (features[:, 0] < 0.3)).astype(np.float32)
    for i in range(1, n):
        if c[i] > o[i] and body[i] > body[i-1] and c[i] > o[i-1] and o[i] < c[i-1]:
            features[i, 6] = 1.0
        if c[i] < o[i] and body[i] > body[i-1] and c[i] < o[i-1] and o[i] > c[i-1]:
            features[i, 7] = 1.0
    for i in range(1, n):
        if h[i] <= h[i-1] and l[i] >= l[i-1]:
            features[i, 8] = 1.0
    if ohlcv.shape[1] > 4:
        vol = ohlcv[:, 4]
        mean_vol = np.convolve(vol, np.ones(20)/20, mode='same') + 1e-8
        features[:, 9] = vol / mean_vol
    for i in range(2, n - 2):
        if h[i] >= h[i-1] and h[i] >= h[i-2] and h[i] >= h[i+1] and h[i] >= h[i+2]:
            features[i, 12] = 1.0
        if l[i] <= l[i-1] and l[i] <= l[i-2] and l[i] <= l[i+1] and l[i] <= l[i+2]:
            features[i, 13] = 1.0
    for i in range(5, n):
        trend = (c[i] - c[i-5]) / (c[i-5] + 1e-8)
        features[i, 14] = np.clip(trend * 100, -1, 1)
    for i in range(n):
        hour_approx = (i % 24)
        features[i, 23] = math.sin(2 * math.pi * hour_approx / 24)
        features[i, 24] = math.cos(2 * math.pi * hour_approx / 24)
        dow_approx = (i // 24) % 5
        features[i, 25] = math.sin(2 * math.pi * dow_approx / 5)
        features[i, 26] = math.cos(2 * math.pi * dow_approx / 5)
    features[1:, 27] = np.log(c[1:] / (c[:-1] + 1e-8))
    return features


def load_all_data() -> dict:
    """Load M5 features + H1/H4 raw OHLCV for all symbols."""
    import polars as pl
    symbol_data = {}
    for feat_file in sorted(DATA_DIR.glob("*_M5_features.parquet")):
        sym_name = feat_file.stem.replace("_M5_features", "")
        df_m5 = pl.read_parquet(feat_file)
        available_cols = [c for c in FEATURE_COLS if c in df_m5.columns]
        m5_features = np.column_stack([
            df_m5[col].fill_null(0.0).fill_nan(0.0).to_numpy().astype(np.float32)
            for col in available_cols
        ])
        if m5_features.shape[1] < N_FEATURES:
            pad = np.zeros((len(m5_features), N_FEATURES - m5_features.shape[1]), dtype=np.float32)
            m5_features = np.hstack([m5_features, pad])
        m5_ohlcv = np.column_stack([
            df_m5["open"].to_numpy().astype(np.float32),
            df_m5["high"].to_numpy().astype(np.float32),
            df_m5["low"].to_numpy().astype(np.float32),
            df_m5["close"].to_numpy().astype(np.float32),
            df_m5["tick_volume"].fill_null(0).to_numpy().astype(np.float32)
            if "tick_volume" in df_m5.columns
            else np.zeros(len(df_m5), dtype=np.float32),
        ])
        m5_times = df_m5["time"].to_list() if "time" in df_m5.columns else []
        h1_path = DATA_DIR / f"{sym_name}_H1.parquet"
        if h1_path.exists():
            df_h1 = pl.read_parquet(h1_path)
            h1_ohlcv = np.column_stack([
                df_h1["open"].to_numpy().astype(np.float32),
                df_h1["high"].to_numpy().astype(np.float32),
                df_h1["low"].to_numpy().astype(np.float32),
                df_h1["close"].to_numpy().astype(np.float32),
                df_h1["tick_volume"].fill_null(0).to_numpy().astype(np.float32)
                if "tick_volume" in df_h1.columns
                else np.zeros(len(df_h1), dtype=np.float32),
            ])
            h1_features = build_htf_features(h1_ohlcv, N_FEATURES)
        else:
            h1_features = np.zeros((100, N_FEATURES), dtype=np.float32)
        h4_path = DATA_DIR / f"{sym_name}_H4.parquet"
        if h4_path.exists():
            df_h4 = pl.read_parquet(h4_path)
            h4_ohlcv = np.column_stack([
                df_h4["open"].to_numpy().astype(np.float32),
                df_h4["high"].to_numpy().astype(np.float32),
                df_h4["low"].to_numpy().astype(np.float32),
                df_h4["close"].to_numpy().astype(np.float32),
                df_h4["tick_volume"].fill_null(0).to_numpy().astype(np.float32)
                if "tick_volume" in df_h4.columns
                else np.zeros(len(df_h4), dtype=np.float32),
            ])
            h4_features = build_htf_features(h4_ohlcv, N_FEATURES)
        else:
            h4_features = np.zeros((100, N_FEATURES), dtype=np.float32)
        ib_path = DATA_DIR / f"{sym_name}_H1_insidebar.parquet"
        h1_ib_times = set()
        if ib_path.exists():
            ib_df = pl.read_parquet(ib_path)
            h1_ib_times = set(
                ib_df.filter(pl.col("inside_bar") == 1.0)["time"].to_list()
            )
        symbol_data[sym_name] = {
            "m5_features": m5_features, "m5_ohlcv": m5_ohlcv,
            "m5_times": m5_times, "h1_features": h1_features,
            "h4_features": h4_features, "h1_ib_times": h1_ib_times,
            "n_m5": len(m5_features), "n_h1": len(h1_features),
            "n_h4": len(h4_features),
        }
    return symbol_data


def normalize_all(symbol_data: dict) -> dict:
    """Z-score normalize M5, H1, H4 features."""
    for tf_key in ["m5_features", "h1_features", "h4_features"]:
        all_feat = np.concatenate([d[tf_key] for d in symbol_data.values()], axis=0)
        mean = np.mean(all_feat, axis=0)
        std = np.std(all_feat, axis=0) + 1e-8
        for sym in symbol_data:
            symbol_data[sym][f"{tf_key}_norm"] = (
                (symbol_data[sym][tf_key] - mean) / std
            ).astype(np.float32)
    return symbol_data


def apply_feature_emphasis(symbol_data: dict, emphasis_indices: list[int]) -> dict:
    """Boost emphasized feature columns by 2x for specialist focus."""
    if not emphasis_indices:
        return symbol_data
    for sym in symbol_data:
        for tf_key in ["m5_features_norm", "h1_features_norm", "h4_features_norm"]:
            for idx in emphasis_indices:
                if idx < symbol_data[sym][tf_key].shape[1]:
                    symbol_data[sym][tf_key][:, idx] *= 2.0
    return symbol_data


# =====================================================================
# VECTORIZED ENVIRONMENT WRAPPER
# =====================================================================

class MultiTFTradeEnv:
    """Single trading env (copied from train_v2 for self-containment)."""

    def __init__(self, m5_features, m5_ohlcv, h1_features, h4_features,
                 m5_times, h1_ib_times, n_m5, n_h1, n_h4,
                 initial_balance=100_000.0, reward_shaper=None):
        self.m5_feat = m5_features
        self.m5_ohlcv = m5_ohlcv
        self.h1_feat = h1_features
        self.h4_feat = h4_features
        self.m5_times = m5_times
        self.h1_ib_times = h1_ib_times
        self.n_m5, self.n_h1, self.n_h4 = n_m5, n_h1, n_h4
        self.initial_balance = initial_balance
        self.reward_shaper = reward_shaper
        self.m5_per_h1 = max(1, n_m5 // max(n_h1, 1))
        self.m5_per_h4 = max(1, n_m5 // max(n_h4, 1))
        self.reset()

    def reset(self):
        max_start = self.n_m5 - LOOKBACK_M5 - 500
        self.m5_idx = np.random.randint(LOOKBACK_M5, max(LOOKBACK_M5 + 1, max_start))
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_step = 0
        self.unrealized_pnl = 0.0
        self.step_count = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_loss = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance
        self.realized_pnl = 0.0
        self.high_vol_steps = 0
        return self._get_obs()

    def _get_m5_window(self):
        start = max(0, self.m5_idx - LOOKBACK_M5)
        w = self.m5_feat[start:self.m5_idx]
        if len(w) < LOOKBACK_M5:
            pad = np.zeros((LOOKBACK_M5 - len(w), N_FEATURES), dtype=np.float32)
            w = np.concatenate([pad, w], axis=0)
        return w

    def _get_h1_window(self):
        h1_idx = min(self.m5_idx // self.m5_per_h1, self.n_h1 - 1)
        start = max(0, h1_idx - LOOKBACK_H1)
        w = self.h1_feat[start:h1_idx]
        if len(w) < LOOKBACK_H1:
            pad = np.zeros((LOOKBACK_H1 - len(w), N_FEATURES), dtype=np.float32)
            w = np.concatenate([pad, w], axis=0)
        return w

    def _get_h4_window(self):
        h4_idx = min(self.m5_idx // self.m5_per_h4, self.n_h4 - 1)
        start = max(0, h4_idx - LOOKBACK_H4)
        w = self.h4_feat[start:h4_idx]
        if len(w) < LOOKBACK_H4:
            pad = np.zeros((LOOKBACK_H4 - len(w), N_FEATURES), dtype=np.float32)
            w = np.concatenate([pad, w], axis=0)
        return w

    def _get_obs(self):
        return self._get_m5_window(), self._get_h1_window(), self._get_h4_window()

    def _is_high_vol(self):
        """Check if current bar has high relative volume."""
        if self.m5_idx < self.n_m5 and self.m5_feat.shape[1] > 9:
            return float(self.m5_feat[self.m5_idx, 9]) > 2.0  # rel_vol > 2x
        return False

    def step(self, action):
        confidence = float(np.clip(action[0], -1, 1))
        risk_frac = float(np.clip(action[1], 0, 1))
        close_now = float(self.m5_ohlcv[self.m5_idx, 3])
        reward = 0.0
        done = False

        if self._is_high_vol():
            self.high_vol_steps += 1

        # Update PnL for open position
        if self.position != 0:
            price_change = (close_now - self.entry_price) / self.entry_price
            self.unrealized_pnl = self.position * price_change * self.balance * risk_frac
            holding_time = self.step_count - self.entry_step

            max_loss = self.balance * 0.003
            if self.unrealized_pnl < -max_loss:
                # Stop-loss hit
                self.balance += self.unrealized_pnl
                self.realized_pnl += self.unrealized_pnl
                self.daily_loss += abs(self.unrealized_pnl) / self.initial_balance
                reward = -1.0
                self.total_trades += 1
                self.position = 0.0
                self.unrealized_pnl = 0.0
            elif self.unrealized_pnl > self.balance * 0.01:
                # Take-profit hit
                self.balance += self.unrealized_pnl
                self.realized_pnl += self.unrealized_pnl
                reward = 2.0
                self.total_trades += 1
                self.winning_trades += 1
                self.position = 0.0
                self.unrealized_pnl = 0.0

                # Reward shaping: quick scalp bonus
                if self.reward_shaper and holding_time <= self.reward_shaper.quick_scalp_max_bars:
                    reward += self.reward_shaper.quick_scalp_bonus
            elif self.position != 0 and self.unrealized_pnl > 0:
                # Reward shaping: trend hold bonus (per step holding winner)
                if self.reward_shaper:
                    reward += self.reward_shaper.trend_hold_bonus

        # Entry / exit logic
        if abs(confidence) > 0.3 and self.position == 0 and self.daily_loss < 0.03:
            self.position = 1.0 if confidence > 0 else -1.0
            self.entry_price = close_now
            self.entry_step = self.step_count
        elif abs(confidence) < 0.1 and self.position != 0:
            pnl = self.unrealized_pnl
            self.balance += pnl
            self.realized_pnl += pnl
            holding_time = self.step_count - self.entry_step
            if pnl > 0:
                reward = 0.5
                self.winning_trades += 1
                if self.reward_shaper and holding_time <= self.reward_shaper.quick_scalp_max_bars:
                    reward += self.reward_shaper.quick_scalp_bonus
            else:
                reward = -0.3
                self.daily_loss += abs(pnl) / self.initial_balance
            self.total_trades += 1
            self.position = 0.0
            self.unrealized_pnl = 0.0

        # Track drawdown
        self.peak_balance = max(self.peak_balance, self.balance)
        current_dd = (self.peak_balance - self.balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_dd)

        self.m5_idx += 1
        self.step_count += 1

        if self.m5_idx >= self.n_m5 - 1 or self.step_count >= 480:
            done = True
        if self.daily_loss >= 0.03 or self.balance < self.initial_balance * 0.95:
            done = True

        # End-of-episode volatility survive bonus
        if done and self.reward_shaper and self.reward_shaper.volatility_survive_bonus > 0:
            if self.high_vol_steps > 20 and self.balance >= self.initial_balance:
                reward += self.reward_shaper.volatility_survive_bonus

        info = {
            "balance": self.balance,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(self.total_trades, 1),
            "realized_pnl": self.realized_pnl,
            "max_drawdown": self.max_drawdown,
        }
        return self._get_obs(), reward, done, info


class VectorizedEnvs:
    """
    Run N environments in a single thread using round-robin stepping.

    Simulates SubprocVecEnv behavior without subprocess overhead.
    Each env resets independently on 'done'. Collects N transitions per step.
    """

    def __init__(self, n_envs: int, env_factory: Callable):
        self.envs = [env_factory() for _ in range(n_envs)]
        self.n_envs = n_envs
        self.observations = [env.reset() for env in self.envs]

    def step_all(self, actions: list[np.ndarray]):
        """
        Step all envs with corresponding actions.

        Returns:
            List of (obs, action, reward, next_obs, done, info) tuples
        """
        transitions = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs = self.observations[i]
            next_obs, reward, done, info = env.step(action)

            transitions.append({
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "info": info,
            })

            if done:
                self.observations[i] = env.reset()
            else:
                self.observations[i] = next_obs

        return transitions


# =====================================================================
# SPECIALIST TRAINING WORKER
# =====================================================================

def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tp.data) * tp.data
                      if False else tau * sp.data + (1 - tau) * tp.data)


def train_specialist(
    spec_config: SpecialistConfig,
    train_steps: int,
    test_mode: bool,
    gpu_id: int = 0,
):
    """
    Train a single specialist agent. Runs as a separate process.

    Args:
        spec_config: SpecialistConfig (name, seed, reward shaping, etc.)
        train_steps: Total training steps
        test_mode: If True, run 5000 steps with offline W&B
        gpu_id: CUDA device index
    """
    # --- Resource pinning ---
    torch.set_num_threads(THREADS_PER_PROCESS)
    torch.manual_seed(spec_config.seed)
    np.random.seed(spec_config.seed)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError(f"[{spec_config.name}] CUDA REQUIRED!")

    # --- W&B Init (grouped) ---
    import wandb
    wandb_mode = "offline" if test_mode else "online"
    run = wandb.init(
        project="rabit-propfirm-drl",
        group="ensemble_sprint5",
        name=spec_config.wandb_name,
        config={
            "specialist": spec_config.name,
            "seed": spec_config.seed,
            "train_steps": train_steps,
            "batch_size": BATCH_SIZE,
            "lr_actor": LR_ACTOR,
            "lr_critic": LR_CRITIC,
            "n_vec_envs": N_VEC_ENVS,
            "threads": THREADS_PER_PROCESS,
            "trend_hold_bonus": spec_config.trend_hold_bonus,
            "quick_scalp_bonus": spec_config.quick_scalp_bonus,
            "vol_survive_bonus": spec_config.volatility_survive_bonus,
            "feature_emphasis": spec_config.feature_emphasis,
            "gpu": torch.cuda.get_device_name(gpu_id),
        },
        mode=wandb_mode,
    )

    print(f"\n{'='*60}")
    print(f"  [{spec_config.name}] Training Started")
    print(f"  Seed: {spec_config.seed} | Device: {device}")
    print(f"  Steps: {train_steps:,} | VecEnvs: {N_VEC_ENVS}")
    print(f"  Threads: {THREADS_PER_PROCESS} | Batch: {BATCH_SIZE}")
    print(f"  {spec_config.describe()}")
    print(f"{'='*60}\n")

    # --- Data ---
    symbol_data = load_all_data()
    symbol_data = normalize_all(symbol_data)

    # Apply feature emphasis (Layer 3 diversity)
    import copy
    symbol_data = apply_feature_emphasis(
        copy.deepcopy(symbol_data),
        spec_config.feature_emphasis,
    )
    symbols = list(symbol_data.keys())

    # --- Model ---
    from agents.sac_policy import SACTransformerActor, SACTransformerCritic
    from training_pipeline.per_buffer import PERBuffer

    actor = SACTransformerActor(
        n_features=N_FEATURES, action_dim=4, embed_dim=128,
        n_heads=4, n_transformer_layers=2, n_cross_layers=1,
        hidden_dims=[256, 256], dropout=0.1,
    ).to(device)

    critic = SACTransformerCritic(
        n_features=N_FEATURES, action_dim=4, embed_dim=128,
        n_heads=4, n_transformer_layers=2, n_cross_layers=1,
        hidden_dims=[256, 256], dropout=0.1,
    ).to(device)

    critic_target = SACTransformerCritic(
        n_features=N_FEATURES, action_dim=4, embed_dim=128,
        n_heads=4, n_transformer_layers=2, n_cross_layers=1,
        hidden_dims=[256, 256], dropout=0.1,
    ).to(device)
    critic_target.load_state_dict(critic.state_dict())

    log_alpha = torch.nn.Parameter(torch.zeros(1, device=device))
    target_entropy = -4

    actor_opt = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)
    alpha_opt = torch.optim.Adam([log_alpha], lr=LR_CRITIC)

    per_buffer = PERBuffer(
        capacity=PER_CAPACITY, alpha=PER_ALPHA,
        beta_start=PER_BETA_START, beta_frames=train_steps,
        seq_m5=LOOKBACK_M5, seq_h1=LOOKBACK_H1, seq_h4=LOOKBACK_H4,
        n_features=N_FEATURES, action_dim=4,
    )

    # --- Vectorized Environments ---
    def make_env():
        sym = symbols[np.random.randint(len(symbols))]
        sd = symbol_data[sym]
        return MultiTFTradeEnv(
            m5_features=sd["m5_features_norm"], m5_ohlcv=sd["m5_ohlcv"],
            h1_features=sd["h1_features_norm"], h4_features=sd["h4_features_norm"],
            m5_times=sd["m5_times"], h1_ib_times=sd["h1_ib_times"],
            n_m5=sd["n_m5"], n_h1=sd["n_h1"], n_h4=sd["n_h4"],
            reward_shaper=spec_config,
        )

    vec_envs = VectorizedEnvs(N_VEC_ENVS, make_env)

    # --- Training Loop ---
    best_eval_reward = -999
    episode_count = 0
    start_time = time.time()
    last_actor_loss = 0.0
    last_critic_loss = 0.0
    global_step = 0

    while global_step < train_steps:
        # --- Collect N_VEC_ENVS transitions per step ---
        if per_buffer.size < WARMUP_STEPS:
            actions = [np.random.uniform(-1, 1, size=4).astype(np.float32)
                       for _ in range(N_VEC_ENVS)]
        else:
            with torch.no_grad():
                batch_actions = []
                for obs in vec_envs.observations:
                    m5_t = torch.FloatTensor(obs[0]).unsqueeze(0).to(device)
                    h1_t = torch.FloatTensor(obs[1]).unsqueeze(0).to(device)
                    h4_t = torch.FloatTensor(obs[2]).unsqueeze(0).to(device)
                    act_t, _ = actor(m5_t, h1_t, h4_t)
                    batch_actions.append(act_t.squeeze(0).cpu().numpy())
                actions = batch_actions

        transitions = vec_envs.step_all(actions)

        for trans in transitions:
            per_buffer.add(
                trans["obs"], trans["action"], trans["reward"],
                trans["next_obs"], trans["done"],
            )
            global_step += 1

            if trans["done"]:
                episode_count += 1
                info = trans["info"]
                wandb.log({
                    "episode/reward": trans["reward"],
                    "episode/win_rate": info["win_rate"],
                    "episode/max_drawdown": info["max_drawdown"],
                    "episode/total_trades": info["total_trades"],
                    "episode/balance": info["balance"],
                    "per/beta": per_buffer.beta,
                    "per/buffer_size": per_buffer.size,
                }, step=global_step)

        # --- SAC Update (every 4 steps) ---
        if per_buffer.size >= WARMUP_STEPS and global_step % 4 == 0:
            batch = per_buffer.sample(BATCH_SIZE, device=device)
            alpha = log_alpha.exp().detach()
            is_w = batch["is_weights"].unsqueeze(-1)

            with torch.no_grad():
                next_a, next_lp = actor(batch["next_m5"], batch["next_h1"], batch["next_h4"])
                q1n, q2n = critic_target(batch["next_m5"], batch["next_h1"], batch["next_h4"], next_a)
                q_next = torch.min(q1n, q2n) - alpha * next_lp
                target_q = batch["rew"].unsqueeze(-1) + GAMMA * (1 - batch["done"].unsqueeze(-1)) * q_next

            q1, q2 = critic(batch["m5"], batch["h1"], batch["h4"], batch["act"])
            critic_loss = (is_w * (q1 - target_q)**2).mean() + (is_w * (q2 - target_q)**2).mean()

            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()
            last_critic_loss = critic_loss.item()

            td_errors = (((q1 - target_q).abs() + (q2 - target_q).abs()) / 2).squeeze(-1).detach().cpu().numpy()
            per_buffer.update_priorities(batch["tree_indices"], td_errors)

            new_a, lp = actor(batch["m5"], batch["h1"], batch["h4"])
            q1_new = critic(batch["m5"], batch["h1"], batch["h4"], new_a)[0]
            actor_loss = (alpha * lp - q1_new).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_opt.step()
            last_actor_loss = actor_loss.item()

            alpha_loss = -(log_alpha.exp() * (lp.detach() + target_entropy)).mean()
            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()

            # Soft target update
            for tp, sp in zip(critic_target.parameters(), critic.parameters()):
                tp.data.copy_(TAU * sp.data + (1 - TAU) * tp.data)

        # --- Evaluation ---
        if global_step > 0 and global_step % EVAL_EVERY < N_VEC_ENVS:
            actor.eval()
            e_rewards = []
            for sym in symbols:
                sd = symbol_data[sym]
                eval_env = MultiTFTradeEnv(
                    m5_features=sd["m5_features_norm"], m5_ohlcv=sd["m5_ohlcv"],
                    h1_features=sd["h1_features_norm"], h4_features=sd["h4_features_norm"],
                    m5_times=sd["m5_times"], h1_ib_times=sd["h1_ib_times"],
                    n_m5=sd["n_m5"], n_h1=sd["n_h1"], n_h4=sd["n_h4"],
                )
                e_obs = eval_env.reset()
                e_total = 0.0
                for _ in range(480):
                    with torch.no_grad():
                        m5_t = torch.FloatTensor(e_obs[0]).unsqueeze(0).to(device)
                        h1_t = torch.FloatTensor(e_obs[1]).unsqueeze(0).to(device)
                        h4_t = torch.FloatTensor(e_obs[2]).unsqueeze(0).to(device)
                        act_t, _ = actor(m5_t, h1_t, h4_t, deterministic=True)
                        e_act = act_t.squeeze(0).cpu().numpy()
                    e_obs, e_rew, e_done, e_info = eval_env.step(e_act)
                    e_total += e_rew
                    if e_done:
                        break
                e_rewards.append(e_total)
            actor.train()

            mean_r = np.mean(e_rewards)
            elapsed = time.time() - start_time
            sps = global_step / max(elapsed, 1)

            print(
                f"  [{spec_config.name}] Step {global_step:>7,} | "
                f"R: {mean_r:>7.2f} | {sps:.0f} sps"
            )

            wandb.log({
                "eval/mean_reward": mean_r,
                "eval/sps": sps,
            }, step=global_step)

            # Save best
            if mean_r > best_eval_reward:
                best_eval_reward = mean_r
                save_name = f"best_{spec_config.name.lower()}.pt"
                save_path = MODEL_DIR / save_name
                torch.save({
                    "actor_state": actor.state_dict(),
                    "critic_state": critic.state_dict(),
                    "specialist": spec_config.name,
                    "seed": spec_config.seed,
                    "step": global_step,
                    "best_reward": best_eval_reward,
                }, save_path)
                print(f"  [{spec_config.name}] >> New best! Saved {save_name}")

                artifact = wandb.Artifact(f"model-{spec_config.name.lower()}", type="model")
                artifact.add_file(str(save_path))
                run.log_artifact(artifact)

    # --- Final save ---
    final_path = MODEL_DIR / f"final_{spec_config.name.lower()}.pt"
    torch.save({
        "actor_state": actor.state_dict(),
        "critic_state": critic.state_dict(),
        "specialist": spec_config.name,
        "seed": spec_config.seed,
        "step": global_step,
        "best_reward": best_eval_reward,
    }, final_path)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  [{spec_config.name}] COMPLETE!")
    print(f"  Steps: {global_step:,} | Time: {elapsed/3600:.1f}h")
    print(f"  Best reward: {best_eval_reward:.2f}")
    print(f"  Saved: {final_path.name}")
    print(f"{'='*60}\n")

    wandb.finish()


# =====================================================================
# MAIN — Launch 3 parallel processes
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Train 3 Specialist Agents in Parallel")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Steps per specialist")
    parser.add_argument("--test", action="store_true", help="Test mode: 5000 steps, offline W&B")
    args = parser.parse_args()

    train_steps = 5000 if args.test else args.steps

    print("=" * 65)
    print("  RABIT-PROPFIRM — Ensemble Specialist Training")
    print("  3 Agents × Parallel Processes × Vectorized Envs")
    print(f"  Steps per agent: {train_steps:,}")
    print(f"  VecEnvs per agent: {N_VEC_ENVS}")
    print(f"  CPU threads per agent: {THREADS_PER_PROCESS}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"  VRAM: {vram_gb:.0f} GB")
    print("=" * 65)

    print("\nSpecialists:")
    for i, spec in enumerate(SPECIALIST_CONFIGS):
        print(f"  [{i+1}] {spec.describe()}")
    print()

    # Spawn 3 parallel processes
    mp.set_start_method("spawn", force=True)
    processes = []

    for spec_config in SPECIALIST_CONFIGS:
        p = mp.Process(
            target=train_specialist,
            args=(spec_config, train_steps, args.test, 0),
            name=spec_config.name,
        )
        p.start()
        processes.append(p)
        print(f"  Launched: {spec_config.name} (PID={p.pid})")

    print(f"\n  All 3 specialists launched! Waiting for completion...")
    print(f"  Monitor: https://wandb.ai/nguyenvu16992-/rabit-propfirm-drl")
    print(f"  Group: ensemble_sprint5\n")

    # Wait for all to finish
    for p in processes:
        p.join()
        print(f"  {p.name} finished (exit code: {p.exitcode})")

    print("\n" + "=" * 65)
    print("  ALL SPECIALISTS TRAINING COMPLETE!")
    print("  Models saved in: models_saved/")
    print("    - best_trendagent.pt")
    print("    - best_rangeagent.pt")
    print("    - best_volatilityagent.pt")
    print("=" * 65)


if __name__ == "__main__":
    main()
