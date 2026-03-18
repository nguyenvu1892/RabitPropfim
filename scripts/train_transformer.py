"""
Train SAC Agent with Transformer Backbone — Sprint 3.6

Key differences from train_agent.py (MLP baseline):
1. Multi-TF observations: M5 (64 bars) + H1 (24 bars) + H4 (30 bars)
2. TransformerFeatureExtractor → 388-dim global_state
3. SACTransformerActor/Critic replace flat MLP networks

H1/H4 features computed on-the-fly from raw OHLCV (same 28 features as M5
where applicable, remaining features zero-filled).

Usage: py -3.11 scripts/train_transformer.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "rabit_propfirm_drl"))

DATA_DIR = project_root / "data"
MODEL_DIR = project_root / "models_saved"
MODEL_DIR.mkdir(exist_ok=True)

# Feature columns (matching M5 feature_builder.py)
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

LOOKBACK_M5 = 64   # M5 window
LOOKBACK_H1 = 24   # H1 window (1 day)
LOOKBACK_H4 = 30   # H4 window (5 days)
N_FEATURES = 28     # Feature dimension

TRAIN_STEPS = 200_000
EVAL_EVERY = 10_000
BATCH_SIZE = 64
LR_ACTOR = 1e-4     # Lower LR for Transformer (more params)
LR_CRITIC = 3e-4
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = 50_000


def build_htf_features(ohlcv: np.ndarray, n_features: int = 28) -> np.ndarray:
    """
    Build simplified features from raw OHLCV for H1/H4.
    Uses same feature structure as M5 (28 dims) for compatibility.
    Features not available from pure OHLCV are zero-filled.
    """
    n = len(ohlcv)
    features = np.zeros((n, n_features), dtype=np.float32)

    o, h, l, c = ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3]
    body = np.abs(c - o) + 1e-8
    full_range = h - l + 1e-8

    # Price Action (idx 0-8)
    features[:, 0] = body / full_range                           # body_ratio
    features[:, 1] = np.where(c >= o, (h - c), (h - o)) / full_range  # upper_wick
    features[:, 2] = np.where(c >= o, (o - l), (c - l)) / full_range  # lower_wick
    features[:, 3] = np.where(c >= o, 1.0, -1.0)               # candle_direction
    # pin_bar_bull (4), pin_bar_bear (5)
    features[:, 4] = ((features[:, 2] > 0.6) & (features[:, 0] < 0.3)).astype(np.float32)
    features[:, 5] = ((features[:, 1] > 0.6) & (features[:, 0] < 0.3)).astype(np.float32)
    # engulfing (6, 7): compare with previous bar
    for i in range(1, n):
        if c[i] > o[i] and body[i] > body[i-1] and c[i] > o[i-1] and o[i] < c[i-1]:
            features[i, 6] = 1.0  # engulfing_bull
        if c[i] < o[i] and body[i] > body[i-1] and c[i] < o[i-1] and o[i] > c[i-1]:
            features[i, 7] = 1.0  # engulfing_bear
    # inside_bar (8)
    for i in range(1, n):
        if h[i] <= h[i-1] and l[i] >= l[i-1]:
            features[i, 8] = 1.0

    # Volume (idx 9-11): use tick_volume if available
    if ohlcv.shape[1] > 4:
        vol = ohlcv[:, 4]
        mean_vol = np.convolve(vol, np.ones(20)/20, mode='same') + 1e-8
        features[:, 9] = vol / mean_vol   # relative_volume
    # vol_delta (10), climax_vol (11) stay at 0 for HTF

    # Swing structure (idx 12-16): simplified from price
    for i in range(2, n - 2):
        if h[i] >= h[i-1] and h[i] >= h[i-2] and h[i] >= h[i+1] and h[i] >= h[i+2]:
            features[i, 12] = 1.0  # swing_high
        if l[i] <= l[i-1] and l[i] <= l[i-2] and l[i] <= l[i+1] and l[i] <= l[i+2]:
            features[i, 13] = 1.0  # swing_low
    # swing_trend (14): rolling direction
    for i in range(5, n):
        trend = (c[i] - c[i-5]) / (c[i-5] + 1e-8)
        features[i, 14] = np.clip(trend * 100, -1, 1)

    # BOS/CHoCH (15, 16), OB/FVG/LIQ (17-22): stay at 0 for HTF (complex calc)

    # Time features (idx 23-26): compute from index (approximation)
    import math
    for i in range(n):
        hour_approx = (i % 24) if n_features == 28 else 0
        features[i, 23] = math.sin(2 * math.pi * hour_approx / 24)  # sin_hour
        features[i, 24] = math.cos(2 * math.pi * hour_approx / 24)  # cos_hour
        dow_approx = (i // 24) % 5
        features[i, 25] = math.sin(2 * math.pi * dow_approx / 5)    # sin_dow
        features[i, 26] = math.cos(2 * math.pi * dow_approx / 5)    # cos_dow

    # log_return (idx 27)
    features[1:, 27] = np.log(c[1:] / (c[:-1] + 1e-8))

    return features


def load_all_data() -> dict:
    """Load M5 features + H1/H4 raw OHLCV for all symbols."""
    import polars as pl

    symbol_data = {}
    for feat_file in sorted(DATA_DIR.glob("*_M5_features.parquet")):
        sym_name = feat_file.stem.replace("_M5_features", "")

        # M5 features (pre-computed)
        df_m5 = pl.read_parquet(feat_file)
        available_cols = [c for c in FEATURE_COLS if c in df_m5.columns]
        m5_features = np.column_stack([
            df_m5[col].fill_null(0.0).fill_nan(0.0).to_numpy().astype(np.float32)
            for col in available_cols
        ])
        # Pad to 28 if fewer columns available
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

        # H1 data
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

        # H4 data
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

        # H1 inside bar timestamps
        ib_path = DATA_DIR / f"{sym_name}_H1_insidebar.parquet"
        h1_ib_times = set()
        if ib_path.exists():
            ib_df = pl.read_parquet(ib_path)
            h1_ib_times = set(
                ib_df.filter(pl.col("inside_bar") == 1.0)["time"].to_list()
            )

        symbol_data[sym_name] = {
            "m5_features": m5_features,
            "m5_ohlcv": m5_ohlcv,
            "m5_times": m5_times,
            "h1_features": h1_features,
            "h4_features": h4_features,
            "h1_ib_times": h1_ib_times,
            "n_m5": len(m5_features),
            "n_h1": len(h1_features),
            "n_h4": len(h4_features),
        }
        print(f"  {sym_name}: M5={len(m5_features):,} | "
              f"H1={len(h1_features):,} | H4={len(h4_features):,}")

    return symbol_data


def normalize_all(symbol_data: dict) -> dict:
    """Z-score normalize M5, H1, H4 features separately."""
    for tf_key in ["m5_features", "h1_features", "h4_features"]:
        all_feat = np.concatenate([d[tf_key] for d in symbol_data.values()], axis=0)
        mean = np.mean(all_feat, axis=0)
        std = np.std(all_feat, axis=0) + 1e-8
        for sym in symbol_data:
            symbol_data[sym][f"{tf_key}_norm"] = (
                (symbol_data[sym][tf_key] - mean) / std
            ).astype(np.float32)

    # Save normalizer
    norm = {}
    for tf_key in ["m5_features", "h1_features", "h4_features"]:
        all_feat = np.concatenate([d[tf_key] for d in symbol_data.values()], axis=0)
        norm[tf_key] = {
            "mean": np.mean(all_feat, axis=0).tolist(),
            "std": (np.std(all_feat, axis=0) + 1e-8).tolist(),
        }
    with open(MODEL_DIR / "normalizer_transformer.json", "w") as f:
        json.dump(norm, f, indent=2)

    return symbol_data


class MultiTFTradeEnv:
    """Trading env with multi-timeframe structured observations."""

    def __init__(
        self,
        m5_features: np.ndarray,
        m5_ohlcv: np.ndarray,
        h1_features: np.ndarray,
        h4_features: np.ndarray,
        m5_times: list,
        h1_ib_times: set,
        n_m5: int,
        n_h1: int,
        n_h4: int,
        initial_balance: float = 100_000.0,
        max_loss_per_trade: float = 0.003,
    ) -> None:
        self.m5_feat = m5_features
        self.m5_ohlcv = m5_ohlcv
        self.h1_feat = h1_features
        self.h4_feat = h4_features
        self.m5_times = m5_times
        self.h1_ib_times = h1_ib_times
        self.n_m5 = n_m5
        self.n_h1 = n_h1
        self.n_h4 = n_h4
        self.initial_balance = initial_balance
        self.max_loss_per_trade = max_loss_per_trade

        # Ratio: how many M5 bars per H1/H4 bar
        self.m5_per_h1 = max(1, n_m5 // max(n_h1, 1))
        self.m5_per_h4 = max(1, n_m5 // max(n_h4, 1))

        self.act_dim = 4
        self.reset()

    def reset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_start = self.n_m5 - LOOKBACK_M5 - 500
        self.m5_idx = np.random.randint(LOOKBACK_M5, max(LOOKBACK_M5 + 1, max_start))
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.step_count = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_loss = 0.0
        return self._get_obs()

    def _get_m5_window(self) -> np.ndarray:
        start = max(0, self.m5_idx - LOOKBACK_M5)
        window = self.m5_feat[start:self.m5_idx]
        if len(window) < LOOKBACK_M5:
            pad = np.zeros((LOOKBACK_M5 - len(window), N_FEATURES), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)
        return window  # (64, 28)

    def _get_h1_window(self) -> np.ndarray:
        h1_idx = min(self.m5_idx // self.m5_per_h1, self.n_h1 - 1)
        start = max(0, h1_idx - LOOKBACK_H1)
        window = self.h1_feat[start:h1_idx]
        if len(window) < LOOKBACK_H1:
            pad = np.zeros((LOOKBACK_H1 - len(window), N_FEATURES), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)
        return window  # (24, 28)

    def _get_h4_window(self) -> np.ndarray:
        h4_idx = min(self.m5_idx // self.m5_per_h4, self.n_h4 - 1)
        start = max(0, h4_idx - LOOKBACK_H4)
        window = self.h4_feat[start:h4_idx]
        if len(window) < LOOKBACK_H4:
            pad = np.zeros((LOOKBACK_H4 - len(window), N_FEATURES), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)
        return window  # (30, 28)

    def _get_obs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._get_m5_window(), self._get_h1_window(), self._get_h4_window()

    def _check_h1_inside_bar(self) -> bool:
        if not self.m5_times or self.m5_idx >= len(self.m5_times):
            return False
        ct = self.m5_times[self.m5_idx]
        if hasattr(ct, 'replace'):
            h1_time = ct.replace(minute=0, second=0, microsecond=0)
            return h1_time in self.h1_ib_times
        return False

    def step(self, action: np.ndarray):
        confidence = float(np.clip(action[0], -1, 1))
        risk_frac = float(np.clip(action[1], 0, 1))
        close_now = float(self.m5_ohlcv[self.m5_idx, 3])
        reward = 0.0
        done = False

        # H1 inside bar exit
        if self.position != 0 and self._check_h1_inside_bar():
            pnl = self.unrealized_pnl
            self.balance += pnl
            if pnl > 0:
                reward = 0.5
                self.winning_trades += 1
            else:
                reward = -0.2
                self.daily_loss += abs(pnl) / self.initial_balance
            self.total_trades += 1
            self.position = 0.0
            self.unrealized_pnl = 0.0
            reward += 0.1

        # Update PnL
        if self.position != 0:
            price_change = (close_now - self.entry_price) / self.entry_price
            self.unrealized_pnl = self.position * price_change * self.balance * risk_frac

            max_loss = self.balance * self.max_loss_per_trade
            if self.unrealized_pnl < -max_loss:
                self.balance += self.unrealized_pnl
                self.daily_loss += abs(self.unrealized_pnl) / self.initial_balance
                reward = -1.0
                self.total_trades += 1
                self.position = 0.0
                self.unrealized_pnl = 0.0
            elif self.unrealized_pnl > self.balance * 0.01:
                self.balance += self.unrealized_pnl
                reward = 2.0
                self.total_trades += 1
                self.winning_trades += 1
                self.position = 0.0
                self.unrealized_pnl = 0.0

        is_h1_ib = self._check_h1_inside_bar()

        if abs(confidence) > 0.3 and self.position == 0 and self.daily_loss < 0.03 and not is_h1_ib:
            self.position = 1.0 if confidence > 0 else -1.0
            self.entry_price = close_now
        elif abs(confidence) < 0.1 and self.position != 0:
            pnl = self.unrealized_pnl
            self.balance += pnl
            if pnl > 0:
                reward = 0.5
                self.winning_trades += 1
            else:
                reward = -0.3
                self.daily_loss += abs(pnl) / self.initial_balance
            self.total_trades += 1
            self.position = 0.0
            self.unrealized_pnl = 0.0

        if self.position != 0 and self.unrealized_pnl > 0:
            reward += 0.01

        self.m5_idx += 1
        self.step_count += 1

        if self.m5_idx >= self.n_m5 - 1:
            done = True
        if self.step_count >= 480:
            done = True
        if self.daily_loss >= 0.03:
            done = True
        if self.balance < self.initial_balance * 0.95:
            done = True

        info = {
            "balance": self.balance,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(self.total_trades, 1),
        }
        return self._get_obs(), reward, done, info


class MultiTFReplayBuffer:
    """Replay buffer storing structured multi-TF observations."""

    def __init__(self, capacity: int, act_dim: int = 4) -> None:
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.act_dim = act_dim

        # Preallocate arrays
        self.m5 = np.zeros((capacity, LOOKBACK_M5, N_FEATURES), dtype=np.float32)
        self.h1 = np.zeros((capacity, LOOKBACK_H1, N_FEATURES), dtype=np.float32)
        self.h4 = np.zeros((capacity, LOOKBACK_H4, N_FEATURES), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_m5 = np.zeros((capacity, LOOKBACK_M5, N_FEATURES), dtype=np.float32)
        self.next_h1 = np.zeros((capacity, LOOKBACK_H1, N_FEATURES), dtype=np.float32)
        self.next_h4 = np.zeros((capacity, LOOKBACK_H4, N_FEATURES), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done):
        m5, h1, h4 = obs
        nm5, nh1, nh4 = next_obs
        self.m5[self.ptr] = m5
        self.h1[self.ptr] = h1
        self.h4[self.ptr] = h4
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_m5[self.ptr] = nm5
        self.next_h1[self.ptr] = nh1
        self.next_h4[self.ptr] = nh4
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device: torch.device = torch.device("cpu")):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "m5": torch.FloatTensor(self.m5[idxs]).to(device),
            "h1": torch.FloatTensor(self.h1[idxs]).to(device),
            "h4": torch.FloatTensor(self.h4[idxs]).to(device),
            "act": torch.FloatTensor(self.act[idxs]).to(device),
            "rew": torch.FloatTensor(self.rew[idxs]).to(device),
            "next_m5": torch.FloatTensor(self.next_m5[idxs]).to(device),
            "next_h1": torch.FloatTensor(self.next_h1[idxs]).to(device),
            "next_h4": torch.FloatTensor(self.next_h4[idxs]).to(device),
            "done": torch.FloatTensor(self.done[idxs]).to(device),
        }


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


def main():
    from agents.sac_policy import SACTransformerActor, SACTransformerCritic

    # ══════════════════════════════════════════════════════
    # FIX #1: Device declaration — CUDA is MANDATORY
    # ══════════════════════════════════════════════════════
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> BAT DAU TRAIN. DANG SU DUNG DEVICE: {device.type.upper()}")

    # FIX #2: RuntimeError guard — KHONG CHO PHEP train tren CPU
    if device.type != "cuda":
        raise RuntimeError(
            "[X] KHONG CO GPU CUDA! Transformer 3.5M params KHONG THE train tren CPU.\n"
            "   Yeu cau: NVIDIA GPU + CUDA toolkit.\n"
            "   Kiem tra: py -3.11 -c \"import torch; print(torch.cuda.is_available())\""
        )

    print("=" * 65)
    print("  RABIT-PROPFIRM -- SAC Training (Transformer Backbone)")
    print("  Sprint 3.6: TransformerSMC + CrossAttentionMTF + RegimeDetector")
    print(f"  Device: {device} | CUDA: {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    print("\n[1/4] Loading multi-TF data (M5 + H1 + H4)...")
    symbol_data = load_all_data()

    print("\n[2/4] Normalizing features per timeframe...")
    symbol_data = normalize_all(symbol_data)

    symbols = list(symbol_data.keys())
    print(f"\n  Training on: {symbols}")

    print("\n[3/4] Initializing Transformer SAC Agent...")

    # FIX #3: All models pushed to GPU with .to(device)
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
    target_entropy = -4  # -action_dim

    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"  Actor params:  {actor_params:,}")
    print(f"  Critic params: {critic_params:,} (x2 for twin)")
    print(f"  Total params:  {actor_params + critic_params * 2:,}")
    print(f"  Device:        {device.type.upper()} [OK]")

    actor_opt = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)
    alpha_opt = torch.optim.Adam([log_alpha], lr=LR_CRITIC)

    # Init env + buffer
    first_sym = symbols[0]
    sd = symbol_data[first_sym]
    env = MultiTFTradeEnv(
        m5_features=sd["m5_features_norm"], m5_ohlcv=sd["m5_ohlcv"],
        h1_features=sd["h1_features_norm"], h4_features=sd["h4_features_norm"],
        m5_times=sd["m5_times"], h1_ib_times=sd["h1_ib_times"],
        n_m5=sd["n_m5"], n_h1=sd["n_h1"], n_h4=sd["n_h4"],
    )
    buffer = MultiTFReplayBuffer(BUFFER_SIZE)

    print(f"\n[4/4] Training for {TRAIN_STEPS:,} steps...")
    print("-" * 65)

    obs = env.reset()
    episode_count = 0
    best_eval_reward = -999
    start_time = time.time()

    for step in range(1, TRAIN_STEPS + 1):
        # Exploration / exploitation
        if buffer.size < 2000:
            action = np.random.uniform(-1, 1, size=4).astype(np.float32)
        else:
            with torch.no_grad():
                # FIX #4: Push tensors to GPU before feeding to model
                m5_t = torch.FloatTensor(obs[0]).unsqueeze(0).to(device)
                h1_t = torch.FloatTensor(obs[1]).unsqueeze(0).to(device)
                h4_t = torch.FloatTensor(obs[2]).unsqueeze(0).to(device)
                act_t, _ = actor(m5_t, h1_t, h4_t)
                action = act_t.squeeze(0).cpu().numpy()

        next_obs, reward, done, info = env.step(action)
        buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs

        if done:
            episode_count += 1
            sym = symbols[episode_count % len(symbols)]
            sd = symbol_data[sym]
            env = MultiTFTradeEnv(
                m5_features=sd["m5_features_norm"], m5_ohlcv=sd["m5_ohlcv"],
                h1_features=sd["h1_features_norm"], h4_features=sd["h4_features_norm"],
                m5_times=sd["m5_times"], h1_ib_times=sd["h1_ib_times"],
                n_m5=sd["n_m5"], n_h1=sd["n_h1"], n_h4=sd["n_h4"],
            )
            obs = env.reset()

        # Training update (every 4 steps)
        if buffer.size >= 2000 and step % 4 == 0:
            batch = buffer.sample(BATCH_SIZE, device=device)
            alpha = log_alpha.exp().detach()

            # ── Critic Update ──
            with torch.no_grad():
                next_a, next_lp = actor(batch["next_m5"], batch["next_h1"], batch["next_h4"])
                q1_next, q2_next = critic_target(
                    batch["next_m5"], batch["next_h1"], batch["next_h4"], next_a
                )
                q_next = torch.min(q1_next, q2_next) - alpha * next_lp
                target_q = batch["rew"].unsqueeze(-1) + GAMMA * (1 - batch["done"].unsqueeze(-1)) * q_next

            q1, q2 = critic(batch["m5"], batch["h1"], batch["h4"], batch["act"])
            critic_loss = ((q1 - target_q)**2).mean() + ((q2 - target_q)**2).mean()
            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()

            # ── Actor Update ──
            new_a, lp = actor(batch["m5"], batch["h1"], batch["h4"])
            q1_new = critic(batch["m5"], batch["h1"], batch["h4"], new_a)[0]
            actor_loss = (alpha * lp - q1_new).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_opt.step()

            # ── Alpha Update ──
            alpha_loss = -(log_alpha.exp() * (lp.detach() + target_entropy)).mean()
            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()

            # ── Soft Target Update ──
            soft_update(critic_target, critic, TAU)

        # Evaluation
        if step % EVAL_EVERY == 0:
            actor.eval()
            eval_rewards, eval_trades, eval_wrs = [], [], []
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
                eval_rewards.append(e_total)
                eval_trades.append(e_info["total_trades"])
                eval_wrs.append(e_info["win_rate"])
            actor.train()

            mean_reward = np.mean(eval_rewards)
            elapsed = time.time() - start_time
            sps = step / elapsed
            print(
                f"  Step {step:>7,} | Reward: {mean_reward:>7.2f} | "
                f"Trades: {np.mean(eval_trades):>4.1f} | WR: {np.mean(eval_wrs):>5.1%} | "
                f"Alpha: {log_alpha.exp().item():.3f} | {sps:.0f} sps"
            )

            if mean_reward > best_eval_reward:
                best_eval_reward = mean_reward
                torch.save({
                    "actor_state": actor.state_dict(),
                    "critic_state": critic.state_dict(),
                    "model_type": "transformer",
                    "eval_reward": mean_reward,
                    "step": step,
                    "n_features": N_FEATURES,
                    "embed_dim": 128,
                }, MODEL_DIR / "best_transformer.pt")
                print(f"    >> New best! Saved to best_transformer.pt")

    # Save final
    torch.save({
        "actor_state": actor.state_dict(),
        "critic_state": critic.state_dict(),
        "model_type": "transformer",
        "step": TRAIN_STEPS,
        "n_features": N_FEATURES,
        "embed_dim": 128,
    }, MODEL_DIR / "final_transformer.pt")

    print("\n" + "=" * 65)
    print("  TRANSFORMER TRAINING COMPLETE!")
    print(f"  Best eval reward: {best_eval_reward:.2f}")
    print(f"  Actor params:  {actor_params:,}")
    print(f"  Critic params: {critic_params * 2:,}")
    print(f"  Models saved: {MODEL_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
