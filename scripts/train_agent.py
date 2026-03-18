"""
Train SAC Agent — End-to-end training on SMC + Volume + Price Action features.

Usage: py -3.11 scripts/train_agent.py

Pipeline:
1. Load M5 SMC feature data from data/ Parquet files
2. Load H1 inside bar data for exit rule
3. Normalize features with z-score
4. Train SAC agent with multi-symbol rotation
5. Evaluate and save best model
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


# ─── Config ───
# SMC + Volume + PA features (matching feature_builder.py FEATURE_COLUMNS)
FEATURE_COLS = [
    # Price Action
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "candle_direction",
    "pin_bar_bull", "pin_bar_bear",
    "engulfing_bull", "engulfing_bear",
    "inside_bar",
    # Volume
    "relative_volume", "vol_delta", "climax_vol",
    # SMC Structure
    "swing_high", "swing_low", "swing_trend",
    "bos", "choch",
    # SMC Zones
    "ob_bull_dist", "ob_bear_dist",
    "fvg_bull_active", "fvg_bear_active",
    "liq_above", "liq_below",
    # Time
    "sin_hour", "cos_hour", "sin_dow", "cos_dow",
    # Raw
    "log_return",
]

LOOKBACK = 64
TRAIN_STEPS = 200_000
EVAL_EVERY = 10_000
BATCH_SIZE = 128
LR = 3e-4
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = 50_000


def load_all_features() -> dict:
    """Load M5 SMC features + H1 inside bar data."""
    import polars as pl

    symbol_data = {}
    for feat_file in sorted(DATA_DIR.glob("*_M5_features.parquet")):
        sym_name = feat_file.stem.replace("_M5_features", "")
        df = pl.read_parquet(feat_file)

        # Feature matrix
        available_cols = [c for c in FEATURE_COLS if c in df.columns]
        features = np.column_stack([
            df[col].fill_null(0.0).fill_nan(0.0).to_numpy().astype(np.float32)
            for col in available_cols
        ])

        # OHLCV
        ohlcv = np.column_stack([
            df["open"].to_numpy().astype(np.float32),
            df["high"].to_numpy().astype(np.float32),
            df["low"].to_numpy().astype(np.float32),
            df["close"].to_numpy().astype(np.float32),
            df["tick_volume"].fill_null(0).to_numpy().astype(np.float32)
            if "tick_volume" in df.columns
            else np.zeros(len(df), dtype=np.float32),
        ])

        # Load H1 inside bar timestamps
        ib_path = DATA_DIR / f"{sym_name}_H1_insidebar.parquet"
        h1_ib_times = set()
        if ib_path.exists():
            ib_df = pl.read_parquet(ib_path)
            h1_ib_times = set(
                ib_df.filter(pl.col("inside_bar") == 1.0)["time"].to_list()
            )

        symbol_data[sym_name] = {
            "features": features,
            "ohlcv": ohlcv,
            "n_bars": len(features),
            "n_features": features.shape[1],
            "h1_ib_times": h1_ib_times,
            "times": df["time"].to_list() if "time" in df.columns else [],
        }
        print(f"  {sym_name}: {len(features):,} bars x {features.shape[1]} features"
              f" | H1 inside bars: {len(h1_ib_times)}")

    return symbol_data


def normalize_features(symbol_data: dict) -> tuple[dict, dict]:
    """Z-score normalization across all symbols."""
    all_features = np.concatenate([d["features"] for d in symbol_data.values()], axis=0)
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0) + 1e-8

    for sym in symbol_data:
        symbol_data[sym]["features_norm"] = (symbol_data[sym]["features"] - mean) / std

    norm_state = {"mean": mean.tolist(), "std": std.tolist()}
    norm_path = MODEL_DIR / "normalizer_state.json"
    with open(norm_path, "w") as f:
        json.dump(norm_state, f, indent=2)
    print(f"  Normalizer saved: {norm_path}")

    return symbol_data, norm_state


class SimpleTradeEnv:
    """Trading env with SMC features + H1 inside bar exit rule."""

    def __init__(
        self,
        features: np.ndarray,
        ohlcv: np.ndarray,
        lookback: int = 64,
        initial_balance: float = 100_000.0,
        max_loss_per_trade: float = 0.003,
        h1_ib_times: set | None = None,
        times: list | None = None,
    ) -> None:
        self.features = features
        self.ohlcv = ohlcv
        self.lookback = lookback
        self.initial_balance = initial_balance
        self.max_loss_per_trade = max_loss_per_trade
        self.h1_ib_times = h1_ib_times or set()
        self.times = times or []

        self.n_features = features.shape[1]
        self.obs_dim = lookback * self.n_features + 3
        self.act_dim = 4

        self.reset()

    def reset(self) -> np.ndarray:
        max_start = len(self.features) - self.lookback - 500
        self.idx = np.random.randint(self.lookback, max(self.lookback + 1, max_start))
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.step_count = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_loss = 0.0
        return self._get_obs()

    def _check_h1_inside_bar(self) -> bool:
        """Check if current M5 bar falls within an H1 inside bar period."""
        if not self.times or self.idx >= len(self.times):
            return False
        current_time = self.times[self.idx]
        # Round down to H1
        h1_time = current_time.replace(minute=0, second=0, microsecond=0)
        return h1_time in self.h1_ib_times

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        confidence = float(np.clip(action[0], -1, 1))
        risk_frac = float(np.clip(action[1], 0, 1))

        close_now = float(self.ohlcv[self.idx, 3])

        reward = 0.0
        done = False

        # === H1 INSIDE BAR EXIT RULE ===
        if self.position != 0 and self._check_h1_inside_bar():
            pnl = self.unrealized_pnl
            self.balance += pnl
            if pnl > 0:
                reward = 0.5
                self.winning_trades += 1
            else:
                reward = -0.2  # Lighter penalty — forced exit, not a mistake
                self.daily_loss += abs(pnl) / self.initial_balance
            self.total_trades += 1
            self.position = 0.0
            self.unrealized_pnl = 0.0
            # Small bonus for obeying exit rule
            reward += 0.1

        # Update unrealized PnL
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

        # Don't open new positions during H1 inside bar
        is_h1_ib = self._check_h1_inside_bar()

        if (abs(confidence) > 0.3
                and self.position == 0
                and self.daily_loss < 0.03
                and not is_h1_ib):
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

        self.idx += 1
        self.step_count += 1

        if self.idx >= len(self.features) - 1:
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

    def _get_obs(self) -> np.ndarray:
        start = max(0, self.idx - self.lookback)
        window = self.features[start:self.idx]

        if len(window) < self.lookback:
            pad = np.zeros((self.lookback - len(window), self.n_features), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)

        flat = window.flatten()
        balance_ratio = self.balance / self.initial_balance
        extra = np.array([balance_ratio, self.position, self.unrealized_pnl / 1000], dtype=np.float32)

        return np.concatenate([flat, extra]).astype(np.float32)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None:
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.FloatTensor(self.obs[idxs]),
            "act": torch.FloatTensor(self.act[idxs]),
            "rew": torch.FloatTensor(self.rew[idxs]),
            "next_obs": torch.FloatTensor(self.next_obs[idxs]),
            "done": torch.FloatTensor(self.done[idxs]),
        }


class SACAgent(torch.nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        )
        self.mu = torch.nn.Linear(hidden, act_dim)
        self.log_std = torch.nn.Linear(hidden, act_dim)

        self.q1 = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + act_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.q2 = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + act_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.q1_target = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + act_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.q2_target = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + act_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.log_alpha = torch.nn.Parameter(torch.zeros(1))
        self.target_entropy = -act_dim

    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            h = self.actor(obs_t)
            mu = self.mu(h)
            if deterministic:
                return torch.tanh(mu).squeeze(0).numpy()
            log_std = torch.clamp(self.log_std(h), -20, 2)
            std = log_std.exp()
            z = mu + std * torch.randn_like(std)
            return torch.tanh(z).squeeze(0).numpy()

    def sample_action(self, obs_batch):
        h = self.actor(obs_batch)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), -20, 2)
        std = log_std.exp()
        z = mu + std * torch.randn_like(std)
        action = torch.tanh(z)
        log_prob = (
            torch.distributions.Normal(mu, std).log_prob(z)
            - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(-1, keepdim=True)
        return action, log_prob


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


def main():
    print("=" * 60)
    print("  RABIT-PROPFIRM -- SAC Training")
    print("  Features: SMC + Volume + Price Action")
    print("=" * 60)

    print("\n[1/4] Loading M5 SMC feature data...")
    symbol_data = load_all_features()

    print("\n[2/4] Normalizing features...")
    symbol_data, norm_state = normalize_features(symbol_data)

    symbols = list(symbol_data.keys())
    print(f"\n  Training on: {symbols}")

    print("\n[3/4] Initializing agent...")
    first_sym = symbols[0]
    env = SimpleTradeEnv(
        features=symbol_data[first_sym]["features_norm"],
        ohlcv=symbol_data[first_sym]["ohlcv"],
        lookback=LOOKBACK,
        h1_ib_times=symbol_data[first_sym]["h1_ib_times"],
        times=symbol_data[first_sym]["times"],
    )

    agent = SACAgent(env.obs_dim, env.act_dim)
    buffer = ReplayBuffer(BUFFER_SIZE, env.obs_dim, env.act_dim)

    actor_opt = torch.optim.Adam(
        list(agent.actor.parameters()) + list(agent.mu.parameters()) + list(agent.log_std.parameters()), lr=LR
    )
    critic_opt = torch.optim.Adam(list(agent.q1.parameters()) + list(agent.q2.parameters()), lr=LR)
    alpha_opt = torch.optim.Adam([agent.log_alpha], lr=LR)

    print(f"  Obs dim: {env.obs_dim} ({LOOKBACK} x {env.n_features} + 3)")
    print(f"  Act dim: {env.act_dim}")
    print(f"  Agent params: {sum(p.numel() for p in agent.parameters()):,}")

    print(f"\n[4/4] Training for {TRAIN_STEPS:,} steps...")
    print("-" * 60)

    obs = env.reset()
    episode_count = 0
    best_eval_reward = -999
    start_time = time.time()

    for step in range(1, TRAIN_STEPS + 1):
        if buffer.size < 1000:
            action = np.random.uniform(-1, 1, size=env.act_dim).astype(np.float32)
        else:
            action = agent.get_action(obs)

        next_obs, reward, done, info = env.step(action)
        buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs

        if done:
            episode_count += 1
            sym = symbols[episode_count % len(symbols)]
            env = SimpleTradeEnv(
                features=symbol_data[sym]["features_norm"],
                ohlcv=symbol_data[sym]["ohlcv"],
                lookback=LOOKBACK,
                h1_ib_times=symbol_data[sym]["h1_ib_times"],
                times=symbol_data[sym]["times"],
            )
            obs = env.reset()

        if buffer.size >= 1000 and step % 4 == 0:
            batch = buffer.sample(BATCH_SIZE)

            with torch.no_grad():
                next_a, next_lp = agent.sample_action(batch["next_obs"])
                sa_next = torch.cat([batch["next_obs"], next_a], dim=-1)
                q_next = torch.min(agent.q1_target(sa_next), agent.q2_target(sa_next)) - agent.log_alpha.exp() * next_lp
                target_q = batch["rew"].unsqueeze(-1) + GAMMA * (1 - batch["done"].unsqueeze(-1)) * q_next

            sa = torch.cat([batch["obs"], batch["act"]], dim=-1)
            critic_loss = ((agent.q1(sa) - target_q)**2).mean() + ((agent.q2(sa) - target_q)**2).mean()
            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(agent.q1.parameters()) + list(agent.q2.parameters()), 1.0)
            critic_opt.step()

            new_a, lp = agent.sample_action(batch["obs"])
            sa_new = torch.cat([batch["obs"], new_a], dim=-1)
            actor_loss = (agent.log_alpha.exp().detach() * lp - torch.min(agent.q1(sa_new), agent.q2(sa_new))).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(agent.actor.parameters()) + list(agent.mu.parameters()) + list(agent.log_std.parameters()), 1.0
            )
            actor_opt.step()

            alpha_loss = -(agent.log_alpha.exp() * (lp.detach() + agent.target_entropy)).mean()
            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()

            soft_update(agent.q1_target, agent.q1, TAU)
            soft_update(agent.q2_target, agent.q2, TAU)

        if step % EVAL_EVERY == 0:
            eval_rewards, eval_trades, eval_wrs = [], [], []
            for sym in symbols:
                eval_env = SimpleTradeEnv(
                    features=symbol_data[sym]["features_norm"],
                    ohlcv=symbol_data[sym]["ohlcv"],
                    lookback=LOOKBACK,
                    h1_ib_times=symbol_data[sym]["h1_ib_times"],
                    times=symbol_data[sym]["times"],
                )
                e_obs = eval_env.reset()
                e_total = 0.0
                for _ in range(480):
                    e_act = agent.get_action(e_obs, deterministic=True)
                    e_obs, e_rew, e_done, e_info = eval_env.step(e_act)
                    e_total += e_rew
                    if e_done:
                        break
                eval_rewards.append(e_total)
                eval_trades.append(e_info["total_trades"])
                eval_wrs.append(e_info["win_rate"])

            mean_reward = np.mean(eval_rewards)
            elapsed = time.time() - start_time
            sps = step / elapsed
            print(
                f"  Step {step:>7,} | Reward: {mean_reward:>7.2f} | "
                f"Trades: {np.mean(eval_trades):>4.1f} | WR: {np.mean(eval_wrs):>5.1%} | "
                f"Alpha: {agent.log_alpha.exp().item():.3f} | {sps:.0f} steps/s"
            )

            if mean_reward > best_eval_reward:
                best_eval_reward = mean_reward
                torch.save({
                    "agent_state": agent.state_dict(),
                    "obs_dim": env.obs_dim,
                    "act_dim": env.act_dim,
                    "eval_reward": mean_reward,
                    "step": step,
                    "norm_state": norm_state,
                    "feature_cols": FEATURE_COLS,
                }, MODEL_DIR / "best_model.pt")
                print(f"    >> New best! Saved to best_model.pt")

    torch.save({
        "agent_state": agent.state_dict(),
        "obs_dim": env.obs_dim, "act_dim": env.act_dim,
        "step": TRAIN_STEPS, "norm_state": norm_state,
        "feature_cols": FEATURE_COLS,
    }, MODEL_DIR / "final_model.pt")

    print("\n" + "=" * 60)
    print(f"  TRAINING COMPLETE!")
    print(f"  Best eval reward: {best_eval_reward:.2f}")
    print(f"  Models saved to: {MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
