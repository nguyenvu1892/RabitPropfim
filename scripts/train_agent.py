"""
Train SAC Agent — End-to-end training on historical data.

Usage: py -3.11 scripts/train_agent.py

Pipeline:
1. Load M15 feature data from data/ Parquet files
2. Normalize features with RunningNormalizer
3. Create PropFirmTradingEnv with curriculum stages
4. Train SAC agent with PER buffer
5. Evaluate and register best model
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
FEATURE_COLS = [
    "log_return", "atr", "vol_ratio", "rsi",
    "body_ratio", "rvol", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
]
LOOKBACK = 64        # Bars of history the model sees
TRAIN_STEPS = 200_000 # Extended training for convergence
EVAL_EVERY = 10_000  # Evaluate every N steps
BATCH_SIZE = 128
LR = 3e-4
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = 50_000


def load_all_features() -> dict[str, np.ndarray]:
    """Load M15 feature data for all symbols."""
    import polars as pl

    symbol_data = {}
    for feat_file in sorted(DATA_DIR.glob("*_M15_features.parquet")):
        sym_name = feat_file.stem.replace("_M15_features", "")
        df = pl.read_parquet(feat_file)

        # Extract feature columns as numpy
        features = np.column_stack([
            df[col].to_numpy().astype(np.float32) for col in FEATURE_COLS
        ])

        # Add OHLCV for the environment
        ohlcv = np.column_stack([
            df["open"].to_numpy().astype(np.float32),
            df["high"].to_numpy().astype(np.float32),
            df["low"].to_numpy().astype(np.float32),
            df["close"].to_numpy().astype(np.float32),
            df["tick_volume"].to_numpy().astype(np.float32),
        ])

        symbol_data[sym_name] = {
            "features": features,
            "ohlcv": ohlcv,
            "n_bars": len(features),
        }
        print(f"  {sym_name}: {len(features):,} bars x {features.shape[1]} features")

    return symbol_data


def normalize_features(symbol_data: dict) -> tuple[dict, dict]:
    """Normalize all features using simple z-score."""
    # Compute global mean/std across all symbols
    all_features = np.concatenate([d["features"] for d in symbol_data.values()], axis=0)
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0) + 1e-8

    # Normalize
    for sym in symbol_data:
        symbol_data[sym]["features_norm"] = (symbol_data[sym]["features"] - mean) / std

    norm_state = {"mean": mean.tolist(), "std": std.tolist()}

    # Save normalizer state
    norm_path = MODEL_DIR / "normalizer_state.json"
    with open(norm_path, "w") as f:
        json.dump(norm_state, f, indent=2)
    print(f"  Normalizer saved: {norm_path}")

    return symbol_data, norm_state


class SimpleTradeEnv:
    """
    Simplified trading environment for training.

    Operates on M15 feature data, simulates position management
    with SL based on 0.3% max loss per trade.
    """

    def __init__(
        self,
        features: np.ndarray,
        ohlcv: np.ndarray,
        lookback: int = 64,
        initial_balance: float = 100_000.0,
        max_loss_per_trade: float = 0.003,
    ) -> None:
        self.features = features
        self.ohlcv = ohlcv
        self.lookback = lookback
        self.initial_balance = initial_balance
        self.max_loss_per_trade = max_loss_per_trade

        # State dimensions
        self.n_features = features.shape[1]
        self.obs_dim = lookback * self.n_features + 3  # + balance_ratio, position, unrealized
        self.act_dim = 4  # confidence, risk_frac, sl_mult, tp_mult

        self.reset()

    def reset(self) -> np.ndarray:
        # Random start (leave room for lookback + episode)
        max_start = len(self.features) - self.lookback - 500
        self.idx = np.random.randint(self.lookback, max(self.lookback + 1, max_start))
        self.balance = self.initial_balance
        self.position = 0.0  # +1 long, -1 short, 0 flat
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.step_count = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_loss = 0.0
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        confidence = float(np.clip(action[0], -1, 1))
        risk_frac = float(np.clip(action[1], 0, 1))

        close_now = float(self.ohlcv[self.idx, 3])
        close_prev = float(self.ohlcv[self.idx - 1, 3])

        reward = 0.0
        done = False

        # Update unrealized PnL if in position
        if self.position != 0:
            price_change = (close_now - self.entry_price) / self.entry_price
            self.unrealized_pnl = self.position * price_change * self.balance * risk_frac

            # Check SL (0.3% max loss)
            max_loss = self.balance * self.max_loss_per_trade
            if self.unrealized_pnl < -max_loss:
                # Force close at SL
                self.balance += self.unrealized_pnl
                self.daily_loss += abs(self.unrealized_pnl) / self.initial_balance
                reward = -1.0  # Penalty for SL hit
                self.total_trades += 1
                self.position = 0.0
                self.unrealized_pnl = 0.0

            # Take profit (1% target)
            elif self.unrealized_pnl > self.balance * 0.01:
                self.balance += self.unrealized_pnl
                reward = 2.0  # Bonus for TP
                self.total_trades += 1
                self.winning_trades += 1
                self.position = 0.0
                self.unrealized_pnl = 0.0

        # Action gating
        if abs(confidence) > 0.3 and self.position == 0 and self.daily_loss < 0.03:
            # Open new position
            self.position = 1.0 if confidence > 0 else -1.0
            self.entry_price = close_now
            reward += 0.0  # Neutral for opening

        elif abs(confidence) < 0.1 and self.position != 0:
            # Close position (low confidence)
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

        # Small reward for holding profitable position
        if self.position != 0 and self.unrealized_pnl > 0:
            reward += 0.01

        # Advance
        self.idx += 1
        self.step_count += 1

        # Episode termination
        if self.idx >= len(self.features) - 1:
            done = True
        if self.step_count >= 480:  # ~5 trading days of M15
            done = True
        if self.daily_loss >= 0.03:
            done = True  # Daily loss cooldown
        if self.balance < self.initial_balance * 0.95:
            done = True  # 5% DD

        info = {
            "balance": self.balance,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(self.total_trades, 1),
        }

        return self._get_obs(), reward, done, info

    def _get_obs(self) -> np.ndarray:
        # Feature window
        start = max(0, self.idx - self.lookback)
        window = self.features[start:self.idx]

        # Pad if needed
        if len(window) < self.lookback:
            pad = np.zeros((self.lookback - len(window), self.n_features), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)

        flat = window.flatten()

        # Account state
        balance_ratio = self.balance / self.initial_balance
        extra = np.array([balance_ratio, self.position, self.unrealized_pnl / 1000], dtype=np.float32)

        return np.concatenate([flat, extra]).astype(np.float32)


class ReplayBuffer:
    """Simple experience replay buffer."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None:
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

    def add(self, obs: np.ndarray, act: np.ndarray, rew: float,
            next_obs: np.ndarray, done: bool) -> None:
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.FloatTensor(self.obs[idxs]),
            "act": torch.FloatTensor(self.act[idxs]),
            "rew": torch.FloatTensor(self.rew[idxs]),
            "next_obs": torch.FloatTensor(self.next_obs[idxs]),
            "done": torch.FloatTensor(self.done[idxs]),
        }


class SACAgent(torch.nn.Module):
    """Simple SAC agent with MLP policy and twin critics."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()

        # Actor
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
        )
        self.mu = torch.nn.Linear(hidden, act_dim)
        self.log_std = torch.nn.Linear(hidden, act_dim)

        # Twin Critics
        self.q1 = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + act_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.q2 = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + act_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

        # Target critics
        self.q1_target = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + act_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.q2_target = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + act_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

        # Copy params
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Auto-entropy
        self.log_alpha = torch.nn.Parameter(torch.zeros(1))
        self.target_entropy = -act_dim

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
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

    def sample_action(self, obs_batch: torch.Tensor) -> tuple:
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


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


def main() -> None:
    print("=" * 60)
    print("  RABIT-PROPFIRM -- SAC Training")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/4] Loading feature data...")
    symbol_data = load_all_features()

    # Step 2: Normalize
    print("\n[2/4] Normalizing features...")
    symbol_data, norm_state = normalize_features(symbol_data)

    # Pick first symbol for training (can iterate later)
    symbols = list(symbol_data.keys())
    print(f"\n  Training on: {symbols}")

    # Step 3: Create env and agent
    print("\n[3/4] Initializing agent and environment...")
    first_sym = symbols[0]
    env = SimpleTradeEnv(
        features=symbol_data[first_sym]["features_norm"],
        ohlcv=symbol_data[first_sym]["ohlcv"],
        lookback=LOOKBACK,
    )

    agent = SACAgent(env.obs_dim, env.act_dim)
    buffer = ReplayBuffer(BUFFER_SIZE, env.obs_dim, env.act_dim)

    actor_opt = torch.optim.Adam(
        list(agent.actor.parameters()) + list(agent.mu.parameters()) + list(agent.log_std.parameters()),
        lr=LR
    )
    critic_opt = torch.optim.Adam(
        list(agent.q1.parameters()) + list(agent.q2.parameters()),
        lr=LR
    )
    alpha_opt = torch.optim.Adam([agent.log_alpha], lr=LR)

    print(f"  Obs dim: {env.obs_dim}")
    print(f"  Act dim: {env.act_dim}")
    print(f"  Agent params: {sum(p.numel() for p in agent.parameters()):,}")

    # Step 4: Training loop
    print(f"\n[4/4] Training for {TRAIN_STEPS:,} steps...")
    print("-" * 60)

    obs = env.reset()
    episode_reward = 0.0
    episode_count = 0
    best_eval_reward = -999

    start_time = time.time()

    for step in range(1, TRAIN_STEPS + 1):
        # Collect experience
        if buffer.size < 1000:
            action = np.random.uniform(-1, 1, size=env.act_dim).astype(np.float32)
        else:
            action = agent.get_action(obs)

        next_obs, reward, done, info = env.step(action)
        buffer.add(obs, action, reward, next_obs, done)
        episode_reward += reward
        obs = next_obs

        if done:
            episode_count += 1
            # Rotate through symbols
            sym = symbols[episode_count % len(symbols)]
            env = SimpleTradeEnv(
                features=symbol_data[sym]["features_norm"],
                ohlcv=symbol_data[sym]["ohlcv"],
                lookback=LOOKBACK,
            )
            obs = env.reset()
            episode_reward = 0.0

        # Train
        if buffer.size >= 1000 and step % 4 == 0:
            batch = buffer.sample(BATCH_SIZE)

            # Critic update
            with torch.no_grad():
                next_a, next_log_prob = agent.sample_action(batch["next_obs"])
                sa_next = torch.cat([batch["next_obs"], next_a], dim=-1)
                q1_next = agent.q1_target(sa_next)
                q2_next = agent.q2_target(sa_next)
                q_next = torch.min(q1_next, q2_next) - agent.log_alpha.exp() * next_log_prob
                target_q = batch["rew"].unsqueeze(-1) + GAMMA * (1 - batch["done"].unsqueeze(-1)) * q_next

            sa = torch.cat([batch["obs"], batch["act"]], dim=-1)
            q1_loss = ((agent.q1(sa) - target_q) ** 2).mean()
            q2_loss = ((agent.q2(sa) - target_q) ** 2).mean()
            critic_loss = q1_loss + q2_loss

            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(agent.q1.parameters()) + list(agent.q2.parameters()), 1.0
            )
            critic_opt.step()

            # Actor update
            new_a, log_prob = agent.sample_action(batch["obs"])
            sa_new = torch.cat([batch["obs"], new_a], dim=-1)
            q_new = torch.min(agent.q1(sa_new), agent.q2(sa_new))
            actor_loss = (agent.log_alpha.exp().detach() * log_prob - q_new).mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(agent.actor.parameters()) + list(agent.mu.parameters()) + list(agent.log_std.parameters()),
                1.0,
            )
            actor_opt.step()

            # Alpha update
            alpha_loss = -(agent.log_alpha.exp() * (log_prob.detach() + agent.target_entropy)).mean()
            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()

            # Soft update targets
            soft_update(agent.q1_target, agent.q1, TAU)
            soft_update(agent.q2_target, agent.q2, TAU)

        # Evaluate
        if step % EVAL_EVERY == 0:
            eval_rewards = []
            eval_trades = []
            eval_wrs = []

            for sym in symbols:
                eval_env = SimpleTradeEnv(
                    features=symbol_data[sym]["features_norm"],
                    ohlcv=symbol_data[sym]["ohlcv"],
                    lookback=LOOKBACK,
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
                f"  Step {step:>6,} | Reward: {mean_reward:>7.2f} | "
                f"Trades: {np.mean(eval_trades):>4.1f} | WR: {np.mean(eval_wrs):>5.1%} | "
                f"Alpha: {agent.log_alpha.exp().item():.3f} | {sps:.0f} steps/s"
            )

            if mean_reward > best_eval_reward:
                best_eval_reward = mean_reward
                ckpt_path = MODEL_DIR / "best_model.pt"
                torch.save({
                    "agent_state": agent.state_dict(),
                    "obs_dim": env.obs_dim,
                    "act_dim": env.act_dim,
                    "eval_reward": mean_reward,
                    "step": step,
                    "norm_state": norm_state,
                }, ckpt_path)
                print(f"    >> New best! Saved to {ckpt_path.name}")

    # Final save
    final_path = MODEL_DIR / "final_model.pt"
    torch.save({
        "agent_state": agent.state_dict(),
        "obs_dim": env.obs_dim,
        "act_dim": env.act_dim,
        "step": TRAIN_STEPS,
        "norm_state": norm_state,
    }, final_path)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print(f"  Best eval reward: {best_eval_reward:.2f}")
    print(f"  Models saved to: {MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
