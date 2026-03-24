"""
V3.3 VIP Harvest -- Collect gold trades from Stage 1 model.

Runs the trained Stage 1 model, captures obs+action at every trade entry,
filters winning trades through SMC (OB proximity, Volume spike, Pin bar,
Breakout), and saves VIP experiences to disk for Stage 2 imitation.

Usage:
    python scripts/harvest_vip.py
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "rabit_propfirm_drl"))

DATA_DIR = project_root / "data"
MODELS_DIR = project_root / "models_saved"
VIP_DIR = MODELS_DIR / "vip_buffer"
VIP_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("vip_harvest")

SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD", "US30_cash", "US100_cash"]


# --- PPO Model (must match train_v33.py) ---
class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim=300, n_actions=3, hidden_dims=None):
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


# --- SMC Filter (enhanced with OHLCV context) ---

def smc_score_trade(
    entry_obs: np.ndarray,
    direction: int,
    m1_window: np.ndarray,  # (5, 50) M1 bars around entry
    atr: float,
) -> tuple[float, dict]:
    """
    Score a winning trade for VIP quality using SMC criteria.

    Checks:
    1. Volume spike (> 2 std from recent mean)
    2. Pin bar / doji pattern (body < 30% of range)
    3. Trend alignment (direction matches recent momentum)
    4. Breakout / BOS (price breaks recent high/low)

    Returns (score 0-1, details dict).
    """
    score = 0.0
    checks = {"volume_spike": False, "pin_bar": False, "trend_aligned": False, "breakout": False}
    n_checks = 4

    if m1_window is None or len(m1_window) < 3:
        return 0.0, checks

    try:
        # Check 1: Volume spike
        # Feature index 4 is typically volume in normalized features
        # Use last column range as proxy for volume activity
        vol_proxy = np.abs(m1_window[:, 4]) if m1_window.shape[1] > 4 else np.abs(m1_window[:, -1])
        if len(vol_proxy) > 2:
            vol_mean = np.mean(vol_proxy[:-1])
            vol_std = np.std(vol_proxy[:-1]) + 1e-8
            if (vol_proxy[-1] - vol_mean) / vol_std > 1.5:  # 1.5 sigma
                score += 1.0
                checks["volume_spike"] = True
    except (IndexError, ValueError):
        pass

    try:
        # Check 2: Pin bar (small body, long wick)
        # In normalized features, first few cols are OHLC-derived
        # Use cols 0-3 as price movement proxies
        last_bar = m1_window[-1]
        body = abs(last_bar[3] - last_bar[0])  # close - open proxy
        wick_up = abs(last_bar[1] - max(last_bar[0], last_bar[3]))
        wick_dn = abs(min(last_bar[0], last_bar[3]) - last_bar[2])
        total_range = abs(last_bar[1] - last_bar[2]) + 1e-8
        body_ratio = body / total_range

        if body_ratio < 0.35 and (wick_up > body * 2 or wick_dn > body * 2):
            score += 1.0
            checks["pin_bar"] = True
    except (IndexError, ValueError):
        pass

    try:
        # Check 3: Trend alignment (momentum matches direction)
        closes = m1_window[:, 3]  # Close price proxy
        momentum = closes[-1] - closes[0]
        if (direction > 0 and momentum > 0) or (direction < 0 and momentum < 0):
            score += 1.0
            checks["trend_aligned"] = True
    except (IndexError, ValueError):
        pass

    try:
        # Check 4: Breakout (last bar breaks recent range = BOS)
        highs = m1_window[:-1, 1]
        lows = m1_window[:-1, 2]
        if direction > 0 and m1_window[-1, 1] > np.max(highs):
            score += 1.0
            checks["breakout"] = True
        elif direction < 0 and m1_window[-1, 2] < np.min(lows):
            score += 1.0
            checks["breakout"] = True
    except (IndexError, ValueError):
        pass

    return score / n_checks, checks


def harvest_vip_trades(n_episodes: int = 20, episode_length: int = 2000, cap_per_symbol: int = 500):
    """Run Stage 1 model, capture trades, filter through SMC."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt_path = MODELS_DIR / "best_v33_stage1.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PPOActorCritic(obs_dim=ckpt.get("obs_dim", 300), n_actions=3).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded Stage 1 model (step=%d)", ckpt.get("step", 0))

    # Load config + data
    import yaml as _yaml
    from data_engine.normalizer import RunningNormalizer
    from environments.prop_env import MultiTFTradingEnv

    config_path = project_root / "rabit_propfirm_drl" / "configs" / "prop_rules.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        env_config = _yaml.safe_load(f)
    env_config["stage1_mode"] = True

    norm_path = DATA_DIR / "normalizer_v3.json"
    with open(norm_path, "r") as f:
        norm_data = json.load(f)
    normalizers = {}
    for tf, state in norm_data.items():
        normalizers[tf] = RunningNormalizer.from_state_dict(state)

    # Collect VIP trades
    all_vip = []  # List of (obs, action, pnl, smc_score, symbol, checks)
    total_evaluated = 0
    total_accepted = 0
    total_rejected_loss = 0
    total_rejected_smc = 0

    for sym in SYMBOLS:
        safe = sym.replace(".", "_")
        logger.info("Harvesting %s (%d episodes)...", sym, n_episodes)

        sym_data = {}
        for tf in ["M1", "M5", "M15", "H1"]:
            arr = np.load(DATA_DIR / f"{safe}_{tf}_50dim.npy")
            sym_data[tf] = normalizers[tf].normalize(arr).astype(np.float32)
            ohlcv_path = DATA_DIR / f"{safe}_{tf}_ohlcv.npy"
            sym_data[f"{tf}_ohlcv"] = np.load(ohlcv_path).astype(np.float32) if ohlcv_path.exists() else None

        env = MultiTFTradingEnv(
            data_m1=sym_data["M1"], data_m5=sym_data["M5"],
            data_m15=sym_data["M15"], data_h1=sym_data["H1"],
            config=env_config, n_features=50, initial_balance=10_000.0,
            episode_length=episode_length,
            ohlcv_m5=sym_data.get("M5_ohlcv"),
            action_mode="discrete",
        )

        sym_vip = 0
        sym_trades = 0

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=100 + ep)
            # Track obs at each step for trade-to-obs mapping
            pending_entry_obs = {}  # ticket -> obs_at_entry
            prev_trade_count = 0

            for step in range(episode_length):
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=10.0, neginf=-10.0)

                with torch.no_grad():
                    logits, _ = model(obs_t)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    action = int(np.argmax(probs))

                # Save obs BEFORE step (this is the observation when action was taken)
                current_obs = obs.copy()

                # Get M1 window for SMC context
                m5_idx = min(env.current_m5_step, env.n_m5_bars - 1)
                m1_end = min(m5_idx * env.m1_per_m5 + env.m1_per_m5 - 1, len(env.data_m1) - 1)
                m1_start = max(0, m1_end - 4)
                m1_window = env.data_m1[m1_start:m1_end + 1].copy()

                obs, reward, terminated, truncated, info = env.step(action)

                # Detect new trade opened
                current_trade_count = len(env.positions) + len(env.trade_history)
                if current_trade_count > prev_trade_count and action in (0, 1):
                    # A trade was just opened
                    direction = 1 if action == 0 else -1
                    for pos in env.positions:
                        if pos.ticket not in pending_entry_obs:
                            pending_entry_obs[pos.ticket] = {
                                "obs": current_obs,
                                "action": action,
                                "direction": direction,
                                "m1_window": m1_window,
                                "atr": float(env.atr_array[min(env.current_m5_step, len(env.atr_array) - 1)]),
                            }
                prev_trade_count = current_trade_count

                # Check closed trades
                for trade in env.trade_history:
                    ticket = trade.get("ticket")
                    if ticket in pending_entry_obs:
                        entry_data = pending_entry_obs.pop(ticket)
                        pnl = trade.get("pnl", 0)
                        total_evaluated += 1
                        sym_trades += 1

                        if pnl <= 0:
                            total_rejected_loss += 1
                            continue

                        # SMC filter
                        smc_s, checks = smc_score_trade(
                            entry_data["obs"],
                            entry_data["direction"],
                            entry_data["m1_window"],
                            entry_data["atr"],
                        )

                        if smc_s >= 0.5:  # At least 2/4 checks pass
                            all_vip.append({
                                "obs": entry_data["obs"],
                                "action": entry_data["action"],
                                "pnl": pnl,
                                "smc_score": smc_s,
                                "symbol": sym,
                                "checks": checks,
                                "direction": entry_data["direction"],
                            })
                            total_accepted += 1
                            sym_vip += 1
                        else:
                            total_rejected_smc += 1

                if terminated or truncated:
                    break

        logger.info("  %s: %d trades evaluated, %d VIP accepted (%.1f%%)",
                     sym, sym_trades, sym_vip,
                     100 * sym_vip / max(sym_trades, 1))

    # Cap VIP per symbol to prevent imbalance
    if cap_per_symbol > 0:
        capped_vip = []
        for sym in SYMBOLS:
            sym_entries = [v for v in all_vip if v["symbol"] == sym]
            if len(sym_entries) > cap_per_symbol:
                # Keep highest SMC score entries
                sym_entries.sort(key=lambda x: x["smc_score"], reverse=True)
                sym_entries = sym_entries[:cap_per_symbol]
                logger.info("  %s: CAPPED %d -> %d", sym, len([v for v in all_vip if v["symbol"] == sym]), cap_per_symbol)
            capped_vip.extend(sym_entries)
        total_before = total_accepted
        all_vip = capped_vip
        total_accepted = len(all_vip)
        logger.info("  VIP CAPPED: %d -> %d (cap=%d/symbol)", total_before, total_accepted, cap_per_symbol)

    # Save VIP buffer
    vip_obs = np.array([v["obs"] for v in all_vip], dtype=np.float32)
    vip_actions = np.array([v["action"] for v in all_vip], dtype=np.int64)
    vip_scores = np.array([v["smc_score"] for v in all_vip], dtype=np.float32)

    np.save(VIP_DIR / "vip_obs.npy", vip_obs)
    np.save(VIP_DIR / "vip_actions.npy", vip_actions)
    np.save(VIP_DIR / "vip_scores.npy", vip_scores)

    # Save metadata
    meta = {
        "total_evaluated": total_evaluated,
        "total_accepted": total_accepted,
        "total_rejected_loss": total_rejected_loss,
        "total_rejected_smc": total_rejected_smc,
        "acceptance_rate": total_accepted / max(total_evaluated, 1),
        "per_symbol": {},
    }
    for sym in SYMBOLS:
        sym_vips = [v for v in all_vip if v["symbol"] == sym]
        meta["per_symbol"][sym] = {
            "vip_count": len(sym_vips),
            "avg_smc_score": float(np.mean([v["smc_score"] for v in sym_vips])) if sym_vips else 0,
            "avg_pnl": float(np.mean([v["pnl"] for v in sym_vips])) if sym_vips else 0,
            "buy_pct": 100 * sum(1 for v in sym_vips if v["action"] == 0) / max(len(sym_vips), 1),
            "sell_pct": 100 * sum(1 for v in sym_vips if v["action"] == 1) / max(len(sym_vips), 1),
            "checks": {
                "volume_spike": sum(1 for v in sym_vips if v["checks"].get("volume_spike")),
                "pin_bar": sum(1 for v in sym_vips if v["checks"].get("pin_bar")),
                "trend_aligned": sum(1 for v in sym_vips if v["checks"].get("trend_aligned")),
                "breakout": sum(1 for v in sym_vips if v["checks"].get("breakout")),
            },
        }

    with open(VIP_DIR / "vip_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("=" * 70)
    logger.info("  VIP HARVEST COMPLETE")
    logger.info("  Evaluated: %d | Accepted: %d (%.1f%%) | Loss: %d | SMC Fail: %d",
                total_evaluated, total_accepted,
                100 * total_accepted / max(total_evaluated, 1),
                total_rejected_loss, total_rejected_smc)
    logger.info("  Saved: %s (obs=%s, actions=%s)",
                VIP_DIR, vip_obs.shape, vip_actions.shape)
    logger.info("=" * 70)

    for sym, info in meta["per_symbol"].items():
        logger.info("  %-12s | VIP=%4d | SMC=%.2f | PnL=%.2f | Vol=%d Pin=%d Trend=%d BOS=%d",
                     sym, info["vip_count"], info["avg_smc_score"], info["avg_pnl"],
                     info["checks"]["volume_spike"], info["checks"]["pin_bar"],
                     info["checks"]["trend_aligned"], info["checks"]["breakout"])

    return meta


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--cap-per-symbol", type=int, default=500)
    args = p.parse_args()
    harvest_vip_trades(n_episodes=args.episodes, episode_length=2000, cap_per_symbol=args.cap_per_symbol)
