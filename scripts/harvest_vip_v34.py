#!/usr/bin/env python3
"""
V3.4 VIP Harvest — Filter MANUAL_CLOSE + WIN + SMC trades.

Only harvests trades where:
  1. Bot manually pressed CLOSE (close_reason == "MANUAL_CLOSE")
  2. Trade was profitable (pnl > 0)
  3. SMC score >= 0.5 (Volume spike, Pin bar, Trend, BOS)

Usage:
    python scripts/harvest_vip_v34.py --cap-per-symbol 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "rabit_propfirm_drl"))

DATA_DIR = project_root / "data"
MODELS_DIR = project_root / "models_saved"
VIP_DIR = MODELS_DIR / "vip_buffer_v34"
VIP_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("vip_harvest_v34")

SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD", "US30_cash", "US100_cash"]


class PPOActorCritic(nn.Module):
    """Must match train_v34.py exactly."""
    def __init__(self, obs_dim=350, n_actions=4, hidden_dims=None):
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


def smc_score_trade(m1_window, direction):
    """Score using SMC criteria: volume spike, pin bar, trend, breakout."""
    score = 0.0
    checks = {"volume_spike": False, "pin_bar": False, "trend_aligned": False, "breakout": False}
    if m1_window is None or len(m1_window) < 3:
        return 0.0, checks

    try:
        vol_proxy = np.abs(m1_window[:, 4]) if m1_window.shape[1] > 4 else np.abs(m1_window[:, -1])
        if len(vol_proxy) > 2:
            vol_mean = np.mean(vol_proxy[:-1])
            vol_std = np.std(vol_proxy[:-1]) + 1e-8
            if (vol_proxy[-1] - vol_mean) / vol_std > 1.5:
                score += 1.0
                checks["volume_spike"] = True
    except (IndexError, ValueError):
        pass

    try:
        last_bar = m1_window[-1]
        body = abs(last_bar[3] - last_bar[0])
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
        closes = m1_window[:, 3]
        momentum = closes[-1] - closes[0]
        if (direction > 0 and momentum > 0) or (direction < 0 and momentum < 0):
            score += 1.0
            checks["trend_aligned"] = True
    except (IndexError, ValueError):
        pass

    try:
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

    return score / 4.0, checks


def harvest(n_episodes=20, episode_length=2000, cap_per_symbol=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = MODELS_DIR / "best_v34_stage1.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PPOActorCritic(obs_dim=350, n_actions=4).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded V3.4 Stage 1 (step=%d)", ckpt.get("step", 0))

    import yaml as _yaml
    from data_engine.normalizer import RunningNormalizer
    from environments.prop_env import MultiTFTradingEnv

    with open(project_root / "rabit_propfirm_drl" / "configs" / "prop_rules.yaml") as f:
        cfg = _yaml.safe_load(f)
    cfg["stage1_mode"] = True

    with open(DATA_DIR / "normalizer_v3.json") as f:
        nd = json.load(f)
    norms = {k: RunningNormalizer.from_state_dict(v) for k, v in nd.items()}

    all_vip = []
    total_eval = 0
    total_win = 0
    total_manual_win = 0
    total_smc_fail = 0
    total_accepted = 0

    for sym in SYMBOLS:
        safe = sym.replace(".", "_")
        logger.info("Harvesting %s (%d episodes)...", sym, n_episodes)

        sd = {}
        for tf in ["M1", "M5", "M15", "H1"]:
            arr = np.load(DATA_DIR / f"{safe}_{tf}_50dim.npy")
            sd[tf] = norms[tf].normalize(arr).astype(np.float32)
            op = DATA_DIR / f"{safe}_{tf}_ohlcv.npy"
            sd[f"{tf}_ohlcv"] = np.load(op).astype(np.float32) if op.exists() else None

        env = MultiTFTradingEnv(
            data_m1=sd["M1"], data_m5=sd["M5"], data_m15=sd["M15"], data_h1=sd["H1"],
            config=cfg, n_features=50, initial_balance=10_000.0,
            episode_length=episode_length, ohlcv_m5=sd.get("M5_ohlcv"),
            action_mode="discrete",
        )

        sym_vip = 0
        sym_trades = 0

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=100 + ep)
            pending = {}
            prev_tc = 0

            for step in range(episode_length):
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=10.0, neginf=-10.0)
                with torch.no_grad():
                    logits, _ = model(obs_t)
                    action = int(torch.argmax(torch.softmax(logits, dim=-1)))

                current_obs = obs.copy()
                m5_idx = min(env.current_m5_step, env.n_m5_bars - 1)
                m1_end = min(m5_idx * env.m1_per_m5 + env.m1_per_m5 - 1, len(env.data_m1) - 1)
                m1_start = max(0, m1_end - 4)
                m1_window = env.data_m1[m1_start:m1_end + 1].copy()

                obs, reward, terminated, truncated, info = env.step(action)

                # Track new positions
                cur_tc = len(env.positions) + len(env.trade_history)
                if cur_tc > prev_tc and action in (0, 1):
                    direction = 1 if action == 0 else -1
                    for pos in env.positions:
                        if pos.ticket not in pending:
                            pending[pos.ticket] = {
                                "obs": current_obs, "action": action,
                                "direction": direction, "m1_window": m1_window,
                            }
                prev_tc = cur_tc

                # Check closed trades
                for trade in env.trade_history:
                    ticket = trade.get("ticket")
                    if ticket in pending:
                        entry_data = pending.pop(ticket)
                        pnl = trade.get("pnl", 0)
                        close_reason = trade.get("close_reason", "")
                        total_eval += 1
                        sym_trades += 1

                        if pnl <= 0:
                            continue
                        total_win += 1

                        # V3.4: Only accept MANUAL_CLOSE wins
                        if close_reason != "MANUAL_CLOSE":
                            continue
                        total_manual_win += 1

                        # SMC filter
                        smc_s, checks = smc_score_trade(
                            entry_data["m1_window"], entry_data["direction"]
                        )
                        if smc_s >= 0.5:
                            all_vip.append({
                                "obs": entry_data["obs"],
                                "action": entry_data["action"],
                                "pnl": pnl, "smc_score": smc_s,
                                "symbol": sym, "checks": checks,
                            })
                            total_accepted += 1
                            sym_vip += 1
                        else:
                            total_smc_fail += 1

                if terminated or truncated:
                    break

        logger.info("  %s: %d trades | %d VIP (%.1f%%)",
                     sym, sym_trades, sym_vip, 100 * sym_vip / max(sym_trades, 1))

    # Cap per symbol
    if cap_per_symbol > 0:
        capped = []
        for sym in SYMBOLS:
            entries = [v for v in all_vip if v["symbol"] == sym]
            if len(entries) > cap_per_symbol:
                entries.sort(key=lambda x: x["smc_score"], reverse=True)
                entries = entries[:cap_per_symbol]
                logger.info("  %s: CAPPED %d -> %d", sym,
                            len([v for v in all_vip if v["symbol"] == sym]), cap_per_symbol)
            capped.extend(entries)
        all_vip = capped
        total_accepted = len(all_vip)

    # Save
    vip_obs = np.array([v["obs"] for v in all_vip], dtype=np.float32)
    vip_actions = np.array([v["action"] for v in all_vip], dtype=np.int64)
    np.save(VIP_DIR / "vip_obs.npy", vip_obs)
    np.save(VIP_DIR / "vip_actions.npy", vip_actions)

    logger.info("=" * 70)
    logger.info("  V3.4 VIP HARVEST COMPLETE")
    logger.info("  Evaluated: %d | Wins: %d | ManualClose Wins: %d | SMC Fail: %d | Accepted: %d",
                total_eval, total_win, total_manual_win, total_smc_fail, total_accepted)
    logger.info("  Saved: %s (obs=%s, actions=%s)", VIP_DIR, vip_obs.shape, vip_actions.shape)
    logger.info("=" * 70)

    for sym in SYMBOLS:
        sym_v = [v for v in all_vip if v["symbol"] == sym]
        if sym_v:
            logger.info("  %-12s | VIP=%4d | SMC=%.2f | PnL=%.2f",
                         sym, len(sym_v),
                         np.mean([v["smc_score"] for v in sym_v]),
                         np.sum([v["pnl"] for v in sym_v]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--cap-per-symbol", type=int, default=500)
    args = p.parse_args()
    harvest(n_episodes=args.episodes, cap_per_symbol=args.cap_per_symbol)
