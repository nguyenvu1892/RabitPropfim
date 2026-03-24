#!/usr/bin/env python3
"""
V3.5 VIP Harvest — 4-TF SMC Filter + STRICT Equal Cap.

Only harvests trades where:
  1. Bot manually pressed CLOSE (close_reason == "MANUAL_CLOSE")
  2. Trade was profitable (pnl > 0)
  3. SMC score >= 0.5 (Volume spike, Pin bar, Trend, BOS)

Strict equal cap: find min VIP count across all symbols, cap ALL at that min.
Result: perfect 1:1:1:1:1 ratio.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "rabit_propfirm_drl"))
DATA_DIR = project_root / "data"
MODELS_DIR = project_root / "models_saved"
VIP_DIR = MODELS_DIR / "vip_buffer_v35"
VIP_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("vip_harvest_v35")
SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD", "US30_cash", "US100_cash"]

class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim=400, n_actions=4, hidden_dims=None):
        super().__init__()
        if hidden_dims is None: hidden_dims = [512, 256, 128]
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.actor_head = nn.Linear(hidden_dims[-1], n_actions)
        self.critic_head = nn.Linear(hidden_dims[-1], 1)
    def forward(self, obs):
        f = self.trunk(obs)
        return self.actor_head(f), self.critic_head(f)

def smc_score_trade(m1_window, direction):
    score, checks = 0.0, {"volume_spike": False, "pin_bar": False, "trend_aligned": False, "breakout": False}
    if m1_window is None or len(m1_window) < 3: return 0.0, checks
    try:
        vol = np.abs(m1_window[:, 4]) if m1_window.shape[1] > 4 else np.abs(m1_window[:, -1])
        if len(vol) > 2 and (vol[-1] - np.mean(vol[:-1])) / (np.std(vol[:-1]) + 1e-8) > 1.5:
            score += 1.0; checks["volume_spike"] = True
    except: pass
    try:
        b = m1_window[-1]; body = abs(b[3]-b[0]); rng = abs(b[1]-b[2]) + 1e-8
        wu = abs(b[1]-max(b[0],b[3])); wd = abs(min(b[0],b[3])-b[2])
        if body/rng < 0.35 and (wu > body*2 or wd > body*2): score += 1.0; checks["pin_bar"] = True
    except: pass
    try:
        mom = m1_window[-1,3] - m1_window[0,3]
        if (direction > 0 and mom > 0) or (direction < 0 and mom < 0): score += 1.0; checks["trend_aligned"] = True
    except: pass
    try:
        if direction > 0 and m1_window[-1,1] > np.max(m1_window[:-1,1]): score += 1.0; checks["breakout"] = True
        elif direction < 0 and m1_window[-1,2] < np.min(m1_window[:-1,2]): score += 1.0; checks["breakout"] = True
    except: pass
    return score / 4.0, checks

def harvest(n_episodes=30, max_cap=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODELS_DIR / "best_v35_stage1.pt", map_location=device, weights_only=False)
    model = PPOActorCritic(obs_dim=400, n_actions=4).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded V3.5 Stage 1 (step=%d)", ckpt.get("step", 0))

    import yaml as _yaml
    from data_engine.normalizer import RunningNormalizer
    from environments.prop_env import MultiTFTradingEnv

    with open(project_root / "rabit_propfirm_drl" / "configs" / "prop_rules.yaml") as f:
        cfg = _yaml.safe_load(f)
    cfg["stage1_mode"] = True
    with open(DATA_DIR / "normalizer_v3.json") as f:
        nd = json.load(f)
    norms = {k: RunningNormalizer.from_state_dict(v) for k, v in nd.items()}

    sym_vips = {s: [] for s in SYMBOLS}

    for sym in SYMBOLS:
        safe = sym.replace(".", "_")
        logger.info("Harvesting %s (%d eps)...", sym, n_episodes)
        sd = {}
        for tf in ["M1", "M5", "M15", "H1"]:
            sd[tf] = norms[tf].normalize(np.load(DATA_DIR / f"{safe}_{tf}_50dim.npy")).astype(np.float32)
            op = DATA_DIR / f"{safe}_{tf}_ohlcv.npy"
            sd[f"{tf}_ohlcv"] = np.load(op).astype(np.float32) if op.exists() else None

        env = MultiTFTradingEnv(
            data_m1=sd["M1"], data_m5=sd["M5"], data_m15=sd["M15"], data_h1=sd["H1"],
            config=cfg, n_features=50, initial_balance=10_000.0,
            episode_length=2000, ohlcv_m5=sd.get("M5_ohlcv"), action_mode="discrete",
        )
        sym_trades = 0
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=100 + ep)
            pending = {}; prev_tc = 0
            for step in range(2000):
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
                obs, reward, term, trunc, info = env.step(action)
                cur_tc = len(env.positions) + len(env.trade_history)
                if cur_tc > prev_tc and action in (0, 1):
                    d = 1 if action == 0 else -1
                    for pos in env.positions:
                        if pos.ticket not in pending:
                            pending[pos.ticket] = {"obs": current_obs, "action": action, "direction": d, "m1_window": m1_window}
                prev_tc = cur_tc
                for trade in env.trade_history:
                    ticket = trade.get("ticket")
                    if ticket in pending:
                        ed = pending.pop(ticket)
                        sym_trades += 1
                        if trade.get("pnl", 0) <= 0: continue
                        if trade.get("close_reason") != "MANUAL_CLOSE": continue
                        smc_s, checks = smc_score_trade(ed["m1_window"], ed["direction"])
                        if smc_s >= 0.5:
                            sym_vips[sym].append({"obs": ed["obs"], "action": ed["action"], "pnl": trade["pnl"], "smc_score": smc_s})
                if term or trunc: break
        logger.info("  %s: %d trades | %d VIP (%.1f%%)", sym, sym_trades, len(sym_vips[sym]), 100*len(sym_vips[sym])/max(sym_trades,1))

    # STRICT EQUAL CAP: min across all symbols
    min_vip = min(len(v) for v in sym_vips.values())
    if max_cap and max_cap < min_vip:
        min_vip = max_cap
    logger.info("STRICT EQUAL CAP: %d per symbol (min across all)", min_vip)

    all_vip = []
    for sym in SYMBOLS:
        entries = sym_vips[sym]
        entries.sort(key=lambda x: x["smc_score"], reverse=True)
        entries = entries[:min_vip]
        all_vip.extend(entries)
        logger.info("  %-12s | VIP=%4d | SMC=%.2f", sym, len(entries), np.mean([v["smc_score"] for v in entries]) if entries else 0)

    vip_obs = np.array([v["obs"] for v in all_vip], dtype=np.float32)
    vip_actions = np.array([v["action"] for v in all_vip], dtype=np.int64)
    np.save(VIP_DIR / "vip_obs.npy", vip_obs)
    np.save(VIP_DIR / "vip_actions.npy", vip_actions)

    logger.info("=" * 70)
    logger.info("  V3.5 VIP HARVEST COMPLETE — STRICT EQUAL %d/sym", min_vip)
    logger.info("  Total: %d VIP (obs=%s)", len(all_vip), vip_obs.shape)
    logger.info("=" * 70)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--max-cap", type=int, default=None)
    args = p.parse_args()
    harvest(n_episodes=args.episodes, max_cap=args.max_cap)
