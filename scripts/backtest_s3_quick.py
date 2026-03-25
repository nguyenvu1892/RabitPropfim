#!/usr/bin/env python3
"""Quick Stage 3 backtest."""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, "/home/user/RabitPropfim/rabit_propfirm_drl")

class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim=300, n_actions=3):
        super().__init__()
        layers = []
        ind = obs_dim
        for h in [512, 256, 128]:
            layers += [nn.Linear(ind, h), nn.LayerNorm(h), nn.ReLU()]
            ind = h
        self.trunk = nn.Sequential(*layers)
        self.actor_head = nn.Linear(128, n_actions)
        self.critic_head = nn.Linear(128, 1)
    def forward(self, obs):
        f = self.trunk(obs)
        return self.actor_head(f), self.critic_head(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("/home/user/RabitPropfim/models_saved/best_v33_stage3.pt", map_location=device, weights_only=False)
m = PPOActorCritic().to(device)
m.load_state_dict(ckpt["model_state_dict"])
m.eval()
print(f"Stage 3 model loaded (step={ckpt.get('step', 0)})")

import yaml
from data_engine.normalizer import RunningNormalizer
from environments.prop_env import MultiTFTradingEnv

with open("/home/user/RabitPropfim/rabit_propfirm_drl/configs/prop_rules.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["stage1_mode"] = False
with open("/home/user/RabitPropfim/data/normalizer_v3.json") as f:
    nd = json.load(f)
norms = {k: RunningNormalizer.from_state_dict(v) for k, v in nd.items()}

DATA = Path("/home/user/RabitPropfim/data")
syms = ["XAUUSD", "BTCUSD", "ETHUSD", "US30_cash", "US100_cash"]
total_all_t = 0
total_all_w = 0

for sym in syms:
    safe = sym.replace(".", "_")
    sd = {}
    for tf in ["M1", "M5", "M15", "H1"]:
        a = np.load(DATA / f"{safe}_{tf}_50dim.npy")
        sd[tf] = norms[tf].normalize(a).astype(np.float32)
        op = DATA / f"{safe}_{tf}_ohlcv.npy"
        sd[f"{tf}_ohlcv"] = np.load(op).astype(np.float32) if op.exists() else None

    env = MultiTFTradingEnv(
        data_m1=sd["M1"], data_m5=sd["M5"], data_m15=sd["M15"], data_h1=sd["H1"],
        config=cfg, n_features=50, initial_balance=10000, episode_length=2000,
        ohlcv_m5=sd.get("M5_ohlcv"), action_mode="discrete",
    )

    buys = sells = holds = 0
    all_t = []
    for ep in range(10):
        obs, _ = env.reset(seed=42 + ep)
        for s in range(2000):
            ot = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            ot = torch.nan_to_num(ot, nan=0.0)
            with torch.no_grad():
                lg, _ = m(ot)
                a = int(torch.argmax(torch.softmax(lg, dim=-1)))
            if a == 0: buys += 1
            elif a == 1: sells += 1
            else: holds += 1
            obs, _, t, tr, _ = env.step(a)
            if t or tr: break
        all_t.extend(env.trade_history)

    tot = len(all_t)
    w = sum(1 for t in all_t if t.get("pnl", 0) > 0)
    wr = 100 * w / max(tot, 1)
    ta = buys + sells + holds
    total_all_t += tot
    total_all_w += w
    st = "PASS" if wr > 40 else "WATCH"
    print(f"{sym:12s} | Trades={tot:5d} | WR={wr:5.1f}% | BUY={100*buys/ta:.1f}% SELL={100*sells/ta:.1f}% HOLD={100*holds/ta:.1f}% | {st}")

print("=" * 70)
print(f"OVERALL: {total_all_t} trades | Win Rate = {100*total_all_w/max(total_all_t,1):.1f}% | Target > 40%")
print("=" * 70)
