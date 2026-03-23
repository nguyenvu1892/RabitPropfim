"""V3 Smoke Test: 100 steps to verify real prices, ATR, SL/TP, and trade execution."""
import sys, numpy as np, torch, yaml
from pathlib import Path
sys.path.insert(0, str(Path("rabit_propfirm_drl")))
from agents.sac_policy import SACTransformerActor
from environments.prop_env import MultiTFTradingEnv

with open("rabit_propfirm_drl/configs/prop_rules.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Load data
data = {}
for tf in ["M1", "M5", "M15", "H1"]:
    arr = np.load("data/XAUUSD_%s_50dim.npy" % tf)
    split = int(len(arr) * 0.8)
    data[tf] = arr[split:].astype(np.float32)

# Load Stage1 model
ckpt = torch.load("models_saved/best_Stage1_Context.pt", map_location="cpu", weights_only=False)
actor = SACTransformerActor(n_features=50, action_dim=5, embed_dim=128, n_heads=4,
    n_cross_layers=1, n_regimes=4, hidden_dims=[256, 256], dropout=0.1)
actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
actor.eval()

# Create env WITHOUT ohlcv (test synthetic fallback)
env = MultiTFTradingEnv(
    data_m1=data["M1"], data_m5=data["M5"],
    data_m15=data["M15"], data_h1=data["H1"],
    config=config, n_features=50, initial_balance=10000,
    episode_length=200, pip_value=0.01,
)
env.m5_threshold = 0.05
env.m1_threshold = 0.05
obs, info = env.reset(seed=42)

print("=" * 80)
print("V3 SMOKE TEST - 100 Steps (synthetic OHLCV from log_return)")
print("Using real OHLCV:", env._use_real_ohlcv)
print("=" * 80)

for step in range(100):
    m1 = torch.from_numpy(obs["m1"]).unsqueeze(0)
    m5 = torch.from_numpy(obs["m5"]).unsqueeze(0)
    m15 = torch.from_numpy(obs["m15"]).unsqueeze(0)
    h1 = torch.from_numpy(obs["h1"]).unsqueeze(0)
    with torch.no_grad():
        action, _ = actor(m1, m5, m15, h1, deterministic=True)
    a = action.numpy().flatten()
    conf = float(a[0])
    et = float(a[1])
    risk = float(np.clip((a[2]+1)/2, 0, 1))
    sl_m = float(np.clip(a[3]*1.25+1.75, 0.5, 3.0))
    tp_m = float(np.clip(a[4]*2.25+2.75, 0.5, 5.0))

    price_before = env._get_current_price()
    atr = env._estimate_atr_pips(price_before)

    env_action = np.array([conf, et, risk, sl_m, tp_m], dtype=np.float32)
    obs, r, term, trunc, info = env.step(env_action)
    price_after = env._get_current_price()

    if step < 10 or step % 20 == 0 or len(env.trade_history) != getattr(env, '_prev_th_len', 0):
        direction = "BUY" if conf > 0 else "SELL"
        entry = "M1" if et > 0 else "M5"
        print("Step %3d | price=%.2f | conf=%+.3f (%s %s) | sl_m=%.2f tp_m=%.2f | "
              "ATR=%.1f pips | pos=%d trades=%d bal=%.2f" %
              (step, price_after, conf, direction, entry, sl_m, tp_m,
               atr, len(env.positions), len(env.trade_history), info["balance"]))
    env._prev_th_len = len(env.trade_history)
    if term or trunc:
        break

print("\n" + "=" * 80)
print("FINAL: balance=%.2f | total_trades=%d | positions=%d" %
      (info["balance"], len(env.trade_history), len(env.positions)))
print("Price range: %.2f - %.2f" % (env.ohlcv_m5[:, 3].min(), env.ohlcv_m5[:, 3].max()))
print("ATR final: %.1f pips" % env._estimate_atr_pips(env._get_current_price()))

if env.trade_history:
    pnls = [t["pnl"] for t in env.trade_history]
    durs = [t["duration"] for t in env.trade_history]
    print("Trade PnLs: min=%.2f max=%.2f avg=%.2f" % (min(pnls), max(pnls), np.mean(pnls)))
    print("Hold durations: min=%d max=%d avg=%.1f" % (min(durs), max(durs), np.mean(durs)))
    sl_mults = [float(np.clip(a[3]*1.25+1.75, 0.5, 3.0)) for a in [action.numpy().flatten()]]
    print("Sample sl_mult=%.2f tp_mult=%.2f" % (sl_m, tp_m))
else:
    print("WARNING: 0 trades in 100 steps")
print("=" * 80)
