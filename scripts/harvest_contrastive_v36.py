#!/usr/bin/env python3
"""
V3.6 Contrastive Memory Harvest — Single-Env (fixes AsyncVectorEnv limitation).

Runs 50 episodes per symbol using the trained AttentionPPO model.
Collects WIN and LOSS trades with regime labels into ContrastiveMemory.
"""
import sys, json, logging, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rabit_propfirm_drl"))

from models.attention_ppo import AttentionPPO
from training_pipeline.contrastive_memory import ContrastiveMemory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("harvest_contrastive")

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
MODELS = PROJECT / "models_saved"
MEMORY_DIR = MODELS / "contrastive_memory_v36"
SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD", "US30_cash", "US100_cash"]

def estimate_regime(data_m15, m5_idx, m5_per_m15=3):
    m15_idx = min(m5_idx // m5_per_m15, len(data_m15) - 1)
    start = max(0, m15_idx - 4)
    w = data_m15[start:m15_idx + 1]
    if len(w) < 2: return "unknown"
    slope = float(np.mean(w[:, min(27, w.shape[1]-1)]))
    if slope > 0.001: return "trending_up"
    elif slope < -0.001: return "trending_down"
    return "ranging"

def main():
    import argparse, yaml
    from data_engine.normalizer import RunningNormalizer
    from environments.prop_env import MultiTFTradingEnv

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-per-sym", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODELS / "best_v36_stage1.pt", map_location=device, weights_only=False)
    obs_dim = ckpt.get("obs_dim", 416)
    model = AttentionPPO(obs_dim=obs_dim, n_actions=4).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # Disable confidence gate for harvest — we want ALL trades (even low-confidence)
    model.confidence_threshold = 0.0
    logger.info("Loaded V3.6 S1 (step=%d, obs_dim=%d, gate=OFF)", ckpt.get("step", 0), obs_dim)

    with open(PROJECT / "rabit_propfirm_drl" / "configs" / "prop_rules.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["stage1_mode"] = True
    with open(DATA / "normalizer_v3.json") as f:
        nd = json.load(f)
    norms = {k: RunningNormalizer.from_state_dict(v) for k, v in nd.items()}

    memory = ContrastiveMemory(max_per_symbol=args.max_per_sym)

    for sym in SYMBOLS:
        safe = sym.replace(".", "_")
        sd = {}
        for tf in ["M1", "M5", "M15", "H1"]:
            sd[tf] = norms[tf].normalize(np.load(DATA / f"{safe}_{tf}_50dim.npy")).astype(np.float32)
            op = DATA / f"{safe}_{tf}_ohlcv.npy"
            sd[f"{tf}_ohlcv"] = np.load(op).astype(np.float32) if op.exists() else None

        env = MultiTFTradingEnv(
            data_m1=sd["M1"], data_m5=sd["M5"], data_m15=sd["M15"], data_h1=sd["H1"],
            config=cfg, n_features=50, initial_balance=10_000.0,
            episode_length=2000, ohlcv_m5=sd.get("M5_ohlcv"), action_mode="discrete",
        )

        total_trades = 0
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=42 + ep)
            pending = {}  # ticket -> {obs, action, m5_idx}
            prev_trade_count = 0

            for step in range(2000):
                # Record obs BEFORE step
                current_obs = obs.copy()
                m5_idx = min(env.current_m5_step, env.n_m5_bars - 1)
                regime = estimate_regime(sd["M15"], m5_idx)

                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                obs_t = torch.nan_to_num(obs_t, nan=0.0)
                with torch.no_grad():
                    logits, _, _ = model(obs_t)
                    probs = torch.softmax(logits, dim=-1).squeeze(0)
                    dist = torch.distributions.Categorical(probs=probs)
                    action = int(dist.sample().item())
                    action_confidence = float(probs[action])

                # Track new positions
                old_tickets = {p.ticket for p in env.positions}

                obs, _, term, trunc, _ = env.step(action)

                # Detect new positions opened
                new_tickets = {p.ticket for p in env.positions} - old_tickets
                for t in new_tickets:
                    pos = next(p for p in env.positions if p.ticket == t)
                    pending[t] = {
                        "obs": current_obs,
                        "action": 0 if pos.direction > 0 else 1,
                        "regime": regime,
                        "confidence": action_confidence,
                    }

                # Check for closed trades
                new_trade_count = len(env.trade_history)
                if new_trade_count > prev_trade_count:
                    for trade in env.trade_history[prev_trade_count:]:
                        ticket = trade.get("ticket")
                        total_trades += 1
                        if ticket in pending:
                            ed = pending.pop(ticket)
                            pnl = trade.get("pnl", 0)
                            memory.add_trade(
                                obs=ed["obs"],
                                action=ed["action"],
                                pnl=pnl,
                                symbol=sym,
                                regime=ed["regime"],
                                confidence=ed["confidence"],
                            )
                prev_trade_count = new_trade_count

                if term or trunc:
                    break

        stats = memory.stats()[sym]
        logger.info("%-12s | %5d trades | WIN=%4d LOSS=%4d",
                     sym, total_trades, stats["wins"], stats["losses"])

    memory.save(MEMORY_DIR)

    logger.info("=" * 70)
    logger.info("  CONTRASTIVE MEMORY HARVEST COMPLETE")
    total = memory.total_entries()
    logger.info("  Total: %d entries", total)
    for sym in SYMBOLS:
        s = memory.stats()[sym]
        logger.info("  %-12s | WIN=%4d | LOSS=%4d", sym, s["wins"], s["losses"])
    logger.info("  Saved to: %s", MEMORY_DIR)
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
