"""
V3.3 Behavioral Backtest — Evaluate PPO discrete model.

Runs the trained model through each symbol and reports:
- Entry Rate (BUY/SELL/HOLD distribution)
- Win Rate
- Total trades, avg PnL
- Action entropy
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "rabit_propfirm_drl"))

DATA_DIR = project_root / "data"
MODELS_DIR = project_root / "models_saved"
REPORTS_DIR = project_root / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("v33_backtest")


# Import PPO model from train script
from scripts.train_v33 import PPOActorCritic

SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD", "US30_cash", "US100_cash"]


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    obs_dim = ckpt.get("obs_dim", 300)
    n_actions = ckpt.get("n_actions", 3)
    model = PPOActorCritic(obs_dim=obs_dim, n_actions=n_actions).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded model (step=%d, obs=%d, actions=%d)", ckpt.get("step", 0), obs_dim, n_actions)
    return model


def run_eval(model, device, n_episodes=10, episode_length=2000):
    """Run evaluation across all symbols."""
    import yaml as _yaml
    from data_engine.normalizer import RunningNormalizer
    from environments.prop_env import MultiTFTradingEnv

    # Load config
    config_path = project_root / "rabit_propfirm_drl" / "configs" / "prop_rules.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        env_config = _yaml.safe_load(f)
    env_config["stage1_mode"] = True

    # Load normalizer
    norm_path = DATA_DIR / "normalizer_v3.json"
    with open(norm_path, "r") as f:
        norm_data = json.load(f)
    normalizers = {}
    for tf, state in norm_data.items():
        normalizers[tf] = RunningNormalizer.from_state_dict(state)

    results = {}

    for sym in SYMBOLS:
        safe = sym.replace(".", "_")
        logger.info("Evaluating %s (%d episodes x %d steps)...", sym, n_episodes, episode_length)

        # Load data
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

        # Run episodes
        all_buys = 0
        all_sells = 0
        all_holds = 0
        all_trades = []
        episode_results = []
        action_probs_list = []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=42 + ep)
            ep_reward = 0.0
            ep_trades = 0

            for step in range(episode_length):
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=10.0, neginf=-10.0)

                with torch.no_grad():
                    logits, _ = model(obs_t)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    action = int(np.argmax(probs))  # Greedy for eval

                action_probs_list.append(probs)

                if action == 0: all_buys += 1
                elif action == 1: all_sells += 1
                else: all_holds += 1

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward

                if terminated or truncated:
                    break

            ep_trades = info.get("total_trades", len(env.trade_history))
            wins = sum(1 for t in env.trade_history if t.get("pnl", 0) > 0)
            losses = sum(1 for t in env.trade_history if t.get("pnl", 0) <= 0)
            total_pnl = sum(t.get("pnl", 0) for t in env.trade_history)

            episode_results.append({
                "episode": ep, "reward": ep_reward, "trades": ep_trades,
                "wins": wins, "losses": losses, "total_pnl": total_pnl,
                "final_balance": env.balance,
            })
            all_trades.extend(env.trade_history)

        # Aggregate
        total_actions = all_buys + all_sells + all_holds
        total_trades_count = len(all_trades)
        wins_total = sum(1 for t in all_trades if t.get("pnl", 0) > 0)
        losses_total = total_trades_count - wins_total
        win_rate = 100 * wins_total / max(total_trades_count, 1)
        avg_pnl = np.mean([t.get("pnl", 0) for t in all_trades]) if all_trades else 0

        # Action probability analysis
        probs_arr = np.array(action_probs_list)
        mean_probs = np.mean(probs_arr, axis=0) if len(probs_arr) > 0 else [0.33, 0.33, 0.33]
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8))

        sym_result = {
            "symbol": sym,
            "total_steps": total_actions,
            "pct_buy": 100 * all_buys / max(total_actions, 1),
            "pct_sell": 100 * all_sells / max(total_actions, 1),
            "pct_hold": 100 * all_holds / max(total_actions, 1),
            "total_trades": total_trades_count,
            "wins": wins_total,
            "losses": losses_total,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "entropy": entropy,
            "mean_probs": mean_probs.tolist(),
            "episodes": episode_results,
        }

        logger.info("  %s: %d trades | WR=%.1f%% | BUY=%.1f%% SELL=%.1f%% HOLD=%.1f%% | H=%.3f | PnL=%.2f",
                     sym, total_trades_count, win_rate,
                     sym_result["pct_buy"], sym_result["pct_sell"], sym_result["pct_hold"],
                     entropy, avg_pnl)
        results[sym] = sym_result

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    ckpt_path = MODELS_DIR / "best_v33_stage1.pt"
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    model = load_model(ckpt_path, device)

    start = time.time()
    results = run_eval(model, device, n_episodes=10, episode_length=2000)
    elapsed = time.time() - start

    # Save JSON
    report_path = REPORTS_DIR / "v33_behavioral_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved: %s", report_path)

    # Summary
    logger.info("=" * 70)
    logger.info("  V3.3 BEHAVIORAL ANALYSIS COMPLETE (%.1fs)", elapsed)
    logger.info("=" * 70)

    total_trades_all = sum(r["total_trades"] for r in results.values())
    total_wins_all = sum(r["wins"] for r in results.values())
    overall_wr = 100 * total_wins_all / max(total_trades_all, 1)

    for sym, r in results.items():
        status = "PASS" if r["win_rate"] > 40 else "WATCH" if r["total_trades"] > 0 else "FAIL"
        logger.info("  %-12s | Trades=%4d | WR=%5.1f%% | BUY=%5.1f%% SELL=%5.1f%% HOLD=%5.1f%% | %s",
                     sym, r["total_trades"], r["win_rate"],
                     r["pct_buy"], r["pct_sell"], r["pct_hold"], status)

    logger.info("-" * 70)
    logger.info("  OVERALL: %d trades | Win Rate = %.1f%% | Target > 40%%", total_trades_all, overall_wr)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
