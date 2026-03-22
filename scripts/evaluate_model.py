"""
Model Evaluation Script — Phase 3 Acceptance Testing.

Loads best_Stage3_FullFusion.pt and runs full backtest across 5 symbols
using the last 20% of data (unseen holdout set) per symbol.

Outputs: structured JSON results + console summary.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "rabit_propfirm_drl"))

DATA_DIR = project_root / "data"
MODELS_DIR = project_root / "models_saved"
CONFIG_PATH = project_root / "rabit_propfirm_drl" / "configs" / "prop_rules.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluator")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. CHECKPOINT INSPECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def inspect_checkpoint(path: Path) -> dict:
    """Load and inspect checkpoint metadata."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    info = {
        "file_size_mb": path.stat().st_size / (1024 * 1024),
        "keys": list(ckpt.keys()),
        "stage": ckpt.get("stage", "UNKNOWN"),
        "step": ckpt.get("step", -1),
    }
    
    # Count parameters
    if "actor_state_dict" in ckpt:
        actor_params = sum(v.numel() for v in ckpt["actor_state_dict"].values())
        info["actor_params"] = actor_params
    if "critic_state_dict" in ckpt:
        critic_params = sum(v.numel() for v in ckpt["critic_state_dict"].values())
        info["critic_params"] = critic_params
    
    # Check for NaN/Inf in weights
    nan_count = 0
    inf_count = 0
    weight_stats = {}
    for key, val in ckpt.get("actor_state_dict", {}).items():
        nan_count += torch.isnan(val).sum().item()
        inf_count += torch.isinf(val).sum().item()
        if "weight" in key and val.dim() >= 2:
            weight_stats[key] = {
                "mean": val.float().mean().item(),
                "std": val.float().std().item(),
                "min": val.float().min().item(),
                "max": val.float().max().item(),
            }
    
    info["nan_params"] = nan_count
    info["inf_params"] = inf_count
    info["weight_health"] = "HEALTHY" if nan_count == 0 and inf_count == 0 else "CORRUPTED"
    info["weight_stats_sample"] = dict(list(weight_stats.items())[:10])
    
    return info, ckpt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. LOAD MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_model(ckpt: dict, device: torch.device):
    """Instantiate actor and load weights."""
    from agents.sac_policy import SACTransformerActor
    
    actor = SACTransformerActor(
        n_features=50,
        action_dim=4,
        embed_dim=128,
        n_heads=4,
        n_cross_layers=1,
        n_regimes=4,
        hidden_dims=[256, 256],
        dropout=0.1,
    ).to(device)
    
    actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
    actor.eval()
    
    total_params = sum(p.numel() for p in actor.parameters())
    logger.info("Actor loaded: %d parameters", total_params)
    
    return actor


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. LOAD DATA (Holdout Split)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYMBOL_FILE_MAP = {
    "XAUUSD": "XAUUSD",
    "BTCUSD": "BTCUSD",
    "ETHUSD": "ETHUSD",
    "US30": "US30_cash",
    "USTEC": "US100_cash",
}

def load_symbol_data(symbol_key: str, holdout_ratio: float = 0.20) -> dict:
    """Load last holdout_ratio of data for evaluation."""
    file_prefix = SYMBOL_FILE_MAP[symbol_key]
    data = {}
    
    for tf in ["M1", "M5", "M15", "H1"]:
        fpath = DATA_DIR / f"{file_prefix}_{tf}_50dim.npy"
        if not fpath.exists():
            logger.warning("Missing %s", fpath.name)
            return None
        arr = np.load(fpath)
        # Take last holdout_ratio as unseen test data
        split_idx = int(len(arr) * (1.0 - holdout_ratio))
        data[tf] = arr[split_idx:].astype(np.float32)
        logger.info("  %s %s: full=%d, holdout=%d (from idx %d)",
                    symbol_key, tf, len(arr), len(data[tf]), split_idx)
    
    return data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. BACKTEST ENGINE (Vectorized Rollout)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def run_backtest(
    actor: torch.nn.Module,
    data: dict,
    config: dict,
    device: torch.device,
    symbol: str,
    n_episodes: int = 10,
    episode_length: int = 2000,
) -> dict:
    """
    Run deterministic backtest episodes and collect trade statistics.
    """
    from environments.prop_env import MultiTFTradingEnv
    
    env = MultiTFTradingEnv(
        data_m1=data["M1"],
        data_m5=data["M5"],
        data_m15=data["M15"],
        data_h1=data["H1"],
        config=config,
        n_features=50,
        initial_balance=10_000.0,
        episode_length=episode_length,
        pip_value=config.get("symbol_configs", {}).get(symbol, {}).get("pip_value", 0.01),
    )
    
    all_trades = []
    episode_results = []
    action_distributions = []
    confidence_values = []
    steps_between_trades_list = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep * 1000 + 42)
        
        ep_reward = 0.0
        ep_steps = 0
        last_trade_step = 0
        
        done = False
        while not done:
            # Prepare tensors
            m1_t = torch.from_numpy(obs["m1"]).unsqueeze(0).to(device)
            m5_t = torch.from_numpy(obs["m5"]).unsqueeze(0).to(device)
            m15_t = torch.from_numpy(obs["m15"]).unsqueeze(0).to(device)
            h1_t = torch.from_numpy(obs["h1"]).unsqueeze(0).to(device)
            
            # Deterministic action
            action, log_prob = actor(m1_t, m5_t, m15_t, h1_t, deterministic=True)
            action_np = action.cpu().numpy().flatten()
            action_distributions.append(action_np.copy())
            confidence_values.append(float(action_np[0]))
            
            # Scale action to env bounds
            env_action = np.array([
                float(np.clip(action_np[0], -1.0, 1.0)),
                float(np.clip((action_np[1] + 1) / 2, 0.0, 1.0)),  # tanh→[0,1]
                float(np.clip(action_np[2] * 1.25 + 1.75, 0.5, 3.0)),  # scale sl_mult
                float(np.clip(action_np[3] * 2.25 + 2.75, 0.5, 5.0)),  # scale tp_mult
            ], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(env_action)
            ep_reward += reward
            ep_steps += 1
            done = terminated or truncated
            
            prev_trades = len(all_trades) + len(env.trade_history)
        
        # Collect episode trades
        for trade in env.trade_history:
            trade["symbol"] = symbol
            trade["episode"] = ep
            all_trades.append(trade)
            
        # Record steps between trades
        if len(env.trade_history) > 0:
            for i, t in enumerate(env.trade_history):
                if i == 0:
                    steps_between_trades_list.append(t.get("entry_step", 0) if "entry_step" in t else 0)
                else:
                    prev_t = env.trade_history[i-1]
                    gap = t.get("duration", 0)
                    steps_between_trades_list.append(gap)
        
        episode_results.append({
            "episode": ep,
            "total_reward": ep_reward,
            "steps": ep_steps,
            "final_balance": info["balance"],
            "final_equity": info["equity"],
            "max_dd_daily": info.get("daily_dd", 0),
            "max_dd_total": info.get("total_dd", 0),
            "n_trades": len(env.trade_history),
            "terminated": terminated,
            "truncated": truncated,
        })
        
        logger.info("  [%s] Episode %d/%d: reward=%.2f, balance=%.2f, trades=%d, steps=%d",
                    symbol, ep+1, n_episodes, ep_reward, info["balance"], 
                    len(env.trade_history), ep_steps)
    
    # ── Aggregate stats ──
    action_dist = np.array(action_distributions) if action_distributions else np.zeros((1,4))
    confidences = np.array(confidence_values) if confidence_values else np.zeros(1)
    
    # Trade analysis
    wins = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    
    total_trades = len(all_trades)
    win_rate = len(wins) / max(total_trades, 1) * 100
    
    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 1
    profit_factor = gross_profit / max(gross_loss, 0.01)
    
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    
    avg_duration = np.mean([t["duration"] for t in all_trades]) if all_trades else 0
    
    # Trading frequency
    total_steps = sum(ep["steps"] for ep in episode_results)
    trades_per_step = total_trades / max(total_steps, 1)
    # M5 bars, approx 288 bars/day (24h * 12 bars/h)
    trades_per_day = trades_per_step * 288
    
    # Confidence distribution
    above_threshold = np.sum(np.abs(confidences) >= 0.3) / max(len(confidences), 1) * 100
    
    # Consecutive losses
    max_consec_loss = 0
    curr_consec = 0
    for t in all_trades:
        if t["pnl"] <= 0:
            curr_consec += 1
            max_consec_loss = max(max_consec_loss, curr_consec)
        else:
            curr_consec = 0
    
    # Balance curve
    balances = [ep["final_balance"] for ep in episode_results]
    avg_final_balance = np.mean(balances) if balances else 10000
    
    return {
        "symbol": symbol,
        "n_episodes": n_episodes,
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "net_pnl": round(gross_profit - gross_loss, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_rr": round(abs(avg_win) / max(abs(avg_loss), 0.01), 2) if avg_loss != 0 else 0,
        "max_consecutive_losses": max_consec_loss,
        "avg_trade_duration_steps": round(avg_duration, 1),
        "trades_per_day": round(trades_per_day, 2),
        "pct_steps_active": round(above_threshold, 1),
        "avg_final_balance": round(avg_final_balance, 2),
        "avg_return_pct": round((avg_final_balance - 10000) / 100, 2),
        "action_stats": {
            "confidence_mean": round(float(action_dist[:, 0].mean()), 4),
            "confidence_std": round(float(action_dist[:, 0].std()), 4),
            "risk_frac_mean": round(float(action_dist[:, 1].mean()), 4),
            "sl_mult_mean": round(float(action_dist[:, 2].mean()), 4),
            "tp_mult_mean": round(float(action_dist[:, 3].mean()), 4),
        },
        "episode_details": episode_results,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. WEIGHT ANALYSIS (Overfitting / Catastrophic Forgetting)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def analyze_weights(ckpt: dict) -> dict:
    """Analyze weight distributions for overfitting signals."""
    actor_sd = ckpt.get("actor_state_dict", {})
    
    layers = {}
    dead_neurons = 0
    total_neurons = 0
    
    for name, tensor in actor_sd.items():
        if "weight" in name and tensor.dim() >= 2:
            t = tensor.float()
            std = t.std().item()
            mean = t.mean().item()
            max_val = t.abs().max().item()
            
            # Check for dead neurons (all near-zero row)
            row_norms = t.norm(dim=-1)
            dead = (row_norms < 1e-6).sum().item()
            dead_neurons += dead
            total_neurons += row_norms.numel()
            
            layers[name] = {
                "shape": list(tensor.shape),
                "mean": round(mean, 6),
                "std": round(std, 6),
                "max_abs": round(max_val, 6),
                "dead_rows": dead,
            }
    
    # Check encoder balance (Catastrophic Forgetting indicator)
    encoder_stds = {}
    for name, info in layers.items():
        if "m1_encoder" in name or "m5_encoder" in name:
            encoder_stds[name] = info["std"]
        elif "m15_encoder" in name or "h1_encoder" in name:
            encoder_stds[name] = info["std"]
    
    # If Stage 1 encoders (M15/H1) have much lower std than Stage 3 (M1),
    # it may indicate catastrophic forgetting
    m1_stds = [v for k, v in encoder_stds.items() if "m1" in k]
    m15h1_stds = [v for k, v in encoder_stds.items() if "m15" in k or "h1" in k]
    
    avg_m1_std = np.mean(m1_stds) if m1_stds else 0
    avg_m15h1_std = np.mean(m15h1_stds) if m15h1_stds else 0
    
    balance_ratio = avg_m15h1_std / max(avg_m1_std, 1e-10) if avg_m1_std > 0 else 1.0
    
    return {
        "total_layers_analyzed": len(layers),
        "dead_neurons": dead_neurons,
        "total_neurons": total_neurons,
        "dead_neuron_pct": round(dead_neurons / max(total_neurons, 1) * 100, 2),
        "m1_encoder_avg_std": round(avg_m1_std, 6),
        "m15h1_encoder_avg_std": round(avg_m15h1_std, 6),
        "encoder_balance_ratio": round(balance_ratio, 3),
        "forgetting_risk": "LOW" if 0.5 < balance_ratio < 2.0 else "HIGH",
        "layer_details_sample": dict(list(layers.items())[:8]),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    
    ckpt_path = MODELS_DIR / "best_Stage3_FullFusion.pt"
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return
    
    # Load config
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 1. Inspect checkpoint
    logger.info("=" * 60)
    logger.info("  CHECKPOINT INSPECTION")
    logger.info("=" * 60)
    ckpt_info, ckpt = inspect_checkpoint(ckpt_path)
    for k, v in ckpt_info.items():
        if k != "weight_stats_sample":
            logger.info("  %s: %s", k, v)
    
    # 2. Weight analysis
    logger.info("=" * 60)
    logger.info("  WEIGHT ANALYSIS (Overfitting / Forgetting)")
    logger.info("=" * 60)
    weight_analysis = analyze_weights(ckpt)
    for k, v in weight_analysis.items():
        if k != "layer_details_sample":
            logger.info("  %s: %s", k, v)
    
    # 3. Load model
    logger.info("=" * 60)
    logger.info("  LOADING MODEL")
    logger.info("=" * 60)
    actor = load_model(ckpt, device)
    
    # 4. Run backtests per symbol
    symbols = ["XAUUSD", "BTCUSD", "ETHUSD", "US30", "USTEC"]
    all_results = {}
    
    for sym in symbols:
        logger.info("=" * 60)
        logger.info("  BACKTEST: %s", sym)
        logger.info("=" * 60)
        
        data = load_symbol_data(sym, holdout_ratio=0.20)
        if data is None:
            logger.error("Skipping %s — missing data", sym)
            continue
        
        results = run_backtest(
            actor=actor,
            data=data,
            config=config,
            device=device,
            symbol=sym,
            n_episodes=10,
            episode_length=2000,
        )
        all_results[sym] = results
    
    # 5. Aggregate
    total_trades = sum(r["total_trades"] for r in all_results.values())
    total_wins = sum(
        int(r["win_rate_pct"] * r["total_trades"] / 100) for r in all_results.values()
    )
    overall_wr = total_wins / max(total_trades, 1) * 100
    
    overall_pf_num = sum(r["gross_profit"] for r in all_results.values())
    overall_pf_den = sum(r["gross_loss"] for r in all_results.values())
    overall_pf = overall_pf_num / max(overall_pf_den, 0.01)
    
    avg_trades_per_day = np.mean([r["trades_per_day"] for r in all_results.values()])
    
    summary = {
        "checkpoint": str(ckpt_path.name),
        "stage": ckpt_info["stage"],
        "training_steps": ckpt_info["step"],
        "weight_health": ckpt_info["weight_health"],
        "forgetting_risk": weight_analysis["forgetting_risk"],
        "encoder_balance_ratio": weight_analysis["encoder_balance_ratio"],
        "dead_neuron_pct": weight_analysis["dead_neuron_pct"],
        "overall_win_rate_pct": round(overall_wr, 2),
        "overall_profit_factor": round(overall_pf, 3),
        "total_trades": total_trades,
        "avg_trades_per_day": round(avg_trades_per_day, 2),
        "per_symbol": all_results,
    }
    
    # Save results
    output_path = MODELS_DIR / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        # Remove episode_details for cleaner JSON
        save_data = json.loads(json.dumps(summary, default=str))
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    logger.info("=" * 60)
    logger.info("  EVALUATION COMPLETE")
    logger.info("  Results saved to: %s", output_path)
    logger.info("=" * 60)
    
    # Print summary
    print("\n" + "=" * 60)
    print("  OVERALL RESULTS")
    print("=" * 60)
    print(f"  Win Rate:       {overall_wr:.1f}%")
    print(f"  Profit Factor:  {overall_pf:.3f}")
    print(f"  Total Trades:   {total_trades}")
    print(f"  Avg Trades/Day: {avg_trades_per_day:.1f}")
    print(f"  Weight Health:  {ckpt_info['weight_health']}")
    print(f"  Forgetting:     {weight_analysis['forgetting_risk']}")
    print()
    
    for sym, r in all_results.items():
        print(f"  {sym:10s} | WR={r['win_rate_pct']:5.1f}% | PF={r['profit_factor']:5.3f} | "
              f"Trades={r['total_trades']:3d} | Net={r['net_pnl']:+8.2f} | "
              f"MaxConsecLoss={r['max_consecutive_losses']}")
    
    print("=" * 60)
    
    return summary


if __name__ == "__main__":
    main()
