"""
Behavioral Analysis Backtest — 3-Stage Model Evaluation.

Loads all 3 curriculum checkpoints (Stage1/Stage2/Stage3) and runs
comprehensive behavioral analysis on the OOS holdout set.

Modules:
    1. Action Distribution (mode collapse detection)
    2. Holding Time vs PnL Scatter (holding patterns)
    3. Signal Reaction (SMC/Volume/PinAction)
    4. Stage Evolution Comparison

Outputs:
    - reports/behavioral_report.json
    - reports/action_distribution.png
    - reports/holding_scatter.png
    - reports/signal_reaction.png
    - reports/stage_evolution.png

Usage:
    python scripts/backtest_behavioral.py           # Full run
    python scripts/backtest_behavioral.py --test     # Quick test (2 eps, 500 steps)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "rabit_propfirm_drl"))

DATA_DIR = project_root / "data"
MODELS_DIR = project_root / "models_saved"
REPORTS_DIR = project_root / "reports"
CONFIG_PATH = project_root / "rabit_propfirm_drl" / "configs" / "prop_rules.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("behavioral")

# ── Feature Column Indices (50-dim = 28 raw + 22 knowledge) ──
# Knowledge features start at index 28
# SMC (7): idx 28-34
# PA  (8): idx 35-42  → is_pinbar=35, is_doji=36, ...
# Vol (5): idx 43-47  → vol_anomaly=43, vol_exhaustion=44, vol_climax=45
# Ctx (2): idx 48-49
IDX_PINBAR = 35
IDX_DOJI = 36
IDX_ENGULFING_BULL = 38
IDX_ENGULFING_BEAR = 39
IDX_HAMMER = 40
IDX_SHOOTING_STAR = 41
IDX_VOL_ANOMALY = 43
IDX_VOL_CLIMAX = 45
# Raw features: ob_bull_dist=18, ob_bear_dist=19
IDX_OB_BULL_DIST = 18
IDX_OB_BEAR_DIST = 19

SYMBOL_FILE_MAP = {
    "XAUUSD": "XAUUSD",
    "BTCUSD": "BTCUSD",
    "ETHUSD": "ETHUSD",
    "US30": "US30_cash",
    "USTEC": "US100_cash",
}

STAGE_CHECKPOINTS = {
    "Stage1_Context": "best_Stage1_Context.pt",
    "Stage2_Precision": "best_Stage2_Precision.pt",
    "Stage3_FullFusion": "best_Stage3_FullFusion.pt",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_symbol_data(symbol_key: str, holdout_ratio: float = 0.20) -> dict | None:
    """Load last holdout_ratio of data for evaluation."""
    file_prefix = SYMBOL_FILE_MAP[symbol_key]
    data = {}
    for tf in ["M1", "M5", "M15", "H1"]:
        fpath = DATA_DIR / f"{file_prefix}_{tf}_50dim.npy"
        if not fpath.exists():
            logger.warning("Missing %s", fpath.name)
            return None
        arr = np.load(fpath)
        split_idx = int(len(arr) * (1.0 - holdout_ratio))
        data[tf] = arr[split_idx:].astype(np.float32)
    return data


def load_model(ckpt_path: Path, device: torch.device):
    """Load actor from checkpoint."""
    from agents.sac_policy import SACTransformerActor

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    actor = SACTransformerActor(
        n_features=50, action_dim=4 if "action_dim" not in str(ckpt.get("stage","")) else 5,
        embed_dim=128, n_heads=4, n_cross_layers=1, n_regimes=4,
        hidden_dims=[256, 256], dropout=0.1,
    )
    # Try 5-dim first, fallback to 4-dim
    try:
        actor = SACTransformerActor(
            n_features=50, action_dim=5, embed_dim=128, n_heads=4,
            n_cross_layers=1, n_regimes=4, hidden_dims=[256, 256], dropout=0.1,
        ).to(device)
        actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
        logger.info("Loaded with action_dim=5")
    except RuntimeError:
        actor = SACTransformerActor(
            n_features=50, action_dim=4, embed_dim=128, n_heads=4,
            n_cross_layers=1, n_regimes=4, hidden_dims=[256, 256], dropout=0.1,
        ).to(device)
        actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
        logger.info("Loaded with action_dim=4 (legacy)")
    actor.eval()
    return actor, ckpt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BACKTEST ENGINE (Fixed 5-dim action decode)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def decode_action_5dim(action_np: np.ndarray) -> dict:
    """Decode 5-dim action vector — CORRECT mapping."""
    return {
        "confidence": float(np.clip(action_np[0], -1.0, 1.0)),
        "entry_type": float(np.clip(action_np[1], -1.0, 1.0)),
        "risk_frac": float(np.clip((action_np[2] + 1) / 2, 0.0, 1.0)),
        "sl_mult": float(np.clip(action_np[3] * 1.25 + 1.75, 0.5, 3.0)),
        "tp_mult": float(np.clip(action_np[4] * 2.25 + 2.75, 0.5, 5.0)),
    }


def decode_action_4dim(action_np: np.ndarray) -> dict:
    """Decode legacy 4-dim action vector."""
    return {
        "confidence": float(np.clip(action_np[0], -1.0, 1.0)),
        "entry_type": 0.0,  # No entry_type in legacy
        "risk_frac": float(np.clip((action_np[1] + 1) / 2, 0.0, 1.0)),
        "sl_mult": float(np.clip(action_np[2] * 1.25 + 1.75, 0.5, 3.0)),
        "tp_mult": float(np.clip(action_np[3] * 2.25 + 2.75, 0.5, 5.0)),
    }


def classify_action(decoded: dict, m5_thresh: float, m1_thresh: float) -> str:
    """Classify decoded action into BUY/SELL/HOLD with dual entry gating."""
    abs_conf = abs(decoded["confidence"])
    et = decoded["entry_type"]

    if et > 0 and abs_conf >= m1_thresh:
        return "BUY" if decoded["confidence"] > 0 else "SELL"
    elif et <= 0 and abs_conf >= m5_thresh:
        return "BUY" if decoded["confidence"] > 0 else "SELL"
    return "HOLD"


@torch.no_grad()
def run_behavioral_backtest(
    actor, data: dict, config: dict, device: torch.device,
    symbol: str, m5_thresh: float, m1_thresh: float,
    n_episodes: int = 10, episode_length: int = 2000,
) -> dict:
    """Run backtest and collect detailed behavioral data."""
    from environments.prop_env import MultiTFTradingEnv

    env = MultiTFTradingEnv(
        data_m1=data["M1"], data_m5=data["M5"],
        data_m15=data["M15"], data_h1=data["H1"],
        config=config, n_features=50, initial_balance=10_000.0,
        episode_length=episode_length,
        pip_value=config.get("symbol_configs", {}).get(symbol, {}).get("pip_value", 0.01),
    )
    # Override thresholds for force-trigger test
    env.m5_threshold = m5_thresh
    env.m1_threshold = m1_thresh

    action_dim = None
    all_decoded_actions = []
    all_classifications = []
    all_trades = []
    all_confidences = []
    all_entry_types = []
    signal_moments = []  # (step, signal_type, action_class, confidence)
    episode_results = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep * 1000 + 42)
        ep_reward = 0.0
        done = False
        step_idx = 0

        while not done:
            m1_t = torch.from_numpy(obs["m1"]).unsqueeze(0).to(device)
            m5_t = torch.from_numpy(obs["m5"]).unsqueeze(0).to(device)
            m15_t = torch.from_numpy(obs["m15"]).unsqueeze(0).to(device)
            h1_t = torch.from_numpy(obs["h1"]).unsqueeze(0).to(device)

            action, _ = actor(m1_t, m5_t, m15_t, h1_t, deterministic=True)
            action_np = action.cpu().numpy().flatten()

            if action_dim is None:
                action_dim = len(action_np)
            decode_fn = decode_action_5dim if action_dim >= 5 else decode_action_4dim
            decoded = decode_fn(action_np)
            action_class = classify_action(decoded, m5_thresh, m1_thresh)

            all_decoded_actions.append(decoded)
            all_classifications.append(action_class)
            all_confidences.append(decoded["confidence"])
            all_entry_types.append(decoded["entry_type"])

            # ── Signal Detection on current M5 obs ──
            m5_bar = obs["m5"][-1]  # Last bar in window
            signals_here = []
            if m5_bar[IDX_PINBAR] > 0.5:
                signals_here.append("pinbar")
            if m5_bar[IDX_HAMMER] > 0.5:
                signals_here.append("hammer")
            if m5_bar[IDX_SHOOTING_STAR] > 0.5:
                signals_here.append("shooting_star")
            if m5_bar[IDX_VOL_ANOMALY] > 0.5:
                signals_here.append("vol_spike")
            if m5_bar[IDX_VOL_CLIMAX] > 0.5:
                signals_here.append("vol_climax")
            if m5_bar[IDX_ENGULFING_BULL] > 0.5:
                signals_here.append("engulfing_bull")
            if m5_bar[IDX_ENGULFING_BEAR] > 0.5:
                signals_here.append("engulfing_bear")
            if m5_bar[IDX_OB_BULL_DIST] < 0.15:
                signals_here.append("near_ob_bull")
            if m5_bar[IDX_OB_BEAR_DIST] < 0.15:
                signals_here.append("near_ob_bear")

            for sig in signals_here:
                signal_moments.append({
                    "ep": ep, "step": step_idx, "signal": sig,
                    "action": action_class, "confidence": decoded["confidence"],
                    "entry_type": decoded["entry_type"],
                })

            # Build env action — pass RAW tanh outputs to env.
            # The env handles its own clipping in step().
            if action_dim >= 5:
                env_action = np.array([
                    float(action_np[0]),   # confidence (raw tanh)
                    float(action_np[1]),   # entry_type (raw tanh)
                    float(np.clip((action_np[2] + 1) / 2, 0.0, 1.0)),  # risk_frac
                    float(np.clip(action_np[3] * 1.25 + 1.75, 0.5, 3.0)),  # sl_mult
                    float(np.clip(action_np[4] * 2.25 + 2.75, 0.5, 5.0)),  # tp_mult
                ], dtype=np.float32)
            else:
                env_action = np.array([
                    float(action_np[0]),     # confidence
                    0.0,                     # entry_type (legacy: M5 default)
                    float(np.clip((action_np[1] + 1) / 2, 0.0, 1.0)),
                    float(np.clip(action_np[2] * 1.25 + 1.75, 0.5, 3.0)),
                    float(np.clip(action_np[3] * 2.25 + 2.75, 0.5, 5.0)),
                ], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(env_action)
            ep_reward += reward
            step_idx += 1
            done = terminated or truncated

        for trade in env.trade_history:
            trade["symbol"] = symbol
            trade["episode"] = ep
            all_trades.append(trade)

        episode_results.append({
            "episode": ep, "total_reward": ep_reward, "steps": step_idx,
            "final_balance": info["balance"], "n_trades": len(env.trade_history),
        })

    # ── Aggregate ──
    confs = np.array(all_confidences)
    etypes = np.array(all_entry_types)
    n_buy = sum(1 for c in all_classifications if c == "BUY")
    n_sell = sum(1 for c in all_classifications if c == "SELL")
    n_hold = sum(1 for c in all_classifications if c == "HOLD")
    total = max(len(all_classifications), 1)

    # Shannon entropy of action distribution
    probs = np.array([n_buy/total, n_sell/total, n_hold/total])
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs))) if len(probs) > 0 else 0.0

    wins = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]

    return {
        "symbol": symbol,
        "action_dim_detected": action_dim,
        "total_steps": total,
        "n_buy": n_buy, "n_sell": n_sell, "n_hold": n_hold,
        "pct_buy": round(n_buy / total * 100, 2),
        "pct_sell": round(n_sell / total * 100, 2),
        "pct_hold": round(n_hold / total * 100, 2),
        "action_entropy": round(entropy, 4),
        "confidence_mean": round(float(confs.mean()), 6),
        "confidence_std": round(float(confs.std()), 6),
        "confidence_abs_mean": round(float(np.abs(confs).mean()), 6),
        "entry_type_mean": round(float(etypes.mean()), 6),
        "total_trades": len(all_trades),
        "win_rate": round(len(wins) / max(len(all_trades), 1) * 100, 2),
        "avg_hold_win": round(float(np.mean([t["duration"] for t in wins])), 1) if wins else 0,
        "avg_hold_loss": round(float(np.mean([t["duration"] for t in losses])), 1) if losses else 0,
        "trades": all_trades,
        "signal_moments": signal_moments,
        "episode_results": episode_results,
        "mode_collapse": n_hold / total > 0.85 or n_buy / total > 0.85 or n_sell / total > 0.85,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 1: Action Distribution Chart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_action_distribution(results_by_stage: dict, report_label: str, save_path: Path):
    """Stacked bar chart of BUY/SELL/HOLD across 3 stages."""
    fig, ax = plt.subplots(figsize=(12, 6))
    stages = list(results_by_stage.keys())
    x = np.arange(len(stages))
    width = 0.6

    buys = [sum(r["pct_buy"] for r in results_by_stage[s].values()) / max(len(results_by_stage[s]), 1) for s in stages]
    sells = [sum(r["pct_sell"] for r in results_by_stage[s].values()) / max(len(results_by_stage[s]), 1) for s in stages]
    holds = [sum(r["pct_hold"] for r in results_by_stage[s].values()) / max(len(results_by_stage[s]), 1) for s in stages]

    ax.bar(x, buys, width, label="BUY", color="#2ecc71")
    ax.bar(x, sells, width, bottom=buys, label="SELL", color="#e74c3c")
    ax.bar(x, holds, width, bottom=[b+s for b,s in zip(buys, sells)], label="HOLD", color="#95a5a6")

    ax.set_ylabel("% Actions")
    ax.set_title(f"Action Distribution — {report_label}\n(>85% single action = MODE COLLAPSE ⚠️)")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 105)

    # Add entropy labels
    for i, s in enumerate(stages):
        avg_ent = np.mean([r["action_entropy"] for r in results_by_stage[s].values()])
        ax.text(i, 102, f"H={avg_ent:.2f}", ha="center", fontsize=8, color="navy")

    for i, s in enumerate(stages):
        for sym, r in results_by_stage[s].items():
            if r["mode_collapse"]:
                ax.text(i, 95, "⚠ COLLAPSE", ha="center", fontsize=10, color="red", weight="bold")
                break

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 2: Holding Time vs PnL Scatter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_holding_scatter(results_by_stage: dict, report_label: str, save_path: Path):
    """Scatter: holding time (X) vs PnL (Y), per stage subplot."""
    stages = list(results_by_stage.keys())
    fig, axes = plt.subplots(1, len(stages), figsize=(6*len(stages), 5), sharey=True)
    if len(stages) == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages):
        all_trades = []
        for r in results_by_stage[stage].values():
            all_trades.extend(r["trades"])

        if not all_trades:
            ax.text(0.5, 0.5, "NO TRADES", transform=ax.transAxes, ha="center", fontsize=14, color="red")
            ax.set_title(f"{stage}")
            continue

        durations = [t["duration"] for t in all_trades]
        pnls = [t["pnl"] for t in all_trades]
        colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in pnls]
        sizes = [max(10, min(80, abs(t.get("lots", 0.01)) * 500)) for t in all_trades]

        ax.scatter(durations, pnls, c=colors, s=sizes, alpha=0.6, edgecolors="k", linewidths=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Holding Time (M5 bars)")
        ax.set_title(f"{stage} ({len(all_trades)} trades)")

        # Stats annotation
        wins = [t for t in all_trades if t["pnl"] > 0]
        losses = [t for t in all_trades if t["pnl"] <= 0]
        avg_hold_win = np.mean([t["duration"] for t in wins]) if wins else 0
        avg_hold_loss = np.mean([t["duration"] for t in losses]) if losses else 0
        ax.text(0.02, 0.98, f"Avg Hold Win: {avg_hold_win:.0f}\nAvg Hold Loss: {avg_hold_loss:.0f}",
                transform=ax.transAxes, va="top", fontsize=8, bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

    axes[0].set_ylabel("PnL ($)")
    fig.suptitle(f"Holding Time vs PnL — {report_label}", fontsize=13, weight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 3: Signal Reaction Heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SIGNAL_TYPES = [
    "pinbar", "hammer", "shooting_star", "vol_spike", "vol_climax",
    "engulfing_bull", "engulfing_bear", "near_ob_bull", "near_ob_bear",
]

def plot_signal_reaction(results_by_stage: dict, report_label: str, save_path: Path):
    """Heatmap: signal type vs action taken, per stage."""
    stages = list(results_by_stage.keys())
    fig, axes = plt.subplots(1, len(stages), figsize=(6*len(stages), 5))
    if len(stages) == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages):
        all_signals = []
        for r in results_by_stage[stage].values():
            all_signals.extend(r["signal_moments"])

        matrix = np.zeros((len(SIGNAL_TYPES), 3))  # rows=signals, cols=BUY/SELL/HOLD
        action_map = {"BUY": 0, "SELL": 1, "HOLD": 2}
        for sm in all_signals:
            if sm["signal"] in SIGNAL_TYPES:
                row = SIGNAL_TYPES.index(sm["signal"])
                col = action_map.get(sm["action"], 2)
                matrix[row, col] += 1

        # Normalize per row
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix_pct = matrix / row_sums * 100

        im = ax.imshow(matrix_pct, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["BUY", "SELL", "HOLD"])
        ax.set_yticks(range(len(SIGNAL_TYPES)))
        ax.set_yticklabels(SIGNAL_TYPES, fontsize=8)
        ax.set_title(f"{stage} (n={len(all_signals)})")

        for i in range(len(SIGNAL_TYPES)):
            for j in range(3):
                count = int(matrix[i, j])
                if count > 0:
                    ax.text(j, i, f"{matrix_pct[i,j]:.0f}%\n({count})",
                            ha="center", va="center", fontsize=7)

    fig.suptitle(f"Signal Reaction — {report_label}", fontsize=13, weight="bold")
    plt.colorbar(im, ax=axes[-1], label="% of Reactions")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 4: Stage Evolution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_stage_evolution(results_by_stage: dict, report_label: str, save_path: Path):
    """Bar chart comparing key metrics across stages."""
    stages = list(results_by_stage.keys())

    def avg_metric(stage, key):
        vals = [r[key] for r in results_by_stage[stage].values() if key in r]
        return np.mean(vals) if vals else 0

    metrics = {
        "Trade Frequency\n(trades total)": lambda s: sum(r["total_trades"] for r in results_by_stage[s].values()),
        "Win Rate (%)": lambda s: avg_metric(s, "win_rate"),
        "Action Entropy\n(diversity)": lambda s: avg_metric(s, "action_entropy"),
        "Confidence |μ|": lambda s: avg_metric(s, "confidence_abs_mean"),
        "% HOLD": lambda s: avg_metric(s, "pct_hold"),
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 5))
    x = np.arange(len(stages))
    colors = ["#3498db", "#e67e22", "#9b59b6"]

    for ax, (title, fn) in zip(axes, metrics.items()):
        vals = [fn(s) for s in stages]
        bars = ax.bar(x, vals, color=colors[:len(stages)], width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([s.split("_")[0] for s in stages], fontsize=8)
        ax.set_title(title, fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(max(vals, default=1), 1),
                    f"{v:.1f}", ha="center", fontsize=8)

    fig.suptitle(f"Stage Evolution — {report_label}", fontsize=13, weight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN — Dual Report (A: Standard, B: Force-Trigger)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_full_analysis(
    thresholds: dict, report_label: str, suffix: str,
    device: torch.device, config: dict,
    symbols: list[str], n_episodes: int, episode_length: int,
) -> dict:
    """Run full behavioral analysis for one threshold setting."""
    m5_thresh = thresholds["m5"]
    m1_thresh = thresholds["m1"]

    logger.info("═" * 60)
    logger.info("  %s — M5 thresh=%.2f, M1 thresh=%.2f", report_label, m5_thresh, m1_thresh)
    logger.info("═" * 60)

    results_by_stage = {}

    for stage_name, ckpt_file in STAGE_CHECKPOINTS.items():
        ckpt_path = MODELS_DIR / ckpt_file
        if not ckpt_path.exists():
            logger.warning("Checkpoint not found: %s — skipping", ckpt_file)
            continue

        logger.info("─── Loading %s ───", stage_name)
        actor, ckpt = load_model(ckpt_path, device)
        results_by_stage[stage_name] = {}

        for sym in symbols:
            data = load_symbol_data(sym, holdout_ratio=0.20)
            if data is None:
                continue

            logger.info("  [%s] %s: %d eps × %d steps", stage_name, sym, n_episodes, episode_length)
            result = run_behavioral_backtest(
                actor, data, config, device, sym,
                m5_thresh=m5_thresh, m1_thresh=m1_thresh,
                n_episodes=n_episodes, episode_length=episode_length,
            )
            results_by_stage[stage_name][sym] = result
            logger.info("    → %d trades, BUY=%.1f%% SELL=%.1f%% HOLD=%.1f%%, entropy=%.2f",
                        result["total_trades"], result["pct_buy"], result["pct_sell"],
                        result["pct_hold"], result["action_entropy"])

    # ── Generate Charts ──
    if results_by_stage:
        plot_action_distribution(results_by_stage, report_label,
                                 REPORTS_DIR / f"action_distribution_{suffix}.png")
        plot_holding_scatter(results_by_stage, report_label,
                            REPORTS_DIR / f"holding_scatter_{suffix}.png")
        plot_signal_reaction(results_by_stage, report_label,
                            REPORTS_DIR / f"signal_reaction_{suffix}.png")
        plot_stage_evolution(results_by_stage, report_label,
                           REPORTS_DIR / f"stage_evolution_{suffix}.png")

    # ── Build summary (strip trades for JSON) ──
    summary = {}
    for stage, sym_results in results_by_stage.items():
        summary[stage] = {}
        for sym, r in sym_results.items():
            s = {k: v for k, v in r.items() if k not in ("trades", "signal_moments")}
            s["n_signal_moments"] = len(r["signal_moments"])
            summary[stage][sym] = s

    return summary


def main():
    parser = argparse.ArgumentParser(description="Behavioral Analysis Backtest")
    parser.add_argument("--test", action="store_true", help="Quick test: 2 eps × 500 steps, XAUUSD only")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.test:
        symbols = ["XAUUSD"]
        n_episodes = 2
        episode_length = 500
    else:
        symbols = ["XAUUSD", "BTCUSD", "ETHUSD", "US30", "USTEC"]
        n_episodes = 10
        episode_length = 2000

    t0 = time.time()

    # ── Report A: Standard Thresholds ──
    report_a = run_full_analysis(
        thresholds={"m5": 0.50, "m1": 0.85},
        report_label="Report A — Standard Thresholds (M5=0.50, M1=0.85)",
        suffix="A_standard",
        device=device, config=config,
        symbols=symbols, n_episodes=n_episodes, episode_length=episode_length,
    )

    # ── Report B: Force-Trigger (ép bóp cò) ──
    report_b = run_full_analysis(
        thresholds={"m5": 0.05, "m1": 0.05},
        report_label="Report B — Force-Trigger (threshold=0.05)",
        suffix="B_force",
        device=device, config=config,
        symbols=symbols, n_episodes=n_episodes, episode_length=episode_length,
    )

    elapsed = time.time() - t0

    # ── Save Combined JSON ──
    full_report = {
        "report_a_standard": report_a,
        "report_b_force_trigger": report_b,
        "config": {
            "m5_standard": 0.50, "m1_standard": 0.85,
            "m5_force": 0.05, "m1_force": 0.05,
            "n_episodes": n_episodes, "episode_length": episode_length,
            "symbols": symbols, "elapsed_seconds": round(elapsed, 1),
        },
    }
    json_path = REPORTS_DIR / "behavioral_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved JSON: %s", json_path)

    # ── Console Summary (ASCII-safe for Windows cp1252) ──
    print("\n" + "=" * 70)
    print("  BEHAVIORAL ANALYSIS COMPLETE")
    print("  Elapsed: %.0fs | Device: %s" % (elapsed, device))
    print("=" * 70)

    for label, report in [("A (Standard)", report_a), ("B (Force)", report_b)]:
        print("\n  -- Report %s --" % label)
        for stage, syms in report.items():
            total_trades = sum(r.get("total_trades", 0) for r in syms.values())
            avg_wr = np.mean([r.get("win_rate", 0) for r in syms.values()]) if syms else 0
            avg_ent = np.mean([r.get("action_entropy", 0) for r in syms.values()]) if syms else 0
            avg_hold = np.mean([r.get("pct_hold", 0) for r in syms.values()]) if syms else 0
            collapse = any(r.get("mode_collapse", False) for r in syms.values())
            flag = "!! COLLAPSE" if collapse else "OK"
            print("    %-25s | Trades=%4d | WR=%5.1f%% | Entropy=%.2f | HOLD=%5.1f%% | %s" %
                  (stage, total_trades, avg_wr, avg_ent, avg_hold, flag))

    print("\n  Charts saved in:", REPORTS_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
