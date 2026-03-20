"""
Ensemble Holdout Backtest — Sprint 5 Final Exam.

Purpose:
    Validate the 3-specialist EnsembleAgent against the FTMO Challenge
    criteria BEFORE allowing deployment to live trading (Sprint 6).

    This script:
    1. Loads normalizer_v2.json
    2. Loads 3 specialist models (best_trendagent.pt, etc.)
    3. Assembles EnsembleAgent with regime-aware voting
    4. Runs deterministic backtest on HOLDOUT data (last 20%)
    5. Computes: Win Rate, Sharpe Ratio, Max Drawdown, Total Return
    6. Prints comparison table vs. FTMO targets
    7. Saves full report to reports/ensemble_backtest_report.txt

Usage:
    python scripts/backtest_ensemble.py
    python scripts/backtest_ensemble.py --episodes 100
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ── Path Setup ──
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "rabit_propfirm_drl"))

DATA_DIR = project_root / "data"
MODEL_DIR = project_root / "models_saved"
REPORT_DIR = project_root / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# ── Constants (must match training) ──
FEATURE_COLS = [
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "candle_direction",
    "pin_bar_bull", "pin_bar_bear", "engulfing_bull", "engulfing_bear", "inside_bar",
    "relative_volume", "vol_delta", "climax_vol",
    "swing_high", "swing_low", "swing_trend", "bos", "choch",
    "ob_bull_dist", "ob_bear_dist", "fvg_bull_active", "fvg_bear_active",
    "liq_above", "liq_below",
    "sin_hour", "cos_hour", "sin_dow", "cos_dow",
    "log_return",
]

LOOKBACK_M5 = 64
LOOKBACK_H1 = 24
LOOKBACK_H4 = 30
N_FEATURES = 28

# FTMO Challenge targets
FTMO_SHARPE_MIN = 1.0
FTMO_MAX_DD = 0.08
FTMO_MIN_WIN_RATE = 0.50

# Specialist model files
SPECIALIST_FILES = [
    ("TrendAgent",       "best_trendagent.pt"),
    ("RangeAgent",       "best_rangeagent.pt"),
    ("VolatilityAgent",  "best_volatilityagent.pt"),
]


# =====================================================================
# DATA LOADING & NORMALIZING (reused from train_specialists.py)
# =====================================================================

def build_htf_features(ohlcv: np.ndarray, n_features: int = 28) -> np.ndarray:
    """Build simplified features from raw OHLCV for H1/H4."""
    n = len(ohlcv)
    features = np.zeros((n, n_features), dtype=np.float32)
    o, h, l, c = ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3]
    body = np.abs(c - o) + 1e-8
    full_range = h - l + 1e-8
    features[:, 0] = body / full_range
    features[:, 1] = np.where(c >= o, (h - c), (h - o)) / full_range
    features[:, 2] = np.where(c >= o, (o - l), (c - l)) / full_range
    features[:, 3] = np.where(c >= o, 1.0, -1.0)
    features[:, 4] = ((features[:, 2] > 0.6) & (features[:, 0] < 0.3)).astype(np.float32)
    features[:, 5] = ((features[:, 1] > 0.6) & (features[:, 0] < 0.3)).astype(np.float32)
    for i in range(1, n):
        if c[i] > o[i] and body[i] > body[i-1] and c[i] > o[i-1] and o[i] < c[i-1]:
            features[i, 6] = 1.0
        if c[i] < o[i] and body[i] > body[i-1] and c[i] < o[i-1] and o[i] > c[i-1]:
            features[i, 7] = 1.0
    for i in range(1, n):
        if h[i] <= h[i-1] and l[i] >= l[i-1]:
            features[i, 8] = 1.0
    if ohlcv.shape[1] > 4:
        vol = ohlcv[:, 4]
        mean_vol = np.convolve(vol, np.ones(20)/20, mode='same') + 1e-8
        features[:, 9] = vol / mean_vol
    for i in range(2, n - 2):
        if h[i] >= h[i-1] and h[i] >= h[i-2] and h[i] >= h[i+1] and h[i] >= h[i+2]:
            features[i, 12] = 1.0
        if l[i] <= l[i-1] and l[i] <= l[i-2] and l[i] <= l[i+1] and l[i] <= l[i+2]:
            features[i, 13] = 1.0
    for i in range(5, n):
        trend = (c[i] - c[i-5]) / (c[i-5] + 1e-8)
        features[i, 14] = np.clip(trend * 100, -1, 1)
    for i in range(n):
        hour_approx = (i % 24)
        features[i, 23] = math.sin(2 * math.pi * hour_approx / 24)
        features[i, 24] = math.cos(2 * math.pi * hour_approx / 24)
        dow_approx = (i // 24) % 5
        features[i, 25] = math.sin(2 * math.pi * dow_approx / 5)
        features[i, 26] = math.cos(2 * math.pi * dow_approx / 5)
    features[1:, 27] = np.log(c[1:] / (c[:-1] + 1e-8))
    return features


def load_all_data() -> dict:
    """Load M5 features + H1/H4 raw OHLCV for all symbols."""
    import polars as pl

    symbol_data = {}
    for feat_file in sorted(DATA_DIR.glob("*_M5_features.parquet")):
        sym_name = feat_file.stem.replace("_M5_features", "")
        df_m5 = pl.read_parquet(feat_file)
        available_cols = [c for c in FEATURE_COLS if c in df_m5.columns]
        m5_features = np.column_stack([
            df_m5[col].fill_null(0.0).fill_nan(0.0).to_numpy().astype(np.float32)
            for col in available_cols
        ])
        if m5_features.shape[1] < N_FEATURES:
            pad = np.zeros(
                (len(m5_features), N_FEATURES - m5_features.shape[1]),
                dtype=np.float32,
            )
            m5_features = np.hstack([m5_features, pad])

        m5_ohlcv = np.column_stack([
            df_m5["open"].to_numpy().astype(np.float32),
            df_m5["high"].to_numpy().astype(np.float32),
            df_m5["low"].to_numpy().astype(np.float32),
            df_m5["close"].to_numpy().astype(np.float32),
            df_m5["tick_volume"].fill_null(0).to_numpy().astype(np.float32)
            if "tick_volume" in df_m5.columns
            else np.zeros(len(df_m5), dtype=np.float32),
        ])

        m5_times = df_m5["time"].to_list() if "time" in df_m5.columns else []

        h1_path = DATA_DIR / f"{sym_name}_H1.parquet"
        if h1_path.exists():
            df_h1 = pl.read_parquet(h1_path)
            h1_ohlcv = np.column_stack([
                df_h1["open"].to_numpy().astype(np.float32),
                df_h1["high"].to_numpy().astype(np.float32),
                df_h1["low"].to_numpy().astype(np.float32),
                df_h1["close"].to_numpy().astype(np.float32),
                df_h1["tick_volume"].fill_null(0).to_numpy().astype(np.float32)
                if "tick_volume" in df_h1.columns
                else np.zeros(len(df_h1), dtype=np.float32),
            ])
            h1_features = build_htf_features(h1_ohlcv, N_FEATURES)
        else:
            h1_features = np.zeros((100, N_FEATURES), dtype=np.float32)

        h4_path = DATA_DIR / f"{sym_name}_H4.parquet"
        if h4_path.exists():
            df_h4 = pl.read_parquet(h4_path)
            h4_ohlcv = np.column_stack([
                df_h4["open"].to_numpy().astype(np.float32),
                df_h4["high"].to_numpy().astype(np.float32),
                df_h4["low"].to_numpy().astype(np.float32),
                df_h4["close"].to_numpy().astype(np.float32),
                df_h4["tick_volume"].fill_null(0).to_numpy().astype(np.float32)
                if "tick_volume" in df_h4.columns
                else np.zeros(len(df_h4), dtype=np.float32),
            ])
            h4_features = build_htf_features(h4_ohlcv, N_FEATURES)
        else:
            h4_features = np.zeros((100, N_FEATURES), dtype=np.float32)

        ib_path = DATA_DIR / f"{sym_name}_H1_insidebar.parquet"
        h1_ib_times = set()
        if ib_path.exists():
            ib_df = pl.read_parquet(ib_path)
            h1_ib_times = set(
                ib_df.filter(pl.col("inside_bar") == 1.0)["time"].to_list()
            )

        symbol_data[sym_name] = {
            "m5_features": m5_features, "m5_ohlcv": m5_ohlcv,
            "m5_times": m5_times, "h1_features": h1_features,
            "h4_features": h4_features, "h1_ib_times": h1_ib_times,
            "n_m5": len(m5_features), "n_h1": len(h1_features),
            "n_h4": len(h4_features),
        }
    return symbol_data


def load_normalizer(path: Path) -> dict:
    """Load pre-computed mean/std from normalizer JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    return {
        key: {
            "mean": np.array(val["mean"], dtype=np.float32),
            "std": np.array(val["std"], dtype=np.float32),
        }
        for key, val in data.items()
    }


def normalize_with_saved(symbol_data: dict, normalizer: dict) -> dict:
    """
    Normalize features using SAVED mean/std from training.
    This ensures holdout data uses the SAME normalization as training.
    """
    tf_map = {
        "m5_features": "m5_features",
        "h1_features": "h1_features",
        "h4_features": "h4_features",
    }
    for tf_key, norm_key in tf_map.items():
        if norm_key not in normalizer:
            print(f"  [WARN] {norm_key} not in normalizer, using inline z-score")
            continue
        mean = normalizer[norm_key]["mean"]
        std = normalizer[norm_key]["std"]
        for sym in symbol_data:
            feat = symbol_data[sym][tf_key]
            # Handle shape mismatch (normalizer might have different cols)
            n_cols = min(feat.shape[1], len(mean))
            normed = (feat[:, :n_cols] - mean[:n_cols]) / (std[:n_cols] + 1e-8)
            if n_cols < feat.shape[1]:
                normed = np.hstack([normed, feat[:, n_cols:]])
            symbol_data[sym][f"{tf_key}_norm"] = normed.astype(np.float32)
    return symbol_data


# =====================================================================
# HOLDOUT BACKTEST ENVIRONMENT
# =====================================================================

class HoldoutBacktestEnv:
    """
    Sequential backtest environment on holdout (last 20%) data.

    Unlike training env:
    - NO random start position — walks sequentially through holdout
    - Deterministic stepping (start → end, no reset mid-run)
    - Tracks per-trade PnL for Sharpe calculation
    - Records full trade log
    """

    def __init__(
        self,
        m5_features: np.ndarray,
        m5_ohlcv: np.ndarray,
        h1_features: np.ndarray,
        h4_features: np.ndarray,
        n_m5: int,
        n_h1: int,
        n_h4: int,
        holdout_pct: float = 0.20,
        initial_balance: float = 100_000.0,
        max_loss_per_trade: float = 0.003,
        confidence_threshold: float = 0.3,
    ):
        # Holdout = last holdout_pct of data
        self.holdout_start = int(n_m5 * (1 - holdout_pct))
        self.m5_feat = m5_features
        self.m5_ohlcv = m5_ohlcv
        self.h1_feat = h1_features
        self.h4_feat = h4_features
        self.n_m5 = n_m5
        self.n_h1 = n_h1
        self.n_h4 = n_h4
        self.m5_per_h1 = max(1, n_m5 // max(n_h1, 1))
        self.m5_per_h4 = max(1, n_m5 // max(n_h4, 1))

        self.initial_balance = initial_balance
        self.max_loss_per_trade = max_loss_per_trade
        self.confidence_threshold = confidence_threshold

        # State
        self.m5_idx = self.holdout_start
        self.balance = initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.peak_balance = initial_balance

        # Tracking
        self.trade_pnls: list[float] = []
        self.trade_log: list[dict] = []
        self.equity_curve: list[float] = [initial_balance]
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0.0
        self.steps_taken = 0

    @property
    def is_done(self) -> bool:
        return self.m5_idx >= self.n_m5 - 1

    @property
    def holdout_bars(self) -> int:
        return self.n_m5 - self.holdout_start

    def get_obs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get current observation (m5, h1, h4) windows."""
        return self._get_m5_window(), self._get_h1_window(), self._get_h4_window()

    def step(self, action: np.ndarray) -> dict:
        """
        Take one step with the given action.

        Args:
            action: (4,) — [direction/confidence, risk_frac, sl_mult, tp_mult]

        Returns:
            Dict with step info
        """
        confidence = float(np.clip(action[0], -1, 1))
        risk_frac = float(np.clip(action[1], 0, 1))
        close_now = float(self.m5_ohlcv[self.m5_idx, 3])
        trade_event = None

        # Update PnL for open position
        if self.position != 0:
            price_change = (close_now - self.entry_price) / self.entry_price
            self.unrealized_pnl = self.position * price_change * self.balance * risk_frac

            max_loss = self.balance * self.max_loss_per_trade
            if self.unrealized_pnl < -max_loss:
                # SL hit
                trade_event = self._close_trade("SL_HIT")
            elif self.unrealized_pnl > self.balance * 0.01:
                # TP hit
                trade_event = self._close_trade("TP_HIT")

        # Entry / exit logic
        if abs(confidence) > self.confidence_threshold and self.position == 0:
            self.position = 1.0 if confidence > 0 else -1.0
            self.entry_price = close_now
        elif abs(confidence) < 0.1 and self.position != 0:
            trade_event = self._close_trade("SIGNAL_EXIT")

        # Track drawdown
        current_equity = self.balance + self.unrealized_pnl
        self.peak_balance = max(self.peak_balance, current_equity)
        current_dd = (self.peak_balance - current_equity) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_dd)

        self.equity_curve.append(current_equity)
        self.m5_idx += 1
        self.steps_taken += 1

        return {
            "balance": self.balance,
            "equity": current_equity,
            "position": self.position,
            "drawdown": current_dd,
            "trade_event": trade_event,
        }

    def finalize(self) -> dict:
        """Force-close any open position and compute final metrics."""
        if self.position != 0:
            self._close_trade("END_OF_DATA")

        return self._compute_metrics()

    # ── Private ──

    def _close_trade(self, reason: str) -> dict:
        """Close current position and record trade."""
        pnl = self.unrealized_pnl
        self.balance += pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1

        self.trade_pnls.append(pnl)
        trade_rec = {
            "direction": "BUY" if self.position > 0 else "SELL",
            "pnl": round(pnl, 2),
            "balance_after": round(self.balance, 2),
            "reason": reason,
            "bar_index": self.m5_idx,
        }
        self.trade_log.append(trade_rec)

        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        return trade_rec

    def _compute_metrics(self) -> dict:
        """Compute all backtest performance metrics."""
        win_rate = self.winning_trades / max(self.total_trades, 1)

        # Sharpe Ratio from trade PnLs
        if len(self.trade_pnls) >= 2:
            pnl_arr = np.array(self.trade_pnls)
            mean_pnl = pnl_arr.mean()
            std_pnl = pnl_arr.std() + 1e-8
            # Annualize: assume ~10 trades/day, 252 trading days
            trades_per_year = min(len(self.trade_pnls), 10) * 252
            sharpe = (mean_pnl / std_pnl) * np.sqrt(trades_per_year)
        else:
            sharpe = 0.0

        total_return = (self.balance - self.initial_balance) / self.initial_balance

        # Max Drawdown from equity curve
        equity_arr = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (running_max - equity_arr) / (running_max + 1e-8)
        max_dd_curve = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0
        max_dd = max(self.max_drawdown, max_dd_curve)

        # Profit Factor
        gross_profit = sum(p for p in self.trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in self.trade_pnls if p < 0))
        profit_factor = gross_profit / (gross_loss + 1e-8)

        # Average RR
        avg_win = np.mean([p for p in self.trade_pnls if p > 0]) if self.winning_trades > 0 else 0
        avg_loss = abs(np.mean([p for p in self.trade_pnls if p < 0])) if (self.total_trades - self.winning_trades) > 0 else 1e-8
        avg_rr = avg_win / (avg_loss + 1e-8)

        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": win_rate,
            "sharpe_ratio": float(sharpe),
            "max_drawdown": max_dd,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "final_balance": self.balance,
            "profit_factor": profit_factor,
            "avg_rr": avg_rr,
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "holdout_bars": self.holdout_bars,
            "steps_taken": self.steps_taken,
        }

    def _get_m5_window(self):
        start = max(0, self.m5_idx - LOOKBACK_M5)
        w = self.m5_feat[start:self.m5_idx]
        if len(w) < LOOKBACK_M5:
            pad = np.zeros((LOOKBACK_M5 - len(w), N_FEATURES), dtype=np.float32)
            w = np.concatenate([pad, w], axis=0)
        return w

    def _get_h1_window(self):
        h1_idx = min(self.m5_idx // self.m5_per_h1, self.n_h1 - 1)
        start = max(0, h1_idx - LOOKBACK_H1)
        w = self.h1_feat[start:h1_idx]
        if len(w) < LOOKBACK_H1:
            pad = np.zeros((LOOKBACK_H1 - len(w), N_FEATURES), dtype=np.float32)
            w = np.concatenate([pad, w], axis=0)
        return w

    def _get_h4_window(self):
        h4_idx = min(self.m5_idx // self.m5_per_h4, self.n_h4 - 1)
        start = max(0, h4_idx - LOOKBACK_H4)
        w = self.h4_feat[start:h4_idx]
        if len(w) < LOOKBACK_H4:
            pad = np.zeros((LOOKBACK_H4 - len(w), N_FEATURES), dtype=np.float32)
            w = np.concatenate([pad, w], axis=0)
        return w


# =====================================================================
# MODEL LOADING & ENSEMBLE ASSEMBLY
# =====================================================================

def load_specialist(
    name: str, path: Path, device: torch.device
) -> torch.nn.Module:
    """Load a single specialist SACTransformerActor from checkpoint."""
    from agents.sac_policy import SACTransformerActor

    actor = SACTransformerActor(
        n_features=N_FEATURES, action_dim=4, embed_dim=128,
        n_heads=4, n_transformer_layers=2, n_cross_layers=1,
        hidden_dims=[256, 256], dropout=0.1,
    )

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["actor_state"])
    actor.to(device)
    actor.eval()

    step = checkpoint.get("step", "?")
    reward = checkpoint.get("best_reward", "?")
    print(f"  Loaded {name}: step={step}, best_reward={reward}")
    return actor


def assemble_ensemble(device: torch.device):
    """Load all 3 specialists and assemble EnsembleAgent."""
    from agents.ensemble_agent import EnsembleAgent

    agents = []
    for name, filename in SPECIALIST_FILES:
        path = MODEL_DIR / filename
        if not path.exists():
            print(f"  [ERROR] Model not found: {path}")
            sys.exit(1)
        actor = load_specialist(name, path, device)
        agents.append(actor)

    ensemble = EnsembleAgent(
        agents=agents,
        regime_detector=None,  # Use agent[0]'s internal RegimeDetector
        action_gating=None,    # We read raw actions for backtesting
        base_weights=[0.40, 0.30, 0.30],
    )
    # Enable regime detection through agent's feature extractor
    ensemble.regime_detector = True  # trigger hasattr check path

    print(f"  EnsembleAgent assembled: {len(agents)} specialists")
    return ensemble


# =====================================================================
# BACKTEST RUNNER
# =====================================================================

def run_backtest_single_symbol(
    ensemble,
    env: HoldoutBacktestEnv,
    device: torch.device,
    symbol: str,
) -> dict:
    """Run full holdout backtest for one symbol."""
    step_count = 0

    while not env.is_done:
        m5_obs, h1_obs, h4_obs = env.get_obs()

        # Convert to tensors
        m5_t = torch.FloatTensor(m5_obs).unsqueeze(0).to(device)
        h1_t = torch.FloatTensor(h1_obs).unsqueeze(0).to(device)
        h4_t = torch.FloatTensor(h4_obs).unsqueeze(0).to(device)

        # Ensemble inference (deterministic)
        with torch.no_grad():
            action = ensemble.get_action(m5_t, h1_t, h4_t, deterministic=True)

        env.step(action)
        step_count += 1

        # Progress indicator every 5000 bars
        if step_count % 5000 == 0:
            print(f"    [{symbol}] {step_count}/{env.holdout_bars} bars "
                  f"({step_count/env.holdout_bars*100:.0f}%)")

    metrics = env.finalize()
    metrics["symbol"] = symbol
    return metrics


def run_full_backtest(
    ensemble,
    symbol_data: dict,
    device: torch.device,
    n_episodes: int = 1,
) -> list[dict]:
    """Run backtest across all symbols."""
    all_results = []

    for sym, sd in symbol_data.items():
        print(f"\n  Backtesting: {sym} ({sd['n_m5']:,} M5 bars)")

        for ep in range(n_episodes):
            env = HoldoutBacktestEnv(
                m5_features=sd["m5_features_norm"],
                m5_ohlcv=sd["m5_ohlcv"],
                h1_features=sd["h1_features_norm"],
                h4_features=sd["h4_features_norm"],
                n_m5=sd["n_m5"],
                n_h1=sd["n_h1"],
                n_h4=sd["n_h4"],
                holdout_pct=0.20,
            )

            metrics = run_backtest_single_symbol(ensemble, env, device, sym)
            metrics["episode"] = ep + 1
            all_results.append(metrics)

            status = "PASS" if (
                metrics["sharpe_ratio"] >= FTMO_SHARPE_MIN
                and metrics["max_drawdown"] < FTMO_MAX_DD
            ) else "REVIEW"

            print(
                f"    [{sym}] ep{ep+1}: WR={metrics['win_rate']:.1%} | "
                f"Sharpe={metrics['sharpe_ratio']:.2f} | "
                f"DD={metrics['max_drawdown']:.2%} | "
                f"Return={metrics['total_return_pct']:+.2f}% | "
                f"Trades={metrics['total_trades']} | [{status}]"
            )

    return all_results


# =====================================================================
# REPORT GENERATION
# =====================================================================

def compute_aggregate(results: list[dict]) -> dict:
    """Compute aggregate metrics across all symbols."""
    if not results:
        return {}

    # Weighted averages by trade count
    total_trades = sum(r["total_trades"] for r in results)
    total_wins = sum(r["winning_trades"] for r in results)

    all_sharpes = [r["sharpe_ratio"] for r in results]
    all_dds = [r["max_drawdown"] for r in results]
    all_returns = [r["total_return_pct"] for r in results]
    all_pfs = [r["profit_factor"] for r in results]
    all_rrs = [r["avg_rr"] for r in results]

    return {
        "total_trades": total_trades,
        "total_wins": total_wins,
        "win_rate": total_wins / max(total_trades, 1),
        "avg_sharpe": np.mean(all_sharpes),
        "min_sharpe": np.min(all_sharpes),
        "max_sharpe": np.max(all_sharpes),
        "avg_max_dd": np.mean(all_dds),
        "worst_dd": np.max(all_dds),
        "avg_return_pct": np.mean(all_returns),
        "avg_profit_factor": np.mean(all_pfs),
        "avg_rr": np.mean(all_rrs),
        "n_symbols": len(results),
    }


def print_report(results: list[dict], agg: dict):
    """Print formatted report to terminal."""
    w = 70

    print("\n" + "=" * w)
    print("  ENSEMBLE HOLDOUT BACKTEST REPORT")
    print("  Sprint 5 Final Exam — FTMO Challenge Validation")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * w)

    # Per-symbol table
    print(f"\n  {'Symbol':<15} {'WinRate':>8} {'Sharpe':>8} {'MaxDD':>8} "
          f"{'Return':>9} {'Trades':>7} {'PF':>6} {'Status':>8}")
    print("  " + "-" * (w - 4))

    for r in results:
        status = "PASS" if (
            r["sharpe_ratio"] >= FTMO_SHARPE_MIN
            and r["max_drawdown"] < FTMO_MAX_DD
        ) else "FAIL"

        print(
            f"  {r['symbol']:<15} "
            f"{r['win_rate']:>7.1%} "
            f"{r['sharpe_ratio']:>8.2f} "
            f"{r['max_drawdown']:>7.2%} "
            f"{r['total_return_pct']:>+8.2f}% "
            f"{r['total_trades']:>7} "
            f"{r['profit_factor']:>6.2f} "
            f"  {status}"
        )

    print("  " + "-" * (w - 4))

    # Aggregate
    print(f"\n  AGGREGATE METRICS:")
    print(f"    Total Trades:      {agg['total_trades']}")
    print(f"    Overall Win Rate:  {agg['win_rate']:.1%}")
    print(f"    Avg Sharpe:        {agg['avg_sharpe']:.2f} "
          f"(min={agg['min_sharpe']:.2f}, max={agg['max_sharpe']:.2f})")
    print(f"    Avg Max DD:        {agg['avg_max_dd']:.2%}")
    print(f"    Worst DD:          {agg['worst_dd']:.2%}")
    print(f"    Avg Return:        {agg['avg_return_pct']:+.2f}%")
    print(f"    Avg Profit Factor: {agg['avg_profit_factor']:.2f}")
    print(f"    Avg Risk/Reward:   {agg['avg_rr']:.2f}")

    # FTMO comparison
    print(f"\n  FTMO CHALLENGE COMPARISON:")
    print(f"  {'Metric':<25} {'Ensemble':>10} {'Target':>10} {'Status':>8}")
    print("  " + "-" * 55)

    checks = [
        ("Win Rate",   f"{agg['win_rate']:.1%}",     f">={FTMO_MIN_WIN_RATE:.0%}",  agg["win_rate"] >= FTMO_MIN_WIN_RATE),
        ("Sharpe Ratio", f"{agg['avg_sharpe']:.2f}",  f">={FTMO_SHARPE_MIN:.1f}",    agg["avg_sharpe"] >= FTMO_SHARPE_MIN),
        ("Max Drawdown", f"{agg['worst_dd']:.2%}",    f"<{FTMO_MAX_DD:.0%}",         agg["worst_dd"] < FTMO_MAX_DD),
    ]

    all_pass = True
    for name, value, target, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name:<25} {value:>10} {target:>10}     {status}")

    print("\n" + "=" * w)
    if all_pass:
        print("  VERDICT: PASS  -- Ensemble APPROVED for Sprint 6 (Live Trading)")
    else:
        print("  VERDICT: REVIEW NEEDED  -- Some metrics below target")
    print("=" * w + "\n")

    return all_pass


def save_report(results: list[dict], agg: dict, path: Path):
    """Save full text report to file."""
    lines = []
    w = 70

    lines.append("=" * w)
    lines.append("  ENSEMBLE HOLDOUT BACKTEST REPORT")
    lines.append("  Sprint 5 Final Exam")
    lines.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * w)
    lines.append("")

    lines.append("AGGREGATE:")
    lines.append(f"  Total Trades:      {agg['total_trades']}")
    lines.append(f"  Win Rate:          {agg['win_rate']:.1%}")
    lines.append(f"  Avg Sharpe:        {agg['avg_sharpe']:.2f}")
    lines.append(f"  Worst Max DD:      {agg['worst_dd']:.2%}")
    lines.append(f"  Avg Return:        {agg['avg_return_pct']:+.2f}%")
    lines.append(f"  Avg Profit Factor: {agg['avg_profit_factor']:.2f}")
    lines.append(f"  Avg RR:            {agg['avg_rr']:.2f}")
    lines.append("")

    lines.append("PER-SYMBOL BREAKDOWN:")
    lines.append(f"  {'Symbol':<15} {'WR':>6} {'Sharpe':>8} {'MaxDD':>8} "
                 f"{'Return':>9} {'Trades':>7} {'PF':>6}")
    lines.append("  " + "-" * 60)

    for r in results:
        lines.append(
            f"  {r['symbol']:<15} "
            f"{r['win_rate']:>5.1%} "
            f"{r['sharpe_ratio']:>8.2f} "
            f"{r['max_drawdown']:>7.2%} "
            f"{r['total_return_pct']:>+8.2f}% "
            f"{r['total_trades']:>7} "
            f"{r['profit_factor']:>6.2f}"
        )

    lines.append("")
    lines.append("TRADE LOGS:")
    for r in results:
        sym = r["symbol"]
        trades = r.get("trade_log", [])
        lines.append(f"\n  --- {sym} ({len(trades)} trades) ---")
        for t in trades[:20]:  # First 20 trades per symbol
            lines.append(
                f"    {t['direction']:<4} PnL=${t['pnl']:>8.2f} "
                f"Bal=${t['balance_after']:>10.2f}  [{t['reason']}]"
            )
        if len(trades) > 20:
            lines.append(f"    ... and {len(trades) - 20} more trades")

    lines.append("\n" + "=" * w)
    lines.append("END OF REPORT")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report saved: {path}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ensemble Holdout Backtest — Sprint 5 Final Exam"
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Number of episodes per symbol (default: 1 = single pass)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cpu', 'cuda', or 'auto'"
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  RABIT-PROPFIRM  --  Ensemble Holdout Backtest")
    print("  Sprint 5 Final Exam: Must PASS before Sprint 6 (Live Trading)")
    print("=" * 65)

    # ── Device ──
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n  Device: {device}")

    # ── Load normalizer ──
    normalizer_path = MODEL_DIR / "normalizer_v2.json"
    if not normalizer_path.exists():
        print(f"  [ERROR] Normalizer not found: {normalizer_path}")
        sys.exit(1)
    print(f"  Loading normalizer: {normalizer_path.name}")
    normalizer = load_normalizer(normalizer_path)

    # ── Load models ──
    print(f"\n  Loading 3 specialists:")
    ensemble = assemble_ensemble(device)

    # ── Load data ──
    print(f"\n  Loading market data from {DATA_DIR}...")
    t0 = time.time()
    symbol_data = load_all_data()
    print(f"  Loaded {len(symbol_data)} symbols in {time.time()-t0:.1f}s")
    for sym, sd in symbol_data.items():
        holdout_start = int(sd["n_m5"] * 0.80)
        holdout_bars = sd["n_m5"] - holdout_start
        print(f"    {sym}: {sd['n_m5']:,} M5 bars "
              f"(holdout: last {holdout_bars:,} bars = 20%)")

    # ── Normalize with SAVED stats (critical!) ──
    print(f"\n  Normalizing with saved stats (not inline)...")
    symbol_data = normalize_with_saved(symbol_data, normalizer)

    # ── Run backtest ──
    print(f"\n  Running holdout backtest ({args.episodes} episode(s) per symbol)...")
    t0 = time.time()
    results = run_full_backtest(ensemble, symbol_data, device, args.episodes)
    elapsed = time.time() - t0
    print(f"\n  Backtest completed in {elapsed:.1f}s")

    # Attach trade logs to results
    # (trade_log is part of HoldoutBacktestEnv.finalize but we need
    # to preserve it through run_full_backtest)

    # ── Compute aggregate ──
    agg = compute_aggregate(results)

    # ── Print report ──
    all_pass = print_report(results, agg)

    # ── Save report ──
    report_path = REPORT_DIR / "ensemble_backtest_report.txt"
    save_report(results, agg, report_path)

    # Exit code: 0 = PASS, 1 = FAIL
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
