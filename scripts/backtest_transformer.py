"""
Backtest Transformer Model — Sprint 3.7 (Dai Chien)

Loads best_transformer.pt and runs walk-forward backtest on holdout 20% data.
Same logic as backtest.py but with multi-TF observations (M5+H1+H4).

Usage: py -3.11 -u scripts/backtest_transformer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "rabit_propfirm_drl"))

DATA_DIR = project_root / "data"
MODEL_DIR = project_root / "models_saved"
REPORT_DIR = project_root / "reports"
REPORT_DIR.mkdir(exist_ok=True)

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
INITIAL_BALANCE = 100_000.0
MAX_LOSS_PCT = 0.003
DAILY_COOLDOWN = 0.03


def build_htf_features(ohlcv: np.ndarray, n_features: int = 28) -> np.ndarray:
    """Build features from raw OHLCV for H1/H4."""
    import math
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


def main():
    import polars as pl
    from agents.sac_policy import SACTransformerActor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  RABIT-PROPFIRM -- TRANSFORMER BACKTEST (Out-of-Sample)")
    print(f"  Device: {device.type.upper()}")
    print("=" * 70)

    # Load Transformer model
    ckpt_path = MODEL_DIR / "best_transformer.pt"
    print(f"\n[1/3] Loading Transformer model from {ckpt_path.name}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    actor = SACTransformerActor(
        n_features=ckpt.get("n_features", N_FEATURES),
        action_dim=4,
        embed_dim=ckpt.get("embed_dim", 128),
        n_heads=4, n_transformer_layers=2, n_cross_layers=1,
        hidden_dims=[256, 256], dropout=0.0,  # No dropout during inference
    ).to(device)
    actor.load_state_dict(ckpt["actor_state"])
    actor.eval()
    print(f"  Loaded: step {ckpt.get('step', '?')}, eval_reward {ckpt.get('eval_reward', '?'):.2f}")

    # Load normalizer
    norm_path = MODEL_DIR / "normalizer_transformer.json"
    with open(norm_path) as f:
        norm = json.load(f)
    m5_mean = np.array(norm["m5_features"]["mean"], dtype=np.float32)
    m5_std = np.array(norm["m5_features"]["std"], dtype=np.float32)
    h1_mean = np.array(norm["h1_features"]["mean"], dtype=np.float32)
    h1_std = np.array(norm["h1_features"]["std"], dtype=np.float32)
    h4_mean = np.array(norm["h4_features"]["mean"], dtype=np.float32)
    h4_std = np.array(norm["h4_features"]["std"], dtype=np.float32)

    print(f"\n[2/3] Running walk-forward backtest on holdout 20%...")
    print("-" * 70)

    all_results = {}

    for feat_file in sorted(DATA_DIR.glob("*_M5_features.parquet")):
        sym = feat_file.stem.replace("_M5_features", "")
        df_m5 = pl.read_parquet(feat_file)

        # M5 features
        available_cols = [c for c in FEATURE_COLS if c in df_m5.columns]
        m5_features = np.column_stack([
            df_m5[col].fill_null(0.0).fill_nan(0.0).to_numpy().astype(np.float32)
            for col in available_cols
        ])
        if m5_features.shape[1] < N_FEATURES:
            pad = np.zeros((len(m5_features), N_FEATURES - m5_features.shape[1]), dtype=np.float32)
            m5_features = np.hstack([m5_features, pad])
        m5_features_norm = ((m5_features - m5_mean) / m5_std).astype(np.float32)

        ohlcv = np.column_stack([
            df_m5["open"].to_numpy().astype(np.float32),
            df_m5["high"].to_numpy().astype(np.float32),
            df_m5["low"].to_numpy().astype(np.float32),
            df_m5["close"].to_numpy().astype(np.float32),
        ])
        times = df_m5["time"].to_list() if "time" in df_m5.columns else []

        # H1 features
        h1_path = DATA_DIR / f"{sym}_H1.parquet"
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
        h1_features_norm = ((h1_features - h1_mean) / h1_std).astype(np.float32)

        # H4 features
        h4_path = DATA_DIR / f"{sym}_H4.parquet"
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
        h4_features_norm = ((h4_features - h4_mean) / h4_std).astype(np.float32)

        # H1 inside bar
        ib_path = DATA_DIR / f"{sym}_H1_insidebar.parquet"
        h1_ib_times = set()
        if ib_path.exists():
            ib_df = pl.read_parquet(ib_path)
            h1_ib_times = set(ib_df.filter(pl.col("inside_bar") == 1.0)["time"].to_list())

        # Holdout: last 20%
        n_m5 = len(m5_features_norm)
        n_h1 = len(h1_features_norm)
        n_h4 = len(h4_features_norm)
        holdout_start_m5 = int(n_m5 * 0.8)
        holdout_m5 = m5_features_norm[holdout_start_m5:]
        holdout_ohlcv = ohlcv[holdout_start_m5:]
        holdout_times = times[holdout_start_m5:] if times else []

        m5_per_h1 = max(1, n_m5 // max(n_h1, 1))
        m5_per_h4 = max(1, n_m5 // max(n_h4, 1))

        print(f"\n  === {sym} === (holdout: {len(holdout_m5):,} bars, "
              f"~{len(holdout_m5) * 5 / 60 / 24:.0f} days)")

        # Walk-forward simulation
        balance = INITIAL_BALANCE
        position = 0.0
        entry_price = 0.0
        risk_frac = 0.0
        trades = []
        equity_curve = [balance]
        daily_loss = 0.0
        current_day = None
        cooled_down = False

        for i in range(LOOKBACK_M5, len(holdout_m5)):
            close_price = float(holdout_ohlcv[i, 3])
            bar_time = holdout_times[i] if i < len(holdout_times) else None

            # Day rollover
            if bar_time:
                day = bar_time.date() if hasattr(bar_time, 'date') else None
                if day and day != current_day:
                    current_day = day
                    daily_loss = 0.0
                    cooled_down = False

            # Build multi-TF observation
            m5_window = holdout_m5[i - LOOKBACK_M5:i]  # (64, 28)

            # Map M5 index to H1/H4 index
            abs_m5_idx = holdout_start_m5 + i
            h1_idx = min(abs_m5_idx // m5_per_h1, n_h1 - 1)
            h1_start = max(0, h1_idx - LOOKBACK_H1)
            h1_window = h1_features_norm[h1_start:h1_idx]
            if len(h1_window) < LOOKBACK_H1:
                pad = np.zeros((LOOKBACK_H1 - len(h1_window), N_FEATURES), dtype=np.float32)
                h1_window = np.concatenate([pad, h1_window], axis=0)

            h4_idx = min(abs_m5_idx // m5_per_h4, n_h4 - 1)
            h4_start = max(0, h4_idx - LOOKBACK_H4)
            h4_window = h4_features_norm[h4_start:h4_idx]
            if len(h4_window) < LOOKBACK_H4:
                pad = np.zeros((LOOKBACK_H4 - len(h4_window), N_FEATURES), dtype=np.float32)
                h4_window = np.concatenate([pad, h4_window], axis=0)

            # Get action from Transformer
            with torch.no_grad():
                m5_t = torch.FloatTensor(m5_window).unsqueeze(0).to(device)
                h1_t = torch.FloatTensor(h1_window).unsqueeze(0).to(device)
                h4_t = torch.FloatTensor(h4_window).unsqueeze(0).to(device)
                act_t, _ = actor(m5_t, h1_t, h4_t, deterministic=True)
                action = act_t.squeeze(0).cpu().numpy()

            confidence = float(np.clip(action[0], -1, 1))
            risk_frac_a = float(np.clip(action[1], 0.01, 1.0))

            # H1 inside bar check
            is_h1_ib = False
            if bar_time and hasattr(bar_time, 'replace'):
                h1_time = bar_time.replace(minute=0, second=0, microsecond=0)
                is_h1_ib = h1_time in h1_ib_times

            # === H1 INSIDE BAR EXIT ===
            if position != 0 and is_h1_ib:
                pnl_pct = position * (close_price - entry_price) / entry_price * risk_frac
                pnl_usd = pnl_pct * balance
                balance += pnl_usd
                if pnl_usd < 0:
                    daily_loss += abs(pnl_pct)
                trades.append({
                    "symbol": sym, "direction": "LONG" if position > 0 else "SHORT",
                    "entry_price": entry_price, "exit_price": close_price,
                    "pnl_pct": pnl_pct * 100, "pnl_usd": pnl_usd,
                    "reason": "H1_INSIDE_BAR",
                })
                position = 0.0

            # === UPDATE UNREALIZED PNL ===
            if position != 0:
                pnl_pct = position * (close_price - entry_price) / entry_price * risk_frac
                pnl_usd = pnl_pct * balance

                if pnl_usd < -(balance * MAX_LOSS_PCT):
                    loss = balance * MAX_LOSS_PCT
                    balance -= loss
                    daily_loss += MAX_LOSS_PCT
                    trades.append({
                        "symbol": sym, "direction": "LONG" if position > 0 else "SHORT",
                        "entry_price": entry_price, "exit_price": close_price,
                        "pnl_pct": -MAX_LOSS_PCT * 100, "pnl_usd": -loss,
                        "reason": "SL_HIT",
                    })
                    position = 0.0
                elif pnl_usd > balance * 0.01:
                    gain = balance * 0.01
                    balance += gain
                    trades.append({
                        "symbol": sym, "direction": "LONG" if position > 0 else "SHORT",
                        "entry_price": entry_price, "exit_price": close_price,
                        "pnl_pct": 1.0, "pnl_usd": gain,
                        "reason": "TP_HIT",
                    })
                    position = 0.0

            # === OPEN NEW POSITION ===
            if abs(confidence) > 0.3 and position == 0 and not cooled_down and not is_h1_ib:
                position = 1.0 if confidence > 0 else -1.0
                entry_price = close_price
                risk_frac = risk_frac_a

            # === CLOSE ON LOW CONFIDENCE ===
            elif abs(confidence) < 0.1 and position != 0:
                pnl_pct = position * (close_price - entry_price) / entry_price * risk_frac
                pnl_usd = pnl_pct * balance
                balance += pnl_usd
                if pnl_usd < 0:
                    daily_loss += abs(pnl_pct)
                trades.append({
                    "symbol": sym, "direction": "LONG" if position > 0 else "SHORT",
                    "entry_price": entry_price, "exit_price": close_price,
                    "pnl_pct": pnl_pct * 100, "pnl_usd": pnl_usd,
                    "reason": "LOW_CONFIDENCE",
                })
                position = 0.0

            if daily_loss >= DAILY_COOLDOWN:
                cooled_down = True
                if position != 0:
                    pnl_pct = position * (close_price - entry_price) / entry_price * risk_frac
                    pnl_usd = pnl_pct * balance
                    balance += pnl_usd
                    trades.append({
                        "symbol": sym, "direction": "LONG" if position > 0 else "SHORT",
                        "entry_price": entry_price, "exit_price": close_price,
                        "pnl_pct": pnl_pct * 100, "pnl_usd": pnl_usd,
                        "reason": "DAILY_COOLDOWN",
                    })
                    position = 0.0

            equity_curve.append(balance)

        # Calculate metrics
        if trades:
            pnls = [t["pnl_usd"] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]

            total_pnl = balance - INITIAL_BALANCE
            total_return = (balance / INITIAL_BALANCE - 1) * 100
            win_rate = len(wins) / len(trades) * 100
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 0
            rr_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")
            profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")

            peak = INITIAL_BALANCE
            max_dd = 0.0
            for eq in equity_curve:
                peak = max(peak, eq)
                dd = (peak - eq) / peak
                max_dd = max(max_dd, dd)

            daily_returns = []
            day_pnl = 0.0
            last_day = None
            for t in trades:
                t_day = str(t.get("exit_price", ""))[:10]
                if t_day != last_day and last_day is not None:
                    daily_returns.append(day_pnl)
                    day_pnl = 0.0
                day_pnl += t["pnl_pct"]
                last_day = t_day
            daily_returns.append(day_pnl)

            sharpe = 0.0
            if len(daily_returns) > 1 and np.std(daily_returns) > 0:
                sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

            reasons = {}
            for t in trades:
                r = t["reason"]
                reasons[r] = reasons.get(r, 0) + 1

            print(f"  Total trades:     {len(trades)}")
            print(f"  Win Rate:         {win_rate:.1f}%")
            print(f"  Total Return:     {total_return:+.2f}%")
            print(f"  Total PnL:        ${total_pnl:+,.2f}")
            print(f"  Avg Win:          ${avg_win:+,.2f}")
            print(f"  Avg Loss:         ${-avg_loss:,.2f}")
            print(f"  Risk:Reward:      1:{rr_ratio:.2f}")
            print(f"  Profit Factor:    {profit_factor:.2f}")
            print(f"  Max Drawdown:     {max_dd*100:.2f}%")
            print(f"  Sharpe Ratio:     {sharpe:.2f}")
            print(f"  Exit reasons:     {reasons}")

            all_results[sym] = {
                "trades": len(trades),
                "win_rate": win_rate,
                "total_return": total_return,
                "total_pnl": total_pnl,
                "profit_factor": profit_factor,
                "max_dd": max_dd * 100,
                "sharpe": sharpe,
                "rr_ratio": rr_ratio,
                "exit_reasons": reasons,
            }
        else:
            print(f"  No trades executed!")
            all_results[sym] = {"trades": 0}

    # === COMBINED REPORT ===
    print("\n" + "=" * 70)
    print("  TRANSFORMER COMBINED BACKTEST REPORT")
    print("=" * 70)

    active = {k: v for k, v in all_results.items() if v.get("trades", 0) > 0}

    if active:
        total_trades = sum(v["trades"] for v in active.values())
        avg_wr = np.mean([v["win_rate"] for v in active.values()])
        avg_return = np.mean([v["total_return"] for v in active.values()])
        avg_pf = np.mean([v["profit_factor"] for v in active.values()])
        avg_dd = np.mean([v["max_dd"] for v in active.values()])
        avg_sharpe = np.mean([v["sharpe"] for v in active.values()])
        avg_rr = np.mean([v["rr_ratio"] for v in active.values()])

        print(f"\n  {'Metric':<25} {'Value':>15}")
        print(f"  {'-'*40}")
        print(f"  {'Total Trades':<25} {total_trades:>15,}")
        print(f"  {'Avg Win Rate':<25} {avg_wr:>14.1f}%")
        print(f"  {'Avg Return':<25} {avg_return:>+14.2f}%")
        print(f"  {'Avg Risk:Reward':<25} {'1:' + f'{avg_rr:.2f}':>13}")
        print(f"  {'Avg Profit Factor':<25} {avg_pf:>15.2f}")
        print(f"  {'Avg Max Drawdown':<25} {avg_dd:>14.2f}%")
        print(f"  {'Avg Sharpe Ratio':<25} {avg_sharpe:>15.2f}")

    # Save report
    report = {
        "timestamp": "2026-03-19",
        "model": "best_transformer.pt",
        "holdout_pct": "20%",
        "results": all_results,
    }
    report_path = REPORT_DIR / "backtest_transformer.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    print("\n" + "=" * 70)
    print("  TRANSFORMER BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
