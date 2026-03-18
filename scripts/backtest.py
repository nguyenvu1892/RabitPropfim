"""
Backtest — Walk-forward evaluation of trained SAC model on holdout data.

Usage: py -3.11 scripts/backtest.py

Tests the trained model on the LAST portion of data (not used in training random starts)
to generate realistic performance metrics and equity curve.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
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

LOOKBACK = 64
INITIAL_BALANCE = 100_000.0
MAX_LOSS_PCT = 0.003   # 0.3% SL per trade
DAILY_COOLDOWN = 0.03  # 3% daily cooldown


class SACAgent(torch.nn.Module):
    """Matching architecture from train_agent.py"""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        )
        self.mu = torch.nn.Linear(hidden, act_dim)
        self.log_std = torch.nn.Linear(hidden, act_dim)
        self.q1 = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + act_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.q2 = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + act_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.q1_target = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + act_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.q2_target = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + act_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.log_alpha = torch.nn.Parameter(torch.zeros(1))
        self.target_entropy = -act_dim

    def get_action(self, obs, deterministic=True):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            h = self.actor(obs_t)
            mu = self.mu(h)
            if deterministic:
                return torch.tanh(mu).squeeze(0).numpy()
            log_std = torch.clamp(self.log_std(h), -20, 2)
            z = mu + log_std.exp() * torch.randn_like(log_std)
            return torch.tanh(z).squeeze(0).numpy()


def main():
    import polars as pl

    print("=" * 70)
    print("  RABIT-PROPFIRM -- BACKTEST REPORT")
    print("  Model: SMC + Volume + Price Action")
    print("=" * 70)

    # Load model
    ckpt_path = MODEL_DIR / "best_model.pt"
    print(f"\n[1/3] Loading model from {ckpt_path.name}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    obs_dim = ckpt["obs_dim"]
    act_dim = ckpt["act_dim"]
    norm_state = ckpt["norm_state"]
    mean = np.array(norm_state["mean"], dtype=np.float32)
    std = np.array(norm_state["std"], dtype=np.float32)

    agent = SACAgent(obs_dim, act_dim)
    agent.load_state_dict(ckpt["agent_state"])
    agent.eval()
    print(f"  Loaded: step {ckpt.get('step', '?')}, eval_reward {ckpt.get('eval_reward', '?'):.2f}")

    # Load data + run backtest per symbol
    print(f"\n[2/3] Running walk-forward backtest...")
    print("-" * 70)

    all_results = {}

    for feat_file in sorted(DATA_DIR.glob("*_M5_features.parquet")):
        sym = feat_file.stem.replace("_M5_features", "")
        df = pl.read_parquet(feat_file)

        # Features
        available_cols = [c for c in FEATURE_COLS if c in df.columns]
        features = np.column_stack([
            df[col].fill_null(0.0).fill_nan(0.0).to_numpy().astype(np.float32)
            for col in available_cols
        ])
        features_norm = (features - mean) / std

        ohlcv = np.column_stack([
            df["open"].to_numpy().astype(np.float32),
            df["high"].to_numpy().astype(np.float32),
            df["low"].to_numpy().astype(np.float32),
            df["close"].to_numpy().astype(np.float32),
        ])
        times = df["time"].to_list() if "time" in df.columns else []

        # H1 inside bar
        ib_path = DATA_DIR / f"{sym}_H1_insidebar.parquet"
        h1_ib_times = set()
        if ib_path.exists():
            ib_df = pl.read_parquet(ib_path)
            h1_ib_times = set(ib_df.filter(pl.col("inside_bar") == 1.0)["time"].to_list())

        # Use LAST 20% of data as holdout (walk-forward)
        n = len(features_norm)
        holdout_start = int(n * 0.8)
        holdout_features = features_norm[holdout_start:]
        holdout_ohlcv = ohlcv[holdout_start:]
        holdout_times = times[holdout_start:] if times else []

        print(f"\n  === {sym} === (holdout: {len(holdout_features):,} bars, "
              f"~{len(holdout_features) * 5 / 60 / 24:.0f} days)")

        # Walk-forward simulation
        balance = INITIAL_BALANCE
        position = 0.0  # +1 long, -1 short, 0 flat
        entry_price = 0.0
        entry_time = None
        risk_frac = 0.0

        trades: list[dict] = []
        equity_curve = [balance]
        daily_loss = 0.0
        current_day = None
        cooled_down = False

        for i in range(LOOKBACK, len(holdout_features)):
            close_price = float(holdout_ohlcv[i, 3])
            bar_time = holdout_times[i] if i < len(holdout_times) else None

            # Day rollover
            if bar_time:
                day = bar_time.date() if hasattr(bar_time, 'date') else None
                if day and day != current_day:
                    current_day = day
                    daily_loss = 0.0
                    cooled_down = False

            # Build observation
            window = holdout_features[i - LOOKBACK:i]
            flat = window.flatten()
            extra = np.array([
                balance / INITIAL_BALANCE,
                position,
                0.0,  # unrealized placeholder
            ], dtype=np.float32)
            obs = np.concatenate([flat, extra])

            # Get action
            action = agent.get_action(obs, deterministic=True)
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
                    "entry_time": str(entry_time) if entry_time else "",
                    "exit_time": str(bar_time) if bar_time else "",
                })
                position = 0.0

            # === UPDATE UNREALIZED PNL ===
            if position != 0:
                pnl_pct = position * (close_price - entry_price) / entry_price * risk_frac
                pnl_usd = pnl_pct * balance

                # SL hit (0.3% max)
                if pnl_usd < -(balance * MAX_LOSS_PCT):
                    loss = balance * MAX_LOSS_PCT
                    balance -= loss
                    daily_loss += MAX_LOSS_PCT
                    trades.append({
                        "symbol": sym, "direction": "LONG" if position > 0 else "SHORT",
                        "entry_price": entry_price, "exit_price": close_price,
                        "pnl_pct": -MAX_LOSS_PCT * 100, "pnl_usd": -loss,
                        "reason": "SL_HIT",
                        "entry_time": str(entry_time) if entry_time else "",
                        "exit_time": str(bar_time) if bar_time else "",
                    })
                    position = 0.0

                # TP hit (1%)
                elif pnl_usd > balance * 0.01:
                    gain = balance * 0.01
                    balance += gain
                    trades.append({
                        "symbol": sym, "direction": "LONG" if position > 0 else "SHORT",
                        "entry_price": entry_price, "exit_price": close_price,
                        "pnl_pct": 1.0, "pnl_usd": gain,
                        "reason": "TP_HIT",
                        "entry_time": str(entry_time) if entry_time else "",
                        "exit_time": str(bar_time) if bar_time else "",
                    })
                    position = 0.0

            # === OPEN NEW POSITION ===
            if (abs(confidence) > 0.3
                    and position == 0
                    and not cooled_down
                    and not is_h1_ib):
                position = 1.0 if confidence > 0 else -1.0
                entry_price = close_price
                entry_time = bar_time
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
                    "entry_time": str(entry_time) if entry_time else "",
                    "exit_time": str(bar_time) if bar_time else "",
                })
                position = 0.0

            # Daily cooldown check
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
                        "entry_time": str(entry_time) if entry_time else "",
                        "exit_time": str(bar_time) if bar_time else "",
                    })
                    position = 0.0

            equity_curve.append(balance)

        # Calculate metrics
        if trades:
            pnls = [t["pnl_usd"] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            pnl_pcts = [t["pnl_pct"] for t in trades]

            total_pnl = balance - INITIAL_BALANCE
            total_return = (balance / INITIAL_BALANCE - 1) * 100
            win_rate = len(wins) / len(trades) * 100
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 0
            rr_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")
            profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")

            # Max drawdown
            peak = INITIAL_BALANCE
            max_dd = 0.0
            for eq in equity_curve:
                peak = max(peak, eq)
                dd = (peak - eq) / peak
                max_dd = max(max_dd, dd)

            # Sharpe (daily returns)
            daily_returns = []
            day_pnl = 0.0
            last_day = None
            for t in trades:
                t_time = t.get("exit_time", "")
                t_day = t_time[:10] if len(t_time) >= 10 else ""
                if t_day != last_day and last_day is not None:
                    daily_returns.append(day_pnl)
                    day_pnl = 0.0
                day_pnl += t["pnl_pct"]
                last_day = t_day
            daily_returns.append(day_pnl)

            sharpe = 0.0
            if len(daily_returns) > 1 and np.std(daily_returns) > 0:
                sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

            # Sortino (downside deviation only)
            neg_returns = [r for r in daily_returns if r < 0]
            sortino = 0.0
            if neg_returns and np.std(neg_returns) > 0:
                sortino = np.mean(daily_returns) / np.std(neg_returns) * np.sqrt(252)

            # Estimate days to 10%
            if total_return > 0 and len(daily_returns) > 0:
                avg_daily_return = np.mean(daily_returns)
                if avg_daily_return > 0:
                    days_to_10pct = 10.0 / avg_daily_return
                else:
                    days_to_10pct = float("inf")
            else:
                days_to_10pct = float("inf")

            # Exit reason breakdown
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
            print(f"  Sortino Ratio:    {sortino:.2f}")
            print(f"  Est. days to 10%: {days_to_10pct:.0f} trading days")
            print(f"  Exit reasons:     {reasons}")

            all_results[sym] = {
                "trades": len(trades),
                "win_rate": win_rate,
                "total_return": total_return,
                "total_pnl": total_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "rr_ratio": rr_ratio,
                "profit_factor": profit_factor,
                "max_dd": max_dd * 100,
                "sharpe": sharpe,
                "sortino": sortino,
                "days_to_10pct": days_to_10pct,
                "equity_curve": equity_curve,
                "trade_log": trades,
                "exit_reasons": reasons,
            }
        else:
            print(f"  No trades executed!")
            all_results[sym] = {"trades": 0}

    # === COMBINED REPORT ===
    print("\n" + "=" * 70)
    print("  COMBINED BACKTEST REPORT")
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
        avg_days = np.mean([v["days_to_10pct"] for v in active.values()
                           if v["days_to_10pct"] < 9999])

        print(f"\n  {'Metric':<25} {'Value':>15}")
        print(f"  {'-'*40}")
        print(f"  {'Total Trades':<25} {total_trades:>15,}")
        print(f"  {'Avg Win Rate':<25} {avg_wr:>14.1f}%")
        print(f"  {'Avg Return':<25} {avg_return:>+14.2f}%")
        print(f"  {'Avg Risk:Reward':<25} {'1:' + f'{avg_rr:.2f}':>13}")
        print(f"  {'Avg Profit Factor':<25} {avg_pf:>15.2f}")
        print(f"  {'Avg Max Drawdown':<25} {avg_dd:>14.2f}%")
        print(f"  {'Avg Sharpe Ratio':<25} {avg_sharpe:>15.2f}")
        print(f"  {'Est. Days to 10%':<25} {avg_days:>14.0f}")

        # V1 PASS assessment
        print("\n  ─── V1 PASS ASSESSMENT ───")
        can_pass = avg_return > 0 and avg_dd < 5.0 and avg_sharpe > 0.5
        if can_pass:
            print(f"  ✅ Model shows POSITIVE expectation — worth paper trading")
        else:
            issues = []
            if avg_return <= 0:
                issues.append("negative return")
            if avg_dd >= 5.0:
                issues.append(f"DD too high ({avg_dd:.1f}%)")
            if avg_sharpe <= 0.5:
                issues.append(f"Sharpe too low ({avg_sharpe:.2f})")
            print(f"  ⚠️ Issues: {', '.join(issues)} — needs improvement")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": str(ckpt_path),
        "holdout_pct": "20%",
        "results": {k: {key: val for key, val in v.items()
                        if key not in ("equity_curve", "trade_log")}
                    for k, v in all_results.items()},
    }
    report_path = REPORT_DIR / "backtest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    # Save trade logs
    all_trades = []
    for sym, data in all_results.items():
        all_trades.extend(data.get("trade_log", []))
    trades_path = REPORT_DIR / "trade_log.json"
    with open(trades_path, "w") as f:
        json.dump(all_trades, f, indent=2, default=str)
    print(f"  Trade log saved: {trades_path} ({len(all_trades)} trades)")

    print("\n" + "=" * 70)
    print("  BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
