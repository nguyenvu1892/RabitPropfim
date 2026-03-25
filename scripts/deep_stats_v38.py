#!/usr/bin/env python3
"""V3.8 Deep Statistics: Timeframe, Trades/Day, R:R Ratio."""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rabit_propfirm_drl"))

import yaml
from models.attention_ppo import AttentionPPO
from data_engine.normalizer import RunningNormalizer
from environments.prop_env import MultiTFTradingEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = Path(__file__).resolve().parent.parent / "data"
MODELS = Path(__file__).resolve().parent.parent / "models_saved"

def main():
    # Load model
    ckpt = torch.load(MODELS / "best_v36_stage3.pt", map_location=device, weights_only=False)
    m = AttentionPPO(obs_dim=432, n_actions=4).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    print(f"Loaded V3.8 S3 (step={ckpt.get('step', 0)})")

    with open(Path(__file__).resolve().parent.parent / "rabit_propfirm_drl" / "configs" / "prop_rules.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["stage1_mode"] = True
    with open(DATA / "normalizer_v3.json") as f:
        nd = json.load(f)
    norms = {k: RunningNormalizer.from_state_dict(v) for k, v in nd.items()}

    syms = ["XAUUSD", "BTCUSD", "ETHUSD", "US30_cash", "US100_cash"]

    print("\n" + "="*80)
    print("  V3.8 DEEP STATISTICS REPORT")
    print("="*80)

    # 1. DATA TIMEFRAME
    print("\n📅 1. DATA TIMEFRAME")
    print("-"*50)
    for sym in syms:
        safe = sym.replace(".", "_")
        m5_data = np.load(DATA / f"{safe}_M5_50dim.npy")
        m5_bars = len(m5_data)
        trading_days = m5_bars / 288  # 288 M5 bars per day
        trading_months = trading_days / 21  # ~21 trading days per month
        print(f"  {sym:12s} | M5 bars: {m5_bars:,} | ~{trading_days:.0f} trading days | ~{trading_months:.1f} months")

    # 2 & 3. RUN BACKTEST WITH PnL TRACKING
    print(f"\n💰 2-3. PER-SYMBOL STATS (Trades/Day + R:R)")
    print("-"*80)

    total_all_trades = 0
    total_all_days = 0
    all_win_pnls = []
    all_loss_pnls = []

    for sym in syms:
        safe = sym.replace(".", "_")
        sd = {}
        for tf in ["M1", "M5", "M15", "H1"]:
            sd[tf] = np.load(DATA / f"{safe}_{tf}_50dim.npy")
        ohlcv = np.load(DATA / f"{safe}_M5_ohlcv.npy")
        env = MultiTFTradingEnv(
            data_m1=sd["M1"], data_m5=sd["M5"], data_m15=sd["M15"], data_h1=sd["H1"],
            config={**cfg, "symbol": sym}, n_features=50, initial_balance=10000,
            episode_length=2000, ohlcv_m5=ohlcv, action_mode="discrete",
        )
        obs, _ = env.reset()
        
        m5_bars = len(sd["M5"])
        trading_days = m5_bars / 288

        trades = 0
        win_pnls = []
        loss_pnls = []

        for ep in range(10):
            obs, _ = env.reset(seed=42 + ep)

            for step in range(2000):
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                obs_t = torch.nan_to_num(obs_t, nan=0.0)
                with torch.no_grad():
                    logits, value, attn_w = m(obs_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = int(dist.sample().item())
                obs, reward, term, trunc, info = env.step(action)

                if term or trunc:
                    break

            # Collect all closed trades from this episode BEFORE next reset
            for trade in env.trade_history:
                pnl = trade.get("pnl", 0)
                trades += 1
                if pnl > 0:
                    win_pnls.append(pnl)
                elif pnl < 0:
                    loss_pnls.append(abs(pnl))

        trades_per_day = trades / max(trading_days, 1)
        avg_win = np.mean(win_pnls) if win_pnls else 0
        avg_loss = np.mean(loss_pnls) if loss_pnls else 1
        rr_ratio = avg_win / max(avg_loss, 1e-8)
        wr = len(win_pnls) / max(trades, 1) * 100

        total_all_trades += trades
        total_all_days += trading_days
        all_win_pnls.extend(win_pnls)
        all_loss_pnls.extend(loss_pnls)

        print(f"  {sym:12s} | Trades: {trades:5d} | Days: {trading_days:.0f} | "
              f"T/Day: {trades_per_day:.1f} | WR: {wr:.1f}% | "
              f"Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f} | R:R = 1:{rr_ratio:.2f}")

    # OVERALL
    print("-"*80)
    overall_tpd = total_all_trades / max(total_all_days, 1)
    overall_avg_win = np.mean(all_win_pnls) if all_win_pnls else 0
    overall_avg_loss = np.mean(all_loss_pnls) if all_loss_pnls else 1
    overall_rr = overall_avg_win / max(overall_avg_loss, 1e-8)
    overall_wr = len(all_win_pnls) / max(len(all_win_pnls) + len(all_loss_pnls), 1) * 100

    print(f"\n📊 OVERALL SUMMARY:")
    print(f"  Total Trades: {total_all_trades:,}")
    print(f"  Total Trading Days: {total_all_days:.0f} ({total_all_days/21:.1f} months)")
    print(f"  Trades/Day (all symbols): {overall_tpd:.1f}")
    print(f"  Trades/Day/Symbol: {overall_tpd/5:.1f}")
    print(f"  Win Rate: {overall_wr:.1f}%")
    print(f"  Avg Win PnL:  ${overall_avg_win:.4f}")
    print(f"  Avg Loss PnL: ${overall_avg_loss:.4f}")
    print(f"  Risk:Reward = 1:{overall_rr:.2f}")
    
    expectancy = (overall_wr/100 * overall_avg_win) - ((100-overall_wr)/100 * overall_avg_loss)
    print(f"  Expectancy/Trade: ${expectancy:.4f}")
    print(f"\n  {'✅ R:R > 1.5 — PASS' if overall_rr >= 1.5 else '⚠️ R:R < 1.5 — NEEDS IMPROVEMENT'}")
    print(f"  {'✅ T/Day/Sym ≤ 5 — NOT OVERTRADING' if overall_tpd/5 <= 5 else '⚠️ T/Day/Sym > 5 — OVERTRADING!'}")
    print("="*80)

if __name__ == "__main__":
    main()
