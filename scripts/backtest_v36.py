#!/usr/bin/env python3
"""
V3.6 Behavioral Backtest + Attention Heatmap Analysis.

Outputs:
  1. Per-symbol WR, MC, SL stats
  2. Attention Weights analysis: which TFs bot looks at for WIN vs LOSS trades
"""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rabit_propfirm_drl"))

from models.attention_ppo import AttentionPPO, TOKEN_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = Path(__file__).resolve().parent.parent / "data"
MODELS = Path(__file__).resolve().parent.parent / "models_saved"

def load_model():
    ckpt = torch.load(MODELS / "best_v36_stage1.pt", map_location=device, weights_only=False)
    m = AttentionPPO(obs_dim=400, n_actions=4).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    print(f"V3.6 Stage 1 loaded (step={ckpt.get('step', 0)}, type={ckpt.get('model_type', 'unknown')})")
    return m

def run_backtest():
    import yaml
    from data_engine.normalizer import RunningNormalizer
    from environments.prop_env import MultiTFTradingEnv

    model = load_model()
    with open(Path(__file__).resolve().parent.parent / "rabit_propfirm_drl" / "configs" / "prop_rules.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["stage1_mode"] = True
    with open(DATA / "normalizer_v3.json") as f:
        nd = json.load(f)
    norms = {k: RunningNormalizer.from_state_dict(v) for k, v in nd.items()}

    syms = ["XAUUSD", "BTCUSD", "ETHUSD", "US30_cash", "US100_cash"]
    ta_t = ta_w = ta_mc = ta_mcw = ta_sl = 0

    # Attention collection
    win_attn_all = []  # Attention weights on WIN trades
    loss_attn_all = []  # Attention weights on LOSS trades
    all_attn = []  # All attention weights

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

        B = S = H = C = 0
        at = []
        pending_attn = {}  # ticket -> attn_weights at entry

        for ep in range(10):
            obs, _ = env.reset(seed=42 + ep)
            prev_tickets = set()

            for s in range(2000):
                ot = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                ot = torch.nan_to_num(ot, nan=0.0)
                with torch.no_grad():
                    logits, value, attn_w = model(ot)
                    a_idx = int(torch.argmax(torch.softmax(logits, dim=-1)))

                # Average attention across heads: (1, n_heads, 8, 8) -> (8, 8)
                avg_attn = attn_w.squeeze(0).mean(dim=0).cpu().numpy()
                all_attn.append(avg_attn)

                if a_idx == 0: B += 1
                elif a_idx == 1: S += 1
                elif a_idx == 2: H += 1
                else: C += 1

                # Track entry attention
                current_tickets = {p.ticket for p in env.positions}
                new_tickets = current_tickets - prev_tickets
                for t in new_tickets:
                    pending_attn[t] = avg_attn

                obs, _, t, tr, _ = env.step(a_idx)

                # Check closed trades
                for trade in list(env.trade_history):
                    ticket = trade.get("ticket")
                    if ticket in pending_attn:
                        entry_attn = pending_attn.pop(ticket)
                        if trade.get("pnl", 0) > 0:
                            win_attn_all.append(entry_attn)
                        else:
                            loss_attn_all.append(entry_attn)

                prev_tickets = current_tickets
                if t or tr:
                    break
            at.extend(env.trade_history)

        tot = len(at)
        w = sum(1 for t in at if t.get("pnl", 0) > 0)
        mc = sum(1 for t in at if t.get("close_reason") == "MANUAL_CLOSE")
        mcw = sum(1 for t in at if t.get("close_reason") == "MANUAL_CLOSE" and t.get("pnl", 0) > 0)
        sl = sum(1 for t in at if t.get("close_reason") == "SL_HIT")
        wr = 100 * w / max(tot, 1)
        ta2 = B + S + H + C
        ta_t += tot; ta_w += w; ta_mc += mc; ta_mcw += mcw; ta_sl += sl
        dur = np.mean([t.get("duration", 0) for t in at]) if at else 0
        st = "PASS" if wr > 40 else "WATCH"
        print(f"{sym:12s} | T={tot:5d} | WR={wr:5.1f}% | MC={mc:4d}(W={mcw}) | SL={sl:4d} | D={dur:.1f} | "
              f"B={100*B/ta2:.0f}% S={100*S/ta2:.0f}% H={100*H/ta2:.0f}% C={100*C/ta2:.0f}% | {st}")

    print("=" * 100)
    twr = 100 * ta_w / max(ta_t, 1)
    mcwr = 100 * ta_mcw / max(ta_mc, 1)
    print(f"OVERALL: {ta_t} trades | WR={twr:.1f}% | MC={ta_mc}(WR={mcwr:.1f}%) | SL={ta_sl}")
    print("=" * 100)

    # --- ATTENTION HEATMAP ANALYSIS ---
    print("\n" + "=" * 70)
    print("  ATTENTION ANALYSIS — What does the bot look at?")
    print("=" * 70)

    if all_attn:
        avg_all = np.mean(all_attn, axis=0)  # (8, 8)
        token_importance = avg_all.sum(axis=0)  # How much attention each token RECEIVES
        token_importance /= token_importance.sum()

        print("\n📊 Token Importance (how much attention each TF receives):")
        for i, name in enumerate(TOKEN_NAMES):
            bar = "█" * int(token_importance[i] * 80)
            print(f"  {name:6s} | {token_importance[i]*100:5.1f}% | {bar}")

    if win_attn_all and loss_attn_all:
        avg_win = np.mean(win_attn_all, axis=0)
        avg_loss = np.mean(loss_attn_all, axis=0)
        diff = avg_win - avg_loss

        win_imp = avg_win.sum(axis=0); win_imp /= win_imp.sum()
        loss_imp = avg_loss.sum(axis=0); loss_imp /= loss_imp.sum()

        print(f"\n🏆 WIN trades attention ({len(win_attn_all)} trades):")
        for i, name in enumerate(TOKEN_NAMES):
            bar = "█" * int(win_imp[i] * 80)
            print(f"  {name:6s} | {win_imp[i]*100:5.1f}% | {bar}")

        print(f"\n❌ LOSS trades attention ({len(loss_attn_all)} trades):")
        for i, name in enumerate(TOKEN_NAMES):
            bar = "█" * int(loss_imp[i] * 80)
            print(f"  {name:6s} | {loss_imp[i]*100:5.1f}% | {bar}")

        print("\n📈 WIN - LOSS difference (positive = bot looks MORE at this TF when winning):")
        for i, name in enumerate(TOKEN_NAMES):
            d = (win_imp[i] - loss_imp[i]) * 100
            arrow = "↑" if d > 0 else "↓" if d < 0 else "="
            bar = "+" * max(0, int(d * 20)) if d > 0 else "-" * max(0, int(-d * 20))
            print(f"  {name:6s} | {d:+5.2f}% {arrow} | {bar}")
    else:
        print("\n⚠️ Not enough WIN/LOSS trades for attention comparison")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    run_backtest()
