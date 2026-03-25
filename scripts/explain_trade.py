#!/usr/bin/env python3
"""
V3.6.1 'Giải Trình' Script (Trade Explainer)

Runs inference using a trained AttentionPPO model (e.g., Stage 2) and outputs
a detailed 'thought process' log for the latest N trades, explaining WHY the
decision was made based on attention weights and feature engineering (SMC).
"""
import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

# Setup Python path to include rabit_propfirm_drl
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "rabit_propfirm_drl"))

from models.attention_ppo import AttentionPPO, TOKEN_NAMES
from environments.prop_env import MultiTFTradingEnv
from data_engine.normalizer import RunningNormalizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("explain_trade")

def analyze_token_features(token_name, m5_idx, env):
    """Trích xuất các tính năng SMC và Trend từ dữ liệu thô."""
    features = []
    
    if token_name == "H1":
        idx = min(m5_idx // (env.m5_per_m15 * env.m15_per_h1), len(env.data_h1) - 1)
        data = env.data_h1
    elif token_name == "M15":
        idx = min(m5_idx // env.m5_per_m15, len(env.data_m15) - 1)
        data = env.data_m15
    elif token_name == "M5":
        idx = min(m5_idx, len(env.data_m5) - 1)
        data = env.data_m5
    elif token_name.startswith("M1_b"):
        m1_end_idx = min(m5_idx * env.m1_per_m5 + env.m1_per_m5 - 1, len(env.data_m1) - 1)
        m1_start_idx = max(0, m1_end_idx - 4)
        offset = int(token_name[-1]) - 1 # b1=0, b2=1 ... b5=4
        idx = m1_start_idx + offset
        data = env.data_m1
    else:
        return ""

    if idx < 0 or idx >= len(data): return ""
    bar = data[idx]

    # Trend (EMA slope or return)
    # We will use the simplified check for trend
    ret = bar[27] if 27 < len(bar) else 0
    if ret > 0.001: features.append("Đang Uptrend (Momentum dương)")
    elif ret < -0.001: features.append("Đang Downtrend (Momentum âm)")
    
    # SMC: BOS / CHOCH
    if 31 < len(bar) and bar[31] > 0.5: features.append("BOS Tăng (Phá vỡ cấu trúc tăng)")
    elif 32 < len(bar) and bar[32] > 0.5: features.append("BOS Giảm (Phá vỡ cấu trúc giảm)")
    if 33 < len(bar) and bar[33] > 0.5: features.append("CHoCH Tăng")
    elif 34 < len(bar) and bar[34] > 0.5: features.append("CHoCH Giảm")

    # Order Block Proximity
    ob_prox = env._compute_ob_proximity(data, idx)
    if ob_prox < 0.1: features.append("Giá CỰC GẦN Order Block!")
    elif ob_prox < 0.3: features.append(f"Có Order Block tiềm năng (Khoảng cách {ob_prox:.2f})")

    # Volume Spike
    vol_ratio = bar[20] if 20 < len(bar) else 0
    spike_mag = env._compute_volume_spike(data, idx)
    if spike_mag > 0.5: features.append(f"Volume NỔ ĐỘT BIẾN (x{vol_ratio:.1f} chuẩn hóa)")
    elif vol_ratio > 1.5: features.append(f"Volume tăng (x{vol_ratio:.1f})")
    
    # Pinbar / Engulfing
    if 45 < len(bar) and bar[45] > 0.5: features.append("Nến Pinbar Tăng")
    elif 46 < len(bar) and bar[46] > 0.5: features.append("Nến Pinbar Giảm")
    if 47 < len(bar) and bar[47] > 0.5: features.append("Engulfing Tăng")
    elif 48 < len(bar) and bar[48] > 0.5: features.append("Engulfing Giảm")

    return " | ".join(features)

def explain_decision(action, probs, attn_weights, obs_dim, env, m5_idx):
    """Generates the explanation string."""
    actions = ["BUY", "SELL", "HOLD", "CLOSE"]
    confidence = float(probs[action]) * 100
    
    # Check if confidence gate was triggered
    raw_best_action = int(torch.argmax(probs))
    raw_confidence = float(probs[raw_best_action]) * 100

    decision = actions[action]
    if action == 2 and raw_best_action in [0, 1] and raw_confidence < 70.0:
        decision = f"TỪ CHỐI {actions[raw_best_action]} (KÍCH HOẠT KHIÊN M1)"
        confidence = raw_confidence

    explanation = [f"\n🎯 QUYẾT ĐỊNH: {decision} | Tự tin: {confidence:.1f}%"]
    if "TỪ CHỐI" in decision:
        explanation.append("   Lý do: Xác suất < 70% (Chỉ báo Khiên M1 bảo vệ). Không đủ điều kiện bắn tỉa.")
    
    # Attention Analysis
    attn_avg = attn_weights.squeeze(0).mean(dim=0).mean(dim=0).cpu().numpy() # [8]
    attn_avg = (attn_avg / attn_avg.sum()) * 100
    
    # Sort top tokens
    top_indices = np.argsort(attn_avg)[::-1][:3] # Top 3 tokens
    
    explanation.append("\n🧠 NHẬT KÝ SUY NGHĨ (Top Attention):")
    for idx in top_indices:
        token_name = TOKEN_NAMES[idx]
        weight = attn_avg[idx]
        feature_desc = analyze_token_features(token_name, m5_idx, env)
        desc = f"      - Nhìn vào {token_name} ({weight:.1f}% chú ý): "
        if feature_desc:
            desc += feature_desc
        else:
            desc += "Không có tín hiệu SMC/Volume nổi bật."
        explanation.append(desc)
    
    return "\n".join(explanation)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models_saved/best_v36_stage2.pt", help="Path to checkpoint")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Symbol to analyze")
    parser.add_argument("--limit", type=int, default=10, help="Number of trades to explain")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = PROJECT_ROOT / args.model
    if not ckpt_path.exists():
        logger.error(f"Cannot find checkpoint: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_dim = ckpt.get("obs_dim", 416)
    model = AttentionPPO(obs_dim=obs_dim, n_actions=4).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded Model {args.model} | Khung {obs_dim}-dim | Cấp độ 3 SMC")

    # Load Data
    cfg_path = PROJECT_ROOT / "rabit_propfirm_drl/configs/prop_rules.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["stage1_mode"] = True

    norm_path = PROJECT_ROOT / "data/normalizer_v3.json"
    with open(norm_path) as f:
        nd = json.load(f)
    norms = {k: RunningNormalizer.from_state_dict(v) for k, v in nd.items()}

    sym = args.symbol
    safe = sym.replace(".", "_")
    sd = {}
    for tf in ["M1", "M5", "M15", "H1"]:
        npy_path = PROJECT_ROOT / f"data/{safe}_{tf}_50dim.npy"
        if not npy_path.exists():
            logger.error(f"Data not found: {npy_path}. Vui lòng kiểm tra data.")
            return
        sd[tf] = norms[tf].normalize(np.load(npy_path)).astype(np.float32)
        op = PROJECT_ROOT / f"data/{safe}_{tf}_ohlcv.npy"
        sd[f"{tf}_ohlcv"] = np.load(op).astype(np.float32) if op.exists() else None

    env = MultiTFTradingEnv(
        data_m1=sd["M1"], data_m5=sd["M5"], data_m15=sd["M15"], data_h1=sd["H1"],
        config=cfg, n_features=50, initial_balance=10000.0,
        episode_length=2000, ohlcv_m5=sd.get("M5_ohlcv"), action_mode="discrete"
    )

    explained_count = 0
    episodes = 0
    max_episodes = 50

    logger.info(f"\n================ GIẢI TRÌNH LỆNH {sym} ================")

    while explained_count < args.limit and episodes < max_episodes:
        obs, _ = env.reset(seed=42 + episodes)
        episodes += 1

        for step in range(2000):
            m5_idx = min(env.current_m5_step, env.n_m5_bars - 1)
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            obs_t = torch.nan_to_num(obs_t, nan=0.0)
            
            with torch.no_grad():
                logits, _, attn_w = model(obs_t)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                dist = torch.distributions.Categorical(probs=probs)
                action = int(dist.sample().item())

            raw_confidence = float(probs[action]) * 100
            
            action_is_meaningful = False
            if action in [0, 1]:
                action_is_meaningful = True

            if action_is_meaningful:
                logger.info(f"\n==================================================")
                logger.info(f"LỆNH #{explained_count+1} | Bước {step}/2000 | Equity: {env.equity:.2f}")
                explanation = explain_decision(action, probs, attn_w, obs_dim, env, m5_idx)
                logger.info(explanation)
                
                explained_count += 1
                if explained_count >= args.limit:
                    break

            obs, _, term, trunc, _ = env.step(action)
            
            if term or trunc:
                break

    logger.info(f"==================================================\n")

if __name__ == "__main__":
    main()
