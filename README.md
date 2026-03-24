# 🐰 RabitPropfim — DRL Trading Bot cho Prop Firm

> **Tầm nhìn:** Tạo ra một BOT AI có khả năng **tự suy nghĩ, tự tiến hóa** dựa vào các kiến thức được nạp vào (SMC + Volume + Pin Action). Bot được thiết kế để giao dịch các quỹ Prop Firm, với nguyên tắc sắt: **mỗi lệnh không được thua quá 0.3% tài khoản.**

---

## 📋 Mục Tiêu Dự Án

### Tầm nhìn cốt lõi
- **Bot AI tự suy nghĩ** — Học cách đọc thị trường thông qua SMC (Smart Money Concepts), Volume Profile, và Pin Action thay vì dùng indicator truyền thống
- **Tự tiến hóa** — Qua hệ thống Self-Imitation Learning (Rương Vàng), bot tự nhặt ra lệnh chuẩn bài để học lại, liên tục nâng cấp tư duy giao dịch
- **Kỷ luật thép Prop Firm** — Tuân thủ tuyệt đối ngưỡng rủi ro: max 0.3% loss/trade, 5% daily DD, 10% total DD

### Chiến lược Khung Thời Gian (Multi-Timeframe)
| Khung TF | Vai trò | Mô tả |
|----------|---------|-------|
| **H1** | 🔭 Xác định xu hướng lớn | Nhìn cấu trúc thị trường (BOS/CHoCH), xác định trend chính |
| **M15** | 🗺️ Xác định vùng cản/OB | Tìm Order Block, Fair Value Gap, vùng supply/demand |
| **M5** | 🎯 Entry chính | Điểm vào lệnh tiêu chuẩn — khi có hợp lưu H1+M15 |
| **M1** | 🔫 Sniper Entry | Chỉ vào lệnh khi **xác suất win > 70%** — Volume nổ + Pin Action chuẩn |

> **Nguyên tắc:** M1 Sniper là vũ khí tối thượng nhưng chỉ bóp cò khi xác suất thắng vượt 70%. Còn lại dùng M5 entry bình thường.

### Mục tiêu cụ thể
1. **Pass Prop Firm Challenge** — Đạt 10% lợi nhuận trong 30 ngày, giữ drawdown trong giới hạn
2. **Bot tự giao dịch 100%** — Từ phân tích → Entry → Quản trị rủi ro → Exit, hoàn toàn tự động
3. **Đa cặp tiền** — XAUUSD, BTCUSD, ETHUSD, US30, US100 (5 symbols)
4. **Kết nối MT5 live** — Giao dịch trực tiếp trên tài khoản FTMO

### Chỉ tiêu kỹ thuật
| Metric | Target | V3.4 S1 Best |
|--------|--------|-------------|
| Win Rate | ≥ 50% | **49.0%** ✅ |
| Max Daily DD | < 5% | ✅ |
| Max Total DD | < 10% | ✅ |
| Sharpe Ratio | > 2.0 | TBD |
| Min Trading Days | ≥ 4 | ✅ |
| Trades/Day | 5-6 chất lượng | ~4.5 |

---

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────┐
│                   MT5 Live Bridge                    │
│  mt5_connector │ inference_pipeline │ paper_trading  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  DRL Agent (PPO)                     │
│  Obs: 350-dim (M15+M5+M1) → Policy → Discrete(4)   │
│  Actions: BUY / SELL / HOLD / CLOSE                  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Trading Environment                     │
│  Auto SL (Swing Point M5) │ Manual CLOSE             │
│  Spread Sim │ Slippage │ Session Control             │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                 Reward Engine                         │
│  PnL-based │ CLOSE Profit ×5 │ No frequency pressure │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  Data Pipeline                       │
│  MT5 → OHLCV → 50-dim Features → ATR Normalization  │
│  M1(250K) + M5(50K) + M15(17K) + H1(4.2K) per sym  │
└─────────────────────────────────────────────────────┘
```

---

## 📦 Cấu Trúc Project

```
RabitPropfim/
├── rabit_propfirm_drl/          # Core package
│   ├── configs/                  # prop_rules.yaml, train_hyperparams.yaml
│   ├── data_engine/              # MT5 fetcher, feature builder, normalizer
│   ├── environments/             # prop_env.py (Gym), reward_engine.py, physics_sim.py
│   ├── models/                   # transformer_smc, cross_attention, actor_critic
│   ├── agents/                   # sac_policy, action_gating, episodic_memory
│   ├── features/                 # knowledge_extractor (22 SMC+PA+Vol features)
│   ├── training_pipeline/        # PER buffer, curriculum, SAC trainer
│   ├── live_execution/           # MT5 connector, inference, paper trading
│   ├── model_registry/           # Versioned checkpoints, rollback
│   └── utils/                    # Telegram alerts, Polars bridge
├── scripts/                      # Training & evaluation scripts
│   ├── train_v34.py              # V3.4 PPO training (Stage 1 + Stage 2)
│   ├── harvest_vip_v34.py        # VIP buffer extraction (SMC + MANUAL_CLOSE)
│   ├── fetch_historical_data.py  # MT5 data fetching
│   └── backtest_*.py             # Behavioral analysis scripts
├── data/                         # .npy feature arrays + OHLCV
├── models_saved/                 # Checkpoints (.pt files)
├── docs/                         # Architecture docs
├── tests/                        # Unit tests (147+ passed)
├── DEVLOG.md                     # Nhật ký phát triển chi tiết
└── README.md                     # (File này)
```

---

## 🔄 Lịch Sử Phiên Bản

### V3.4 — "Quản Trị Rủi Ro" (24/03/2026) ← HIỆN TẠI
- **Obs Space**: 350-dim (M15 + M5 + 5×M1) — 3 khung thời gian
- **Action Space**: Discrete(4) — BUY / SELL / HOLD / **CLOSE**
- **SL**: Auto Swing Point từ M5 (không fix cứng)
- **TP**: Không có — bot tự CLOSE bằng tay
- **Reward**: PnL + CLOSE profit ×5 (không inaction/trade_bonus)
- **Kết quả tốt nhất (Stage 1)**: WR=49%, Manual Close WR=60.8%

### V3.3 — "Rương Vàng" (24/03/2026)
- Discrete(3) BUY/SELL/HOLD, PPO
- ATR Normalization, Frame Stacking M5+M1
- VIP Buffer (Self-Imitation Learning)
- Stage 1: WR=45.3% → Stage 3: WR=77.3% (22 trades)

### V3.2 — "Anti-Mode-Collapse" (23/03/2026)
- SAC 5-dim continuous → Dual Entry (M1 Sniper + M5 Normal)
- Cognitive Architecture: TransformerSMC + CrossAttention + RegimeDetector
- 50-dim features: 28 raw + 22 knowledge (SMC + PA + Volume)

### V2 — SAC MLP Baseline (18-19/03/2026)
- SAC 4-dim continuous, MLP 256×256
- Best WR=49.4%, Sharpe=4.6, ETHUSD +56.7%

### V1 — Foundation (18/03/2026)
- Project setup, 9 packages, 7 sprints planned
- Prop firm rules, Gymnasium env, reward engine

---

## 🖥️ Hạ Tầng

| Component | Spec |
|-----------|------|
| **Training Server** | RTX 4090 24GB (38.224.253.180) |
| **Dev Machine** | Windows, VS Code |
| **Broker** | FTMO Demo (MT5) |
| **Data** | 5 symbols × 4 TF × 50-dim features |
| **Framework** | PyTorch + Gymnasium |
| **Algorithm** | PPO (V3.3+), SAC (V2-V3.2) |

---

## 🚀 Cách Chạy

### 1. Fetch Data
```bash
python scripts/fetch_historical_data.py
```

### 2. Train V3.4
```bash
# Stage 1: Pure PPO (750K steps)
python scripts/train_v34.py --stage 1 --n-envs 12

# Harvest VIP (MANUAL_CLOSE + SMC filter)
python scripts/harvest_vip_v34.py --episodes 50 --cap-per-symbol 40

# Stage 2: PPO + Self-Imitation Learning (500K steps)
python scripts/train_v34.py --stage 2 --n-envs 12 --il-coef 0.10
```

### 3. Backtest
```bash
python scripts/backtest_v34_quick.py
```

---

## 📊 Kết Quả Training V3.4

### Stage 1 (Pure PPO) ← Best Deploy
| Symbol | Trades | WR | Manual Close WR |
|--------|--------|----|----------------|
| XAUUSD | 337 | **57.3%** | 63.3% |
| BTCUSD | 674 | **55.3%** | 66.5% |
| ETHUSD | 352 | 45.5% | 58.8% |
| US30 | 186 | 45.7% | 49.7% |
| US100 | 728 | 41.8% | 57.4% |
| **TỔNG** | **2,277** | **49.0%** | **60.8%** |

### Stage 3 (Balanced VIP + IL)
| Metric | Value |
|--------|-------|
| Trades | 15,547 |
| WR | 38.3% |
| Manual Close WR | **63.5%** |
| VIP Distribution | 40/sym (balanced) |

---

## 📅 Roadmap

- [x] Sprint 1-6: Foundation → Live Execution Engine
- [x] V3.2: Cognitive Architecture (Transformer + SMC)
- [x] V3.3: PPO + VIP Buffer (Self-Imitation Learning)
- [x] V3.4: 3-TF Obs + Auto SL + CLOSE action
- [ ] **V3.5**: Live Paper Trading trên FTMO Demo
- [ ] **V4.0**: Pass FTMO Challenge → Real Account

---

## 👥 Team
- **Anh Vũ** — Project Owner & Strategy
- **Tech Lead** — Architecture Review
- **An (AI)** — Implementation & Training

---

*Cập nhật lần cuối: 24/03/2026 23:30 UTC+7*
