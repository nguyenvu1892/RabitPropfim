# 🏗️ MASTER PLAN FINAL — RABIT-PROPFIRM DRL SYSTEM
> **Phiên bản:** v4.0 (Cập nhật 18/03/2026 — SMC + Volume + Price Action)  
> **Mục tiêu:** Xây dựng hệ thống AI **có trí tuệ trading** (Transformer + Cross-Attention + Regime Detector) để pass quỹ Prop Firm  
> **Thời gian:** 14 tuần (7 Sprints × 2 tuần)  
> **Nguyên tắc:** Tuyệt đối không hardcode tham số. Mọi config nằm ở `.yaml` duy nhất, validate bằng Pydantic  
> **Symbols:** XAUUSD, US100.cash, US30.cash, ETHUSD, BTCUSD (5 symbols, không bỏ cái nào)  
> **Primary TF:** M5 (với H1/H4 multi-TF context)  
> **Features:** SMC (BOS, CHoCH, OB, FVG, Liquidity) + Volume (Delta, Climax) + Price Action (Pin Bar, Engulfing, Inside Bar)

---

## TIẾN ĐỘ HIỆN TẠI

| Sprint | Status | Kết quả |
|--------|--------|--------|
| Sprint 1 — Data Engine | ✅ **DONE** | 28 SMC features, 50K bars/symbol, 161 tests passed |
| Sprint 2 — Gym Environment | ✅ **DONE** (cơ bản) | SimpleTradeEnv + Risk rules (0.3% SL, 3% daily, H1 IB exit) |
| MLP Prototype | ✅ **DONE** | SAC 200K steps, backtest +18.7%, WR 44.8%, PF 1.22 |
| Sprint 3 — Neural Architecture | 🔴 **ĐANG LÀM** | TransformerSMC + CrossAttention + RegimeDetector |
| Sprint 4 — Training Pipeline | ⬜ Chưa | PER + Curriculum Learning |
| Sprint 5 — Ensemble | ⬜ Chưa | 3 models × 2/3 voting |
| Sprint 6 — Paper Trading | ⬜ Chưa | 5+ ngày MT5 Demo |
| Sprint 7 — Live | ⬜ Chưa | FTMO Challenge |  

---

## MỤC LỤC

1. [Tech Stack](#1-tech-stack-chuẩn)
2. [Kiến trúc Monorepo](#2-kiến-trúc-monorepo)
3. [Sơ đồ Hệ thống](#3-sơ-đồ-hệ-thống-tổng-thể)
4. [Sprint 1 — Data Engine](#sprint-1--data-engine--feature-pipeline-tuần-1-2)
5. [Sprint 2 — Gym Environment](#sprint-2--gymnasium-environment--reward-engine-tuần-3-5)
6. [Sprint 3 — Neural Architecture](#sprint-3--neural-architecture--action-gating-tuần-6-7)
7. [Sprint 4 — Training Pipeline](#sprint-4--training-pipeline--curriculum-learning-tuần-8-9)
8. [Sprint 5 — Ensemble & Backtest](#sprint-5--ensemble--out-of-sample-validation-tuần-10-11)
9. [Sprint 6 — Paper Trading](#sprint-6--paper-trading--safety-validation-tuần-12)
10. [Sprint 7 — Live Deployment](#sprint-7--live-deployment--self-evolution-tuần-13-14)

---

## 1. TECH STACK CHUẨN

| Layer | Công nghệ | Lý do chọn |
|-------|-----------|------------|
| **Ngôn ngữ** | Python 3.10+ (Type Hinting 100%) | Ecosystem ML/DRL mạnh nhất, type hints bắt buộc cho team maintenance |
| **Data Processing** | Polars + NumPy | Polars nhanh hơn Pandas 10-50x cho time-series, tiết kiệm RAM cho tick data 5 năm |
| **Simulation** | Gymnasium (OpenAI) | Standard interface cho RL environments, thay thế gym cũ |
| **Deep Learning** | PyTorch 2.x | torch.compile, FSDP, cần thiết cho custom Transformer + DRL |
| **DRL Framework** | Ray RLlib hoặc Stable-Baselines3 | Algo chính: SAC (Soft Actor-Critic) cho continuous action space |
| **Broker API** | MetaTrader5 Python | API chính thức, lấy tick data + đẩy lệnh live |
| **Experiment Tracking** | Weights & Biases (W&B) | Training loss/reward real-time trên web, team giám sát từ xa |
| **Data Versioning** | DVC (Data Version Control) | Quản lý phiên bản dataset, reproduce được training results |
| **Config Validation** | Pydantic v2 | Validate yaml trước khi chạy, catch lỗi config trước khi mất tiền |
| **Alert System** | Telegram Bot API | Push notification real-time cho mọi event quan trọng |

---

## 2. KIẾN TRÚC MONOREPO

```
rabit_propfirm_drl/
│
├── configs/                          # ⚙️ NƠI DUY NHẤT CHỨA THAM SỐ
│   ├── prop_rules.yaml               #    Max DD, Trading Hours, Max Lots
│   ├── train_hyperparams.yaml         #    SAC config, LR, Batch size, Gamma
│   └── validator.py                   #    Pydantic schema — validate trước khi chạy
│
├── data_engine/                      # 🧑‍💻 [TEAM DATA] ✅
│   ├── mt5_fetcher.py                 #    Kéo M5 từ MT5 → .parquet (5 symbols)
│   ├── feature_builder.py             #    ✅ SMC + Volume + PA → 28 features
│   ├── normalizer.py                  #    ✅ Welford's Running Normalizer
│   └── multi_tf_builder.py            #    ✅ Build Multi-Timeframe features (M5/M15/H1/H4)
│
├── environments/                     # 🧑‍💻 [TEAM QUANT/BACKEND] — TRÁI TIM
│   ├── prop_env.py                    #    Custom Gymnasium Env (State, Action, Step, Reset)
│   ├── physics_sim.py                 #    Variable Spread, Slippage, Latency, Partial Fill
│   └── reward_engine.py               #    Multi-component Reward + Exponential DD Penalty
│
├── models/                           # 🧠 [TEAM AI/ML]
│   ├── cross_attention.py             #    H1/H4 Context × M15 Query (Multi-TF Transformer)
│   ├── transformer_smc.py             #    Self-Attention quét FVG / Order Block
│   └── regime_detector.py             #    HMM Market Regime (Trend/Range/Volatile)
│
├── agents/                           # 🤖 [TEAM AI/ML]
│   ├── sac_policy.py                  #    SAC → Continuous [Confidence, Risk, SL, TP]
│   ├── action_gating.py               #    HOLD enforcement nếu |Confidence| < threshold
│   └── ensemble.py                    #    Multi-model voting (2/3 consensus)
│
├── training_pipeline/                # 🚀 [TEAM MLOps]
│   ├── per_buffer.py                  #    Prioritized Experience Replay
│   ├── curriculum_runner.py           #    4-stage training (Easy → Hard)
│   └── safe_retrain.py                #    Nightly retrain + validation gate + rollback
│
├── live_execution/                   # ⚡ [TEAM SYSTEM]
│   ├── mt5_bridge.py                  #    Socket gửi/nhận lệnh Live (< 5ms)
│   ├── risk_killswitch.py             #    Hard-stop nếu DD chạm -4.5%
│   ├── watchdog.py                    #    Cron watchdog — backup protection layer
│   ├── monitoring.py                  #    Live metrics logging (JSONL → Dashboard)
│   └── nightly_retrain.py             #    Orchestrate safe_retrain + deploy/reject
│
├── model_registry/                   # 📦 [TEAM MLOps]
│   ├── registry.py                    #    Version tracking + rollback capability
│   └── models/                        #    Stored model weights + config snapshots
│       └── current -> v001/           #    Symlink tới model đang live
│
├── utils/                            # 🔧 SHARED UTILITIES
│   ├── polars_bridge.py               #    Polars DataFrame ↔ torch.Tensor conversion
│   ├── alert_bot.py                   #    Telegram alert cho mọi event
│   └── shap_analysis.py               #    Feature importance visualization
│
├── tests/                            # 🧪 BẮT BUỘC TRƯỚC KHI MERGE (11 files)
│   ├── test_config_validation.py
│   ├── test_feature_builder.py
│   ├── test_normalizer.py
│   ├── test_env_step.py
│   ├── test_spread_model.py
│   ├── test_reward_hack.py
│   ├── test_overnight_penalty.py
│   ├── test_action_gating.py
│   ├── test_killswitch.py
│   ├── test_nightly_retrain.py
│   └── test_ensemble_voting.py
│
├── .dvc/                             # Data Version Control
├── .github/workflows/                # CI: Auto-run tests on PR
└── README.md
```

---

## 3. SƠ ĐỒ HỆ THỐNG TỔNG THỂ

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MARKET DATA LAYER                              │
│  MT5 API → Tick/M1/M5/M15/H1/H4                                    │
│  + Economic Calendar + Variable Spread Feed                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                   PERCEPTION LAYER (State Space)                     │
│                                                                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────┐  │
│  │ Price       │ │ Volume      │ │ SMC         │ │ Regime       │  │
│  │ Relative    │ │ RVol +      │ │ Self-Attn   │ │ HMM (Trend/  │  │
│  │ Ratios      │ │ Profile     │ │ FVG/OB      │ │ Range/Vol)   │  │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬───────┘  │
│         └───────────┬───┴───────────────┘                │          │
│                     │  Multi-TF Cross-Attention           │          │
│                     │  (H1/H4=Context, M15=Query)         │          │
│                     ▼                                     │          │
│              ┌──────────────┐                              │          │
│              │ Running      │◄─────────────────────────────┘          │
│              │ Normalizer   │ (Welford's Online Stats)               │
│              └──────┬───────┘                                        │
└─────────────────────┼────────────────────────────────────────────────┘
                      │ Normalized State Vector
┌─────────────────────▼────────────────────────────────────────────────┐
│                    DECISION ENGINE                                    │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │              Time-Series Transformer Encoder                   │  │
│  │         (Multi-Head Self-Attention + Positional Enc)            │  │
│  └───────────────────────┬────────────────────────────────────────┘  │
│                          │                                           │
│  ┌───────────────────────▼────────────────────────────────────────┐  │
│  │                    SAC Agent                                    │  │
│  │  Output: [confidence, risk_fraction, sl_mult, tp_mult]          │  │
│  └───────────────────────┬────────────────────────────────────────┘  │
│                          │                                           │
│  ┌───────────────────────▼────────────────────────────────────────┐  │
│  │              Action Gating Layer                                │  │
│  │  |confidence| < 0.3 → FORCE HOLD (chống overtrading)           │  │
│  │  confidence > +0.3  → BUY  (risk scaled by confidence)         │  │
│  │  confidence < -0.3  → SELL (risk scaled by confidence)         │  │
│  └───────────────────────┬────────────────────────────────────────┘  │
│                          │                                           │
│  ┌───────────────────────▼────────────────────────────────────────┐  │
│  │         Ensemble Voting (3 models, 2/3 consensus)               │  │
│  └───────────────────────┬────────────────────────────────────────┘  │
└──────────────────────────┼───────────────────────────────────────────┘
                           │ Action Vector
┌──────────────────────────▼───────────────────────────────────────────┐
│                     REWARD ENGINE (8 components)                      │
│                                                                      │
│  + realized_pnl              (khi đóng lệnh)                        │
│  + 0.1 × unrealized_shaping  (mark-to-market mỗi step)              │
│  - α × exp(β × dd/max_dd)   (exponential drawdown penalty)          │
│  - overnight_penalty         (giữ lệnh qua phiên)                   │
│  - spread_cost - commission  (chi phí thực tế)                       │
│  + rr_bonus if RR > 1.5     (thưởng risk/reward tốt)                │
│  - overtrading_penalty       (quá nhiều lệnh/ngày)                   │
│  - inaction_nudge            (nudge nhẹ nếu idle quá lâu)            │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│               GYMNASIUM ENVIRONMENT (Realistic Simulation)            │
│                                                                      │
│  Variable Spread │ Slippage theo Lot │ Latency 50-150ms              │
│  Partial Fill    │ Requote           │ Session Hours                  │
│  Max Positions   │ Gap Risk          │ News Calendar                  │
│  ★ Prop Firm: 5% Daily DD │ 10% Total DD │ Intraday Only             │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                   SAFETY & OPERATIONS LAYER                           │
│                                                                      │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────────────┐   │
│  │ Layer 1:       │ │ Layer 2:       │ │ Layer 3:               │   │
│  │ risk_killswitch│ │ Broker-side SL │ │ watchdog.py            │   │
│  │ DD > 4.5%      │ │ Max 1.5%/trade │ │ Cron 60s — close all   │   │
│  │ → Force Close  │ │ (broker auto)  │ │ if process dead        │   │
│  └────────────────┘ └────────────────┘ └────────────────────────┘   │
│                                                                      │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────────────┐   │
│  │ Model Registry │ │ Live Monitor   │ │ Nightly Safe Retrain   │   │
│  │ Version + Roll │ │ Equity/DD/PnL  │ │ 20% new + 80% old      │   │
│  │ back at will   │ │ → Dashboard    │ │ Validation gate         │   │
│  └────────────────┘ └────────────────┘ └────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │ Telegram Alert Bot: Trade/DD Warning/Killswitch/Retrain      │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## SPRINT 1 — DATA ENGINE & FEATURE PIPELINE (Tuần 1-2) ✅ DONE

**Mục tiêu:** Biến data thô thành ngôn ngữ "Tương đối" mà AI hiểu được. Zero-hardcoded. Reproducible.

**Team phụ trách:** TEAM DATA

> **⚡ CẬP NHẬT:** Feature builder đã chuyển sang **SMC + Volume + Price Action** (28 features).  
> Symbols: XAUUSD, US100.cash, US30.cash, ETHUSD, BTCUSD. Primary TF: **M5**.  
> Thêm rules: **0.3% max loss/trade, 3% daily cooldown, H1 inside bar → chốt hết lệnh**.

### Task list:

#### 1.1 — Project Setup & Config Foundation ✅
- [x] T1.1.1 — Init Git repo với monorepo structure đầy đủ
- [x] T1.1.2 — Viết `configs/prop_rules.yaml` (+ max_loss_per_trade 0.3%, daily_cooldown 3%, h1_inside_bar_exit)
- [x] T1.1.3 — Viết `configs/train_hyperparams.yaml`
- [x] T1.1.4 — Viết `configs/validator.py` (Pydantic schema)
- [x] T1.1.5 — Viết `tests/test_config_validation.py` (19 tests PASS)

#### 1.2 — MT5 Data Fetcher ✅
- [x] T1.2.1 — Viết `scripts/fetch_historical_data.py`:
  - Kết nối MT5 Python API (FTMO Demo)
  - Kéo **M5** data 50K bars/symbol cho **XAUUSD, US100.cash, US30.cash, ETHUSD, BTCUSD**
  - Resample → M15, H1, H4
  - Lưu compressed `.parquet` (Polars) — 40 files, 32.8 MB
- [x] T1.2.2 — Build SMC + Volume + PA features cho M5
- [x] T1.2.3 — Build H1 inside bar data cho exit rule

#### 1.3 — Feature Builder (SMC + Volume + Price Action) ✅
- [x] T1.3.1 — Viết `data_engine/feature_builder.py` — **28 features**:
  - **Price Action:** `candle_ratios()`, `pin_bar()`, `engulfing()`, `inside_bar()`
  - **Volume:** `relative_volume()`, `volume_delta()`, `climax_volume()`
  - **SMC:** `swing_structure()`, `bos_choch()`, `order_blocks()`, `fair_value_gaps()`, `liquidity_zones()`
  - **Time:** `time_encoding()` (sin/cos hour + dow)
  - **Raw:** `log_return()`
  - ~~`returns_features()` (RSI, ATR)~~ → **ĐÃ XÓA** (indicator truyền thống)
- [x] T1.3.2 — Viết `data_engine/multi_tf_builder.py`: Resample M5 → M15, H1, H4
- [x] T1.3.3 — Viết `tests/test_feature_builder.py`: **23 tests PASS**
  - Test candle ratios, pin bar, engulfing, inside bar, volume delta, climax vol
  - Test swing structure, BOS/CHoCH, order blocks, FVG, liquidity zones
  - Test full pipeline produces all 28 feature columns

#### 1.4 — Running Normalizer ✅
- [x] T1.4.1 — Viết `data_engine/normalizer.py` (Welford's + z-score)
- [x] T1.4.2 — Viết `tests/test_normalizer.py` (11 tests PASS)
- [x] T1.4.3 — Save `normalizer_state.json` cho live inference

#### 1.5 — Utilities + Safety ✅
- [x] T1.5.1 — Viết `utils/polars_bridge.py`
- [x] T1.5.2 — Viết `live_execution/killswitch.py` (Killswitch + EquityWatchdog + DailyLossGate)
- [x] T1.5.3 — Viết `tests/test_safety.py` (25 tests PASS)

#### 1.6 — MLP Prototype Training ✅
- [x] T1.6.1 — Viết `scripts/train_agent.py` (SAC MLP 256×256, 200K steps)
- [x] T1.6.2 — Backtest trên holdout 20%: **+18.7% return, WR 44.8%, PF 1.22, DD 4.26%**
- [x] T1.6.3 — Viết `scripts/backtest.py` (walk-forward evaluation, Sharpe/DD/PF report)

---

**Sprint 1 Definition of Done:** ✅ **ALL PASSED**
> ✅ Fetched M5 data 50K bars × 5 symbols → `.parquet`  
> ✅ `feature_builder.py` chuyển OHLCV → **28 SMC + Volume + PA features**  
> ✅ `normalizer.py` normalize output ≈ N(0,1) và serialize được  
> ✅ `validator.py` catch được yaml config sai (incl. risk rules)  
> ✅ MLP prototype backtest: **+18.7% return, WR 44.8%, Sharpe 4.6**  
> ✅ **161 tests PASS, 2 skipped**

---

## SPRINT 2 — GYMNASIUM ENVIRONMENT & REWARD ENGINE (Tuần 3-5)

**⚠️ SPRINT RỦI RO NHẤT — Nếu Gym sai, ra live sẽ cháy tài khoản. Buffer thêm 1 tuần.**

**Mục tiêu:** Hoàn thiện "Sàn Giao Dịch Ảo" với physics simulation thực tế.

**Team phụ trách:** TEAM QUANT/BACKEND

### Task list:

#### 2.1 — Physics Simulation
- [ ] T2.1.1 — Viết `environments/physics_sim.py` class `MarketPhysics`:
  - `variable_spread(hour, volatility, is_news)`:
    - Base spread (ví dụ 1.5 pip EURUSD)
    - News multiplier: ×5-10 trong ±5 phút quanh news
    - Session multiplier: ×2-3 lúc Asian low liquidity
  - `slippage(lot_size, liquidity)`:
    - Proportional to lot size: slippage_pips = base + k × log(lot_size)
  - `execution_delay()`:
    - Random uniform [50ms, 150ms]
    - Ảnh hưởng: fill price = price_at_signal + price_change_during_delay
  - `partial_fill(lot_size)`:
    - 5% probability cho lots > 5.0
    - Fill ratio: random uniform [0.5, 1.0]
  - `requote(volatility)`:
    - 2% probability khi volatility > 2× average
    - Requote price = new market price
- [ ] T2.1.2 — Viết `tests/test_spread_model.py`:
  - Test spread tăng lúc 3:00 UTC (Asian low liquidity)
  - Test spread spike khi is_news=True
  - Test slippage tăng theo lot_size
  - Test execution delay nằm trong [50, 150]ms

#### 2.2 — Reward Engine (8 components)
- [ ] T2.2.1 — Viết `environments/reward_engine.py` class `RewardEngine`:
  ```
  Component 1: realized_pnl         → normalize bằng /account_balance
  Component 2: unrealized_shaping   → 0.1 × delta_unrealized_pnl
  Component 3: dd_penalty           → α × exp(β × (dd / max_dd)), bắt đầu từ 2%
  Component 4: overnight_penalty    → -5.0 nếu hold qua trading_end_utc
  Component 5: spread_commission    → -(spread_cost + commission) khi mở lệnh
  Component 6: rr_bonus             → +0.3 × (rr_ratio - 1.0) nếu RR > 1.5
  Component 7: overtrading_penalty  → -0.5 × (trades_today - max_per_day) nếu quá
  Component 8: inaction_nudge       → -0.01 nếu idle > threshold steps
  ```
- [ ] T2.2.2 — Tất cả weights/thresholds đọc từ `prop_rules.yaml` (không hardcode!)
- [ ] T2.2.3 — Viết `tests/test_reward_hack.py`:
  - Test: agent mở lệnh rồi ngâm unrealized PnL → reward KHÔNG tăng vô hạn (component 2 bounded)
  - Test: agent trade 50 lệnh/ngày → penalty lớn (component 7)
  - Test: agent giữ lệnh qua 21:00 UTC → penalty -5.0 (component 4)
- [ ] T2.2.4 — Viết `tests/test_overnight_penalty.py`:
  - Test penalty = 0 lúc 15:00 UTC (trong session)
  - Test penalty = -5.0 lúc 22:00 UTC (ngoài session)

#### 2.3 — Custom Gymnasium Environment
- [ ] T2.3.1 — Viết `environments/prop_env.py` class `PropFirmTradingEnv(gymnasium.Env)`:
  - `__init__()`:
    - Load config từ yaml (qua validator)
    - Init observation_space (Box, shape = total_feature_dim)
    - Init action_space (Box, shape = 4: confidence, risk, sl_mult, tp_mult)
    - Init MarketPhysics, RewardEngine
  - `reset()`:
    - Random start point trong dataset (tránh look-ahead bias)
    - Reset equity, daily_pnl, trade_count, open_positions
    - Return initial observation
  - `step(action)`:
    - Interpret action qua ActionGating logic
    - Apply MarketPhysics (spread, slippage, delay)
    - Update positions, equity, drawdown
    - Calculate multi-component reward
    - Check termination: DD limit hit, session end, episode length
    - Return (obs, reward, terminated, truncated, info)
  - `_get_observation()`:
    - Concatenate: price_features + volume_features + time_encoding + regime_features
    - Apply RunningNormalizer
- [ ] T2.3.2 — Implement Prop Firm rules:
  - `done = True` nếu daily_dd > max_daily_drawdown (từ yaml)
  - `done = True` nếu total_dd > max_total_drawdown (từ yaml)
  - Max positions enforcement (reject new trade nếu đạt limit)
  - Max lots per trade enforcement
- [ ] T2.3.3 — Viết `tests/test_env_step.py`:
  - Test: `env.reset()` trả observation đúng shape
  - Test: `env.step()` trả (obs, reward, done, truncated, info) đúng type
  - Test: equity giảm 6% → env terminates (done=True)
  - Test: 100 steps liên tục không crash, không memory leak
  - Test: observation values nằm trong range hợp lý sau normalize

---

**Sprint 2 Definition of Done:**
> ✅ `prop_env.py` pass gymnasium compatibility check (`gymnasium.utils.env_checker.check_env`)  
> ✅ Variable spread + slippage hoạt động đúng  
> ✅ Reward engine 8 components tính đúng, weights từ yaml  
> ✅ Prop Firm DD rules enforce termination  
> ✅ Tất cả tests PASS (4 test files)

---

## SPRINT 3 — NEURAL ARCHITECTURE & ACTION GATING (Tuần 6-7) 🔴 ĐANG LÀM

**★ SPRINT QUAN TRỌNG NHẤT — Đây là "trí tuệ" của bot. MLP prototype chỉ là baseline.**

**Mục tiêu:** Dựng Transformer đọc Multi-Timeframe + hệ thống lọc hành động → AI **biết suy nghĩ** trước khi trade.

**Team phụ trách:** TEAM AI/ML

**Tại sao cần:**
- MLP flatten 64 bars → coi mọi bar quan trọng bằng nhau. **Transformer Self-Attention** tự học bar nào quan trọng
- MLP chỉ nhìn M5. **Cross-Attention MTF** nhìn H4/H1 context trước rồi mới quyết định trên M5
- MLP không biết thị trường trend/range. **Regime Detector** thay đổi chiến thuật theo trạng thái

### Task list:

#### 3.1 — Transformer SMC Module
- [ ] T3.1.1 — Viết `models/transformer_smc.py` class `TransformerSMC(nn.Module)`:
  - Multi-Head Self-Attention (4-8 heads)
  - Positional Encoding (sinusoidal)
  - Input: sequence of candle features (lookback window)
  - Output: latent representation capturing FVG / Order Block patterns
  - Params: embed_dim, n_heads, n_layers, dropout — tất cả từ yaml
- [ ] T3.1.2 — Unit test: random input → output shape đúng, gradients flow

#### 3.2 — Cross-Attention Multi-Timeframe
- [ ] T3.2.1 — Viết `models/cross_attention.py` class `CrossAttentionMTF(nn.Module)`:
  - **Context Encoder**: nhận H1 (24 bars = 1 day) + H4 (30 bars = 5 days) features → context vectors
  - **Query Encoder**: nhận M5 features (64 bars) → query vectors
  - **Cross-Attention**: M5 queries attend to H1/H4 context
  - Giải quyết RAM: H4 sequence ngắn (30 bars), M5 dài hơn (64 bars)
- [ ] T3.2.2 — Unit test:
  - Input M5 (batch, 64, feat_dim) + H4 (batch, 30, feat_dim) → output shape đúng
  - Memory usage < 2GB cho batch_size=64

#### 3.3 — Market Regime Detector
- [ ] T3.3.1 — Viết `models/regime_detector.py` class `RegimeDetector`:
  - Feature-based input: volatility percentile, trend strength (from swing_trend), range width
  - 4 regimes: trending-up, trending-down, ranging, volatile
  - `fit(historical_data)`: train offline trên SMC features
  - `predict(current_features)` → regime_id + regime_probabilities (4-dim vector)
  - Output regime probabilities append vào state vector
  - **Không dùng ATR/ADX** (indicator truyền thống) → dùng SMC-derived features
- [ ] T3.3.2 — Unit test: fit trên synthetic data → predict regime ≠ random

#### 3.4 — SAC Policy Network
- [ ] T3.4.1 — Viết `agents/sac_policy.py`:
  - Actor: state → (mean, log_std) → sample action [confidence, risk_frac, sl_mult, tp_mult]
  - Critic: (state, action) → Q-value (twin critics cho SAC)
  - Temperature (alpha) auto-tuning
  - Integrate TransformerSMC + CrossAttentionMTF as feature extractor
- [ ] T3.4.2 — Unit test: forward pass, backward pass, action trong bounds

#### 3.5 — Action Gating
- [ ] T3.5.1 — Viết `agents/action_gating.py` class `ActionGating`:
  - threshold đọc từ yaml (`confidence_threshold: 0.3`)
  - |confidence| < threshold → return HOLD
  - confidence > threshold → BUY, risk scaled by (|c| - threshold) / (1 - threshold)
  - confidence < -threshold → SELL, same scaling
- [ ] T3.5.2 — Viết `tests/test_action_gating.py`:
  - Test: confidence=0.1 → HOLD
  - Test: confidence=0.5 → BUY với risk > 0
  - Test: confidence=-0.8 → SELL với risk scaled đúng
  - Test: confidence=0.29 → HOLD (edge case)
  - Test: confidence=0.31 → BUY (edge case)

---

**Sprint 3 Definition of Done:**
> ✅ Transformer chạy forward/backward pass không lỗi  
> ✅ Cross-Attention Multi-TF hoạt động (M5 × H1/H4), RAM < 2GB  
> ✅ Regime Detector phân loại được market states  
> ✅ SAC policy output action đúng range  
> ✅ Action Gating enforce HOLD đúng threshold  
> ✅ **Backtest Transformer > MLP baseline** (WR/Sharpe/PF cải thiện)  
> ✅ Tất cả components kết nối: Data → SMC Features → Transformer → Cross-Attention → Regime → SAC → Action

---

## SPRINT 4 — TRAINING PIPELINE & CURRICULUM LEARNING (Tuần 8-9)

**Mục tiêu:** Đưa AI lên Cloud GPU, bắt đầu học từ Mẫu giáo đến Đại học.

**Team phụ trách:** TEAM MLOps

### Task list:

#### 4.1 — Prioritized Experience Replay
- [ ] T4.1.1 — Viết `training_pipeline/per_buffer.py` class `PrioritizedReplayBuffer`:
  - SumTree data structure cho O(log n) sampling
  - `add(experience, td_error)`: lưu với priority = |td_error|^alpha
  - `sample(batch_size)`: proportional sampling + importance sampling weights
  - `update_priorities(indices, new_td_errors)`: update sau mỗi training step
  - Beta annealing: beta tăng dần từ 0.4 → 1.0 qua training
  - `buffer_size` đọc từ yaml (default 1M experiences)
- [ ] T4.1.2 — Unit test: add 10K experiences → sample → priorities ảnh hưởng distribution

#### 4.2 — Curriculum Learning Runner
- [ ] T4.2.1 — Viết `training_pipeline/curriculum_runner.py`:
  - **Stage 1 — "Mẫu giáo"** (0-50K steps):
    - Fixed spread, no slippage, no commission
    - Chỉ dùng trending market data
    - DD limit relaxed: 20%
    - Mục tiêu: Học BUY/SELL cơ bản
  - **Stage 2 — "Tiểu học"** (50K-200K steps):
    - Variable spread, nhỏ slippage
    - Mix trending + ranging data
    - DD limit: 15%
    - Mục tiêu: Học cost-aware trading
  - **Stage 3 — "Trung học"** (200K-500K steps):
    - Full realistic spread, slippage, commission
    - All market regimes
    - DD limit: 10% (Prop Firm actual)
    - Mục tiêu: Học survival
  - **Stage 4 — "Đại học"** (500K+ steps):
    - Full realism + news events, gaps, requotes
    - DD limit: 5% daily + 10% total (Prop Firm exact)
    - Random execution delays [50, 200]ms
    - Mục tiêu: Battle-tested
  - Auto-promote: chuyển stage khi reward mean > threshold liên tục 1000 episodes
- [ ] T4.2.2 — Stage configs trong `train_hyperparams.yaml` (không hardcode!)

#### 4.3 — W&B Integration
- [ ] T4.3.1 — Integrate W&B logging:
  - Log mỗi episode: reward, PnL, drawdown, trade_count, win_rate
  - Log mỗi stage transition: stage_id, episodes_to_graduate
  - Log model checkpoints
  - Custom dashboard: reward curve, DD distribution, action distribution
- [ ] T4.3.2 — Setup W&B project, invite team members

#### 4.4 — Cloud GPU Setup
- [ ] T4.4.1 — Setup training server (RunPod/AWS):
  - GPU: A100 40GB (hoặc RTX 4090) cho training
  - Docker image với PyTorch 2.x + Ray RLlib + dependencies
  - Mount data volume (parquet files)
- [ ] T4.4.2 — Test: chạy 1000 episodes Stage 1 → verify reward tăng → verify W&B graphs

#### 4.5 — Safe Nightly Retrain Module
- [ ] T4.5.1 — Viết `training_pipeline/safe_retrain.py`:
  - Fetch new data → add vào PER Buffer (không replace)
  - Sample batch: 20% new data + 80% historical
  - Fine-tune: lr=1e-5 (10× thấp hơn), max 5 epochs, gradient clip 0.5
  - **Validation Gate**: backtest new model trên 30 ngày gần nhất
    - Deploy chỉ khi: `new_sharpe >= old_sharpe × 0.9 AND new_max_dd <= old_max_dd × 1.1`
    - Reject + alert nếu model mới tệ hơn
  - Backup old model trước khi deploy mới
- [ ] T4.5.2 — Viết `tests/test_nightly_retrain.py`:
  - Test: model mới tốt hơn → deploy
  - Test: model mới tệ hơn → reject, giữ cũ
  - Test: có backup file cho model cũ

---

**Sprint 4 Definition of Done:**
> ✅ PER buffer sampling ưu tiên high-TD-error experiences  
> ✅ Curriculum 4 stages auto-promote khi agent đạt threshold  
> ✅ W&B dashboard hiển thị training metrics real-time  
> ✅ Agent hoàn thành Stage 1+2 với reward positive  
> ✅ Safe retrain module có validation gate hoạt động đúng

---

## SPRINT 5 — ENSEMBLE & OUT-OF-SAMPLE VALIDATION (Tuần 10-11)

**Mục tiêu:** Kiểm định "chống đạn" trước khi cắm vào broker.

**Team phụ trách:** TEAM AI + TEAM BACKEND

### Task list:

#### 5.1 — Complete Curriculum Training
- [ ] T5.1.1 — Train agent qua Stage 3 và Stage 4 (trên cloud GPU)
- [ ] T5.1.2 — Monitor W&B: reward stable, DD within limits, win_rate acceptable
- [ ] T5.1.3 — Nếu không converge: adjust hyperparams, retrain

#### 5.2 — Ensemble System
- [ ] T5.2.1 — Viết `agents/ensemble.py` class `EnsembleTradingAgent`:
  - Train 3 models với random seeds khác nhau (cùng architecture, cùng data)
  - `decide(state)`:
    - Lấy action từ 3 models
    - Count votes: ≥2/3 cùng BUY → BUY, ≥2/3 cùng SELL → SELL, else → HOLD
    - Average parameters (lot, SL, TP) từ các models đồng thuận
  - Consensus threshold đọc từ yaml
- [ ] T5.2.2 — Viết `tests/test_ensemble_voting.py`:
  - Test: 3 BUY → BUY
  - Test: 2 BUY + 1 SELL → BUY
  - Test: 1 BUY + 1 SELL + 1 HOLD → HOLD
  - Test: 3 HOLD → HOLD

#### 5.3 — Out-of-Sample Backtest
- [ ] T5.3.1 — Backtest trên data **chưa bao giờ dùng trong training** (VD: 6 tháng gần nhất)
- [ ] T5.3.2 — Xuất Quant Tearsheet:
  - Sharpe Ratio, Sortino Ratio
  - Max Drawdown, Calmar Ratio
  - Win Rate, Avg Win/Loss, Profit Factor
  - Monthly returns table, equity curve
  - Trade duration distribution
  - Drawdown underwater plot
- [ ] T5.3.3 — So sánh: Single model vs Ensemble → Ensemble phải tốt hơn hoặc bằng
- [ ] T5.3.4 — **Gate**: chỉ qua Sprint 6 nếu:
  - Sharpe > 1.0
  - Max DD < 8% (buffer trước ngưỡng 10%)
  - Win Rate > 40% (với avg_win > avg_loss)

#### 5.4 — Model Registry Setup
- [ ] T5.4.1 — Viết `model_registry/registry.py`:
  - `register(model, metrics, config)` → save weights + metrics snapshot
  - `get_current()` → return model đang dùng live
  - `rollback(version)` → switch về model cũ
  - `list_versions()` → show all versions + metrics
- [ ] T5.4.2 — Register 3 ensemble models vào registry

#### 5.5 — SHAP Analysis
- [ ] T5.5.1 — Viết `utils/shap_analysis.py`:
  - Chạy SHAP trên trained model
  - Top 10 most important features → verify không dựa vào noise
  - Attention weight visualization → model nhìn timeframe nào nhiều nhất
- [ ] T5.5.2 — Export report dạng HTML cho team review

---

**Sprint 5 Definition of Done:**
> ✅ 3 models trained với random seeds khác nhau  
> ✅ Ensemble voting logic hoạt động đúng  
> ✅ Out-of-sample backtest: Sharpe > 1.0, Max DD < 8%  
> ✅ SHAP analysis confirm model học features có ý nghĩa  
> ✅ Model registry lưu 3 models + backtest results

---

## SPRINT 6 — PAPER TRADING & SAFETY VALIDATION (Tuần 12)

**★ SPRINT MỚI — Không có trong plan gốc. Bắt buộc trước khi live.**

**Mục tiêu:** Chạy agent trên MT5 Demo Account (tiền ảo, data thật) — validate end-to-end.

**Team phụ trách:** TEAM SYSTEM + TEAM AI

### Task list:

#### 6.1 — MT5 Bridge
- [ ] T6.1.1 — Viết `live_execution/mt5_bridge.py`:
  - Connect MT5 Demo Account
  - `send_order(symbol, direction, lots, sl, tp)` → market order
  - `close_position(ticket)` → close by ticket
  - `get_positions()`, `get_equity()`, `get_account_info()`
  - Measure latency: signal → fill (target < 5ms code-side)
- [ ] T6.1.2 — Test: mở/đóng 1 lệnh EURUSD trên demo → verify

#### 6.2 — Risk Killswitch (Triple-Layer)
- [ ] T6.2.1 — Viết `live_execution/risk_killswitch.py`:
  - Monitor equity mỗi 500ms
  - Daily DD > 4.5% → force-close ALL positions → ngắt API → alert
  - Chạy trong subprocess riêng biệt
- [ ] T6.2.2 — Viết `live_execution/watchdog.py`:
  - Cron mỗi 60 giây
  - Check: killswitch process alive? live_execution process alive?
  - Nếu chết → close ALL positions → Telegram alert
- [ ] T6.2.3 — Config MT5: mọi lệnh có broker-side SL (max 1.5% equity)
- [ ] T6.2.4 — Viết `tests/test_killswitch.py`:
  - Test: equity giảm 4.6% → killswitch triggers
  - Test: equity giảm 3% → killswitch KHÔNG trigger
  - Test: watchdog detect process dead → close positions

#### 6.3 — Live Monitoring
- [ ] T6.3.1 — Viết `live_execution/monitoring.py`:
  - Log mỗi trade: timestamp, symbol, direction, confidence, lots, entry, sl, tp, pnl, latency, equity, dd, model_version
  - Output: `logs/live_trades.jsonl`
  - DD warning alert khi > 3%
- [ ] T6.3.2 — Dashboard (Streamlit local hoặc Grafana):
  - Equity curve real-time
  - Daily PnL bar chart
  - Current DD gauge
  - Trade log table

#### 6.4 — Paper Trading Execution
- [ ] T6.4.1 — Chạy full system trên MT5 Demo Account:
  - Ensemble 3 models → voting → action gating → mt5_bridge → execute
  - killswitch + watchdog chạy song song
  - monitoring logging
- [ ] T6.4.2 — Chạy liên tục **tối thiểu 5 ngày trading**
- [ ] T6.4.3 — Thu thập metrics:
  - Latency thực tế: signal → fill
  - Slippage thực tế vs simulated
  - Spread thực tế vs model
  - Memory usage sau 8h chạy liên tục
  - Số lệnh/ngày, win rate, PnL
- [ ] T6.4.4 — **Gate**: chỉ qua Sprint 7 nếu:
  - Không crash/memory leak trong 5 ngày
  - Sharpe > 1.0 trên demo
  - Max DD < 4% trên demo
  - Latency < 100ms consistently
  - Killswitch test pass (tắt thử → watchdog catch)

---

**Sprint 6 Definition of Done:**
> ✅ Agent chạy 5 ngày liên tục trên demo không crash  
> ✅ Triple-layer protection hoạt động đúng  
> ✅ Monitoring dashboard hiển thị metrics  
> ✅ Paper trading Sharpe > 1.0, Max DD < 4%

---

## SPRINT 7 — LIVE DEPLOYMENT & SELF-EVOLUTION (Tuần 13-14)

**Mục tiêu:** Nối AI với money thật + cơ chế tự tiến hóa ban đêm.

**Team phụ trách:** TEAM SYSTEM + ALL TEAMS

### Task list:

#### 7.1 — Live Deployment
- [ ] T7.1.1 — Kết nối MT5 **Live Account** (Prop Firm challenge account)
- [ ] T7.1.2 — Deploy ensemble models từ registry (`current` symlink)
- [ ] T7.1.3 — Start killswitch + watchdog (PHẢI chạy trước khi bật trading)
- [ ] T7.1.4 — Start monitoring + Telegram alerts
- [ ] T7.1.5 — **Ngày 1-2**: Trade lot size tối thiểu (0.01) — verify end-to-end
- [ ] T7.1.6 — **Ngày 3+**: Scale lot size dần theo config

#### 7.2 — Nightly Self-Evolution Pipeline
- [ ] T7.2.1 — Viết `live_execution/nightly_retrain.py` orchestrator:
  ```
  Schedule: 23:30 UTC daily (sau khi đóng phiên Mỹ)
  
  1. Kéo data hôm nay từ MT5
  2. Convert → features (feature_builder)
  3. Gọi safe_retrain.py:
     - Add vào PER Buffer
     - Fine-tune: 20% new + 80% old
     - Validation gate: backtest 30 ngày
     - Deploy hoặc reject
  4. Update model_registry
  5. Alert kết quả qua Telegram
  6. Sẵn sàng cho phiên Á sáng hôm sau (01:00 UTC)
  ```
- [ ] T7.2.2 — Viết cron job / Windows Task Scheduler cho nightly pipeline
- [ ] T7.2.3 — Test: chạy thủ công 1 lần → verify new model registered + oldmodel backed up

#### 7.3 — Operational Procedures
- [ ] T7.3.1 — Viết runbook: "Cách rollback model khi thua"
  - Step 1: Check model_registry → list versions
  - Step 2: rollback(target_version)
  - Step 3: Verify via monitoring dashboard
- [ ] T7.3.2 — Viết runbook: "Cách xử lý khi killswitch kích hoạt"
  - Step 1: Check Telegram alert
  - Step 2: Review trade logs
  - Step 3: Decide: resume trading (reset DD) hoặc stop for the day
- [ ] T7.3.3 — Viết runbook: "Weekly model assessment"
  - So sánh PnL thực tế vs backtest expected
  - Review SHAP analysis: features importance thay đổi?
  - Decision: continue / retrain from scratch / adjust config

#### 7.4 — Prop Firm Challenge Execution
- [ ] T7.4.1 — Bắt đầu Prop Firm challenge Phase 1 (thường yêu cầu +8-10% profit)
- [ ] T7.4.2 — Monitor daily: DD headroom, profit target progress
- [ ] T7.4.3 — Weekly review meeting: performance vs expectations
- [ ] T7.4.4 — Pass Phase 1 → Phase 2 → Funded Account

---

**Sprint 7 Definition of Done:**
> ✅ Agent trade live trên Prop Firm challenge account  
> ✅ Nightly self-evolution pipeline chạy tự động mỗi đêm  
> ✅ Rollback tested và hoạt động  
> ✅ Runbooks documented cho mọi tình huống  
> ✅ Prop Firm Phase 1 profit target progress

---

## TỔNG KẾT TASK COUNT

| Sprint | Tuần | Tasks | Tests | Status |
|--------|------|-------|-------|--------|
| Sprint 1 — Data Engine + SMC | 1-2 | 20 tasks | 3 test files | ✅ DONE |
| Sprint 2 — Gym Environment | 3-5 | 12 tasks | 4 test files | ✅ Cơ bản |
| **Sprint 3 — Neural Architecture** | **6-7** | **12 tasks** | **1 test file** | **🔴 TIẾP THEO** |
| Sprint 4 — Training Pipeline | 8-9 | 11 tasks | 1 test file | ⬜ |
| Sprint 5 — Ensemble & Validation | 10-11 | 12 tasks | 1 test file | ⬜ |
| Sprint 6 — Paper Trading | 12 | 13 tasks | 1 test file | ⬜ |
| Sprint 7 — Live & Self-Evolution | 13-14 | 13 tasks | — | ⬜ |
| **TỔNG** | **14 tuần** | **93 tasks** | **11 test files** | |

---

## TRADING RULES (Bổ sung)

| Rule | Giá trị | Config |
|------|---------|--------|
| Max loss per trade | 0.3% balance | `max_loss_per_trade_pct: 0.003` |
| Daily loss cooldown | 3% → stop trading | `daily_loss_cooldown_pct: 0.03` |
| H1 Inside Bar exit | Chốt hết lệnh | `h1_inside_bar_exit: true` |
| Killswitch DD | 4.5% → force close | `killswitch_dd_threshold: 0.045` |
| Confidence threshold | \|c\| < 0.3 → HOLD | `confidence_threshold: 0.3` |

---

> **Mỗi task có prefix T[Sprint].[Section].[Number]** (ví dụ T2.1.1) để dễ reference trong standup/review meetings.  
> **Nguyên tắc:** Viết xong Unit Test mới được merge code sang module khác.  
> **Gate giữa các Sprint:** Mỗi Sprint có "Definition of Done" rõ ràng — PHẢI pass hết mới qua Sprint tiếp.  
> **Features:** TUYỆT ĐỐI không dùng indicator truyền thống (RSI, ATR, Bollinger, MA). Chỉ SMC + Volume + Price Action.
