# 📝 NHẬT KÝ DỰ ÁN — RABIT-PROPFIRM DRL SYSTEM

> Mỗi thay đổi PHẢI được ghi vào đây. Người sau đọc file này sẽ biết dự án đang ở đâu, đã làm gì, chỉnh sửa gì.

---

## 22/03/2026

### [HOTFIX] Alpha Reset Bug — Mandatory Final Checkpoint Save — 22/03/2026 22:10
- **Branch:** `phase3.1/dual-entry-system` → **merge vào `main`**
- **Bug:** `log_alpha` reset về 1.0 khi chuyển Stage (RunPod đang chạy code cũ chưa merge)
- **Fix:** Thêm MANDATORY final checkpoint save sau khi training loop kết thúc
  - Đảm bảo `log_alpha` + optimizer states LUÔN LUÔN được lưu ở bước cuối cùng
  - Log xác nhận: `FINAL checkpoint -> best_TestStage.pt (log_alpha=-2.0300, alpha=0.1313)`
- **Test:** 100 steps, EXIT CODE 0 ✅

### [FIX] Phase 3.1 — Đại Phẫu: Dual Entry System + Root Cause Surgery — 22/03/2026 17:35
- **Branch:** `phase3.1/dual-entry-system`
- **Nguyên nhân:** Phase 3 (1M steps) thất bại do **Mean Collapse** — bot không vào lệnh.
  - Root cause: `log_alpha` reset về 1.0 mỗi khi chuyển Stage (không save trong checkpoint)
  - Stage 1 train trên Dummy Data → policy saturation (confidence = +1.0 100% thời gian)
  - Stage 2 overcorrect → policy mean collapse xuống 0.065 (dưới threshold 0.3)

**6 file đã sửa:**

#### NV1: Vá Mất trí nhớ + Cấm Dummy Data (`train_curriculum.py`)
- `log_alpha` + 3 optimizer states (`actor_optim`, `critic_optim`, `alpha_optim`) → save trong TẤT CẢ checkpoint
- Warm-start khôi phục đầy đủ từ checkpoint trước
- Default init `log_alpha = -2.0` (alpha=0.135) thay vì 0 (alpha=1.0)
- **Xóa sạch** dummy data loop → thay bằng **real MultiTFTradingEnv rollouts** từ `data/`
- `target_entropy`: -4 → **-5** (cho 5-dim action)
- `action_dim`: 4 → **5**

#### NV2: Dual Entry System — Sếp Vũ chỉ đạo: M1 = Sniper, M5 = Bộ binh
- **`sac_policy.py`**: Action space **4→5 dim**: `[confidence, entry_type, risk_frac, sl_mult, tp_mult]`
  - `entry_type < 0` = M5 Normal, `entry_type > 0` = M1 Sniper
  - Constants: `M5_NORMAL_THRESHOLD = 0.50`, `M1_SNIPER_THRESHOLD = 0.85`
- **`prop_env.py`**: Action space Box 5-dim, dual threshold gating
  - `|conf| ≥ 0.85 AND entry_type > 0` → M1 Sniper Entry
  - `|conf| ≥ 0.50 AND entry_type ≤ 0` → M5 Normal Entry
  - Else → Standby
- **`prop_rules.yaml`**: Thêm `m5_normal_threshold: 0.50`, `m1_sniper_threshold: 0.85`

#### NV3: Kỷ luật Thép (`reward_engine.py`)
- Reward 8→**11 components**:
  - (C9) **FOMO Oracle**: -5 penalty khi standby mà giá chạy ≥0.5% trong 50 bars (TRAINING ONLY)
  - (C10) **Sniper Miss**: 3x loss penalty cho Sniper M1 thua
  - (C11) **Sniper Win**: 5x bonus cho Sniper M1 thắng
- `inaction_nudge`: -0.01 → **-0.5** (ép bot vào lệnh)
- `confidence_threshold`: 0.3 → **0.15** (fallback)

#### NV4: Script Phẫu thuật Não (NEW `scripts/extract_encoder.py`)
- Đọc `best_Stage2_Precision.pt` → giữ 97 encoder layers, re-init 43 policy layers
- Output: `pretrained_eyes_only.pt` (12.47 MB)
- Cũng có `scripts/evaluate_model.py` (NEW) cho đánh giá checkpoint

#### Verification:
- **100-step test**: EXIT CODE 0 ✅
- Params: 1,440,398 (tăng 514 so với 4-dim — đúng)
- Alpha: 0.1353 → 0.1314 (ổn định, không explode)
- 5 env × 4 TF loaded real data, checkpoint 36.2MB full state

---

## 20/03/2026

### [COGNITIVE] Phase 3: Data Pipeline & Curriculum Training — 20/03/2026 11:52
- **Sửa `scripts/fetch_historical_data.py`** v3.0:
  - Fetch trực tiếp 4 TF từ MT5 (không resample): M1 × 250K, M5 × 50K, M15 × 17K, H1 × 4200
  - M1 fetch batched (50K/batch) → tránh MT5 timeout
  - Build 28 raw features (build_features) + 22 knowledge (KnowledgeExtractor) = **50-dim per bar**
  - Generate `normalizer_v3.json` (per-TF Welford stats)
  - Output: `{symbol}_{TF}_50dim.npy` files
- **Viết `scripts/train_curriculum.py`** — 3-stage curriculum:

| Stage | Name | Steps | TFs | Freeze | LR | EpisodicMemory |
|-------|------|-------|-----|--------|----|----------------|
| 1 | Context | 200K | M15+H1 | M1,M5,Entry | 3e-4 | OFF |
| 2 | Precision | 300K | M5+M15+H1 | M1,Structure | 1e-4 | OFF |
| 3 | Full Fusion | 500K | ALL | None | 5e-5 | ON |

  - Progressive Freezing: Anti-Catastrophic-Forgetting
  - Warm-start chain: Stage N loads best weights từ Stage N-1
  - `--test` flag: 100 steps quick validation
  - `--resume-stage N`: resume từ bất kỳ stage nào
- **Tests:** Syntax OK (2/2), `--test` PASS (100 steps, 2 checkpoints, best_TestStage.pt 12.5MB)

- **Sửa `agents/sac_policy.py`**: `HierarchicalFeatureExtractor` (516-dim global state)
  - 4-TF inputs: M1/M5/M15/H1 × 50-dim (28 raw + 22 knowledge)
  - Actor: 1,439,884 params → action (B, 4) + log_prob
  - Critic: 1,638,534 params → twin Q-values
  - `apply_episodic_memory_bonus()`: NGOÀI gradient graph, ±30% confidence modifier
- **Sửa `environments/prop_env.py`**: `MultiTFTradingEnv` (Dict obs space)
  - Obs: {m1: (128,50), m5: (64,50), m15: (48,50), h1: (24,50)}
  - M5 drives stepping, M1/M15/H1 aligned by timestamp ratio
  - Session close tất cả lệnh khi hour ≥ 22:00
  - Zero-padding khi window ngắn hơn lookback
- **Sửa `live_execution/data_feed.py`**: 4-TF polling
  - M1 × 128 bars, M5 × 96, M15 × 48, H1 × 24
  - Poll interval: 3s (nhanh hơn v1: 5s)
  - Xóa hoàn toàn H4
- **Tests:** Syntax OK (3/3), Functional PASS (3/3), Env stepped OK (4 trades/10 steps)

### [PIVOT] Cognitive Architecture — 3 Trụ Cột Nhận Thức — 20/03/2026 10:10
- **Branch:** `feature/cognitive-architecture` (checkout từ `main`)
- **Quyết định:** Pivot từ Swing (M5/H1/H4, WR=34.9%) sang Intraday/Scalping (M1/M5/M15/H1, target WR≥55%)
- **Thiết kế:** `docs/COGNITIVE_ARCHITECTURE.md` — Top-Down Analysis + Curriculum Learning
- **3 Module mới:**
  - `features/knowledge_extractor.py`: **22 biến ngữ nghĩa** (SMC 7 + PA 8 + Vol 5 + Ctx 2)
    - SMC: `distance_to_ob`, `is_in_fvg`, `trend_state` (BOS/CHoCH), `liquidity_grab`, `swing_distance`
    - PA: `is_pinbar`, `is_doji`, `is_inside_bar`, `is_engulfing_bull/bear`, `is_hammer`, `is_shooting_star`, `candle_strength`
    - Vol: `vol_anomaly`, `vol_exhaustion`, `vol_climax`, `vol_trend`, `delta_approx`
    - NumPy vectorized, <1ms/bar ✅
  - `agents/episodic_memory.py`: **k-NN Memory Bank** (500 entries, cosine similarity)
    - Bonus ∈ [-0.3, +0.3] — auxiliary signal, không override Agent
    - Cold start protection: bonus=0 khi <50 entries
    - JSON save/load persistence ✅
  - `models/cross_attention.py`: **HierarchicalCrossAttentionMTF** (1,226,496 params)
    - Entry cluster: M1(Q=128 bars) × M5(KV=64 bars) → entry_latent (128-dim)
    - Structure cluster: M15(Q=48 bars) × H1(KV=24 bars) → structure_latent (128-dim)
    - M1 encoder: TransformerSMC 2-layer (deep, lọc noise)
    - M5/M15/H1 encoders: ContextEncoder 1-layer (light)
    - Gradient flow all 4 inputs ✅
- Feature vector: **50-dim** (28 raw + 22 knowledge) per timeframe
- Backward compat: Old `CrossAttentionMTF` (3-TF) untouched
- **Tests:** Syntax OK (3/3), Functional PASS (3/3)
- **Trạng thái:** Phase 1 DONE. Phase 2 (Policy+DataPipeline) pending.

### [PIVOT] Intraday Architecture Plan — 20/03/2026 09:30
- **Branch:** `feature/intraday-pivot` (docs only, archived)
- `docs/PIVOT_INTRADAY_PLAN.md`: Thiết kế 4-TF intraday (M1/M5/M15/H1)
- Quyết định: Hierarchical Fusion thay Flat 4-Way Attention
- VRAM estimate: ~8MB (L40 48GB dư sức)

### [FIX] Xóa Hardcoded Fixed Lot XAUUSD — 20/03/2026 08:00
- `docs/SPRINT_6_MT5_BRIDGE_DESIGN.md`: Bỏ `[!IMPORTANT]` fixed_lot 0.01 → `[!NOTE]` Dynamic Sizing
- `live_execution/order_calculator.py`: Xóa `symbol_overrides`, `fixed_lot` check
- **Tất cả symbols dùng 100% Dynamic Lot Sizing (ATR-based)**

### [TEST] Ensemble Holdout Backtest — 5/5 PASS — 20/03/2026 08:34
- `scripts/backtest_ensemble.py`: Holdout backtest cho 3 Specialist models
- **Kết quả:**

| Symbol | WR | Sharpe | MaxDD | Return | Trades | PF |
|--------|-----|--------|-------|--------|--------|-----|
| BTCUSD | 30.8% | 2.07 | 6.42% | +3.32% | 172 | 1.15 |
| ETHUSD | 30.6% | 4.38 | 4.98% | +13.56% | 232 | 1.33 |
| US100 | 39.6% | 9.33 | 2.13% | +7.48% | 106 | 2.03 |
| US30 | 45.3% | 7.99 | 1.46% | +2.29% | 53 | 1.73 |
| XAUUSD | 38.2% | 8.30 | 5.17% | +16.64% | 204 | 1.76 |

- **Aggregate:** Sharpe=6.41, Worst DD=6.42% (<8% FTMO), PF=1.60, avg RR=1.87
- FTMO Gate: **Sharpe ✅ PASS, MaxDD ✅ PASS**, WR ⚠️ 34.9% (trend-following profile)
- Report: `reports/ensemble_backtest_report.txt`

### [SPRINT 6] MT5 Live Execution Utility Modules — 20/03/2026 01:00
- `docs/SPRINT_6_MT5_BRIDGE_DESIGN.md`: Kiến trúc cầu nối MT5
- `live_execution/data_feed.py`: DataFeedManager (M5/H1/H4 polling, bar alignment)
- `live_execution/connection_guard.py`: ConnectionGuard (3-tier auto-reconnect)
- `live_execution/order_calculator.py`: OrderCalculator + SlippageHandler (dynamic lot, requote handling)

### [SPRINT 5] Train 3 Specialist Models — 19/03/2026 23:00
- `scripts/train_specialists.py`: Parallel training (TrendAgent, RangeAgent, VolatilityAgent)
- Models saved: `best_trendagent.pt` (930K), `best_rangeagent.pt` (520K), `best_volatilityagent.pt` (740K)
- Trained on L40 with specialized reward shaping per regime

---

### [SPRINT 4.5] Safe Nightly Retrain -- 19/03/2026 20:15
- `training_pipeline/safe_retrain.py`: MixedSampler (20/80) + SafeNightlyRetrainer
  - Fine-tune: lr=1e-5, grad_clip=0.5, max 5 epochs
  - Validation Gate: new_sharpe >= old*0.9 AND new_dd <= old*1.1
  - Model backup before overwrite, rollback support
- `tests/test_nightly_retrain.py`: **16/16 PASSED**
- **SPRINT 4 COMPLETE!**

### [SPRINT 4.4] Cloud GPU Setup (RunPod L40) -- 19/03/2026 16:55
- `requirements.txt` + `scripts/setup_runpod.sh`: 1-click deploy
- `RabitPropfim_RunPod.zip`: 41MB archive (code + data + best_v2.pt)

### [SPRINT 4.3] W&B Integration + Training Script v2 -- 19/03/2026 06:25
- `scripts/train_v2.py`: PER + Curriculum + W&B monitoring
  - W&B logs: reward, realized_pnl, max_drawdown, win_rate, stage, actor/critic loss, PER beta
  - Checkpoint auto-upload: on promotion + best reward -> W&B Artifacts
  - IS-weighted critic loss (PER), curriculum env overrides (spread/slippage/commission)
  - --test flag: 1000-step dry run (offline W&B)
- Test run OK: All metrics logged, stage=Kindergarten, beta=0.4

### [SPRINT 4.1+4.2] PER Buffer + CurriculumRunner -- 19/03/2026 06:15
- `training_pipeline/per_buffer.py`: SumTree O(log n) + Multi-TF PER (M5+H1+H4)
  - IS weights with beta annealing (0.4 -> 1.0), stratified sampling, GPU-ready
- `training_pipeline/curriculum_runner.py`: 4-stage auto-promote
  - Kindergarten (no cost) -> Elementary (variable spread) -> High School (real costs) -> University (news/gap)
  - Auto-promote: rolling 1000-episode window, configurable thresholds
- `tests/test_sprint4.py`: **30/30 PASSED** (SumTree=9, PER=9, Curriculum=12)

### [SPRINT 3.7] Backtest Transformer vs MLP -- 19/03/2026 01:55
- `backtest_transformer.py`: holdout 20% trên 5 symbols
- Transformer: WR 47.9%, PF 1.10, Sharpe 0.59, MaxDD 2.72%
- MLP baseline: WR 44.8%, PF 1.22, Sharpe 4.61, MaxDD 4.26%
- Transformer thắng WR (+3.1%) và MaxDD (-1.54%), MLP thắng PF và Sharpe
- Transformer trade bảo thủ hơn: 822 trades vs 3,002 trades (MLP)

### [SPRINT 3.6] Retrain Transformer 200K steps trên CUDA -- 19/03/2026 01:44
- `train_transformer.py`: Multi-TF env (M5+H1+H4) + SACTransformerActor/Critic
- GPU fix: device=CUDA (GTX 1660 SUPER), RuntimeError guard nếu CPU
- Best checkpoint: step 140K, reward 4.34, WR 61.1%
- Total params: 3,504,408 (Actor 1.06M + Critic 2x1.22M)
- Training time: ~2.5h on GTX 1660 SUPER @ 20-22 sps

### [SPRINT 3.4+3.5] SAC Policy + ActionGating ✅ — 18/03/2026 22:45
- NEW `agents/sac_policy.py`: Transformer backbone thay MLP
  - `TransformerFeatureExtractor`: TransformerSMC + CrossAttentionMTF + RegimeDetector → **388-dim global_state**
  - `SACTransformerActor`: Squashed Gaussian policy (mean + log_std → tanh)
  - `SACTransformerCritic`: Twin Q-networks (Q1, Q2, min_q)
- NEW `agents/action_gating.py`: confidence threshold gating
  - |c| < 0.3 → HOLD (mandatory), c > 0.3 → BUY, c < -0.3 → SELL
  - Risk scaled by: (|c| - threshold) / (1 - threshold)
  - SL/TP multipliers mapped [-1,1] → [0.5, 2.0]
- NEW `tests/test_action_gating.py`: **14 tests** (HOLD/BUY/SELL/edges/batch/risk scaling)
- Tests: **52 passed** in 3.13s ✅

### [SPRINT 3.3] RegimeDetector ✅ — 18/03/2026 22:35
- Rewrite `models/regime_detector.py`: dual-mode (statistical GMM + neural MLP)
- `SMCFeatureExtractor`: 5 features from SMC data (trend_strength, vol_pctl, range_ratio, bos_freq, vol_climax)
- **ZERO traditional indicators** (no ATR/ADX/RSI/MA)
- `GaussianMixtureRegime`: custom EM algorithm, K-means++ init, auto-labels regimes from centroids
- 4 regimes: trend_up, trend_down, ranging, volatile
- Logic test PASSED: synthetic uptrend data → GMM correctly predicts trend_up with highest probability
- Tests: **38 passed** in 3.18s ✅

### [SPRINT 3.2] CrossAttentionMTF ✅ — 18/03/2026 22:25
- Rewrite `models/cross_attention.py`: M5 (Q) × H1+H4 (K,V) full-sequence cross-attention
- `ContextEncoder` cho H1/H4 (1-layer lightweight Transformer)
- Attn matrix = 64×54 = 3,456 entries (~3.4 MB cho batch=64, **well under 2GB**)
- `get_cross_attention_weights()` cho interpretability
- Legacy wrappers (CrossAttentionFusion, MultiTimeframeEncoder) giữ backward compat
- Tests: **32 passed** in 3.07s ✅

### [SPRINT 3.1] TransformerSMC ✅ — 18/03/2026 22:10
- NEW `models/transformer_smc.py`: Sinusoidal PE + `nn.TransformerEncoder` (2 layers, 4 heads, GELU, pre-norm) + Mean Pooling
- Input: (B, 64, 28) → Output: (B, 128) — latent SMC pattern representation
- `get_attention_weights()` cho SHAP analysis
- Zero hardcode — tất cả params qua `__init__`
- Tests: **26 passed** (8 mới: init, shape, gradient, PE values, seq lengths, mask, attention weights)

### [DOCS] MASTER_PLAN v4.0 — 18/03/2026 21:50
- Cập nhật Sprint 1 tất cả ✅ DONE, Sprint 3 🔴 ĐANG LÀM
- Thêm bảng TIẾN ĐỘ HIỆN TẠI, TRADING RULES
- Đổi symbols, TF, features đúng với hệ thống SMC

### [BACKTEST] Walk-Forward Holdout 20% — 18/03/2026 21:35
- `scripts/backtest.py`: 3,002 trades trên 5 symbols (35 ngày holdout)
- ETHUSD: **+56.7%**, WR 44.8%, PF 1.53, Sharpe 11.0 — pass V1 **~8 ngày**
- BTCUSD: **+25.1%**, WR 42.8%, PF 1.32, Sharpe 7.0 — pass V1 **~16 ngày**
- AVG: +18.7%, WR 44.8%, PF 1.22, DD 4.26%, Sharpe 4.6
- Reports saved: `reports/backtest_report.json`, `reports/trade_log.json`

### [TRAIN] SAC on SMC + Volume + PA Features ✅
- **200K steps** trên 5 symbols, M5 primary TF (50K bars/symbol, 260 days)
- Convergence: 41.8% WR @50K → **49.4% @90K** → **42.8% WR @130K (best reward 3.05)**
- H1 inside bar exit rule: chốt hết lệnh khi H1 nến inside bar
- Model mới: 2.63M params (1,795 obs dim = 64×28 features + 3 state)

### [REFACTOR] Feature Pipeline → SMC + Volume + Price Action ✅
- **Xóa:** RSI, ATR, MA distance, rolling vol (indicator truyền thống ❌)
- **Thêm SMC:** swing_structure, BOS/CHoCH, order_blocks, fair_value_gaps, liquidity_zones
- **Thêm PA:** pin_bar, engulfing, inside_bar
- **Thêm Volume:** vol_delta, climax_vol (ngoài relative_volume hiện có)
- **Giữ:** candle_ratios, relative_volume, time_encoding, log_return
- **28 features** tổng (thay 14 cũ), đổi TF chính M1 → **M5**
- Config: thêm `h1_inside_bar_exit: true` vào prop_rules.yaml
- Tests: **161 passed, 2 skipped** ✅

### [TRAIN] SAC Agent — 200K Training Run ✅
- `scripts/train_agent.py`: SAC (MLP 256x256, 1.16M params, twin critics, auto-α)
- **200K steps** trên 5 symbols (XAUUSD, US100, US30, ETHUSD, BTCUSD)
- Convergence curve: 0% WR (10K) → 31% (110K) → **49.4% (150K, best)** → 45.1% (200K)
- Best eval reward: **5.46** tại step 150K
- Saved `models_saved/best_model.pt` (best) + `final_model.pt` (200K)

### [DATA] Fetch Historical Data + Build Features ✅
- `scripts/fetch_historical_data.py`: fetch M1 từ MT5 (copy_rates_from_pos, 50K-batch)
- 5 symbols × 100K M1 bars = 500K total (~105 ngày/symbol)
- Resample M15 (6,700+ bars), H1 (1,687 bars), H4 (442 bars)
- Build 16 features: log_return, ATR, RSI, vol_ratio, body_ratio, RVol, time encoding
- 25 Parquet files, 10.4 MB

### [FEAT] Per-Trade & Daily Loss Limits ✅
- `prop_rules.yaml`: thêm `max_loss_per_trade_pct: 0.003` (0.3%) + `daily_loss_cooldown_pct: 0.03` (3%)
- `killswitch.py`: thêm class `DailyLossGate` — max SL tính toán, cooldown auto-reset ngày mới
- `validator.py`: thêm 2 trường mới vào `PropRulesConfig`
- `test_safety.py`: thêm 7 test cases cho DailyLossGate — all PASSED ✅
- **Nguyên tắc:** Thua 1 lệnh max 0.3%, thua cả ngày 3% → dừng đến ngày hôm sau

### [INFRA] Merge all Sprint branches → main + MT5 FTMO Connection ✅
- Merged 7 branches vào `main` (fast-forward, no conflicts)
- Tạo `.env` chứa FTMO Demo credentials (gitignored)
- Tạo `.env.example` template (safe to commit)
- Viết `scripts/test_mt5_connection.py`: full MT5 diagnostic
- **Test kết quả:** Login OK, Balance $100,000, EURUSD available, Leverage 1:100
- Tick = 0.0 do thị trường đóng cửa cuối tuần

### [FEAT] T6.1-T6.3 — Live Execution Engine ✅
- **Branch:** `sprint6/T6.1-T6.3-live-execution`
- Viết `live_execution/mt5_connector.py`: Market orders, position mgmt (close/modify/close-all), account queries, auto-reconnect
- Viết `live_execution/inference_pipeline.py`: Feature→Model→Decision pipeline, killswitch integration, trading hours check
- Viết `live_execution/paper_trading.py`: Session tracking, Prop Firm pass/fail validation (DD, profit target, min days), JSON report
- Viết `tests/test_live_execution.py`: 20 test cases — all PASSED ✅
- **Total test suite:** 147/147 PASSED ✅
- **Trạng thái:** Sprint 6 DONE

### [FEAT] T5.1-T5.3 — Safety Layer & Self-Evolution ✅
- **Branch:** `sprint5/T5.1-T5.3-safety-layer`
- Viết `live_execution/killswitch.py`: Triple-layer protection (soft/hard/emergency), alert callback, daily reset, EquityWatchdog
- Viết `model_registry/registry.py`: Versioned checkpoints, best model tracking, rollback, JSON manifest
- Viết `training_pipeline/safe_retrain.py`: Validation gate (5% improvement threshold), shadow training, eval pipeline
- Viết `tests/test_safety.py`: 18 test cases — all PASSED ✅
- **Total test suite:** 127/127 PASSED ✅
- **Trạng thái:** Sprint 5 DONE

### [FEAT] T4.1-T4.3 — Training Pipeline ✅
- **Branch:** `sprint4/T4.1-T4.3-training-pipeline`
- Viết `training_pipeline/per_buffer.py`: SumTree O(log n), proportional PER, IS weights, beta annealing
- Viết `training_pipeline/curriculum.py`: 4-stage progressive difficulty, auto-promote khi reward đạt threshold
- Viết `training_pipeline/sac_trainer.py`: SAC update loop (twin Q-critic target, auto-α entropy, gradient clip, checkpoint)
- Viết `tests/test_training.py`: 21 test cases — all PASSED ✅
- **Total test suite:** 109/109 PASSED ✅
- **Trạng thái:** Sprint 4 DONE

### [FEAT] T3.1-T3.3 — Neural Network Models ✅
- **Branch:** `sprint3/T3.1-T3.3-neural-network-models`
- Viết `models/transformer.py`: Pre-norm Transformer Encoder, CLS token, learnable positional encoding
- Viết `models/cross_attention.py`: Cross-Attention fusion + MultiTimeframeEncoder (M15 → H1/H4)
- Viết `models/actor_critic.py`: SAC Actor (squashed Gaussian + tanh + log prob correction) + Twin Q-Critic (clipped double-Q)
- Viết `models/regime_detector.py`: Neural regime classifier (4 regimes) + learnable regime embeddings
- Viết `tests/test_models.py`: 18 test cases — all PASSED ✅
- **Total test suite:** 88/88 PASSED ✅
- **Trạng thái:** Sprint 3 DONE

### [FEAT] T2.1-T2.3 — Gymnasium Environment & Reward Engine ✅
- **Branch:** `sprint2/T2.1-T2.3-gym-environment`
- Viết `environments/physics_sim.py`: variable spread (session/news/vol), log slippage, exec delay, partial fill, requote
- Viết `environments/reward_engine.py`: 8-component reward (PnL, delta-unrealized, exp-DD penalty, overnight, spread cost, R/R bonus, overtrading, inaction) + episode termination
- Viết `environments/prop_env.py`: custom Gymnasium env — continuous action [confidence, risk_frac, sl_mult, tp_mult], action gating (HOLD if conf < 0.3), position SL/TP tracking, DD termination
- Viết `tests/test_environment.py`: 26 test cases — all PASSED ✅
- **Total test suite:** 70/70 PASSED ✅
- **Trạng thái:** Sprint 2 DONE

### [FEAT] T1.2-T1.5 — Data Engine & Utilities ✅
- **Branch:** `sprint1/T1.2-T1.5-data-engine-utils`
- Viết `data_engine/mt5_fetcher.py`: graceful import, incremental fetch, multi-symbol batch, Polars Parquet zstd compression
- Viết `data_engine/feature_builder.py`: candle ratios, RVol, sin/cos time encoding, log returns, ATR-normalized volatility, price position
- Viết `data_engine/multi_tf_builder.py`: resample M1 → M5/M15/H1/H4, asof join alignment (chống look-ahead bias)
- Viết `data_engine/normalizer.py`: Welford's online algorithm, Chan's batch merge, ±5σ clipping, JSON serialize/load
- Viết `utils/polars_bridge.py`: Polars ↔ PyTorch conversion + sliding window sequence builder
- Viết `utils/alert_bot.py`: Telegram Bot async/sync, convenience methods cho mọi event
- Viết `tests/test_feature_builder.py`: 14 test cases covering all feature functions
- Viết `tests/test_normalizer.py`: 11 test cases covering normalization, clipping, serialization, edge cases
- **Fix:** Polars `map_batches` API → dùng numpy conversion trực tiếp cho time_encoding
- **Fix:** Synthetic OHLCV generator — ensure high >= max(open,close), low <= min(open,close)
- **Tests:** 44/44 PASSED ✅ (config: 19 + features: 14 + normalizer: 11)
- **Trạng thái:** Sprint 1 DONE. Sẵn sàng merge → main

### [FEAT] T1.1 — Project Setup & Config Foundation ✅
- **Branch:** `sprint1/T1.1-project-setup-config`
- Tạo monorepo structure: 9 packages (`configs`, `data_engine`, `environments`, `models`, `agents`, `training_pipeline`, `live_execution`, `model_registry`, `utils`)
- Viết `configs/prop_rules.yaml`: 38 tham số (DD limits, reward weights, execution sim, action gating, killswitch)
- Viết `configs/train_hyperparams.yaml`: 95+ tham số (SAC config, Transformer arch, PER buffer, Curriculum 4 stages, Ensemble, Nightly retrain, W&B)
- Viết `configs/validator.py`: Pydantic v2 schema validation — catch sai format, cross-field conflicts (DD logic, embed_dim%heads, seeds count)
- Viết `tests/test_config_validation.py`: 19 test cases — all PASSED ✅
- Viết `pyproject.toml` (layered deps: core/dev/ml/live), `.gitignore`
- **Trạng thái:** T1.1 DONE. Tiếp tục T1.2-T1.5 (data fetcher, features, normalizer, utils)

## 18/03/2026

### [PLAN] Khởi tạo dự án & Lập kế hoạch

- **Tạo `DRL_TRADING_SKILLS.md`** — Chọn lọc 13/1,265 skills tối ưu từ catalog, chia 3 tầng (Critical/Essential/Infrastructure), map từng skill vào component của hệ thống DRL.
- **Tạo `MASTER_PLAN_FINAL.md`** — Bản Master Plan hoàn chỉnh v3.0: 7 Sprints, 14 tuần, 89 tasks chi tiết. Đã tích hợp toàn bộ cải tiến (multi-component reward, action gating, regime detection, safe nightly retrain, triple-layer protection, paper trading phase, model registry).
- **Tạo `.agent/workflows/git-branching.md`** — Quy tắc branching: không code trên main, mỗi task 1 branch `sprint{N}/T{task_id}-{mô tả}`.
- **Tạo `DEVLOG.md`** (file này) — Nhật ký dự án, ghi chép mọi thay đổi.

**Trạng thái hiện tại:** Chưa bắt đầu code. Plan đã approved. Sẵn sàng bắt đầu Sprint 1.
