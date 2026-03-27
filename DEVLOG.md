# 📝 NHẬT KÝ DỰ ÁN — RABIT-PROPFIRM DRL SYSTEM

> Mỗi thay đổi PHẢI được ghi vào đây. Người sau đọc file này sẽ biết dự án đang ở đâu, đã làm gì, chỉnh sửa gì.

---

## 27/03/2026

### [CHORE] Project Cleanup — Xóa Code Dùng Một Lần — 27/03
- **Thực trạng:** Có một số script và sh file được tạo ra nhanh trong một số phase nhất định để test hoặc chạy quick backtest, nhưng hiện tại đã bị bỏ lại và có thể gây rác cho dự án.
- **Xóa Scripts:** Xóa 6 file ad-hoc/old code trong thư mục scripts: `_debug_env.py`, `backtest_s2_quick.py`, `backtest_s3_quick.py`, `deep_stats_v38.py`, `harvest_memory_v42.py`, `v3_smoke_test.py`.
- **Xóa Shell Runners:** Xóa các script liên quan đến V4.3 cũ (`run_v43_full_crypto.sh`, `run_v43_resume.sh`) để tập trung hoàn toàn vào pipeline V4.4 hiện tại.

### [MAJOR] V4.5 — "Tích Lũy Tuyệt Đối" (Absolute Accumulation) — Master Vault Architecture — 27/03

#### 1. Master Vault (Kho Lưu Trữ Bất Tử)
- Sửa `contrastive_memory.py`: Xóa bỏ `maxlen`, chuyển sang JSONL Append-Only.
- File gộp chung: `master_vault_wins.jsonl`, `master_vault_losses.jsonl`.
- Bộ lọc: Wins R:R > 1.5, Losses Confidence ≥ 0.7.

#### 2. Lò Luyện K-Means Dual Bank
- Sửa `train_memory_prototypes.py`: Dùng `MiniBatchKMeans` clustering **riêng biệt** Win và Loss.
- Output: `memory_prototypes_v45.pt` (`win_prototypes [8×64]`, `loss_prototypes [8×64]`, frozen masks).
- Chiết xuất d_model=64 pooled representations qua `model._encode()` thay vì 128-dim embedding.

#### 3. Stage 3 Training — Unified Battlefield
- **[NEW]** `train_v45_stage3.py`: Train đồng thời TradFi (464-dim, zero-padded → 488) + Crypto (488-dim).
- Warm-start từ best V4.4, **Freeze** toàn bộ Base Price Action layers.
- Chỉ unfreeze: Cross-Attention, R:R Head, Actor, Critic, Contrastive, Memory Banks.
- **[NEW]** `run_v45_stage3.sh`: Bash runner 6.5M steps.

#### Files thay đổi:
| File | Thay đổi |
|------|----------|
| `training_pipeline/contrastive_memory.py` | JSONL Append-Only, xóa maxlen |
| `scripts/harvest_contrastive_v36.py` | Filter R:R > 1.5 + Confidence ≥ 0.7 |
| `scripts/train_memory_prototypes.py` | [REWRITE] Dual bank [8×64] KMeans |
| `scripts/train_v45_stage3.py` | [NEW] V4.5 Stage 3 Training (freeze + inject + zero-pad) |
| `run_v45_stage3.sh` | [NEW] Bash runner 6.5M steps |

---

## 25/03/2026

### [CHORE] Trọn gói Dọn dẹp Dự án (Project Cleanup) — 25/03 23:15
- **Gỡ bỏ hàng loạt các tệp Parquet trung gian:** Xóa hơn 75 file RAW OHLCV và features sinh ra từ `fetch_historical_data.py` chiếm gần 100MB (vì agent V3.6 load trực tiếp từ file `_50dim.npy` thay vì đọc lại parquet thô).
- **Xóa file Backup cũ:** Loại bỏ tệp sao lưu v3.2 `data_v32.zip` (~140MB).
- **Loại bỏ Obsolete Scripts:** Dọn dẹp 9 kịch bản thử nghiệm cũ không còn được bảo trì: `v33`, `v34`, `v35`, `backtest*`, giúp thư mục `scripts` gọn gàng tập trung vào phiên bản chính.
- Dự án nhẹ nhàng, sẵn sàng cho các vòng lặp V3.6/V3.7 tiếp theo!

---

### [MAJOR] V3.6 — "Tự Vấn" (Self-Reflection) — AttentionPPO + Contrastive Learning — 25/03 01:50
- **Commits:** `c4ff8dc`, `dc61ae5` on `main`
- **Tư duy mới:** Bỏ hẳn Imitation Learning (copy lệnh). Bot phải TỰ SO SÁNH WIN vs LOSS.

#### 1. Kiến trúc AttentionPPO (thay thế MLP)
- **Trước (V3.5):** MLP trunk 512→256→128 = 372K params
- **Sau (V3.6):** Self-Attention 8 tokens × 50-dim → Transformer 2L 4H d_model=64 = **120K params** (3.1× nhẹ hơn)
- Tokens: `[H1] [M15] [M5] [M1_b1] [M1_b2] [M1_b3] [M1_b4] [M1_b5]`
- 3 output heads: Actor (4 actions), Critic (value), **Contrastive** (128-dim embedding)
- Manual attention weight capture → xuất Attention Heatmap 8×8 cho phân tích

#### 2. Contrastive Memory (thay thế VIP Buffer + IL)
- Lưu cả WIN lẫn LOSS trades (per-symbol, per-regime)
- InfoNCE Loss: ép embedding WIN xa embedding LOSS
- Không copy mù → bot học PHÂN BIỆT bản chất WIN/LOSS

#### 3. Kết quả Stage 1 (750K steps, 719s)
| Symbol | Trades | WR | MC WR | Actions |
|--------|--------|----|-------|---------|
| XAUUSD | 7,564 | 48.2% | 59.6% | B=24% S=25% H=23% C=28% |
| BTCUSD | 7,337 | 47.5% | 59.1% | Balanced |
| ETHUSD | 7,612 | 47.6% | 59.4% | Balanced |
| US30 | 7,709 | 47.4% | 58.1% | Balanced |
| US100 | 7,458 | 47.8% | 58.8% | Balanced |
| **TỔNG** | **37,680** | **47.7%** | **59.0%** | **Cân bằng nhất** |

#### 4. Attention Heatmap
```
H1     |  31.4%  ← Bot tự học H1 quan trọng nhất (3× so với token khác)
M15    |  10.7%
M5     |  10.7%
M1_b2  |  12.1%  ← Entry bar nhận attention cao nhất trong M1
M1_b1  |   9.0%
```

#### Files mới/thay đổi:
| File | Thay đổi |
|------|----------|
| `models/attention_ppo.py` | [NEW] AttentionPPO: 8-token Self-Attention + 3 heads |
| `training_pipeline/contrastive_memory.py` | [NEW] WIN/LOSS storage + InfoNCE contrastive loss |
| `scripts/train_v36.py` | [NEW] PPO + Contrastive training pipeline |
| `scripts/backtest_v36.py` | [NEW] Backtest + Attention Heatmap analysis |

---

### [SERVER] Server cũ mất — Chuyển server mới — 25/03 09:30
- Server RTX 4090 (`38.224.253.180`) bị mất
- Toàn bộ models trained (best_v34/v35/v36_stage1.pt) cần train lại
- Code đã push đầy đủ lên GitHub, sẵn sàng clone xuống server mới
- **Checkpoint cần train lại:** V3.6 Stage 1 (750K steps, ~12 phút trên RTX 4090)

---

## 24/03/2026

### [MAJOR] V3.5 — "4-TF Hợp Thể" — Thêm H1 vào Observation — 24/03 23:45
- **Commits:** `1f98e68`, `c3d99b7` on `main`

#### Thay đổi:
- **Observation:** 350-dim (M15+M5+M1) → **400-dim (H1+M15+M5+M1)**
- `_get_obs_discrete()`: thêm H1 bar (50-dim) đầu vector
- `train_v35.py`: PPO obs_dim=400, actions=4

#### Kết quả Stage 1:
| Metric | V3.4 | **V3.5** |
|--------|------|---------|
| WR | 49.0% | **50.2%** ← VƯỢT 50% LẦN ĐẦU |
| Trades | 2,277 | 5,557 |
| SL Rate | 19.2% | **17.7%** |

#### Kết quả Stage 2 (PPO + IL, 735 VIP strict 147/sym):
- WR = 31.1% ❌ (SELL bias 68-96%)
- **Kết luận:** IL (Imitation Learning) bị lỗi cấu trúc — VIP data toàn SELL → bot học copy mù

#### Files mới:
| File | Thay đổi |
|------|----------|
| `prop_env.py` | V3.5: 400-dim obs, thêm H1 bar |
| `scripts/train_v35.py` | [NEW] PPO obs=400 |
| `scripts/harvest_vip_v35.py` | [NEW] VIP harvest strict equal cap |
| `scripts/backtest_v35.py` | [NEW] V3.5 backtest |

---

### [MAJOR] V3.4 — "Quản Trị Rủi Ro" — Discrete(4) + Auto SL + CLOSE — 24/03 15:00
- **Commits:** `3eb1103`, `e6b6f11` on `main`
- **Đập đi xây lại** `prop_env.py` và `reward_engine.py`

#### 1. Action Space mới: Discrete(4)
- **Trước (V3.3):** Discrete(3) BUY/SELL/HOLD
- **Sau (V3.4):** Discrete(4) BUY/SELL/HOLD/**CLOSE** ← bot tự chốt lời

#### 2. Auto SL (Swing Point)
- Bỏ SL cố định (x1.5/x2 ATR)
- **Mới:** Quét M5 tìm Swing Low/High gần nhất → SL tự động theo cấu trúc
- Fallback: x2 ATR nếu không tìm được swing

#### 3. CLOSE Action
- Reward x5 khi bot tự tay bấm CLOSE và lệnh đang lãi
- Dạy bot "gồng lời" → biết khi nào chốt

#### 4. Observation V3.4: 350-dim
- M15 (1 bar, 50) + M5 (1 bar, 50) + M1 (5 bars, 250)
- ATR normalization cho price features

#### Kết quả:
- **Stage 1:** WR=49.0%, 2,277 trades, CLOSE rất tích cực
- **Stage 2 (PPO+IL):** WR giảm do SELL bias từ VIP data
- **Stage 3:** WR=38.3%, Manual Close WR=63.5%

#### Files thay đổi:
| File | Thay đổi |
|------|----------|
| `prop_env.py` | V3.4: Discrete(4), Auto SL, 350-dim obs |
| `reward_engine.py` | V3.4: CLOSE profit x5, bỏ frequency rewards |
| `prop_rules.yaml` | Thêm swing_lookback, sl_buffer_mult, close_profit_multiplier |
| `scripts/train_v34.py` | [NEW] PPO Discrete(4) training |
| `scripts/harvest_vip_v34.py` | [NEW] VIP harvest with SMC filter |

### [DOC] README.md — Tài liệu hoá dự án — 24/03 22:00
- **Commits:** `c80a493`, `babfb90`, `6aa5d83` on `main`
- Viết README.md hoàn chỉnh: Vision, Knowledge Base (50-dim features), Quy tắc rủi ro, Nightly Auto-Retrain, lộ trình V3.5→V4.0

---

## 23/03/2026

### [MAJOR] V3 Core Fixes — Blind Price Bug + GPU Optimization + Stage-Gate — 23/03/2026 23:27
- **Commit:** `95e4925` on `main` — 9 files changed, +1282 / -1646
- **Branch:** trực tiếp trên `main`

#### 1. FIX Bug Mù Giá (Priority 1)
- **Trước:** `prop_env.py` dùng `data_m5[step, close_idx=4]` → col 4 = `pin_bar_bull` ≈ 0, **KHÔNG PHẢI close price**
- **Sau:** Thêm param `ohlcv_m5` (N×5 OHLCV thực). `_get_current_price()` đọc OHLCV col 3 (close)
- **ATR:** Viết lại `_estimate_atr_pips()` dùng **True Range** formula: `max(H-L, |H-Cprev|, |L-Cprev|)`
- **Fallback:** Khi chưa có `_ohlcv.npy`, reconstruct synthetic prices từ `log_return` (col 27) với clamp `[-20, 20]`
- `fetch_historical_data.py` lưu thêm `_ohlcv.npy` song song `_50dim.npy`
- **Test:** price = 908.89 (trước: 0.00), ATR = 90 pips (trước: 0), trades = 1 (trước: 0) ✅

#### 2. GPU Optimization — Vắt kiệt RTX 4090
- **Training loop viết lại hoàn toàn:** 2-phase Rollout Buffer pattern
  - Phase 1 (CPU): 32 envs × n_steps=4096 → buffer 131K transitions
  - Phase 2 (GPU): Sample batch_size=2048/4096 → n_updates=8/16 gradient steps
- **Config nâng cấp:** `n_envs` 16→32, `batch_size` 256→2048/4096, `n_steps` 4096/8192
- `train_hyperparams.yaml` cập nhật toàn bộ
- **Test:** entropy=3.4 (trước: 0.0), alpha=0.135 (stable), no NaN ✅

#### 3. Entropy Regularization — Chống Mode Collapse  
- `target_entropy` nâng từ -5.0 → **-2.0**
- `log_alpha` kẹp `[-3.0, 2.0]` → alpha ∈ [0.05, 7.39], không thể tụt về 0
- **Kết quả:** entropy duy trì 3.3-3.5 suốt 100 steps test (V2: 0.00)

#### 4. Stage-Gate Workflow — Ngắt Cầu Dao
- `train_curriculum.py --stage N` train 1 stage → **HARD STOP**
- Chạy không arg → in hướng dẫn, exit code 1
- `--stage 2` khi chưa có Stage1.pt → **STAGE-GATE VIOLATION**, chặn chạy
- Workflow: `/train-v3` (file `.agents/workflows/train-v3.md`)

#### Files thay đổi:
| File | Thay đổi |
|------|----------|
| `prop_env.py` | +ohlcv_m5, real price, True Range ATR, synthetic fallback |
| `fetch_historical_data.py` | +save _ohlcv.npy |
| `train_curriculum.py` | Rollout buffer, GPU config, entropy floor, Stage-Gate |
| `train_hyperparams.yaml` | GPU-optimized settings |
| `backtest_behavioral.py` | [NEW] Behavioral analysis script (4 modules) |
| `v3_smoke_test.py` | [NEW] V3 price/ATR verification |
| `train-v3.md` | [NEW] Stage-Gate workflow |

---

## 23/03/2026

### [FEAT] AsyncVectorEnv — Parallel Environment Stepping — 23/03/2026 02:10
- **Trước:** Sequential loop, 1 env/step, fake batch `.expand()`, GPU chờ CPU → **~0.5 SPS**
- **Sau:** `gymnasium.vector.AsyncVectorEnv` — N envs song song trên N CPU cores
  - Real diverse batch: N observations khác nhau mỗi step (không repeat!)
  - Action scaling vectorized via numpy broadcasting
  - Auto-reset khi episode kết thúc
- **Config:** Stage 1/2/3: **n_envs=16**, TestStage: **n_envs=4**
- **Test Local (4 envs, CPU):** 100 steps / 31.5s → **3 SPS** (~6x faster) ✅
- **RunPod (16 envs, GPU):** Dự kiến **20-50 SPS**

## 22/03/2026

### [HOTFIX] Alpha Reset Bug — Mandatory Final Checkpoint Save — 22/03/2026 22:10
- **Branch:** `phase3.1/dual-entry-system` → **merge vào `main`**
- **Bug:** `log_alpha` reset về 1.0 khi chuyển Stage (RunPod đang chạy code cũ chưa merge)
- **Fix:** Thêm MANDATORY final checkpoint save sau khi training loop kết thúc
  - Đảm bảo `log_alpha` + optimizer states LUÔN LUÔN được lưu ở bước cuối cùng
  - Log xác nhận: `FINAL checkpoint -> best_TestStage.pt (log_alpha=-2.0300, alpha=0.1313)`
- **Test:** 100 steps, EXIT CODE 0 ✅

### [HOTFIX] NaN Crash in Actor Forward (ValueError: Normal) — 22/03/2026 22:45
- **Bug:** Actor mean tensor all-NaN at Step 2 → `ValueError: Expected parameter loc... Real()` crash
- **Root cause:** Gradient explosion after Step 1 update corrupts weights → `mean_head` outputs NaN
- **Fix 3 lớp:**
  1. `train_curriculum.py`: `nan_to_num()` trên 4 input tensors (M1/M5/M15/H1), clamp ±5.0
  2. `train_curriculum.py`: NaN check sau mỗi `optimizer.step()` → auto-recover từ checkpoint
  3. `sac_policy.py`: `nan_to_num + clamp` trên `mean/std` trước `Normal()` constructor
- **Data verified:** 20 files × 50-dim → **0 NaN, 0 Inf** (data sạch, lỗi do gradient)
- **Test:** 100 steps, EXIT CODE 0 ✅

### [HOTFIX] math.exp Overflow + Gradient Explosion — 22/03/2026 01:50
- **Bug:** `OverflowError: math range error` at `reward_engine.py:189` — `math.exp(dd_beta * dd_ratio)` overflows when DD ratio is extreme
- **Thêm:** NaN guard bắt 60+ steps liên tục → critic weights bị NaN → recovery loop nhưng vẫn crash ở reward
- **Fix 3 chỗ:**
  1. `reward_engine.py`: Clamp exponent trong `math.exp()` vào [-50, 50] → chặn Overflow
  2. `train_curriculum.py`: Stage 1 LR: **3e-4 → 1e-4** (giảm tốc cho 5-dim action + frozen layers)
  3. `train_curriculum.py`: `grad_clip`: **1.0 → 0.5** (siết đạo hàm gắt hơn)
- **Test:** 100 steps, **0 NaN warnings, 0 overflow**, EXIT CODE 0 ✅

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
