# 📝 NHẬT KÝ DỰ ÁN — RABIT-PROPFIRM DRL SYSTEM

> Mỗi thay đổi PHẢI được ghi vào đây. Người sau đọc file này sẽ biết dự án đang ở đâu, đã làm gì, chỉnh sửa gì.

---

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
