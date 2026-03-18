# 📝 NHẬT KÝ DỰ ÁN — RABIT-PROPFIRM DRL SYSTEM

> Mỗi thay đổi PHẢI được ghi vào đây. Người sau đọc file này sẽ biết dự án đang ở đâu, đã làm gì, chỉnh sửa gì.

---

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
