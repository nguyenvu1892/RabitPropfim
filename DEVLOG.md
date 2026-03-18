# 📝 NHẬT KÝ DỰ ÁN — RABIT-PROPFIRM DRL SYSTEM

> Mỗi thay đổi PHẢI được ghi vào đây. Người sau đọc file này sẽ biết dự án đang ở đâu, đã làm gì, chỉnh sửa gì.

---

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
