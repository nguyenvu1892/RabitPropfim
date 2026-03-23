---
description: V3 Stage-Gate training workflow for DRL trading bot
---

# V3 Stage-Gate Training Workflow

// turbo-all

## Prerequisites
1. Data fetched with OHLCV: `python scripts/fetch_historical_data.py`
2. Verify `_ohlcv.npy` files exist in `data/` folder

## Stage 1: Context Recognition
```bash
python scripts/train_curriculum.py --stage 1
```
- Trains M15+H1 only (200K steps)
- Output: `models_saved/best_Stage1_Context.pt`
- **HARD STOP — do NOT proceed!**

### Validate Stage 1
```bash
python scripts/backtest_behavioral.py
```
- Check: Action distribution balanced (not 90%+ one direction)
- Check: Entropy > 1.0 (no mode collapse)
- Check: Bot understands market context (buys in uptrend, sells in downtrend)
- **If FAIL:** Tune hyperparams, re-train `--stage 1`

## Stage 2: Precision Entry
```bash
python scripts/train_curriculum.py --stage 2
```
- Adds M5 timeframe (300K steps)
- Warm-starts from Stage 1 checkpoint
- Output: `models_saved/best_Stage2_Precision.pt`
- **HARD STOP — do NOT proceed!**

### Validate Stage 2
```bash
python scripts/backtest_behavioral.py
```
- Check: SL/TP multipliers vary (sl_mult not constant)
- Check: Entry timing improves over Stage 1
- Check: Winrate > 45%
- **If FAIL:** Tune hyperparams, re-train `--stage 2`

## Stage 3: Full Fusion
```bash
python scripts/train_curriculum.py --stage 3
```
- All 4 timeframes + EpisodicMemory (500K steps)
- Warm-starts from Stage 2 checkpoint
- Output: `models_saved/best_Stage3_FullFusion.pt`

### Validate Stage 3
```bash
python scripts/backtest_behavioral.py
```
- Check: Winrate >= 55%
- Check: Max Drawdown < 10%
- Check: Diverse action patterns across market regimes
- **If PASS → Deploy to paper trading**

## GPU Monitoring (RunPod)
```bash
watch -n 1 nvidia-smi
```
- Target: GPU-Util >= 85%, VRAM usage > 50%
- If OOM: reduce `batch_size` in `train_curriculum.py` STAGES config
