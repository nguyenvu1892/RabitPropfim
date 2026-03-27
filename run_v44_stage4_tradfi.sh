#!/bin/bash
set -e
export PYTHONPATH=/home/user/RabitPropfim
cd /home/user/RabitPropfim

echo "=============================="
echo "  V4.4 Stage 4 TradFi Fine-tune"
echo "  ATR-Normalized R:R + Close Unlocked"
echo "=============================="

echo "[1/2] Re-Harvesting with R:R > 1.5 Filter (Close UNLOCKED, 50 episodes)"
/home/user/venv/bin/python scripts/harvest_contrastive_v36.py --episodes 50 --ab-group A

echo "[2/2] Stage 4 Fine-tune (2M steps, TradFi sub-set)"
/home/user/venv/bin/python scripts/train_v36.py --stage 3 --total-steps 2000000 --ab-group A

echo "=============================="
echo "  V4.4 Stage 4 Complete!"
echo "=============================="
