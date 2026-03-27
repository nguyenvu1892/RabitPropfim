#!/bin/bash
set -e
export PYTHONPATH=/home/user/RabitPropfim
cd /home/user/RabitPropfim

echo "=============================="
echo "  V4.4 TradFi RESUME v2"
echo "  (Action Masking Harvest)"
echo "=============================="

echo "[1/3] Harvesting with Action Masking (Close→Hold, 50 episodes)"
/home/user/venv/bin/python scripts/harvest_contrastive_v36.py --episodes 50 --ab-group A

echo "[2/3] Starting Stage 2 (500k steps)"
/home/user/venv/bin/python scripts/train_v36.py --stage 2 --total-steps 500000 --ab-group A

echo "[3/3] Starting Stage 3 (5M steps) — Close UNLOCKED"
/home/user/venv/bin/python scripts/train_v36.py --stage 3 --total-steps 5000000 --ab-group A

echo "=============================="
echo "  V4.4 TradFi Pipeline Done!"
echo "=============================="
