#!/bin/bash
set -e
export PYTHONPATH=/home/user/RabitPropfim
cd /home/user/RabitPropfim

echo "=============================="
echo "  V4.4 TradFi Pipeline (Group A)"
echo "=============================="

echo "[1/4] Starting Stage 1 (750k steps)"
/home/user/venv/bin/python scripts/train_v36.py --stage 1 --total-steps 750000 --ab-group A

echo "[2/4] Harvesting Contrastive Memory (50 episodes x 3 symbols)"
/home/user/venv/bin/python scripts/harvest_contrastive_v36.py --episodes 50 --ab-group A

echo "[3/4] Starting Stage 2 (500k steps)"
/home/user/venv/bin/python scripts/train_v36.py --stage 2 --total-steps 500000 --ab-group A

echo "[4/4] Starting Stage 3 (5M steps)"
/home/user/venv/bin/python scripts/train_v36.py --stage 3 --total-steps 5000000 --ab-group A

echo "=============================="
echo "  V4.4 Pipeline Completed!"
echo "=============================="
