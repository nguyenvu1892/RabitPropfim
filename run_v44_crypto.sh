#!/bin/bash
set -e
export PYTHONPATH=/home/user/RabitPropfim
cd /home/user/RabitPropfim

echo "=============================="
echo "  V4.4 Full Pipeline (Real Futures)"
echo "  Warm-start from V4.2 + Freeze"
echo "=============================="

echo "[1/4] Stage 1 — Fine-tune on Real Futures (750k steps, frozen price layers)"
/home/user/venv/bin/python -u scripts/train_v36.py --stage 1 --total-steps 750000 --ab-group B

echo "[2/4] Harvesting Contrastive Memory"
/home/user/venv/bin/python -u scripts/harvest_contrastive_v36.py --episodes 50 --ab-group B

echo "[3/4] Stage 2 — Contrastive Learning (500k steps)"
/home/user/venv/bin/python -u scripts/train_v36.py --stage 2 --total-steps 500000 --ab-group B

echo "[4/4] Stage 3 — Deep PPO + R:R Head (5M steps)"
/home/user/venv/bin/python -u scripts/train_v36.py --stage 3 --total-steps 5000000 --ab-group B

echo "=============================="
echo "  V4.4 Pipeline Completed!"
echo "=============================="
