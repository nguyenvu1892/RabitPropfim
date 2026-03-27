#!/bin/bash
set -e
export PYTHONPATH=/home/user/RabitPropfim
cd /home/user/RabitPropfim

echo "=============================="
echo "  V4.5 Master Vault Pipeline"
echo "=============================="

echo "[1/2] Generating Master Vault Memory Prototypes (K-Means)"
/home/user/venv/bin/python scripts/train_memory_prototypes.py --obs-dim 488

echo "[2/2] Starting V4.5 Stage 3 (6.5M steps — All Symbols)"
/home/user/venv/bin/python scripts/train_v45_stage3.py --total-steps 6500000

echo "=============================="
echo "  V4.5 Pipeline Completed!"
echo "=============================="
