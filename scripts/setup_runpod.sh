#!/bin/bash
# ================================================================
# RABIT-PROPFIRM DRL — RunPod Setup Script
# Run: bash scripts/setup_runpod.sh
# ================================================================

set -e

echo ""
echo "==========================================================="
echo "  RABIT-PROPFIRM DRL — RunPod Setup"
echo "==========================================================="
echo ""

# ---- 1. System info ----
echo "[1/5] System Info"
echo "  OS:    $(uname -s -r)"
echo "  GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"
echo "  CUDA:  $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'N/A')"
echo "  Python: $(python3 --version)"
echo ""

# ---- 2. Install dependencies ----
echo "[2/5] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Done! Installed: torch, polars, wandb, numpy, pytest"
echo ""

# ---- 3. Verify CUDA ----
echo "[3/5] Verifying CUDA..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    print('  [ERROR] No CUDA GPU! Check RunPod template.')
    exit(1)
"
echo ""

# ---- 4. W&B Login ----
echo "[4/5] Weights & Biases Login"
echo "  Get your API key from: https://wandb.ai/authorize"
echo ""
read -p "  Enter W&B API Key (or press Enter to skip): " WANDB_KEY

if [ -n "$WANDB_KEY" ]; then
    wandb login "$WANDB_KEY"
    echo "  W&B logged in!"
else
    echo "  Skipped. Run 'wandb login' later, or use offline mode."
    export WANDB_MODE=offline
fi
echo ""

# ---- 5. Quick sanity test ----
echo "[5/5] Running quick import test..."
cd "$(dirname "$0")/.."
PYTHONPATH=rabit_propfirm_drl python3 -c "
from agents.sac_policy import SACTransformerActor, SACTransformerCritic
from training_pipeline.per_buffer import PERBuffer
from training_pipeline.curriculum_runner import CurriculumRunner
print('  All modules imported OK!')
"
echo ""

# ---- Done! ----
echo "==========================================================="
echo "  SETUP COMPLETE!"
echo "==========================================================="
echo ""
echo "  Quick Start:"
echo "  ----------------------------------------"
echo "  # Start training (2M steps, warm-start from best_v2.pt):"
echo "  tmux new -s train"
echo "  cd $(pwd)"
echo "  python3 -u scripts/train_v2.py --steps 2000000"
echo ""
echo "  # Detach tmux: Ctrl+B, then D"
echo "  # Reattach:    tmux attach -t train"
echo "  ----------------------------------------"
echo ""
echo "  TIPS:"
echo "  - Use 'tmux' to keep training alive after closing SSH"
echo "  - Monitor GPU: watch -n1 nvidia-smi"
echo "  - W&B dashboard: https://wandb.ai/nguyenvu16992-/rabit-propfirm-drl"
echo ""
