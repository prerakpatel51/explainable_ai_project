#!/bin/bash
#SBATCH --job-name=xai_real
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=200GB
#SBATCH --output=/home1/ppatel2025/xai_project_2/logs/real_%j.out
#SBATCH --error=/home1/ppatel2025/xai_project_2/logs/real_%j.err

# =============================================================================
# XAI Project 2 - Real Domain Pipeline
# Train on real, evaluate on real+sketch, explain with real model
# =============================================================================

# Create log directory
mkdir -p /home1/ppatel2025/xai_project_2/logs

# Activate conda environment (disable nounset temporarily —
# conda hook scripts may reference unset variables)
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate xai
set -euo pipefail

# Project directory
cd /home1/ppatel2025/xai_project_2

echo "============================================================"
echo "  XAI Project 2 - Real Domain Pipeline"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $(hostname)"
echo "  GPUs:   $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Date:   $(date)"
echo "============================================================"

# ---- Step 1: Prepare data ----
echo ""
echo ">>> Step 1/6: Preparing data splits ..."
# python prepare_data.py
echo ">>> Data preparation complete."

# ---- Step 2: Train on real domain ----
echo ""
echo ">>> Step 2/6: Training on REAL domain ..."
# python train.py --domain real
echo ">>> Training complete."

# ---- Step 2: Evaluate real model on real test set ----
echo ""
echo ">>> Step 3/6: Evaluating real model on REAL test set ..."
# python evaluate.py \
#     --checkpoint output/models/real/best_model.pt \
#     --test_domain real
echo ">>> Evaluation (real->real) complete."

# ---- Step 3: Evaluate real model on sketch test set ----
echo ""
echo ">>> Step 4/6: Evaluating real model on SKETCH test set ..."
# python evaluate.py \
#     --checkpoint output/models/real/best_model.pt \
#     --test_domain sketch
echo ">>> Evaluation (real->sketch) complete."

# ---- Step 4: XAI on real model ----
echo ""
echo ">>> Step 5/6: Running XAI pipeline on real model ..."
python explain.py \
    --checkpoint output/models/real/best_model.pt \
    --output_dir output/xai/real_model
echo ">>> XAI (real model) complete."

# ---- Done ----
echo ""
echo "============================================================"
echo "  Real Domain Pipeline Complete!"
echo "  Date: $(date)"
echo "============================================================"
echo ""
echo "Output structure:"
echo "  output/models/real/              - trained model + checkpoints"
echo "  output/evaluation/real_model_on_real/   - in-domain eval"
echo "  output/evaluation/real_model_on_sketch/ - cross-domain eval"
echo "  output/xai/real_model/           - XAI results"
