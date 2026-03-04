#!/bin/bash
#SBATCH --job-name=xai_sketch
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=200GB

#SBATCH --output=/home1/ppatel2025/xai_project_2/logs/sketch_%j.out
#SBATCH --error=/home1/ppatel2025/xai_project_2/logs/sketch_%j.err

# =============================================================================
# XAI Project 2 - Sketch Domain Pipeline
# Train on sketch, evaluate on sketch+real, explain with sketch model
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
echo "  XAI Project 2 - Sketch Domain Pipeline"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $(hostname)"
echo "  GPUs:   $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Date:   $(date)"
echo "============================================================"

# ---- Step 1: Prepare data ----
echo ""
echo ">>> Step 1/6: Preparing data splits ..."
python prepare_data.py
echo ">>> Data preparation complete."

# ---- Step 2: Train on sketch domain ----
echo ""
echo ">>> Step 2/6: Training on SKETCH domain ..."
python train.py --domain sketch
echo ">>> Training complete."

# ---- Step 2: Evaluate sketch model on sketch test set ----
echo ""
echo ">>> Step 3/6: Evaluating sketch model on SKETCH test set ..."
python evaluate.py \
    --checkpoint output/models/sketch/best_model.pt \
    --test_domain sketch
echo ">>> Evaluation (sketch->sketch) complete."

# ---- Step 3: Evaluate sketch model on real test set ----
echo ""
echo ">>> Step 4/6: Evaluating sketch model on REAL test set ..."
python evaluate.py \
    --checkpoint output/models/sketch/best_model.pt \
    --test_domain real
echo ">>> Evaluation (sketch->real) complete."

# ---- Step 4: XAI on sketch model ----
echo ""
echo ">>> Step 5/6: Running XAI pipeline on sketch model ..."
python explain.py \
    --checkpoint output/models/sketch/best_model.pt \
    --output_dir output/xai/sketch_model
echo ">>> XAI (sketch model) complete."

# ---- Done ----
echo ""
echo "============================================================"
echo "  Sketch Domain Pipeline Complete!"
echo "  Date: $(date)"
echo "============================================================"
echo ""
echo "Output structure:"
echo "  output/models/sketch/                    - trained model + checkpoints"
echo "  output/evaluation/sketch_model_on_sketch/ - in-domain eval"
echo "  output/evaluation/sketch_model_on_real/   - cross-domain eval"
echo "  output/xai/sketch_model/                  - XAI results"
