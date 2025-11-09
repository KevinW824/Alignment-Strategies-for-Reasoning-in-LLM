#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --partition=gpucluster
#SBATCH --output=grpo_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Check GPU status
srun --partition=gpucluster nvidia-smi

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2,1

# Parse command line arguments
# Usage: sbatch run_grpo_slurm.sh [group_size] [rollout_batch_size] [loss_type] [n_steps]
# Example: sbatch run_grpo_slurm.sh 8 128 reinforce_with_baseline 200
GROUP_SIZE=${1:-8}
ROLLOUT_BATCH_SIZE=${2:-128}  # Optimized default: reduced from 256
LOSS_TYPE=${3:-reinforce_with_baseline}
N_STEPS=${4:-200}

# Validate inputs
if ! [[ "$GROUP_SIZE" =~ ^[0-9]+$ ]]; then
    echo "Error: group_size must be a positive integer"
    echo "Usage: sbatch run_grpo_slurm.sh [group_size] [rollout_batch_size] [loss_type] [n_steps]"
    echo "Example: sbatch run_grpo_slurm.sh 8 256 reinforce_with_baseline 200"
    exit 1
fi

if ! [[ "$ROLLOUT_BATCH_SIZE" =~ ^[0-9]+$ ]]; then
    echo "Error: rollout_batch_size must be a positive integer"
    echo "Usage: sbatch run_grpo_slurm.sh [group_size] [rollout_batch_size] [loss_type] [n_steps]"
    echo "Example: sbatch run_grpo_slurm.sh 8 256 reinforce_with_baseline 200"
    exit 1
fi

if [[ "$LOSS_TYPE" != "no_baseline" && "$LOSS_TYPE" != "reinforce_with_baseline" && "$LOSS_TYPE" != "grpo_clip" ]]; then
    echo "Error: loss_type must be one of: no_baseline, reinforce_with_baseline, grpo_clip"
    echo "Usage: sbatch run_grpo_slurm.sh [group_size] [rollout_batch_size] [loss_type] [n_steps]"
    echo "Example: sbatch run_grpo_slurm.sh 8 256 reinforce_with_baseline 200"
    exit 1
fi

if ! [[ "$N_STEPS" =~ ^[0-9]+$ ]]; then
    echo "Error: n_steps must be a positive integer"
    echo "Usage: sbatch run_grpo_slurm.sh [group_size] [rollout_batch_size] [loss_type] [n_steps]"
    echo "Example: sbatch run_grpo_slurm.sh 8 256 reinforce_with_baseline 200"
    exit 1
fi

echo "=========================================="
echo "GRPO Training Configuration (Optimized for A100):"
echo "  Group size (G): $GROUP_SIZE"
echo "  Rollout batch size: $ROLLOUT_BATCH_SIZE"
echo "  Loss type: $LOSS_TYPE"
echo "  GRPO steps: $N_STEPS"
echo "  Learning rate: 1e-5"
echo "  Train batch size: 128 (optimized)"
echo "  Gradient accumulation steps: 64 (optimized)"
echo "  Max tokens: 512 (optimized)"
echo "  Eval every: 20 steps (optimized)"
echo "  Use std normalization: True"
echo "=========================================="

# Generate run name with key hyperparameters
RUN_NAME="grpo_G${GROUP_SIZE}_B${ROLLOUT_BATCH_SIZE}_${LOSS_TYPE}_S${N_STEPS}"
OUTPUT_DIR="outputs/grpo_G${GROUP_SIZE}_B${ROLLOUT_BATCH_SIZE}_${LOSS_TYPE}_S${N_STEPS}"

echo "Wandb run name: $RUN_NAME"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run GRPO training with optimized settings for A100
srun --partition=gpucluster .venv/bin/python scripts/train_grpo.py \
    --n_grpo_steps "$N_STEPS" \
    --rollout_batch_size "$ROLLOUT_BATCH_SIZE" \
    --group_size "$GROUP_SIZE" \
    --epochs_per_rollout_batch 1 \
    --train_batch_size 128 \
    --gradient_accumulation_steps 64 \
    --learning_rate 1e-5 \
    --advantage_eps 1e-6 \
    --use_std_normalization \
    --loss_type "$LOSS_TYPE" \
    --sampling_temperature 1.0 \
    --sampling_min_tokens 4 \
    --sampling_max_tokens 512 \
    --eval_every_n_steps 20 \
    --num_eval_examples 200 \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --policy_device cuda:0 \
    --vllm_device cuda:1 \
    --gpu_memory_utilization 0.85 \
    --save_every_n_steps 100

echo "Job completed at $(date)"

