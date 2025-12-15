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
export CUDA_VISIBLE_DEVICES=3,1

# Parse command line arguments
# Usage: sbatch run_grpo_slurm.sh [group_size] [rollout_batch_size] [loss_type] [n_steps]
# Example: sbatch run_grpo_slurm.sh 8 128 reinforce_with_baseline 200
GROUP_SIZE=${1:-8}
ROLLOUT_BATCH_SIZE=${2:-128}  # Default: 128
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

# New hyperparameters
EPOCHS_PER_ROLLOUT_BATCH=1
GRADIENT_ACCUMULATION_STEPS=32
TRAIN_BATCH_SIZE=128
LEARNING_RATE=0.00001
SAMPLING_TEMPERATURE=1.0

echo "=========================================="
echo "GRPO Training Configuration:"
echo "  Group size (G): $GROUP_SIZE"
echo "  Rollout batch size: $ROLLOUT_BATCH_SIZE"
echo "  Loss type: $LOSS_TYPE"
echo "  GRPO steps: $N_STEPS"
echo "  Epochs per rollout batch: $EPOCHS_PER_ROLLOUT_BATCH"
echo "  Train batch size: $TRAIN_BATCH_SIZE"
echo "  Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Sampling temperature: $SAMPLING_TEMPERATURE"
echo "  Max tokens: 512"
echo "  Eval every: 20 steps"
echo "  Use std normalization: True"
echo "=========================================="

# Generate run name with key hyperparameters
RUN_NAME="grpo_G${GROUP_SIZE}_B${ROLLOUT_BATCH_SIZE}_E${EPOCHS_PER_ROLLOUT_BATCH}_BS${TRAIN_BATCH_SIZE}_GA${GRADIENT_ACCUMULATION_STEPS}_LR${LEARNING_RATE}_T${SAMPLING_TEMPERATURE}_${LOSS_TYPE}_S${N_STEPS}"
OUTPUT_DIR="outputs/grpo_G${GROUP_SIZE}_B${ROLLOUT_BATCH_SIZE}_E${EPOCHS_PER_ROLLOUT_BATCH}_BS${TRAIN_BATCH_SIZE}_GA${GRADIENT_ACCUMULATION_STEPS}_LR${LEARNING_RATE}_T${SAMPLING_TEMPERATURE}_${LOSS_TYPE}_S${N_STEPS}"

echo "Wandb run name: $RUN_NAME"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run GRPO training with new hyperparameters
srun --partition=gpucluster .venv/bin/python scripts/train_grpo.py \
    --n_grpo_steps "$N_STEPS" \
    --rollout_batch_size "$ROLLOUT_BATCH_SIZE" \
    --group_size "$GROUP_SIZE" \
    --epochs_per_rollout_batch "$EPOCHS_PER_ROLLOUT_BATCH" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --advantage_eps 1e-6 \
    --use_std_normalization \
    --loss_type "$LOSS_TYPE" \
    --sampling_temperature "$SAMPLING_TEMPERATURE" \
    --sampling_min_tokens 4 \
    --sampling_max_tokens 512 \
    --eval_every_n_steps 25 \
    --num_eval_examples 1319 \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --policy_device cuda:0 \
    --vllm_device cuda:1 \
    --gpu_memory_utilization 0.70 \
    --save_every_n_steps 100

echo "Job completed at $(date)"

