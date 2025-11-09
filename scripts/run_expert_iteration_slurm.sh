#!/bin/bash
#SBATCH --job-name=expert_iteration
#SBATCH --partition=gpucluster
#SBATCH --output=expert_iteration_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Check GPU status
srun --partition=gpucluster nvidia-smi

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2,1

# Parse command line arguments
# Usage: sbatch run_expert_iteration_slurm.sh <G> <batch_size> [epochs]
# Example: sbatch run_expert_iteration_slurm.sh 8 1024 2
G=${1:-8}
BATCH_SIZE=${2:-1024}
EPOCHS=${3:-2}

# Validate inputs
if ! [[ "$G" =~ ^[0-9]+$ ]]; then
    echo "Error: G (rollouts per question) must be a positive integer"
    echo "Usage: sbatch run_expert_iteration_slurm.sh <G> <batch_size> [epochs]"
    echo "Example: sbatch run_expert_iteration_slurm.sh 8 1024 2"
    exit 1
fi

if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]]; then
    echo "Error: batch_size (questions per step) must be a positive integer"
    echo "Usage: sbatch run_expert_iteration_slurm.sh <G> <batch_size> [epochs]"
    echo "Example: sbatch run_expert_iteration_slurm.sh 8 1024 2"
    exit 1
fi

if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]]; then
    echo "Error: epochs must be a positive integer"
    echo "Usage: sbatch run_expert_iteration_slurm.sh <G> <batch_size> [epochs]"
    echo "Example: sbatch run_expert_iteration_slurm.sh 8 1024 2"
    exit 1
fi

# Validate batch size is in {512, 1024, 2048}
if [[ "$BATCH_SIZE" != "512" && "$BATCH_SIZE" != "1024" && "$BATCH_SIZE" != "2048" ]]; then
    echo "Warning: batch_size should be one of {512, 1024, 2048}, but got $BATCH_SIZE"
    echo "Continuing anyway..."
fi

echo "=========================================="
echo "Expert Iteration Configuration:"
echo "  Rollouts per question (G): $G"
echo "  Batch size (questions per step): $BATCH_SIZE"
echo "  SFT epochs per step: $EPOCHS"
echo "  EI steps: 5"
echo "=========================================="

# Generate run name with G and batch size
RUN_NAME="ei_G${G}_B${BATCH_SIZE}_E${EPOCHS}"
OUTPUT_DIR="outputs/expert_iteration_G${G}_B${BATCH_SIZE}_E${EPOCHS}"

echo "Wandb run name: $RUN_NAME"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run Expert Iteration training
srun --partition=gpucluster .venv/bin/python scripts/train_expert_iteration.py \
    --n_ei_steps 5 \
    --rollouts_per_question "$G" \
    --questions_per_step "$BATCH_SIZE" \
    --sft_epochs_per_step "$EPOCHS" \
    --learning_rate 5e-6 \
    --batch_size 16 \
    --microbatch_size 4 \
    --temperature 0.7 \
    --eval_every_n_steps 1 \
    --num_eval_examples 500 \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --policy_device cuda:0 \
    --vllm_device cuda:1

echo "Job completed at $(date)"

