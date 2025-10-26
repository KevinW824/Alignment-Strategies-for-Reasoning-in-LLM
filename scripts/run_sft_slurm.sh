#!/bin/bash
#SBATCH --job-name=sft_training
#SBATCH --partition=gpucluster
#SBATCH --output=sft_training_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Check GPU status
srun --partition=gpucluster nvidia-smi

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2,3  # Use physical GPUs 2 and 3

# Parse command line arguments for experiment type
EXPERIMENT_TYPE=${1:-"full"}

echo "Running SFT experiment: $EXPERIMENT_TYPE"

case $EXPERIMENT_TYPE in
    "128")
        echo "Training with 128 examples..."
        srun --partition=gpucluster .venv/bin/python scripts/train_sft.py \
            --num_train_examples 128 \
            --learning_rate 1e-5 \
            --batch_size 4 \
            --microbatch_size 1 \
            --num_epochs 5 \
            --eval_every_n_steps 50 \
            --output_dir outputs/sft_128 \
            --run_name sft_128_examples \
            --policy_device cuda:0 \
            --vllm_device cuda:1
        ;;
    
    "256")
        echo "Training with 256 examples..."
        srun --partition=gpucluster .venv/bin/python scripts/train_sft.py \
            --num_train_examples 256 \
            --learning_rate 1e-5 \
            --batch_size 4 \
            --microbatch_size 1 \
            --num_epochs 5 \
            --eval_every_n_steps 50 \
            --output_dir outputs/sft_256 \
            --run_name sft_256_examples \
            --policy_device cuda:0 \
            --vllm_device cuda:1
        ;;
    
    "512")
        echo "Training with 512 examples..."
        srun --partition=gpucluster .venv/bin/python scripts/train_sft.py \
            --num_train_examples 512 \
            --learning_rate 1e-5 \
            --batch_size 4 \
            --microbatch_size 1 \
            --num_epochs 3 \
            --eval_every_n_steps 100 \
            --output_dir outputs/sft_512 \
            --run_name sft_512_examples \
            --policy_device cuda:0 \
            --vllm_device cuda:1
        ;;
    
    "1024")
        echo "Training with 1024 examples..."
        srun --partition=gpucluster .venv/bin/python scripts/train_sft.py \
            --num_train_examples 1024 \
            --learning_rate 1e-5 \
            --batch_size 4 \
            --microbatch_size 1 \
            --num_epochs 3 \
            --eval_every_n_steps 100 \
            --output_dir outputs/sft_1024 \
            --run_name sft_1024_examples \
            --policy_device cuda:0 \
            --vllm_device cuda:1
        ;;
    
    "full")
        echo "Training with full dataset..."
        srun --partition=gpucluster .venv/bin/python scripts/train_sft.py \
            --learning_rate 1e-5 \
            --batch_size 8 \
            --microbatch_size 2 \
            --num_epochs 3 \
            --eval_every_n_steps 100 \
            --output_dir outputs/sft_full \
            --run_name sft_full_dataset \
            --policy_device cuda:0 \
            --vllm_device cuda:1
        ;;
    
    *)
        echo "Unknown experiment type: $EXPERIMENT_TYPE"
        echo "Valid options: 128, 256, 512, 1024, full"
        exit 1
        ;;
esac

echo "Job completed at $(date)"

