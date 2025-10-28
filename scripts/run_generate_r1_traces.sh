#!/bin/bash
#SBATCH --job-name=generate_r1_traces
#SBATCH --partition=gpucluster            # Partition name
#SBATCH --time=8:00:00                   # Time limit hrs:min:sec
#SBATCH --output=generate_r1_traces_%j.out   # Standard output and error log
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task

# Check GPU status
srun --partition=gpucluster nvidia-smi

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3  # Use GPU 3 for generation

# Parse arguments
MODEL_NAME=${1:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"}
NUM_EXAMPLES=${2:-""}  # Empty = all examples

echo "="*80
echo "Generating R1-style Reasoning Traces"
echo "Model: $MODEL_NAME"
if [ -n "$NUM_EXAMPLES" ]; then
    echo "Number of examples: $NUM_EXAMPLES"
else
    echo "Processing all examples"
fi
echo "="*80

# Build command
CMD="srun --partition=gpucluster .venv/bin/python scripts/generate_r1_traces.py \
    --model_name $MODEL_NAME \
    --input_path data/gsm8k/train.jsonl \
    --output_path_correct data/gsm8k/sft_r1_correct.jsonl \
    --output_path_all data/gsm8k/sft_r1_all.jsonl \
    --prompt_template_path scripts/prompts/r1_zero.prompt \
    --use_vllm \
    --gpu_device cuda:0 \
    --temperature 0.7 \
    --max_tokens 2048"

# Add num_examples if specified
if [ -n "$NUM_EXAMPLES" ]; then
    CMD="$CMD --num_examples $NUM_EXAMPLES"
fi

# Run the command
echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "Job completed at $(date)"

