#!/bin/bash
# Script to run SFT experiments with different dataset sizes

# Experiment 1: 128 examples
python scripts/train_sft.py \
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

# Experiment 2: 256 examples
python scripts/train_sft.py \
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

# Experiment 3: 512 examples
python scripts/train_sft.py \
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

# Experiment 4: 1024 examples
python scripts/train_sft.py \
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

# Experiment 5: Full dataset
python scripts/train_sft.py \
    --learning_rate 1e-5 \
    --batch_size 8 \
    --microbatch_size 2 \
    --num_epochs 3 \
    --eval_every_n_steps 100 \
    --output_dir outputs/sft_full \
    --run_name sft_full_dataset \
    --policy_device cuda:0 \
    --vllm_device cuda:1

# Experiment 6: Filtered dataset (correct examples only)
# First filter the dataset
python scripts/filter_sft_data.py \
    --input_path data/gsm8k/sft.jsonl \
    --output_path data/gsm8k/sft_correct.jsonl

# Then train on filtered dataset
python scripts/train_sft.py \
    --sft_data_path data/gsm8k/sft_correct.jsonl \
    --learning_rate 1e-5 \
    --batch_size 8 \
    --microbatch_size 2 \
    --num_epochs 3 \
    --eval_every_n_steps 100 \
    --output_dir outputs/sft_correct \
    --run_name sft_correct_only \
    --policy_device cuda:0 \
    --vllm_device cuda:1

