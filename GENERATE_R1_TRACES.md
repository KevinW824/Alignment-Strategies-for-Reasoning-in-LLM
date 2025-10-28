# Generating R1-Style Reasoning Traces

## Overview

This guide explains how to generate high-quality reasoning traces similar to DeepSeek-R1 for your SFT dataset.

## What Are R1 Traces?

R1 (DeepSeek-R1) produces detailed, step-by-step reasoning that:
- Shows the full thought process
- Uses natural language reasoning (not just calculations)
- Self-corrects and explores multiple approaches
- Is much longer and more detailed than standard solutions

## Quick Start

### Option 1: Generate with Qwen2.5-72B-Instruct (Recommended)

```bash
# Generate traces for first 1000 examples (for testing)
sbatch scripts/run_generate_r1_traces.sh "Qwen/Qwen2.5-72B-Instruct" 1000

# Generate traces for ALL training examples (~7.5k)
sbatch scripts/run_generate_r1_traces.sh "Qwen/Qwen2.5-72B-Instruct"
```

### Option 2: Use Other Strong Reasoning Models

```bash
# Using Qwen2.5-32B-Instruct (smaller, faster)
sbatch scripts/run_generate_r1_traces.sh "Qwen/Qwen2.5-32B-Instruct" 1000

# Using DeepSeek-R1 (if you have access)
sbatch scripts/run_generate_r1_traces.sh "deepseek-ai/DeepSeek-R1" 1000
```

### Option 3: Local Testing (Interactive)

```bash
# Generate just 10 examples for testing
srun --partition=gpucluster .venv/bin/python scripts/generate_r1_traces.py \
    --model_name Qwen/Qwen2.5-32B-Instruct \
    --num_examples 10 \
    --gpu_device cuda:0
```

## Output

The script will create **TWO files**:

### 1. Correct-Only Dataset (for Standard SFT)
- **File**: `data/gsm8k/sft_r1_correct.jsonl`
- **Content**: Only traces that produce correct answers
- **Use**: Standard supervised fine-tuning

```json
{
    "prompt": "A conversation between User and Assistant...\nUser: [question]\nAssistant: <think>",
    "response": "<detailed reasoning process here> </think> <answer>42</answer>"
}
```

### 2. Full Dataset with Labels (for Expert Iteration)
- **File**: `data/gsm8k/sft_r1_all.jsonl`  
- **Content**: ALL traces (correct + incorrect) with labels
- **Use**: Expert Iteration (ExIt) experiments

```json
{
    "prompt": "A conversation between User and Assistant...\nUser: [question]\nAssistant: <think>",
    "response": "<reasoning> </think> <answer>42</answer>",
    "ground_truth": "42",
    "predicted": "42",
    "is_correct": true
}
```

**Key Difference**: The `sft_r1_all.jsonl` file includes metadata so you can:
- Filter by correctness for Expert Iteration
- Analyze what makes correct vs incorrect reasoning
- Compare training on filtered vs unfiltered data

## Quality Filtering

The script automatically:
1. ✅ Generates reasoning traces for each question
2. ✅ Extracts the predicted answer from the trace
3. ✅ Validates against ground truth
4. ✅ **Only saves traces that produce correct answers**

This ensures your SFT dataset contains only high-quality, correct reasoning examples.

## Expected Results

- **Accuracy**: Expect 70-90% of generated traces to be correct (depends on model)
- **Output size**: If 80% correct, you'll get ~6000 examples from 7473
- **Time**: 
  - 1000 examples: ~30-60 minutes
  - Full dataset: ~4-6 hours (with Qwen2.5-72B)

## After Generation

### For Standard SFT (Experiment 1)

Use the correct-only dataset:

```bash
# Backup old dataset
mv data/gsm8k/sft.jsonl data/gsm8k/sft_original.jsonl

# Use new R1 traces (correct only)
cp data/gsm8k/sft_r1_correct.jsonl data/gsm8k/sft.jsonl

# Run training as normal
sbatch scripts/run_sft_slurm.sh full
```

### For Expert Iteration (Experiment 2)

Use the full dataset with labels:

```bash
# The sft_r1_all.jsonl file contains:
# - Correct examples (70-90% of data)
# - Incorrect examples (10-30% of data)
# - Labels (is_correct field)

# You can now:
# 1. Train on ALL data (correct + incorrect)
# 2. Train on FILTERED data (correct only)
# 3. Compare the two approaches!
```

## Comparing Results

You now have **three datasets** to compare:

| Dataset | File | Reasoning Type | Correctness | Use Case |
|---------|------|----------------|-------------|----------|
| Original | `sft_original.jsonl` | GSM8K (short) | 100% | Baseline |
| R1 Correct | `sft_r1_correct.jsonl` | R1 (detailed) | 100% | SFT with R1 |
| R1 All | `sft_r1_all.jsonl` | R1 (detailed) | 70-90% | Expert Iteration |

### Experiments You Can Run:

1. **Baseline vs R1**: Original vs R1 traces (both 100% correct)
   - Shows impact of detailed reasoning

2. **Expert Iteration**: R1 all vs R1 filtered
   - Shows value of filtering incorrect examples
   - **THIS WAS NOT POSSIBLE BEFORE** (original data was 100% correct!)

3. **Dataset Size**: Vary number of examples
   - Now possible with R1 traces!

## Troubleshooting

### Model too large for GPU
```bash
# Use smaller model
sbatch scripts/run_generate_r1_traces.sh "Qwen/Qwen2.5-32B-Instruct" 1000
```

### vLLM fails
The script will automatically fall back to HuggingFace Transformers (slower but works).

### Low accuracy (<50%)
- Try a stronger model (72B or larger)
- Adjust temperature (default 0.7)
- Check if model is good at math reasoning

## Model Recommendations

| Model | Size | Speed | Quality | Memory |
|-------|------|-------|---------|--------|
| Qwen2.5-72B-Instruct | 72B | Slow | Best | ~150GB |
| Qwen2.5-32B-Instruct | 32B | Medium | Good | ~70GB |
| Qwen2.5-14B-Instruct | 14B | Fast | OK | ~30GB |

## Notes

- The generated traces will be **longer** than original GSM8K solutions
- This is expected and beneficial - R1-style reasoning is verbose
- Longer traces = better reasoning = better student model performance
- All traces are validated for correctness before being saved

