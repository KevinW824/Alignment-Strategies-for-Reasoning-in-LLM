# SFT Implementation Summary

## Overview

This document summarizes the complete SFT (Supervised Fine-Tuning) implementation for training Qwen 2.5 Math 1.5B on reasoning traces.

## What Was Implemented

### 1. Core SFT Functions (`scripts/sft.py`)

All the building blocks for SFT training:

- **`tokenize_prompt_and_output`**: Tokenizes prompt-response pairs and creates response masks
- **`compute_entropy`**: Computes per-token entropy for monitoring model confidence
- **`get_response_log_probs`**: Extracts log-probabilities from the model
- **`masked_normalize`**: Utility for masked summation and normalization
- **`sft_microbatch_train_step`**: Performs a single SFT training step with gradient accumulation
- **`log_generations`**: Generates responses and logs detailed metrics during training

### 2. Full Training Script (`scripts/train_sft.py`)

A complete, production-ready training script that:

- Loads SFT dataset with reasoning traces
- Initializes policy model on GPU 0
- Initializes vLLM for evaluation on GPU 1
- Implements the full SFT training loop with:
  - Gradient accumulation
  - Learning rate scheduling with warmup
  - Gradient clipping (value: 1.0)
  - Periodic evaluation on GSM8K validation set
  - In-the-loop generation logging
  - Checkpoint saving
  - Wandb logging with separate train/eval metrics

**Key Features**:
- Two-GPU setup (policy + vLLM)
- Efficient gradient accumulation
- Comprehensive logging
- Flexible hyperparameter configuration
- Automatic checkpoint management

### 3. Dataset Filtering Script (`scripts/filter_sft_data.py`)

Filters SFT dataset to only include correct examples:

- Loads full SFT dataset
- Extracts ground truth from responses
- Uses reward function to check correctness
- Saves filtered dataset
- Reports filtering statistics

### 4. Experiment Scripts

**Bash script** (`scripts/run_sft_experiments.sh`):
- Runs all experiments sequentially
- Covers dataset sizes: 128, 256, 512, 1024, full, and filtered

**Slurm script** (`scripts/run_sft_slurm.sh`):
- Submits jobs to cluster
- Requests 2 GPUs per job
- Supports all experiment types
- Includes proper resource allocation

### 5. Documentation

## Experiments to Run

### Required Experiments

1. **Dataset Size Ablation** (5 experiments)
   - 128 examples
   - 256 examples
   - 512 examples
   - 1024 examples
   - Full dataset

2. **Filtered Dataset** (1 experiment)
   - Train on correct examples only

### Success Criteria

- Achieve **>15% validation accuracy** on full dataset
- Generate validation accuracy curves for all dataset sizes
- Compare filtered vs full dataset performance
- Report filtered dataset size and retention rate

## How to Run

### Quick Start

```bash
# Single experiment (full dataset)
sbatch scripts/run_sft_slurm.sh full

# All experiments
sbatch scripts/run_sft_slurm.sh 128
sbatch scripts/run_sft_slurm.sh 256
sbatch scripts/run_sft_slurm.sh 512
sbatch scripts/run_sft_slurm.sh 1024
sbatch scripts/run_sft_slurm.sh full
sbatch scripts/run_sft_slurm.sh correct
```

### Monitor Progress

```bash
# Check job status
squeue -u $USER

# View logs
tail -f sft_training_<JOB_ID>.out

# View metrics
# Visit https://wandb.ai/
```

## Expected Outputs

### 1. Model Checkpoints

```
outputs/
├── sft_128/final/          # Trained model (128 examples)
├── sft_256/final/          # Trained model (256 examples)
├── sft_512/final/          # Trained model (512 examples)
├── sft_1024/final/         # Trained model (1024 examples)
├── sft_full/final/         # Trained model (full dataset)
└── sft_correct/final/      # Trained model (correct only)
```

### 2. Training Metrics (Wandb)

For each experiment:
- Training loss curves
- Validation accuracy curves
- Learning rate schedule
- Token entropy trends
- Response length statistics

### 3. Filtered Dataset

```
data/sft_correct.jsonl      # Filtered dataset (correct examples only)
```

With statistics:
- Original size
- Filtered size
- Retention rate

## Key Implementation Details

### Gradient Accumulation

The implementation correctly handles gradient accumulation:

```python
# Loss is scaled by gradient_accumulation_steps AND batch_size
loss = masked_normalize(
    tensor=-policy_log_probs,
    mask=response_mask,
    normalize_constant=normalize_constant * gradient_accumulation_steps * batch_size,
    dim=None,
)
loss.backward()  # Accumulates scaled gradients
```

This ensures that:
- Gradients accumulate correctly across microbatches
- Final gradient is the average over the full batch
- Batch size normalization is handled properly

### Two-GPU Setup

The training script uses two GPUs efficiently:

- **GPU 0 (cuda:0)**: Policy model for training
  - Forward pass
  - Backward pass
  - Gradient updates

- **GPU 1 (cuda:1)**: vLLM for evaluation
  - Fast batch generation
  - Periodic evaluation
  - No gradient computation

This allows:
- Continuous training without blocking on evaluation
- Fast evaluation with vLLM's optimized inference
- Efficient GPU utilization

### Evaluation Strategy

Evaluation happens at two levels:

1. **Fast vLLM evaluation** (every 100 steps):
   - Uses vLLM for fast batch generation
   - Evaluates on 100 validation examples
   - Computes accuracy and reward metrics

2. **Detailed generation logging** (every 200 steps):
   - Uses HF generate for detailed analysis
   - Logs 5 sample generations
   - Computes entropy and response length statistics

### Wandb Logging

Metrics are organized with separate x-axes:

```python
wandb.define_metric("train_step")
wandb.define_metric("eval_step")
wandb.define_metric("train/*", step_metric="train_step")
wandb.define_metric("eval/*", step_metric="eval_step")
```

This allows:
- Training metrics aligned with training steps
- Evaluation metrics aligned with evaluation steps
- Clean, interpretable plots in Wandb

## Hyperparameter Recommendations

### For Full Dataset (to achieve >15% accuracy)

**Baseline** (start here):
```
Learning rate: 1e-5
Batch size: 8
Microbatch size: 2
Epochs: 3
Gradient clip: 1.0
Warmup steps: 100
```

**If accuracy is low** (< 15%):
- Increase learning rate to `2e-5`
- Train for more epochs (`5-10`)
- Increase batch size to `16`

**If training is unstable**:
- Decrease learning rate to `5e-6`
- Increase warmup steps to `200`
- Reduce batch size to `4`

### For Smaller Datasets

Use more epochs to compensate for less data:
- 128 examples: 5 epochs
- 256 examples: 5 epochs
- 512 examples: 3-5 epochs
- 1024 examples: 3 epochs

## Analysis Guidelines

After running experiments, analyze:

### 1. Dataset Size vs Accuracy

Plot validation accuracy curves for all dataset sizes:
- X-axis: Training steps (or epochs)
- Y-axis: Validation accuracy
- Lines: One per dataset size

**Expected trend**: Larger datasets → higher accuracy

### 2. Filtered Dataset Analysis

Compare filtered vs full dataset:
- Report filtered dataset size
- Compare final accuracy
- Compare learning dynamics (loss curves)
- Analyze if filtering helps or hurts

**Possible outcomes**:
- **Filtered better**: Removing incorrect examples improves learning
- **Full better**: More data outweighs noise from incorrect examples
- **Similar**: Dataset size matters more than correctness

### 3. Learning Dynamics

Analyze training curves:
- Loss should decrease steadily
- Accuracy should increase (may plateau)
- Entropy should decrease (model becomes confident)
- Response lengths should stabilize

### 4. Sample Generations

Examine logged generations:
- Are responses well-formatted?
- Do responses include reasoning traces?
- Are incorrect answers shorter/longer?
- Is the model confident (low entropy) or uncertain (high entropy)?

## Troubleshooting

### Common Issues

1. **OOM Error**
   - Reduce batch size
   - Reduce sequence length
   - Use gradient checkpointing

2. **Low Accuracy**
   - Increase learning rate
   - Train longer
   - Use more data
   - Check data quality

3. **vLLM Errors**
   - Verify GPU 1 is available
   - Reduce `gpu_memory_utilization`
   - Check model path

4. **Slow Training**
   - Increase CPUs for data loading
   - Reduce evaluation frequency
   - Use larger batch size

## Next Steps

After completing SFT experiments:

1. **Analyze results**: Create plots and tables comparing all experiments
2. **Select best model**: Choose the model with best accuracy/efficiency trade-off
3. **Report findings**: Document dataset size effects and filtered dataset results
4. **Prepare for RL**: Use best SFT model as initialization for RL training

## Files Created

### Core Implementation
- `scripts/sft.py` (updated with all SFT functions)
- `scripts/train_sft.py` (full training script)
- `scripts/filter_sft_data.py` (dataset filtering)

### Experiment Scripts
- `scripts/run_sft_experiments.sh` (bash script for all experiments)
- `scripts/run_sft_slurm.sh` (Slurm submission script)

### Documentation
- `SFT_IMPLEMENTATION.md` (detailed function documentation)
- `SFT_TRAINING_GUIDE.md` (comprehensive training guide)
- `SLURM_SUBMISSION_GUIDE.md` (quick reference for job submission)
- `SFT_IMPLEMENTATION_SUMMARY.md` (this file)

## Summary

You now have a complete, production-ready SFT implementation that:

✅ Implements all required SFT functions  
✅ Provides a full training script with proper gradient accumulation  
✅ Supports two-GPU training (policy + vLLM)  
✅ Includes comprehensive logging and evaluation  
✅ Supports all required experiments (dataset sizes + filtering)  
✅ Includes detailed documentation and guides  
✅ Ready to submit to cluster via Slurm  

**To start training**: `sbatch scripts/run_sft_slurm.sh full`


