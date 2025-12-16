# Alignment Strategies for Reasoning in LLMs

A comprehensive implementation of alignment strategies to enhance mathematical reasoning capabilities in Large Language Models. This project implements **Supervised Fine-Tuning (SFT)**, **Group Relative Policy Optimization (GRPO)**, and **Expert Iteration** for training on the GSM8K math reasoning benchmark.

## Overview

This repository provides implementations of three key alignment strategies:

| Method | Description | Use Case |
|--------|-------------|----------|
| **SFT** | Train on reasoning traces using cross-entropy loss | Bootstrap initial reasoning capability |
| **GRPO** | Group-relative policy gradient with reward normalization | On-policy RL fine-tuning |
| **Expert Iteration** | Generate → Filter correct → Train cycle | Self-improvement through iterative refinement |

All methods target the **Qwen 2.5 Math 1.5B** model and use **vLLM** for efficient inference during training.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   SFT        │───▶│    GRPO      │───▶│   Expert     │      │
│  │ (Supervised) │    │  (On-Policy) │    │  Iteration   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Policy Model (cuda:0)                       │   │
│  │              Qwen 2.5 Math 1.5B                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│         ┌──────────────────┼──────────────────┐                │
│         ▼                  ▼                  ▼                │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│  │ vLLM Engine  │   │   Reward     │   │   W&B        │       │
│  │  (cuda:1)    │   │   Grading    │   │   Logging    │       │
│  └──────────────┘   └──────────────┘   └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.11 or 3.12
- CUDA-compatible GPU (2x GPUs recommended for parallel training/inference)

### Setup

1. **Install dependencies** (flash-attn requires special handling):
```bash
uv sync --no-install-package flash-attn
uv sync
```

2. **Run unit tests** to verify installation:
```bash
uv run pytest
```

## Project Structure

```
.
├── scripts/
│   ├── sft.py                    # SFT core functions (tokenization, loss, training step)
│   ├── grpo.py                   # GRPO core functions (reward normalization, policy gradient)
│   ├── expert_iteration.py       # Expert Iteration core functions
│   ├── train_sft.py              # Full SFT training script
│   ├── train_grpo.py             # Full GRPO training script
│   ├── train_expert_iteration.py # Full Expert Iteration training script
│   ├── drgrpo_grader.py          # Math answer grading with format validation
│   ├── math_baseline.py          # Baseline evaluation utilities
│   └── prompts/
│       └── r1_zero.prompt        # R1-Zero style prompt template
├── data/
│   └── gsm8k/
│       ├── train.jsonl           # Training questions
│       ├── test.jsonl            # Test/validation questions
│       └── sft.jsonl             # SFT dataset with reasoning traces
├── outputs/                      # Model checkpoints and results
├── tests/                        # Unit tests
└── pyproject.toml                # Project dependencies
```

## Training Methods

### 1. Supervised Fine-Tuning (SFT)

Train the model on high-quality reasoning traces using standard cross-entropy loss.

```bash
python scripts/train_sft.py \
    --sft_data_path data/gsm8k/sft.jsonl \
    --val_data_path data/gsm8k/test.jsonl \
    --learning_rate 1e-5 \
    --batch_size 4 \
    --num_epochs 3 \
    --output_dir outputs/sft
```

**Key Features:**
- Response-only loss masking (ignore prompt tokens)
- Periodic vLLM evaluation on validation set
- Gradient accumulation for larger effective batch sizes
- W&B logging with separate train/eval step metrics

### 2. Group Relative Policy Optimization (GRPO)

On-policy reinforcement learning with group-normalized rewards for stable training.

```bash
python scripts/train_grpo.py \
    --n_grpo_steps 200 \
    --rollout_batch_size 256 \
    --group_size 8 \
    --loss_type reinforce_with_baseline \
    --learning_rate 1e-5 \
    --output_dir outputs/grpo
```

**Supported Loss Types:**
| Loss Type | Description |
|-----------|-------------|
| `no_baseline` | Vanilla policy gradient with raw rewards |
| `reinforce_with_baseline` | Group-mean subtracted advantages |
| `grpo_clip` | PPO-style clipping on importance sampling ratio |

**GRPO Algorithm:**
1. Sample questions from training set
2. Generate G rollouts per question using current policy
3. Compute rewards and group-normalize (subtract mean, optionally divide by std)
4. Update policy using policy gradient loss
5. Repeat for N steps

### 3. Expert Iteration

Self-improvement through iterative generation and filtering of correct solutions.

```bash
python scripts/train_expert_iteration.py \
    --n_ei_steps 5 \
    --rollouts_per_question 8 \
    --questions_per_step 1024 \
    --sft_epochs_per_step 2 \
    --output_dir outputs/expert_iteration
```

**Expert Iteration Algorithm:**
1. Sample questions from training set
2. Generate multiple rollouts per question
3. Grade rollouts using reward function
4. Filter to keep only correct solutions
5. Train on filtered data using SFT
6. Repeat for N iterations

## Reward Function

The reward grading system (`drgrpo_grader.py`) evaluates model outputs on two criteria:

| Reward Type | Criterion | Value |
|-------------|-----------|-------|
| `format_reward` | Response uses `<think>...</think> <answer>...</answer>` format | 0 or 1 |
| `answer_reward` | Extracted answer matches ground truth | 0 or 1 |
| `reward` | Total reward (format × answer) | 0 or 1 |

The grader uses multiple strategies to verify mathematical equivalence:
- String normalization and matching
- SymPy symbolic comparison
- LaTeX parsing and evaluation
- Numeric tolerance checking

## Prompt Format

The R1-Zero style prompt template encourages step-by-step reasoning:

```
A conversation between User and Assistant. The User asks a question, 
and the Assistant solves it. The Assistant first thinks about the 
reasoning process in the mind and then provides the User with the answer. 
The reasoning process is enclosed within <think> </think> and answer is 
enclosed within <answer> </answer> tags, respectively.

User: {question}
Assistant: <think>
```

## Multi-GPU Setup

The training scripts use a dual-GPU configuration:
- **cuda:0**: Policy model (training with gradients)
- **cuda:1**: vLLM inference engine (fast rollout generation)

This separation allows efficient on-policy training without memory conflicts.

## SLURM Job Submission

For cluster environments, use the provided SLURM scripts:

```bash
# SFT Training
sbatch scripts/run_sft_slurm.sh

# GRPO Training  
sbatch scripts/run_grpo_slurm.sh

# Expert Iteration
sbatch scripts/run_expert_iteration_slurm.sh
```

## Configuration Reference

### SFT Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-5 | AdamW learning rate |
| `batch_size` | 4 | Effective batch size |
| `microbatch_size` | 1 | Gradient accumulation unit |
| `num_epochs` | 3 | Training epochs |
| `warmup_steps` | 100 | LR warmup steps |
| `gradient_clip_value` | 1.0 | Max gradient norm |

### GRPO Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_grpo_steps` | 200 | Number of GRPO iterations |
| `rollout_batch_size` | 256 | Questions per rollout batch |
| `group_size` | 8 | Rollouts per question (G) |
| `loss_type` | reinforce_with_baseline | Policy gradient variant |
| `sampling_temperature` | 1.0 | Generation temperature |
| `advantage_eps` | 1e-6 | Numerical stability epsilon |

### Expert Iteration Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_ei_steps` | 5 | Number of EI iterations |
| `rollouts_per_question` | 8 | Rollouts per question |
| `questions_per_step` | 1024 | Questions per EI step |
| `sft_epochs_per_step` | 2 | SFT epochs on filtered data |
| `temperature` | 0.7 | Generation temperature |

## Logging & Monitoring

All training scripts integrate with **Weights & Biases** for experiment tracking:

```python
# Metrics logged during training
wandb.log({
    "train/loss": loss,
    "train/learning_rate": lr,
    "eval/accuracy": accuracy,
    "eval/format_correct_rate": format_rate,
    "rollout/mean_reward": mean_reward,
})
```

## Dependencies

Core dependencies managed via `pyproject.toml`:

- `torch` - PyTorch framework
- `transformers>=4.50.0` - HuggingFace Transformers
- `vllm==0.7.2` - Fast LLM inference
- `flash-attn==2.7.4.post1` - Flash Attention
- `accelerate>=1.5.2` - Training utilities
- `wandb>=0.19.8` - Experiment tracking
- `math-verify>=0.7.0` - Mathematical answer verification
- `tqdm>=4.67.1` - Progress bars

## References

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [Understanding R1-Zero](https://github.com/sail-sg/understand-r1-zero)
- [GSM8K: Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)

## License

This project includes code adapted from:
- [understand-r1-zero](https://github.com/sail-sg/understand-r1-zero) (Apache 2.0)
- [math-verify](https://github.com/huggingface/math-verify)
