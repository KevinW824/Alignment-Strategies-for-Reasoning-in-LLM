# Alignment Strategies for Reasoning in LLMs

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20|%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/🤗%20Transformers-4.50+-FFD21E?style=for-the-badge" alt="Transformers">
  <img src="https://img.shields.io/badge/vLLM-0.7.2-00ADD8?style=for-the-badge" alt="vLLM">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PEFT-LoRA-9cf?style=flat-square&logo=huggingface" alt="PEFT">
  <img src="https://img.shields.io/badge/Flash%20Attention-2.7-orange?style=flat-square" alt="Flash Attention">
  <img src="https://img.shields.io/badge/W%26B-Logging-FFCC33?style=flat-square&logo=weightsandbiases&logoColor=black" alt="Weights & Biases">
  <img src="https://img.shields.io/badge/CUDA-Multi--GPU-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square" alt="License">
</p>

<p align="center">
  <b>SFT</b> · <b>LoRA</b> · <b>GRPO</b> · <b>Expert Iteration</b> · <b>Representation Engineering</b>
</p>

---

A comprehensive implementation of alignment strategies to enhance mathematical reasoning capabilities in Large Language Models. This project implements **Supervised Fine-Tuning (SFT)**, **LoRA Fine-Tuning**, **Group Relative Policy Optimization (GRPO)**, **Expert Iteration**, and **Representation Engineering** for training and steering on the GSM8K math reasoning benchmark.

## Overview

This repository provides implementations of five key alignment strategies:

| Method | Description | Use Case |
|--------|-------------|----------|
| **SFT** | Train on reasoning traces using cross-entropy loss | Bootstrap initial reasoning capability |
| **LoRA** | Parameter-efficient fine-tuning with low-rank adapters | Memory-efficient training, rapid experimentation |
| **GRPO** | Group-relative policy gradient with reward normalization | On-policy RL fine-tuning |
| **Expert Iteration** | Generate → Filter correct → Train cycle | Self-improvement through iterative refinement |
| **Representation Engineering** | Activation steering via control vectors | Inference-time behavior modification |

All methods target the **Qwen 2.5 Math 1.5B** model and use **vLLM** for efficient inference during training.

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         Training & Inference Pipeline                     │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────── Training ───────────────────────────┐   │
│  │                                                                    │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │   │
│  │  │    SFT     │  │   LoRA     │  │   GRPO     │  │  Expert    │    │   │
│  │  │ (Full FT)  │  │ (PEFT)     │  │ (On-Policy)│  │ Iteration  │    │   │
│  │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘    │   │
│  │        └───────────────┴───────────────┴───────────────┘           │   │
│  │                                │                                   │   │
│  │                                ▼                                   │   │
│  │              ┌─────────────────────────────────┐                   │   │
│  │              │     Policy Model (cuda:0)       │                   │   │
│  │              │     Qwen 2.5 Math 1.5B          │                   │   │
│  │              └─────────────────────────────────┘                   │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  ┌────────────────────── Inference-Time Steering ─────────────────────┐   │
│  │                                                                    │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │              Representation Engineering                     │   │   │
│  │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │   │   │
│  │  │   │ Contrastive  │───▶│  PCA Control │───▶│   Inject at  │  │   │   │
│  │  │   │   H+ - H-    │    │   Vectors    │    │  Layer L     │  │   │   │
│  │  │   └──────────────┘    └──────────────┘    └──────────────┘  │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  ┌─────────────────────────── Shared Components ──────────────────────┐   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │   │
│  │  │ vLLM Engine  │   │   Reward     │   │   W&B        │            │   │
│  │  │  (cuda:1)    │   │   Grading    │   │   Logging    │            │   │
│  │  └──────────────┘   └──────────────┘   └──────────────┘            │   │
│  └────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────┘
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
│   ├── train_lora.py             # LoRA fine-tuning script
│   ├── sweep_lora_train.py       # LoRA hyperparameter sweep
│   ├── sweep_lora_eval.py        # LoRA evaluation sweep
│   ├── drgrpo_grader.py          # Math answer grading with format validation
│   ├── math_baseline.py          # Baseline evaluation utilities
│   └── prompts/
│       └── r1_zero.prompt        # R1-Zero style prompt template
├── re/                           # Representation Engineering module
│   ├── src/
│   │   ├── train_re.py           # Control vector extraction via contrastive PCA
│   │   ├── vector_injector.py    # Runtime control vector injection
│   │   ├── validate.py           # Alpha sweep validation
│   │   ├── dataset.py            # Dataset utilities
│   │   └── create_negative_set.py # Generate contrastive negative examples
│   ├── outputs/                  # Saved control vectors and results
│   └── data/                     # Prompts for different models
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

### 2. LoRA Fine-Tuning

Parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA) via the PEFT library.

```bash
python scripts/train_lora.py \
    --sft_data_path data/gsm8k/sft.jsonl \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target attention \
    --num_epochs 3 \
    --output_dir outputs/sft-lora
```

**Target Module Aliases:**
| Alias | Modules |
|-------|---------|
| `attention` | q_proj, k_proj, v_proj, o_proj |
| `ffn` / `mlp` | gate_proj, up_proj, down_proj |
| `attn_qkv` | q_proj, k_proj, v_proj |
| `all` | All attention + FFN modules |

**Key Features:**
- Configurable rank, alpha, and dropout
- Target module selection via aliases or explicit names
- DoRA (Weight-Decomposed LoRA) support
- Hyperparameter sweep utilities (`sweep_lora_train.py`)

**Hyperparameter Sweep:**
```bash
# Run sweep across multiple LoRA configurations
python scripts/sweep_lora_train.py

# Evaluate all sweep checkpoints
python scripts/sweep_lora_eval.py
```

### 3. Group Relative Policy Optimization (GRPO)

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

### 4. Expert Iteration

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

### 5. Representation Engineering

Inference-time activation steering using contrastive control vectors extracted via PCA.

#### Step 1: Extract Control Vectors

```bash
cd re/src
python train_re.py
```

This extracts control vectors by:
1. Computing hidden states for positive examples (correct reasoning traces)
2. Computing hidden states for negative examples (incorrect/malformed traces)
3. Computing contrastive difference: `H+ - H-`
4. Running PCA to extract principal direction
5. Scaling vectors to match activation norms

#### Step 2: Validate with Alpha Sweep

```bash
python validate.py
```

This evaluates accuracy across different injection strengths (α values) to find the optimal steering coefficient.

**Control Vector Injection:**
```python
from vector_injector import ControlVectorInjector

# Load control vectors
control_vectors = torch.load("outputs/re/contrastive_pca_vectors.pth")

# Create injector (injects at specified layers)
injector = ControlVectorInjector(
    model=model,
    control_vectors=control_vectors,
    alpha=0.5,           # Injection strength
    layers=[14],         # Target layer(s)
)

# Generate with steering
output = model.generate(**inputs)

# Remove hooks when done
injector.remove()
```

**Key Features:**
- Contrastive PCA for direction extraction
- Layer-specific injection via forward hooks
- Adjustable injection strength (α)
- No additional training required—works at inference time

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

# LoRA Sweep
sbatch scripts/run_sweep_lora_slurm.sh

# Representation Engineering Validation
sbatch re/run_validations_slurm.sh
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

### LoRA Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_rank` | 16 | Rank of LoRA update matrices |
| `lora_alpha` | 32 | Scaling factor for LoRA updates |
| `lora_dropout` | 0.05 | Dropout applied to LoRA layers |
| `lora_bias` | none | Bias handling (none/all/lora_only) |
| `lora_target` | attention | Target module alias |
| `use_dora` | False | Enable DoRA (weight decomposition) |

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

### Representation Engineering Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_start` | -1.0 | Start of alpha sweep range |
| `alpha_end` | 1.0 | End of alpha sweep range |
| `alpha_step` | 0.1 | Alpha increment step |
| `injection_layer` | None | Layer for injection (None = middle layer) |
| `batch_size` | 16 | Batch size for validation |

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
- `peft` - Parameter-Efficient Fine-Tuning (LoRA)
- `vllm==0.7.2` - Fast LLM inference
- `flash-attn==2.7.4.post1` - Flash Attention
- `accelerate>=1.5.2` - Training utilities
- `wandb>=0.19.8` - Experiment tracking
- `math-verify>=0.7.0` - Mathematical answer verification
- `scikit-learn` - PCA for Representation Engineering
- `tqdm>=4.67.1` - Progress bars

## References

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [Understanding R1-Zero](https://github.com/sail-sg/understand-r1-zero)
- [GSM8K: Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)

## License

This project includes code adapted from:
- [understand-r1-zero](https://github.com/sail-sg/understand-r1-zero) (Apache 2.0)
- [math-verify](https://github.com/huggingface/math-verify)
