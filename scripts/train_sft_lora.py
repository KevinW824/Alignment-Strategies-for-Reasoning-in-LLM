"""
Implementation utilities for LoRA-augmented supervised fine-tuning.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Set, TYPE_CHECKING, cast

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, # type: ignore
    AutoTokenizer, # type: ignore
    PreTrainedModel, # type: ignore
    get_linear_schedule_with_warmup, # type: ignore
)
import wandb
from tqdm import tqdm

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from peft import PeftModel  # type: ignore

# Allow `python scripts/sft-lora.py` to import project modules when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.sft import (  # noqa: E402
    SFTDataset,
    compute_sft_loss,
    get_response_log_probs,
    log_generations,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)
from scripts.drgrpo_grader import r1_zero_reward_fn  # noqa: E402
from scripts.math_baseline import (  # noqa: E402
    extract_ground_truth_answer,
    format_prompts,
    load_jsonl_data,
)


__all__ = [
    "LoraTrainingConfig",
    "apply_lora_adapters",
    "resolve_lora_target_modules",
    "train",
]


LORA_TARGET_ALIASES = {
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "ffn": ["gate_proj", "up_proj", "down_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
    "attn_qkv": ["q_proj", "k_proj", "v_proj"],
    "attn_output": ["o_proj"],
    "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def _flatten_csv(values: Optional[Sequence[str]]) -> List[str]:
    """Allow both repeated options and comma-separated tokens."""
    if not values:
        return []
    flattened: List[str] = []
    for value in values:
        if not value:
            continue
        flattened.extend(tok.strip() for tok in value.split(",") if tok.strip())
    return flattened


def resolve_lora_target_modules(
    alias_names: Sequence[str],
    explicit_modules: Optional[Sequence[str]],
) -> List[str]:
    """
    Resolve high-level aliases (e.g., "attention", "ffn") into concrete module names.
    """
    resolved: Set[str] = set()

    for alias in _flatten_csv(alias_names):
        key = alias.lower()
        if key not in LORA_TARGET_ALIASES:
            available = ", ".join(sorted(LORA_TARGET_ALIASES))
            raise ValueError(f"Unknown LoRA target alias '{alias}'. Available: {available}")
        resolved.update(LORA_TARGET_ALIASES[key])

    for module_name in _flatten_csv(explicit_modules):
        resolved.add(module_name)

    if not resolved:
        raise ValueError(
            "No LoRA target modules configured. "
            "Provide --lora-target or --lora-module to select submodules."
        )

    return sorted(resolved)


def apply_lora_adapters(model: PreTrainedModel, *, config: "LoraTrainingConfig") -> PreTrainedModel:
    """Attach LoRA adapters to the model and print trainable parameter summary."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise ImportError(
            "The 'peft' package is required for LoRA training. "
            "Install it with `uv add peft`."
        ) from exc

    target_modules = resolve_lora_target_modules(config.lora_target, config.lora_modules)

    try:
        task_type = TaskType.CAUSAL_LM
    except AttributeError:  # pragma: no cover - compatibility
        task_type = "CAUSAL_LM"

    lora_config = LoraConfig(  # type: ignore[call-arg]
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        target_modules=target_modules,
        task_type=task_type,
        use_dora=config.use_dora,
    )

    peft_wrapped = get_peft_model(model, lora_config)
    peft_wrapped.print_trainable_parameters()
    return cast(PreTrainedModel, peft_wrapped)


@dataclass
class LoraTrainingConfig:
    """Configuration for LoRA-based SFT training."""

    # Model and data paths
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    sft_data_path: str = "data/gsm8k/sft.jsonl"
    val_data_path: Optional[str] = "data/gsm8k/test.jsonl"
    prompt_template_path: Optional[str] = "scripts/prompts/r1_zero.prompt"

    # Training hyperparameters
    num_train_examples: Optional[int] = None
    learning_rate: float = 1e-4
    batch_size: int = 4
    microbatch_size: int = 1
    num_epochs: int = 3
    gradient_clip_value: float = 1.0
    warmup_steps: int = 100

    # Logging / evaluation cadence
    eval_every_n_steps: int = 200
    log_generations_every_n_steps: int = 400
    num_eval_examples: int = 100
    num_log_examples: int = 3

    # Hardware
    policy_device: str = "cuda:0"

    # Saving / logging
    output_dir: str = "outputs/sft-lora"
    project_name: Optional[str] = "sft-lora"
    run_name: Optional[str] = None
    seed: int = 42

    # LoRA-specific knobs
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: Literal["none", "all", "lora_only"] = "none"
    lora_target: List[str] = field(default_factory=lambda: ["attention"])
    lora_modules: Optional[List[str]] = None
    use_dora: bool = False


def build_dataloader(
    dataset: SFTDataset,
    microbatch_size: int,
    *,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader that yields dictionaries of prompts/responses."""
    return DataLoader(dataset, batch_size=microbatch_size, shuffle=shuffle)


def evaluate_validation_loss(
    model: PreTrainedModel,
    tokenizer,
    dataset: SFTDataset,
    device: torch.device,
    *,
    max_examples: Optional[int] = None,
) -> float:
    """Compute the average response-token loss over the validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0.0

    with torch.no_grad():
        total_count = len(dataset)
        limit = total_count if max_examples is None else min(total_count, max_examples)

        for idx in range(limit):
            example = dataset[idx]

            tokenized = tokenize_prompt_and_output(
                [example["prompt"]],
                [example["response"]],
                tokenizer,
            )

            input_ids = tokenized["input_ids"].to(device)
            labels = tokenized["labels"].to(device)
            response_mask = tokenized["response_mask"].to(device)

            loss = compute_sft_loss(model, input_ids, labels, response_mask)
            response_tokens = response_mask.sum().item()
            total_loss += loss.item() * response_tokens
            total_tokens += response_tokens

    model.train()
    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens


def maybe_log_generations(
    model: PreTrainedModel,
    tokenizer,
    prompts: List[str],
    ground_truths: List[str],
    *,
    max_examples: int,
    step: int,
) -> None:
    """Generate a few samples for qualitative inspection and log to stdout / wandb."""
    if not prompts:
        return

    model.eval()
    log_data = log_generations(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts[:max_examples],
        ground_truths=ground_truths[:max_examples],
        reward_fn=r1_zero_reward_fn,
        max_examples=max_examples,
    )
    model.train()

    print(f"\nSample generations at step {step}:")
    for idx, sample in enumerate(log_data["examples"]):
        print(f"Example {idx + 1}:")
        print(f"  Prompt: {sample['prompt'][:80]}...")
        print(f"  Response: {sample['response'][:160]}...")
        print(f"  Rewards: format={sample['format_reward']}, answer={sample['answer_reward']}")
        print()

    wandb.log(
        {
            "generations/avg_reward": log_data["metrics"].get("avg_reward"),
            "generations/avg_answer_reward": log_data["metrics"].get("avg_answer_reward"),
            "generations/avg_format_reward": log_data["metrics"].get("avg_format_reward"),
            "train_step": step,
        }
    )


def train(config: LoraTrainingConfig) -> None:
    """Full training loop with LoRA adapters applied to the policy model."""
    torch.manual_seed(config.seed)

    os.makedirs(config.output_dir, exist_ok=True)

    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config=vars(config),
    )
    wandb.define_metric("train_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("generations/*", step_metric="train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    print(f"Loading tokenizer from {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(config.policy_device)
    print(f"Loading base model on {config.policy_device}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.train()

    model = apply_lora_adapters(model, config=config)

    train_dataset = SFTDataset(config.sft_data_path, config.num_train_examples)
    print(f"Loaded {len(train_dataset)} SFT examples from {config.sft_data_path}")

    eval_dataset: Optional[SFTDataset]
    if config.num_eval_examples is None or config.num_eval_examples <= 0:
        eval_dataset = None
        print("No eval dataset configured; skipping loss evaluation.")
    else:
        eval_dataset = SFTDataset(config.sft_data_path, config.num_eval_examples)
        print(f"Using {len(eval_dataset)} examples for loss evaluation.")

    val_prompts: List[str] = []
    val_answers: List[str] = []

    if config.val_data_path and config.prompt_template_path:
        print(f"Loading validation data from {config.val_data_path}...")
        raw_val_examples = load_jsonl_data(config.val_data_path)[: config.num_eval_examples]
        val_answers = [extract_ground_truth_answer(ex["answer"]) for ex in raw_val_examples]
        prompt_template = Path(config.prompt_template_path).read_text(encoding="utf-8")
        val_prompts = format_prompts(raw_val_examples, prompt_template)
        print(f"Prepared {len(val_prompts)} validation prompts for qualitative evaluation.")
    else:
        print("Validation data not provided; skipping eval/log generations based on validation set.")

    gradient_accumulation_steps = config.batch_size // config.microbatch_size
    if gradient_accumulation_steps < 1:
        raise ValueError("microbatch_size must divide batch_size (or be equal) for gradient accumulation.")

    steps_per_epoch = math.ceil(len(train_dataset) / config.batch_size)
    total_steps = steps_per_epoch * config.num_epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )

    resolved_targets = resolve_lora_target_modules(config.lora_target, config.lora_modules)
    print("\nLoRA configuration:")
    print(f"  Target modules: {resolved_targets}")
    print(
        "  "
        f"Rank: {config.lora_rank}, Alpha: {config.lora_alpha}, Dropout: {config.lora_dropout}, "
        f"Bias: {config.lora_bias}, Use DoRA: {config.use_dora}"
    )

    print("\nTraining configuration:")
    print(f"  Total train examples: {len(train_dataset)}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Microbatch size: {config.microbatch_size}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total optimizer steps: {total_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Warmup steps: {config.warmup_steps}")

    global_step = 0
    eval_step = 0

    for epoch in range(config.num_epochs):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'=' * 80}")
        dataloader = build_dataloader(train_dataset, config.microbatch_size, shuffle=True)

        running_loss = 0.0
        optimizer.zero_grad()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False)
        epoch_loss_total = 0.0
        optimizer_steps = 0

        for batch_idx, batch in enumerate(progress_bar):
            prompts: List[str] = batch["prompt"]
            responses: List[str] = batch["response"]

            tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
            input_ids = tokenized["input_ids"].to(device)
            labels = tokenized["labels"].to(device)
            response_mask = tokenized["response_mask"].to(device)

            outputs = get_response_log_probs(model, input_ids, labels)
            _, metadata = sft_microbatch_train_step(
                policy_log_probs=outputs["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                normalize_constant=1.0,
            )

            running_loss += metadata["unscaled_loss"].item()

            should_step = (batch_idx + 1) % gradient_accumulation_steps == 0
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_value)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                avg_loss = running_loss / gradient_accumulation_steps
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train_step": global_step,
                    }
                )
                running_loss = 0.0
                epoch_loss_total += avg_loss
                optimizer_steps += 1
                progress_bar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

                if config.log_generations_every_n_steps > 0 and global_step % config.log_generations_every_n_steps == 0:
                    maybe_log_generations(
                        model,
                        tokenizer,
                        val_prompts,
                        val_answers,
                        max_examples=config.num_log_examples,
                        step=global_step,
                    )

                if (
                    eval_dataset is not None
                    and config.eval_every_n_steps > 0
                    and global_step % config.eval_every_n_steps == 0
                ):
                    print(f"\nRunning validation at step {global_step}...")
                    val_loss = evaluate_validation_loss(
                        model=model,
                        tokenizer=tokenizer,
                        dataset=eval_dataset,
                        device=device,
                        max_examples=config.num_eval_examples,
                    )
                    eval_step += 1
                    wandb.log({"eval/loss": val_loss, "eval_step": eval_step})
                    print(f"Validation loss: {val_loss:.4f}")

        progress_bar.close()

        if eval_dataset is not None:
            val_loss = evaluate_validation_loss(
                model=model,
                tokenizer=tokenizer,
                dataset=eval_dataset,
                device=device,
                max_examples=config.num_eval_examples,
            )
            eval_step += 1
            wandb.log({"eval/loss": val_loss, "eval_step": eval_step})
            print(f"[Epoch {epoch + 1}] Validation loss: {val_loss:.4f}")

        if optimizer_steps > 0:
            epoch_avg_loss = epoch_loss_total / optimizer_steps
            print(f"Epoch {epoch + 1} completed. Average loss: {epoch_avg_loss:.4f}")

    print(f"\nSaving LoRA adapter to {config.output_dir} ...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    wandb.finish()
    print("Training complete.")
