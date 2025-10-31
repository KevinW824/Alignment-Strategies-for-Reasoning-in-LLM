#!/usr/bin/env python3
"""CLI entry point for LoRA-augmented supervised fine-tuning."""

from typing import List, Literal, Optional

import typer

from scripts.train_sft_lora import LoraTrainingConfig, train


app = typer.Typer(help="LoRA-augmented SFT training.")


@app.command()
def main(
    sft_data_path: str = typer.Option("data/gsm8k/sft.jsonl", help="Path to SFT training data."),
    val_data_path: Optional[str] = typer.Option(
        "data/gsm8k/test.jsonl",
        help="Optional path to validation data. Set to None to disable eval.",
    ),
    prompt_template_path: Optional[str] = typer.Option(
        "scripts/prompts/r1_zero.prompt",
        help="Prompt template used for validation generations.",
    ),
    model_name: str = typer.Option("Qwen/Qwen2.5-Math-1.5B", help="Base model to finetune."),
    output_dir: str = typer.Option("outputs/sft-lora", help="Directory for checkpoints/adapters."),
    learning_rate: float = typer.Option(1e-4, help="Optimizer learning rate."),
    batch_size: int = typer.Option(4, help="Effective batch size (microbatch * grad_accum)."),
    microbatch_size: int = typer.Option(1, help="Microbatch size for gradient accumulation."),
    num_epochs: int = typer.Option(3, help="Number of training epochs."),
    warmup_steps: int = typer.Option(100, help="Warmup steps for the LR scheduler."),
    gradient_clip_value: float = typer.Option(1.0, help="Gradient clipping norm."),
    eval_every_n_steps: int = typer.Option(200, help="Evaluation interval (in optimizer steps)."),
    log_generations_every_n_steps: int = typer.Option(
        400,
        help="Interval for qualitative generation logging. Set 0 to disable.",
    ),
    policy_device: str = typer.Option("cuda:0", help="Device for the policy model."),
    num_train_examples: Optional[int] = typer.Option(
        None,
        help="Optional limit on number of training examples.",
    ),
    num_eval_examples: int = typer.Option(100, help="Validation examples to evaluate/log."),
    num_log_examples: int = typer.Option(3, help="Validation examples used for generation logs."),
    project_name: Optional[str] = typer.Option("sft-lora", help="wandb project name."),
    run_name: Optional[str] = typer.Option(None, help="wandb run name."),
    seed: int = typer.Option(42, help="Random seed."),
    lora_rank: int = typer.Option(16, help="Rank of the LoRA update matrices."),
    lora_alpha: int = typer.Option(32, help="Scaling factor for LoRA updates."),
    lora_dropout: float = typer.Option(0.05, help="Dropout applied to LoRA layers."),
    lora_bias: Literal["none", "all", "lora_only"] = typer.Option(
        "none",
        help="Bias handling for LoRA ('none', 'lora_only', 'all').",
    ),
    lora_target: List[str] = typer.Option(
        ["attention"],
        "--lora-target",
        "-t",
        help="High-level aliases for target modules (repeatable or comma-separated).",
    ),
    lora_modules: Optional[List[str]] = typer.Option(
        None,
        "--lora-module",
        "-m",
        help="Explicit module names to wrap with LoRA (repeatable or comma-separated).",
    ),
    use_dora: bool = typer.Option(False, help="Enable DoRA (weight decomposition) for LoRA layers."),
) -> None:
    config = LoraTrainingConfig(
        model_name=model_name,
        sft_data_path=sft_data_path,
        val_data_path=val_data_path,
        prompt_template_path=prompt_template_path,
        num_train_examples=num_train_examples,
        learning_rate=learning_rate,
        batch_size=batch_size,
        microbatch_size=microbatch_size,
        num_epochs=num_epochs,
        warmup_steps=warmup_steps,
        gradient_clip_value=gradient_clip_value,
        eval_every_n_steps=eval_every_n_steps,
        log_generations_every_n_steps=log_generations_every_n_steps,
        num_eval_examples=num_eval_examples,
        num_log_examples=num_log_examples,
        policy_device=policy_device,
        output_dir=output_dir,
        project_name=project_name,
        run_name=run_name,
        seed=seed,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_bias=lora_bias,
        lora_target=list(lora_target),
        lora_modules=list(lora_modules) if lora_modules is not None else None,
        use_dora=use_dora,
    )

    train(config)


if __name__ == "__main__":
    app()
