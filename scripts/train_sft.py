#!/usr/bin/env python3
"""
Full SFT Training Script for Qwen 2.5 Math 1.5B on reasoning traces.

This script implements Algorithm 1 from the assignment:
1. Loads SFT dataset with reasoning traces
2. Finetunes the model using cross-entropy loss on responses
3. Periodically evaluates on GSM8K validation set using vLLM
4. Saves checkpoints and logs metrics to wandb
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from unittest.mock import patch
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import wandb
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our implementations
from scripts.sft import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    log_generations,
)
from scripts.drgrpo_grader import r1_zero_reward_fn
from scripts.math_baseline import load_jsonl_data, extract_ground_truth_answer


@dataclass
class TrainingConfig:
    """Configuration for SFT training."""
    # Model and data
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    sft_data_path: str = "data/gsm8k/sft.jsonl"
    val_data_path: str = "data/gsm8k/test.jsonl"
    prompt_template_path: str = "scripts/prompts/r1_zero.prompt"
    
    # Training hyperparameters
    num_train_examples: Optional[int] = None  # None = use all
    learning_rate: float = 1e-5
    batch_size: int = 4
    microbatch_size: int = 1
    num_epochs: int = 3
    max_seq_length: int = 2048
    gradient_clip_value: float = 1.0
    warmup_steps: int = 100
    
    # Evaluation
    eval_every_n_steps: int = 100
    log_generations_every_n_steps: int = 200
    num_eval_examples: int = 100
    num_log_examples: int = 5
    
    # vLLM settings
    vllm_device: str = "cuda:1"
    policy_device: str = "cuda:0"
    gpu_memory_utilization: float = 0.85
    
    # Logging and saving
    output_dir: str = "outputs/sft"
    save_every_n_steps: int = 500
    project_name: str = "sft-qwen-math"
    run_name: Optional[str] = None
    seed: int = 42


class SFTDataset(Dataset):
    """Dataset for SFT training with prompt-response pairs."""
    
    def __init__(self, data_path: str, max_examples: Optional[int] = None):
        """Load SFT dataset from JSONL file."""
        self.examples = []
        with open(data_path, 'r') as f:
            for line in f:
                self.examples.append(json.loads(line))
        
        if max_examples is not None:
            self.examples = self.examples[:max_examples]
        
        print(f"Loaded {len(self.examples)} SFT examples from {data_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process using vLLM on a separate GPU.
    
    Args:
        model_id: Model identifier or path
        device: Device to place vLLM model on (e.g., "cuda:1")
        seed: Random seed
        gpu_memory_utilization: GPU memory utilization for vLLM
    
    Returns:
        LLM instance
    """
    vllm_set_random_seed(seed)
    
    # Monkeypatch from TRL to make vLLM work on specific device
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Load policy weights into vLLM instance for evaluation.
    
    Copied from TRL: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_on_gsm8k(
    llm: LLM,
    prompts: List[str],
    ground_truths: List[str],
    sampling_params: SamplingParams,
) -> Dict[str, float]:
    """
    Evaluate model on GSM8K using vLLM.
    
    Args:
        llm: vLLM instance
        prompts: List of formatted prompts
        ground_truths: List of ground truth answers
        sampling_params: Sampling parameters for generation
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Generate responses
    outputs = llm.generate(prompts, sampling_params)
    
    # Compute rewards
    all_rewards = []
    for output, ground_truth in zip(outputs, ground_truths):
        response = output.outputs[0].text
        reward_dict = r1_zero_reward_fn(response, ground_truth)
        all_rewards.append(reward_dict)
    
    # Aggregate metrics
    num_examples = len(all_rewards)
    metrics = {
        "accuracy": sum(r["answer_reward"] for r in all_rewards) / num_examples,
        "format_correct_rate": sum(r["format_reward"] for r in all_rewards) / num_examples,
        "avg_total_reward": sum(r["reward"] for r in all_rewards) / num_examples,
        "num_examples": num_examples,
    }
    
    return metrics


def train_sft(config: TrainingConfig):
    """Main SFT training loop."""
    
    # Set random seeds
    torch.manual_seed(config.seed)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config=vars(config),
        mode="offline",  # Use offline mode (no API key needed)
    )
    
    # Setup wandb metrics with separate x-axes for train and eval
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    
    # Load tokenizer
    print(f"Loading tokenizer from {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load policy model
    print(f"Loading policy model on {config.policy_device}...")
    policy = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(config.policy_device)
    policy.train()
    
    # Load SFT dataset
    train_dataset = SFTDataset(config.sft_data_path, config.num_train_examples)
    
    # Load validation data
    print(f"Loading validation data from {config.val_data_path}...")
    val_examples = load_jsonl_data(config.val_data_path)[:config.num_eval_examples]
    val_ground_truths = [extract_ground_truth_answer(ex["answer"]) for ex in val_examples]
    
    # Load prompt template
    with open(config.prompt_template_path, 'r') as f:
        prompt_template = f.read()
    val_prompts = [prompt_template.replace("{question}", ex["question"]) for ex in val_examples]
    
    # Initialize vLLM for evaluation
    print(f"Initializing vLLM on {config.vllm_device}...")
    vllm_model = init_vllm(
        model_id=config.model_name,
        device=config.vllm_device,
        seed=config.seed,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )
    
    # Sampling parameters for evaluation
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)
    
    # Calculate gradient accumulation steps
    gradient_accumulation_steps = config.batch_size // config.microbatch_size
    
    # Calculate total training steps
    steps_per_epoch = len(train_dataset) // config.batch_size
    total_steps = steps_per_epoch * config.num_epochs
    
    # Setup learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Total examples: {len(train_dataset)}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Microbatch size: {config.microbatch_size}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Warmup steps: {config.warmup_steps}\n")
    
    # Training loop
    global_step = 0
    eval_step = 0
    
    for epoch in range(config.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'='*80}\n")
        
        # Create dataloader
        dataloader = DataLoader(
            train_dataset,
            batch_size=config.microbatch_size,
            shuffle=True,
        )
        
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Tokenize batch
            prompts = batch["prompt"]
            responses = batch["response"]
            
            tokenized = tokenize_prompt_and_output(
                prompt_strs=prompts,
                output_strs=responses,
                tokenizer=tokenizer,
            )
            
            # Move to device
            input_ids = tokenized['input_ids'].to(config.policy_device)
            labels = tokenized['labels'].to(config.policy_device)
            response_mask = tokenized['response_mask'].to(config.policy_device)
            
            # Get log probabilities
            outputs = get_response_log_probs(
                model=policy,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True,
            )
            
            # Microbatch train step
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=outputs['log_probs'],
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                normalize_constant=1.0,
            )
            
            epoch_loss += loss.item()
            
            # Optimizer step after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip_value)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Log training metrics
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train_step": global_step,
                })
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })
                
                # Periodic evaluation
                if global_step % config.eval_every_n_steps == 0:
                    print(f"\n\nEvaluating at step {global_step}...")
                    policy.eval()
                    
                    # Load current policy into vLLM
                    load_policy_into_vllm_instance(policy, vllm_model)
                    
                    # Evaluate
                    eval_metrics = evaluate_on_gsm8k(
                        llm=vllm_model,
                        prompts=val_prompts,
                        ground_truths=val_ground_truths,
                        sampling_params=eval_sampling_params,
                    )
                    
                    # Log evaluation metrics
                    for key, value in eval_metrics.items():
                        wandb.log({f"eval/{key}": value, "eval_step": eval_step})
                    
                    print(f"Evaluation Results:")
                    print(f"  Accuracy: {eval_metrics['accuracy']:.2%}")
                    print(f"  Format Correct Rate: {eval_metrics['format_correct_rate']:.2%}")
                    
                    eval_step += 1
                    policy.train()
                
                # Periodic generation logging
                if global_step % config.log_generations_every_n_steps == 0:
                    print(f"\n\nLogging generations at step {global_step}...")
                    
                    # Sample a few validation examples
                    sample_prompts = val_prompts[:config.num_log_examples]
                    sample_ground_truths = val_ground_truths[:config.num_log_examples]
                    
                    # Log generations (uses HF generate, not vLLM)
                    log_data = log_generations(
                        model=policy,
                        tokenizer=tokenizer,
                        prompts=sample_prompts,
                        ground_truths=sample_ground_truths,
                        reward_fn=r1_zero_reward_fn,
                        max_examples=config.num_log_examples,
                    )
                    
                    # Log to wandb
                    for key, value in log_data["metrics"].items():
                        wandb.log({f"generations/{key}": value, "train_step": global_step})
                    
                    # Print sample
                    print(f"\nSample Generation:")
                    example = log_data["examples"][0]
                    print(f"  Prompt: {example['prompt'][:100]}...")
                    print(f"  Response: {example['response'][:200]}...")
                    print(f"  Rewards: format={example['format_reward']}, answer={example['answer_reward']}")
                    print()
                
                # Periodic checkpoint saving
                if global_step % config.save_every_n_steps == 0:
                    checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                    print(f"Saving checkpoint to {checkpoint_dir}...")
                    policy.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    # Final evaluation
    print(f"\n\nFinal evaluation...")
    policy.eval()
    load_policy_into_vllm_instance(policy, vllm_model)
    
    final_metrics = evaluate_on_gsm8k(
        llm=vllm_model,
        prompts=val_prompts,
        ground_truths=val_ground_truths,
        sampling_params=eval_sampling_params,
    )
    
    print(f"\nFinal Results:")
    print(f"  Accuracy: {final_metrics['accuracy']:.2%}")
    print(f"  Format Correct Rate: {final_metrics['format_correct_rate']:.2%}")
    
    # Save final model
    final_dir = os.path.join(config.output_dir, "final")
    print(f"\nSaving final model to {final_dir}...")
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    wandb.finish()
    print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description="SFT Training for Qwen 2.5 Math 1.5B")
    
    # Data arguments
    parser.add_argument("--sft_data_path", type=str, default="data/gsm8k/sft.jsonl")
    parser.add_argument("--val_data_path", type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--num_train_examples", type=int, default=None)
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--microbatch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_clip_value", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    
    # Evaluation arguments
    parser.add_argument("--eval_every_n_steps", type=int, default=100)
    parser.add_argument("--num_eval_examples", type=int, default=100)
    
    # Device arguments
    parser.add_argument("--policy_device", type=str, default="cuda:0")
    parser.add_argument("--vllm_device", type=str, default="cuda:1")
    
    # Logging arguments
    parser.add_argument("--output_dir", type=str, default="outputs/sft")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--project_name", type=str, default="sft-qwen-math")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(**vars(args))
    
    # Run training
    train_sft(config)


if __name__ == "__main__":
    main()

