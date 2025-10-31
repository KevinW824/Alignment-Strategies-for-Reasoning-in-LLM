#!/usr/bin/env python3
"""
Expert Iteration Training Script for MATH Dataset.

This script implements Algorithm 2 (Expert Iteration) from the assignment:
1. Generates rollouts from current policy
2. Evaluates and filters correct responses
3. Trains on filtered data using SFT
4. Iterates N times for progressive improvement

"""

import json
import os
import sys
import random
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

from sft import (
    tokenize_prompt_and_output,
    sft_microbatch_train_step,
    get_response_log_probs,
)
from drgrpo_grader import r1_zero_reward_fn
from math_baseline import load_jsonl_data, extract_ground_truth_answer

from expert_iteration import (
    generate_rollouts,
    compute_rewards,
    filter_correct_outputs
)


@dataclass
class TrainingConfig:
    """Configuration for Expert Iteration training."""
    # Model and data TODO: use Final model from SFT output (how?)
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    train_data_path: str = "data/gsm8k/train.jsonl"  # GSM8K training data
    val_data_path: str = "data/gsm8k/test.jsonl"  # GSM8K test set (used as validation)
    sft_data_path: str = "data/gsm8k/sft.jsonl"  # SFT dataset for initial training (if needed)
    prompt_template_path: str = "scripts/prompts/r1_zero.prompt"
    
    # Expert Iteration hyperparameters
    n_ei_steps: int = 5  # Number of EI iterations
    rollouts_per_question: int = 8  # G in the algorithm
    questions_per_step: int = 1024  # Batch size for each EI step
    sft_epochs_per_step: int = 2  # Epochs to train on filtered data
    
    # Training hyperparameters
    learning_rate: float = 5e-6
    microbatch_size: int = 4
    batch_size: int = 16  # Effective batch size
    max_seq_length: int = 4096
    gradient_clip_value: float = 1.0
    warmup_steps: int = 100
    
    # Generation hyperparameters
    temperature: float = 0.7
    max_gen_tokens: int = 2048
    
    # Evaluation
    eval_every_n_steps: int = 1  # Eval after each EI step
    num_eval_examples: int = 500
    
    # vLLM settings
    vllm_device: str = "cuda:1"
    policy_device: str = "cuda:0"
    gpu_memory_utilization: float = 0.85
    
    # Logging and saving
    output_dir: str = "outputs/expert_iteration"
    save_every_n_steps: int = 1  # Save after each EI step
    project_name: str = "expert-iteration-gsm8k"
    run_name: Optional[str] = None
    seed: int = 42


class FilteredSFTDataset(Dataset):
    """Dataset for filtered EI examples"""
    
    def __init__(self, filtered_examples: List[Dict[str, str]]):
        """
        Args:
            filtered_examples: List of dicts with 'prompt' and 'response' keys
        """
        self.examples = filtered_examples
        print(f"Created filtered dataset with {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Initialize vLLM for rollout generation (same as SFT portion).
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
    Load policy weights into vLLM instance.
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
    Evaluate model on GSM8K test set.
    
    Args:
        llm: vLLM instance
        prompts: List of formatted prompts
        ground_truths: List of ground truth answers
        sampling_params: Sampling parameters for generation
    
    Returns:
        Dictionary with evaluation metrics
    """
    outputs = llm.generate(prompts, sampling_params)

    all_rewards = []
    for output, ground_truth in zip(outputs, ground_truths):
        response = output.outputs[0].text
        reward_dict = r1_zero_reward_fn(response, ground_truth, True)
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


def train_expert_iteration(config: TrainingConfig):
    """Main Expert Iteration training loop"""
    print("=" * 80)
    print("Expert Iteration Training for GSM8K")
    print("=" * 80)

    # Set random seed and create output directory
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Wandb setup
    if config.run_name is None:
        config.run_name = f"ei_G{config.rollouts_per_question}_B{config.questions_per_step}_E{config.sft_epochs_per_step}"
   
    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config=vars(config),
        mode="offline"
    )
    wandb.define_metric("ei_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="ei_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # Load tokenizer (same as SFT)
    print(f"Loading tokenizer from {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load policy model (same as SFT)
    print(f"Loading policy model on {config.policy_device}...")
    policy = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(config.policy_device)
    policy.train()

    # Load training data
    print(f"Loading GSM8K training data from {config.train_data_path}...")
    train_examples = load_jsonl_data(config.train_data_path)
    print(f"Loaded {len(train_examples)} training examples")

    # Load validation data
    print(f"Loading validation data from {config.val_data_path}...")
    val_examples = load_jsonl_data(config.val_data_path)[:config.num_eval_examples]
    # Extract ground truth answers using the same method as SFT
    val_ground_truths = [extract_ground_truth_answer(ex["answer"]) for ex in val_examples]

    # Load prompt template (same as SFT)
    with open(config.prompt_template_path, 'r') as f:
        prompt_template = f.read()
    val_prompts = [prompt_template.replace("{question}", ex["question"]) for ex in val_examples]
    
    # Initialize vLLM for rollout generation (same as SFT)
    print(f"Initializing vLLM on {config.vllm_device}...")
    vllm_model = init_vllm(
        model_id=config.model_name,
        device=config.vllm_device,
        seed=config.seed,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )
    
    # Sampling parameters for rollout generation
    rollout_sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=1.0,
        max_tokens=config.max_gen_tokens,
        min_tokens=4,  # Prevent empty strings
        n=config.rollouts_per_question,  # G rollouts per question
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=config.seed,
    )
    
    # Sampling parameters for evaluation (same as SFT)
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Setup optimizer (same as SFT)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)
    
    # Calculate gradient accumulation steps (same as SFT)
    gradient_accumulation_steps = config.batch_size // config.microbatch_size
    
    # Calculate total training steps
    steps_per_ei_iteration = config.questions_per_step // config.batch_size
    total_sft_steps_per_iteration = steps_per_ei_iteration * config.sft_epochs_per_step
    total_steps = total_sft_steps_per_iteration * config.n_ei_steps
    
    # Setup learning rate scheduler (same as SFT)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"\nExpert Iteration Configuration:")
    print(f"  Number of EI steps: {config.n_ei_steps}")
    print(f"  Questions per step: {config.questions_per_step}")
    print(f"  Rollouts per question (G): {config.rollouts_per_question}")
    print(f"  SFT epochs per step: {config.sft_epochs_per_step}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Microbatch size: {config.microbatch_size}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Total SFT steps per iteration: {total_sft_steps_per_iteration}")
    print(f"  Total steps: {total_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Warmup steps: {config.warmup_steps}\n")
    
    # Run initial evaluation (baseline)
    print("=" * 80)
    print("Initial Evaluation (Baseline)")
    print("=" * 80)
    load_policy_into_vllm_instance(policy, vllm_model)
    initial_metrics = evaluate_on_gsm8k(
        vllm_model,
        val_prompts,
        val_ground_truths,
        eval_sampling_params,
    )
    print(f"Initial accuracy: {initial_metrics['accuracy']:.4f}")
    print(f"Initial format rate: {initial_metrics['format_correct_rate']:.4f}")
    
    wandb.log({
        "eval/accuracy": initial_metrics["accuracy"],
        "eval/format_correct_rate": initial_metrics["format_correct_rate"],
        "eval/avg_total_reward": initial_metrics["avg_total_reward"],
        "eval_step": 0,
    })
    
    # ===================================================================
    # Expert Iteration Loop
    # ===================================================================
    global_step = 0
    eval_step = 0

    for ei_step in range(1, config.n_ei_steps + 1):
        print(f"\n{'='*80}")
        print(f"Expert Iteration Step {ei_step}/{config.n_ei_steps}")
        print(f"{'='*80}\n")
        
        # ===================================================================
        # STEP 1: Sample questions for this EI iteration
        # ===================================================================
        random.seed(config.seed + ei_step)
        step_examples = random.sample(train_examples, config.questions_per_step)
        step_questions = [ex["question"] for ex in step_examples]
        # Extract ground truth
        step_ground_truths = [extract_ground_truth_answer(ex["answer"]) for ex in step_examples]
        step_prompts = [prompt_template.replace("{question}", q) for q in step_questions]
        print(f"Sampled {len(step_prompts)} questions for EI step {ei_step}")
        
        # ===================================================================
        # STEP 2: Load current policy into vLLM and generate rollouts
        # ===================================================================
        print("\nLoading policy weights into vLLM...")
        load_policy_into_vllm_instance(policy, vllm_model)
        
        print(f"Generating {config.rollouts_per_question} rollouts per question...")
        rollouts = generate_rollouts(
            vllm_model=vllm_model,
            questions=step_prompts,
            sampling_params=rollout_sampling_params,
            show_progress=True
        )
        
        total_rollouts = len(rollouts) * len(rollouts[0])
        print(f"Generated {total_rollouts} total rollouts")
        
        # ===================================================================
        # STEP 3: Compute rewards for all rollouts
        # ===================================================================
        print("Computing rewards...")
        rewards = compute_rewards(
            questions=step_prompts,
            rollouts=rollouts,
            ground_truths=step_ground_truths,
            reward_function=lambda q, r, gt: r1_zero_reward_fn(r, gt),
        )
        
        # ===================================================================
        # STEP 4: Filter to keep only correct outputs
        # ===================================================================
        print("Filtering correct outputs...")
        filtered_dataset, filter_stats = filter_correct_outputs(
            questions=step_prompts,
            rollouts=rollouts,
            rewards=rewards,
            threshold=1.0
        )
        
        print(f"\nFiltering Statistics:")
        print(f"  Total rollouts: {filter_stats['total_rollouts']}")
        print(f"  Correct rollouts: {filter_stats['correct_rollouts']}")
        print(f"  Success rate: {filter_stats['success_rate']:.3f}")
        
        # Log filtering metrics
        wandb.log({
            "train/total_rollouts": filter_stats["total_rollouts"],
            "train/correct_rollouts": filter_stats["correct_rollouts"],
            "train/success_rate": filter_stats["success_rate"],
            "ei_step": ei_step,
        })
        
        # Check if we have any correct examples to train on
        if len(filtered_dataset) == 0:
            print("\nWARNING: No correct rollouts found! Skipping SFT for this step.")
            continue
        
        # ===================================================================
        # STEP 5: Train on filtered data using SFT
        # ===================================================================
        print(f"\nRunning SFT on {len(filtered_dataset)} filtered examples...")
        
        # Create filtered dataset
        sft_dataset = FilteredSFTDataset(filtered_dataset)
        
        # Create dataloader
        dataloader = DataLoader(
            sft_dataset,
            batch_size=config.microbatch_size,
            shuffle=True,
        )
        
        # SFT training loop for this EI step
        epoch_losses = []
        
        for epoch in range(config.sft_epochs_per_step):
            print(f"  SFT Epoch {epoch + 1}/{config.sft_epochs_per_step}")
            
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(dataloader, desc=f"  Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Tokenize batch
                prompts = [ex["prompt"] for ex in batch]
                responses = [ex["response"] for ex in batch]
                
                tokenized = tokenize_prompt_and_output(
                    prompts,
                    responses,
                    tokenizer
                )
                
                # Move to device
                input_ids = tokenized["input_ids"].to(config.policy_device)
                labels = tokenized["labels"].to(config.policy_device)
                response_mask = tokenized["response_mask"].to(config.policy_device)
                
                # Get log probabilities (same as SFT)
                outputs = get_response_log_probs(
                    model=policy,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )
                
                # SFT training step (same as SFT)
                loss, metadata = sft_microbatch_train_step(
                    policy_log_probs=outputs['log_probs'],
                    response_mask=response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    normalize_constant=1.0,
                )
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Optimizer step after accumulating gradients (same as SFT)
                if num_batches % gradient_accumulation_steps == 0:
                    # Gradient clipping (required by assignment)
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip_value)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            epoch_losses.append(avg_epoch_loss)
            print(f"    Average loss: {avg_epoch_loss:.4f}")
        
        avg_sft_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        
        # Log SFT metrics
        wandb.log({
            "train/sft_loss": avg_sft_loss,
            "train/filtered_dataset_size": len(filtered_dataset),
            "ei_step": ei_step,
        })

        # Reload policy weights into vLLM
        print(f"\nReloading policy weights into vLLM after EI step {ei_step}...")
        load_policy_into_vllm_instance(policy, vllm_model)

        # ===================================================================
        # STEP 6: Evaluate on validation set
        # ===================================================================
        if ei_step % config.eval_every_n_steps == 0:
            print(f"\nRunning evaluation at EI step {ei_step}...")
            policy.eval()
            
            load_policy_into_vllm_instance(policy, vllm_model)
            eval_metrics = evaluate_on_gsm8k(
                vllm_model,
                val_prompts,
                val_ground_truths,
                eval_sampling_params,
            )
            
            eval_step += 1
            
            print(f"Validation accuracy: {eval_metrics['accuracy']:.4f}")
            print(f"Format correct rate: {eval_metrics['format_correct_rate']:.4f}")
            
            wandb.log({
                "eval/accuracy": eval_metrics["accuracy"],
                "eval/format_correct_rate": eval_metrics["format_correct_rate"],
                "eval/avg_total_reward": eval_metrics["avg_total_reward"],
                "eval_step": eval_step,
            })
            
            policy.train()
        
        # ===================================================================
        # STEP 7: Save checkpoint
        # ===================================================================
        if ei_step % config.save_every_n_steps == 0:
            checkpoint_dir = Path(config.output_dir) / f"checkpoint-step-{ei_step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            policy.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            
            print(f"Saved checkpoint to {checkpoint_dir}")

    # Final evaluation
    print("\n" + "=" * 80)
    print("Final Evaluation")
    print("=" * 80)
    
    policy.eval()
    load_policy_into_vllm_instance(policy, vllm_model)
    final_metrics = evaluate_on_gsm8k(
        vllm_model,
        val_prompts,
        val_ground_truths,
        eval_sampling_params,
    )
    
    print(f"Final accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final format rate: {final_metrics['format_correct_rate']:.4f}")
    
    # Save final model (same as SFT)
    final_checkpoint_dir = Path(config.output_dir) / "final_model"
    final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_checkpoint_dir)
    tokenizer.save_pretrained(final_checkpoint_dir)
    print(f"Saved final model to {final_checkpoint_dir}")
    
    wandb.finish()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Expert Iteration Training for GSM8K")
    
    # Data arguments
    parser.add_argument("--train_data_path", type=str, default="data/gsm8k/train.jsonl")
    parser.add_argument("--val_data_path", type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--prompt_template_path", type=str, default="scripts/prompts/r1_zero.prompt")
    
    # EI-specific arguments
    parser.add_argument("--n_ei_steps", type=int, default=5)
    parser.add_argument("--rollouts_per_question", type=int, default=8)
    parser.add_argument("--questions_per_step", type=int, default=1024)
    parser.add_argument("--sft_epochs_per_step", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    
    # Training arguments (same as SFT)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--microbatch_size", type=int, default=4)
    parser.add_argument("--gradient_clip_value", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    
    # Evaluation arguments (same as SFT)
    parser.add_argument("--eval_every_n_steps", type=int, default=1)
    parser.add_argument("--num_eval_examples", type=int, default=500)
    
    # Device arguments (same as SFT)
    parser.add_argument("--policy_device", type=str, default="cuda:0")
    parser.add_argument("--vllm_device", type=str, default="cuda:1")
    
    # Logging arguments (same as SFT)
    parser.add_argument("--output_dir", type=str, default="outputs/expert_iteration")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--project_name", type=str, default="expert-iteration-gsm8k")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Create config (same as SFT)
    config = TrainingConfig(**vars(args))
    
    # Run training
    train_expert_iteration(config)

if __name__ == "__main__":
    main()