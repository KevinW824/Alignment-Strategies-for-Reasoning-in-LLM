#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) Training Script

This script implements GRPO training following the algorithm:
1. Generates rollouts from current policy
2. Computes rewards and group-normalizes them
3. Trains policy using GRPO loss on rollouts
4. Iterates for progressive improvement
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
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
)
from tqdm import tqdm
import wandb
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our implementations
from scripts.grpo import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
)
from scripts.sft import (
    tokenize_prompt_and_output,
    get_response_log_probs,
)
from scripts.drgrpo_grader import r1_zero_reward_fn
from scripts.math_baseline import load_jsonl_data, extract_ground_truth_answer


@dataclass
class TrainingConfig:
    """Configuration for GRPO training."""
    # Model and data
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    train_data_path: str = "data/gsm8k/train.jsonl"
    val_data_path: str = "data/gsm8k/test.jsonl"
    prompt_template_path: str = "scripts/prompts/r1_zero.prompt"
    
    # GRPO hyperparameters
    n_grpo_steps: int = 200
    rollout_batch_size: int = 256  # Number of questions per rollout batch
    group_size: int = 8  # Number of rollouts per question (G)
    epochs_per_rollout_batch: int = 1  # On-policy training
    train_batch_size: int = 256  # On-policy batch size
    gradient_accumulation_steps: int = 128  # Microbatch size = 2
    
    # Training hyperparameters
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    use_std_normalization: bool = True
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline"
    
    # Generation hyperparameters
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024
    
    # Evaluation
    eval_every_n_steps: int = 20  # Reduced frequency from 10
    num_eval_examples: int = 200  # Reduced from 500 for faster evaluation
    
    # vLLM settings
    vllm_device: str = "cuda:1"
    policy_device: str = "cuda:0"
    gpu_memory_utilization: float = 0.85
    
    # Logging and saving
    output_dir: str = "outputs/grpo"
    save_every_n_steps: int = 100  # Reduced frequency from 50
    project_name: str = "grpo-qwen-math"
    run_name: Optional[str] = None
    seed: int = 42


class RolloutDataset(Dataset):
    """Dataset for GRPO rollouts."""
    
    def __init__(self, rollouts: List[Dict[str, Any]]):
        """
        Args:
            rollouts: List of dicts with 'prompt', 'response', 'advantage', 'raw_reward' keys
        """
        self.examples = rollouts
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_rollout_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for rollout batches."""
    return {
        "prompt": [ex["prompt"] for ex in batch],
        "response": [ex["response"] for ex in batch],
        "advantage": [ex["advantage"] for ex in batch],
        "raw_reward": [ex["raw_reward"] for ex in batch],
    }


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """Initialize vLLM for rollout generation."""
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
            max_model_len=512,  # Reduced from default 4096 to prevent OOM, matches max_tokens=512
            trust_remote_code=True,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """Load policy weights into vLLM instance."""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_on_gsm8k(
    llm: LLM,
    prompts: List[str],
    ground_truths: List[str],
    sampling_params: SamplingParams,
) -> Dict[str, float]:
    """Evaluate model on GSM8K using vLLM."""
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


def generate_rollouts_batch(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    group_size: int,
) -> List[List[str]]:
    """
    Generate rollouts for a batch of prompts.
    
    Args:
        llm: vLLM instance
        prompts: List of prompt strings
        sampling_params: Sampling parameters (should have n=group_size)
        group_size: Number of rollouts per prompt
    
    Returns:
        List of lists, where rollouts[i] contains group_size responses for prompts[i]
    """
    # vLLM with n=group_size already generates group_size rollouts per prompt
    # So we don't need to repeat prompts - just pass them directly
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract rollouts - each output has group_size completions
    rollouts = []
    for output in outputs:
        prompt_rollouts = [completion.text for completion in output.outputs]
        rollouts.append(prompt_rollouts)
    
    return rollouts


def train_grpo(config: TrainingConfig):
    """Main GRPO training loop."""
    
    # Set random seeds
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config=vars(config),
    )
    
    # Setup wandb metrics
    wandb.define_metric("grpo_step")
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    wandb.define_metric("rollout/*", step_metric="grpo_step")
    
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
    
    # Load training data
    print(f"Loading training data from {config.train_data_path}...")
    train_examples = load_jsonl_data(config.train_data_path)
    train_questions = [ex["question"] for ex in train_examples]
    train_ground_truths = [extract_ground_truth_answer(ex["answer"]) for ex in train_examples]
    
    # Load validation data
    print(f"Loading validation data from {config.val_data_path}...")
    val_examples = load_jsonl_data(config.val_data_path)[:config.num_eval_examples]
    val_ground_truths = [extract_ground_truth_answer(ex["answer"]) for ex in val_examples]
    
    # Load prompt template
    with open(config.prompt_template_path, 'r') as f:
        prompt_template = f.read()
    val_prompts = [prompt_template.replace("{question}", ex["question"]) for ex in val_examples]
    
    # Initialize vLLM for rollout generation and evaluation
    print(f"Initializing vLLM on {config.vllm_device}...")
    vllm_model = init_vllm(
        model_id=config.model_name,
        device=config.vllm_device,
        seed=config.seed,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )
    
    # Sampling parameters for rollouts
    rollout_sampling_params = SamplingParams(
        temperature=config.sampling_temperature,
        top_p=1.0,
        n=config.group_size,  # Generate group_size rollouts per prompt
        min_tokens=config.sampling_min_tokens,
        max_tokens=config.sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
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
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    
    # Calculate microbatch size
    microbatch_size = config.train_batch_size // config.gradient_accumulation_steps
    
    print(f"\nGRPO Training Configuration:")
    print(f"  Total GRPO steps: {config.n_grpo_steps}")
    print(f"  Rollout batch size: {config.rollout_batch_size}")
    print(f"  Group size: {config.group_size}")
    print(f"  Epochs per rollout batch: {config.epochs_per_rollout_batch}")
    print(f"  Train batch size: {config.train_batch_size}")
    print(f"  Microbatch size: {microbatch_size}")
    print(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Loss type: {config.loss_type}")
    print(f"  Use std normalization: {config.use_std_normalization}")
    print(f"  Advantage eps: {config.advantage_eps}\n")
    
    # Training loop
    global_step = 0
    eval_step = 0
    
    for grpo_step in range(config.n_grpo_steps):
        print(f"\n{'='*80}")
        print(f"GRPO Step {grpo_step + 1}/{config.n_grpo_steps}")
        print(f"{'='*80}\n")
        
        # Step 1: Sample questions for this rollout batch
        if len(train_questions) >= config.rollout_batch_size:
            # Sample random questions
            indices = torch.randperm(len(train_questions))[:config.rollout_batch_size].tolist()
            batch_questions = [train_questions[i] for i in indices]
            batch_ground_truths = [train_ground_truths[i] for i in indices]
        else:
            # Use all available questions
            batch_questions = train_questions
            batch_ground_truths = train_ground_truths
        
        # Format prompts
        batch_prompts = [prompt_template.replace("{question}", q) for q in batch_questions]
        
        # Step 2: Load current policy into vLLM
        print("Loading policy weights into vLLM...")
        load_policy_into_vllm_instance(policy, vllm_model)
        
        # Step 3: Generate rollouts
        print(f"Generating {config.group_size} rollouts per question for {len(batch_prompts)} questions...")
        rollouts = generate_rollouts_batch(
            llm=vllm_model,
            prompts=batch_prompts,
            sampling_params=rollout_sampling_params,
            group_size=config.group_size,
        )
        
        # Flatten rollouts for reward computation
        rollout_responses = []
        repeated_ground_truths = []
        for prompt_rollouts, ground_truth in zip(rollouts, batch_ground_truths):
            rollout_responses.extend(prompt_rollouts)
            repeated_ground_truths.extend([ground_truth] * config.group_size)
        
        # Step 4: Compute rewards and group-normalize
        print("Computing rewards and group-normalizing...")
        normalized_rewards, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=config.group_size,
            advantage_eps=config.advantage_eps,
            normalize_by_std=config.use_std_normalization,
        )
        
        # Log rollout metrics
        wandb.log({
            "rollout/mean_reward": reward_metadata["mean_reward"],
            "rollout/std_reward": reward_metadata["std_reward"],
            "rollout/min_reward": reward_metadata["min_reward"],
            "rollout/max_reward": reward_metadata["max_reward"],
            "rollout/mean_normalized_reward": reward_metadata["mean_normalized_reward"],
            "rollout/std_normalized_reward": reward_metadata["std_normalized_reward"],
            "grpo_step": grpo_step,
        })
        
        print(f"  Mean reward: {reward_metadata['mean_reward']:.4f}")
        print(f"  Mean normalized reward: {reward_metadata['mean_normalized_reward']:.4f}")
        
        # Step 5: Prepare training data
        # Create dataset with prompts, responses, advantages, and old log probs
        training_examples = []
        advantages_list = normalized_rewards.tolist()
        raw_rewards_list = raw_rewards.tolist()
        
        for i, (prompt, response) in enumerate(zip(
            [p for p in batch_prompts for _ in range(config.group_size)],
            rollout_responses
        )):
            training_examples.append({
                "prompt": prompt,
                "response": response,
                "advantage": advantages_list[i],
                "raw_reward": raw_rewards_list[i],
            })
        
        # Step 6: Train on rollouts
        print(f"Training on {len(training_examples)} rollouts for {config.epochs_per_rollout_batch} epochs...")
        
        for epoch in range(config.epochs_per_rollout_batch):
            # Create dataloader
            dataset = RolloutDataset(training_examples)
            dataloader = DataLoader(
                dataset,
                batch_size=microbatch_size,
                shuffle=True,
                collate_fn=collate_rollout_batch,
            )
            
            epoch_loss = 0.0
            optimizer.zero_grad()
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.epochs_per_rollout_batch}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Extract data from batch (now properly collated as dict of lists)
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
                
                # Prepare advantages/rewards
                advantages = torch.tensor(
                    batch["advantage"],
                    dtype=torch.float32,
                    device=config.policy_device,
                ).unsqueeze(1)  # (batch_size, 1)
                
                raw_rewards_tensor = torch.tensor(
                    batch["raw_reward"],
                    dtype=torch.float32,
                    device=config.policy_device,
                ).unsqueeze(1)  # (batch_size, 1)
                
                # Get old log probs (for GRPO-Clip) - compute before forward pass with gradients
                old_log_probs = None
                if config.loss_type == "grpo_clip":
                    with torch.no_grad():
                        old_outputs = get_response_log_probs(
                            model=policy,
                            input_ids=input_ids,
                            labels=labels,
                            return_token_entropy=False,
                        )
                        old_log_probs = old_outputs['log_probs']
                
                # Get new log probs (with gradients)
                outputs = get_response_log_probs(
                    model=policy,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )
                policy_log_probs = outputs['log_probs']
                
                # GRPO microbatch train step
                if config.loss_type == "no_baseline":
                    loss, metadata = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=response_mask,
                        gradient_accumulation_steps=config.gradient_accumulation_steps,
                        loss_type=config.loss_type,
                        raw_rewards=raw_rewards_tensor,
                        advantages=None,
                        old_log_probs=None,
                        cliprange=None,
                        normalize_constant=None,
                    )
                elif config.loss_type == "reinforce_with_baseline":
                    loss, metadata = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=response_mask,
                        gradient_accumulation_steps=config.gradient_accumulation_steps,
                        loss_type=config.loss_type,
                        raw_rewards=None,
                        advantages=advantages,
                        old_log_probs=None,
                        cliprange=None,
                        normalize_constant=None,
                    )
                elif config.loss_type == "grpo_clip":
                    # For GRPO-Clip, we need cliprange (typically 0.1 or 0.2)
                    cliprange = 0.1  # Default cliprange for GRPO-Clip
                    loss, metadata = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=response_mask,
                        gradient_accumulation_steps=config.gradient_accumulation_steps,
                        loss_type=config.loss_type,
                        raw_rewards=None,
                        advantages=advantages,
                        old_log_probs=old_log_probs,
                        cliprange=cliprange,
                        normalize_constant=None,
                    )
                
                epoch_loss += loss.item()
                
                # Optimizer step after accumulating gradients
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Log training metrics
                    wandb.log({
                        "train/loss": metadata["unscaled_loss"].item(),
                        "train/scaled_loss": loss.item(),
                        "train/num_response_tokens": metadata["num_response_tokens"].item(),
                        "train_step": global_step,
                    })
                    
                    if config.loss_type == "grpo_clip" and "clip_fraction" in metadata:
                        wandb.log({
                            "train/clip_fraction": metadata["clip_fraction"].item(),
                            "train_step": global_step,
                        })
                    
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'step': global_step
                    })
        
        avg_epoch_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        print(f"  Average epoch loss: {avg_epoch_loss:.4f}")
        
        # Step 7: Evaluation
        if (grpo_step + 1) % config.eval_every_n_steps == 0:
            print(f"\nEvaluating on validation set...")
            load_policy_into_vllm_instance(policy, vllm_model)
            
            eval_metrics = evaluate_on_gsm8k(
                llm=vllm_model,
                prompts=val_prompts,
                ground_truths=val_ground_truths,
                sampling_params=eval_sampling_params,
            )
            
            print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
            print(f"  Format correct rate: {eval_metrics['format_correct_rate']:.4f}")
            print(f"  Avg reward: {eval_metrics['avg_total_reward']:.4f}")
            
            wandb.log({
                "eval/accuracy": eval_metrics["accuracy"],
                "eval/format_correct_rate": eval_metrics["format_correct_rate"],
                "eval/avg_total_reward": eval_metrics["avg_total_reward"],
                "eval_step": eval_step,
            })
            eval_step += 1
        
        # Step 8: Save checkpoint
        if (grpo_step + 1) % config.save_every_n_steps == 0:
            checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-step-{grpo_step + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            policy.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"\n✓ Checkpoint saved to {checkpoint_dir}")
    
    # Save final model
    final_dir = os.path.join(config.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n✓ Final model saved to {final_dir}")

    # Final full evaluation
    print("\nRunning final evaluation on full test set...")
    load_policy_into_vllm_instance(policy, vllm_model)
    final_metrics = evaluate_on_gsm8k(
        llm=vllm_model,
        prompts=val_prompts,
        ground_truths=val_ground_truths,
        sampling_params=eval_sampling_params,
        max_examples=None,  # Full test set
    )
    print(f"Final Accuracy (full test set): {final_metrics['accuracy']:.4f}")
    wandb.log({
        "eval/final_accuracy": final_metrics["accuracy"],
        "eval/final_format_correct_rate": final_metrics["format_correct_rate"],
    })
    
    print("\n" + "=" * 80)
    print("GRPO Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Training for GSM8K")
    
    # Model and data
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data_path", type=str, default="data/gsm8k/train.jsonl")
    parser.add_argument("--val_data_path", type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--prompt_template_path", type=str, default="scripts/prompts/r1_zero.prompt")
    
    # GRPO hyperparameters
    parser.add_argument("--n_grpo_steps", type=int, default=200)
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--use_std_normalization", action="store_true")
    parser.add_argument("--loss_type", type=str, default="reinforce_with_baseline",
                       choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    
    # Generation hyperparameters
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_min_tokens", type=int, default=4)
    parser.add_argument("--sampling_max_tokens", type=int, default=1024)
    
    # Evaluation
    parser.add_argument("--eval_every_n_steps", type=int, default=10)
    parser.add_argument("--num_eval_examples", type=int, default=500)
    
    # vLLM settings
    parser.add_argument("--vllm_device", type=str, default="cuda:1")
    parser.add_argument("--policy_device", type=str, default="cuda:0")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    
    # Logging and saving
    parser.add_argument("--output_dir", type=str, default="outputs/grpo")
    parser.add_argument("--save_every_n_steps", type=int, default=50)
    parser.add_argument("--project_name", type=str, default="grpo-qwen-math")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        model_name=args.model_name,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        prompt_template_path=args.prompt_template_path,
        n_grpo_steps=args.n_grpo_steps,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        epochs_per_rollout_batch=args.epochs_per_rollout_batch,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        advantage_eps=args.advantage_eps,
        use_std_normalization=args.use_std_normalization,
        loss_type=args.loss_type,
        sampling_temperature=args.sampling_temperature,
        sampling_min_tokens=args.sampling_min_tokens,
        sampling_max_tokens=args.sampling_max_tokens,
        eval_every_n_steps=args.eval_every_n_steps,
        num_eval_examples=args.num_eval_examples,
        vllm_device=args.vllm_device,
        policy_device=args.policy_device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        output_dir=args.output_dir,
        save_every_n_steps=args.save_every_n_steps,
        project_name=args.project_name,
        run_name=args.run_name,
        seed=args.seed,
    )
    
    train_grpo(config)

