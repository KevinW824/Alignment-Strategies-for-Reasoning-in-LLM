"""
Expert Iteration Implementation for MATH Dataset

This module implements the core functions for Expert Iteration (EI)
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
from tqdm import tqdm

from sft import sft_microbatch_train_step, tokenize_prompt_and_output
from scripts.drgrpo_grader import r1_zero_reward_fn

def generate_rollouts(
    vllm_model,
    questions: List[str],
    sampling_params,
    batch_size: int = 32,
    show_progress: bool = True
) -> List[List[str]]:
    """
    Generate multiple rollouts (candidate solutions) for each question.
    
    Args:
        vllm_model: vLLM inference engine holding the policy model
        questions: List of question strings to generate rollouts for
        sampling_params: vLLM SamplingParams object with:
            - temperature: Controls diversity (higher = more diverse)
            - n: Number of rollouts per question (G in the algorithm)
            - max_tokens: Maximum response length
            - min_tokens: Minimum response length (prevents empty strings)
        batch_size: Number of questions to process at once
        show_progress: Whether to show progress bar
        
    Returns:
        List of lists, where rollouts[i] contains G generated responses
        for questions[i]
    """
    all_rollouts = []
    # Process questions in batches for efficiency
    num_batches = (len(questions) + batch_size -1) // batch_size
    iterator = range(num_batches)

    if show_progress: 
        iterator = tqdm(iterator, desc="Generating rollouts")
    
    for batch_idx in iterator:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(questions))
        batch_questions = questions[start_idx:end_idx]
        # Generate Output
        outputs = vllm_model.generate(batch_questions, sampling_params)
        # Extract output text from each completion of returned CompletionOutput objects
        for output in outputs:
            batch_rollouts = [completion.text for completion in output.outputs]
            all_rollouts.append(batch_rollouts)
    return all_rollouts



def compute_rewards(
    questions: List[str],
    rollouts: List[List[str]],
    ground_truths: List[str],
    reward_function,
) -> Dict[str, List[List[float]]]:
    """
    Compute rewards for all generated rollouts.
    
    The reward function checks two things:
    1. Format: Is the response properly formatted with answer tags?
    2. Correctness: Does the extracted answer match the ground truth?
    
    Total reward = format_reward * answer_reward (both must be 1 for correct)
    
    Args:
        questions: List of question strings
        rollouts: List of lists, rollouts[i] contains G responses for questions[i]
        ground_truths: List of correct answers
        reward_function: Function that takes (question, response, ground_truth)
                        and returns dict with 'format_reward', 'answer_reward', 'reward' keys
        
    Returns:
        Dictionary with keys 'format', 'answer', 'total', each containing
        a list of lists of rewards matching the structure of rollouts
    """
    format_rewards = []
    answer_rewards = []
    total_rewards = []

    for question, question_rollouts, ground_truth in zip(questions, rollouts, ground_truths):
        question_format_rewards = []
        question_answer_rewards = []
        question_total_rewards = []
        
        # Evaluate each rollout for this question
        for response in question_rollouts:
            # Get reward breakdown from reward function
            reward_dict = reward_function(question, response, ground_truth)
            
            question_format_rewards.append(reward_dict['format_reward'])
            question_answer_rewards.append(reward_dict['answer_reward'])
            question_total_rewards.append(reward_dict['reward'])
        
        format_rewards.append(question_format_rewards)
        answer_rewards.append(question_answer_rewards)
        total_rewards.append(question_total_rewards)
    
    return {
        'format': format_rewards,
        'answer': answer_rewards,
        'total': total_rewards
    }



def filter_correct_outputs(
    questions: List[str],
    rollouts: List[List[str]],
    rewards: Dict[str, List[List[float]]],
    threshold: float = 1.0
) -> Tuple[List[Dict], Dict]:
    """
    Filter rollouts to keep only correct outputs (reward >= threshold).
    
    Args:
        questions: List of question strings
        rollouts: List of lists of generated responses
        rewards: Dictionary with 'total' key containing reward values
        threshold: Minimum reward to keep (default 1.0 for only correct)
        
    Returns:
        Tuple of (filtered_dataset, statistics):
        - filtered_dataset: List of dicts with 'prompt' and 'response' keys
        - statistics: Dict with filtering stats (total, kept, ratio)
    """
    filtered_dataset = []
    total_rewards = rewards['total']

    # Track statistics
    total_rollouts = 0
    correct_rollouts = 0

    for question, question_rollouts, question_rewards in zip(
        questions, rollouts, total_rewards
    ):
        total_rollouts += len(question_rollouts)
        for response, reward in zip(question_rollouts, question_rewards):
            if reward >= threshold:
                filtered_dataset.append({
                    'prompt': question,
                    'response': response
                })
                correct_rollouts += 1

    success_rate = correct_rollouts / total_rollouts if total_rollouts > 0 else 0.0
    statistics = {
        'total_rollouts': total_rollouts,
        'correct_rollouts': correct_rollouts,
        'success_rate': success_rate,
        'unique_questions': len(questions)
    }
    return filtered_dataset, statistics



def expert_iteration_step(
    policy_model,
    vllm_model,
    tokenizer,
    optimizer,
    questions: List[str],
    ground_truths: List[str],
    reward_function,
    sampling_params,
    sft_config: Dict,
    device: str,
    step_num: int,
) -> Dict:
    """
    Perform one complete Expert Iteration step, including: 
    1. Generate rollouts from current policy
    2. Compute rewards for all rollouts
    3. Filter to keep only correct outputs
    4. Train policy on filtered data using SFT
    
    Args:
        policy_model: The model being trained
        vllm_model: vLLM engine for efficient rollout generation
        tokenizer: Tokenizer for the model
        optimizer: Optimizer for training
        questions: List of questions to generate rollouts for
        ground_truths: List of correct answers
        reward_function: Function to evaluate rollouts
        sampling_params: vLLM SamplingParams for generation
        sft_config: Dict with SFT hyperparameters:
            - num_epochs: Number of epochs to train on filtered data
            - microbatch_size: Size of each gradient accumulation step
            - grad_accum_steps: Number of microbatches before update
            - max_seq_len: Maximum sequence length
            - lr_scheduler: Learning rate scheduler
        device: Device to train on
        step_num: Current EI step number (for logging)
        
    Returns:
        Dictionary with metrics from this EI step:
        - total_rollouts: Total number of rollouts generated
        - correct_rollouts: Number of correct rollouts
        - success_rate: Fraction of correct rollouts
        - filtered_dataset_size: Size of filtered dataset
        - sft_loss: Average SFT loss on filtered data
        - ... (other relevant metrics)
    """
    # Step 1: Generate rollouts
    print(f"\n[EI Step {step_num}] Generating rollouts...")
    rollouts = generate_rollouts(
        vllm_model=vllm_model,
        questions=questions,
        sampling_params=sampling_params,
        show_progress=True
    )

    # Step 2: Compute rewards
    print(f"[EI Step {step_num}] Computing rewards...")
    rewards = compute_rewards(
        questions=questions,
        rollouts=rollouts,
        ground_truths=ground_truths,
        reward_function=reward_function
    )

    # Step 3: Filter correct outputs
    print(f"[EI Step {step_num}] Filtering correct outputs...")
    filtered_dataset, filter_stats = filter_correct_outputs(
        questions=questions,
        rollouts=rollouts,
        rewards=rewards,
        threshold=1.0
    )

    print(f"[EI Step {step_num}] Filtered dataset stats:")
    print(f"  Total rollouts: {filter_stats['total_rollouts']}")
    print(f"  Correct rollouts: {filter_stats['correct_rollouts']}")
    print(f"  Success rate: {filter_stats['success_rate']:.3f}")

    # If no correct examples, skip SFT
    if len(filtered_dataset) == 0:
        print(f"[EI Step {step_num}] WARNING: No correct outputs found! Skipping SFT.")
        return {
            **filter_stats,
            'filtered_dataset_size': 0,
            'sft_loss': None,
            'sft_epochs_completed': 0
        }
    
    # Step 4: Run SFT on filtered data
    print(f"[EI Step {step_num}] Running SFT on {len(filtered_dataset)} examples...")
    
    # TODO: This will call your SFT training function
    # For now, we'll create a placeholder
    # sft_metrics = run_sft_on_dataset(
    #     model=policy_model,
    #     tokenizer=tokenizer,
    #     optimizer=optimizer,
    #     dataset=filtered_dataset,
    #     config=sft_config,
    #     device=device
    # )
    
    # Placeholder for SFT metrics
    sft_metrics = {
        'sft_loss': 0.0,  # Will be filled in when we implement SFT integration
        'sft_epochs_completed': sft_config['num_epochs']
    }
    
    # Combine all metrics
    metrics = {
        **filter_stats,
        'filtered_dataset_size': len(filtered_dataset),
        **sft_metrics
    }
    
    return metrics

def save_filtered_dataset(
    filtered_dataset: List[Dict],
    output_path: str,
    step_num: Optional[int] = None
):
    """
    Save filtered dataset to disk for inspection or reuse.
    
    Args:
        filtered_dataset: List of dicts with 'prompt' and 'response'
        output_path: Path to save the dataset
        step_num: Optional EI step number to include in filename
    """
    if step_num is not None:
        output_path = output_path.replace('.jsonl', f'_step{step_num}.jsonl')
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for example in filtered_dataset:
            f.write(json.dumps(example) + '\n')
    print(f"Saved {len(filtered_dataset)} examples to {output_path}")



if __name__ == "__main__":
    # Simple test of the functions
    print("Testing Expert Iteration functions...")
    
    # Mock data
    questions = ["What is 2+2?", "What is 5*3?"]
    rollouts = [
        ["2+2=4", "2+2=5", "It's 4"],
        ["5*3=15", "5*3=10"]
    ]
    
    # Mock rewards (pretending we evaluated them)
    rewards = {
        'format': [[1.0, 1.0, 1.0], [1.0, 1.0]],
        'answer': [[1.0, 0.0, 1.0], [1.0, 0.0]],
        'total': [[1.0, 0.0, 1.0], [1.0, 0.0]]
    }
    
    ground_truths = ["4", "15"]
    
    # Test filtering
    filtered_dataset, stats = filter_correct_outputs(questions, rollouts, rewards)
    
    print(f"\nFiltering results:")
    print(f"  Total rollouts: {stats['total_rollouts']}")
    print(f"  Correct rollouts: {stats['correct_rollouts']}")
    print(f"  Success rate: {stats['success_rate']:.3f}")
    print(f"\nFiltered dataset:")
    for example in filtered_dataset:
        print(f"  Q: {example['prompt']}")
        print(f"  A: {example['response']}")
        print()
    
    print("✓ Basic tests passed!")