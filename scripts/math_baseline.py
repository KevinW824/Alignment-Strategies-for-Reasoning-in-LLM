#!/usr/bin/env python3
"""
Zero-shot baseline evaluation script for GSM8K dataset.

This script:
1. Loads GSM8K test examples (used as validation set)
2. Formats them using the r1_zero prompt
3. Generates outputs using vLLM
4. Calculates evaluation metrics
5. Serializes results to disk
"""

import json
import os
import sys
from typing import List, Dict, Any, Callable
from pathlib import Path

import typer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the reward function from the grader module
try:
    from scripts.drgrpo_grader import r1_zero_reward_fn
except ImportError:
    from drgrpo_grader import r1_zero_reward_fn


app = typer.Typer()


def load_jsonl_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from JSONL file.
    
    Args:
        data_path: Path to the JSONL file
        
    Returns:
        List of dictionaries
    """
    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def extract_ground_truth_answer(answer_str: str) -> str:
    """
    Extract the final numerical answer from GSM8K answer format.
    GSM8K answers are in format: "reasoning steps ... #### final_answer"
    
    Args:
        answer_str: The full answer string from GSM8K
        
    Returns:
        The final answer after ####
    """
    if "####" in answer_str:
        return answer_str.split("####")[1].strip()
    return answer_str.strip()


def format_prompts(examples: List[Dict[str, Any]], prompt_template: str) -> List[str]:
    """
    Format examples using the provided prompt template.
    
    Args:
        examples: List of examples with 'question' key
        prompt_template: Template string with {question} placeholder
        
    Returns:
        List of formatted prompt strings
    """
    prompts = []
    for example in examples:
        prompt = prompt_template.replace("{question}", example["question"])
        prompts.append(prompt)
    return prompts


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    output_dir: str = None,
) -> Dict[str, Any]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    
    Args:
        vllm_model: The vLLM model instance
        reward_fn: Function to compute rewards (format_reward, answer_reward, reward)
        prompts: List of formatted prompt strings
        ground_truths: List of ground truth answers
        eval_sampling_params: Sampling parameters for generation
        output_dir: Directory to save results (optional)
        
    Returns:
        Dictionary containing:
            - generations: List of generated texts
            - rewards: List of reward dictionaries
            - metrics: Aggregated metrics
    """
    print(f"\nGenerating {len(prompts)} completions...")
    
    # Generate outputs using vLLM
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    # Extract generated texts
    generations = [output.outputs[0].text for output in outputs]
    
    # Compute rewards for each generation
    print("Computing rewards...")
    rewards = []
    for generation, ground_truth in zip(tqdm(generations), ground_truths):
        reward_dict = reward_fn(generation, ground_truth)
        rewards.append(reward_dict)
    
    # Aggregate metrics
    total_examples = len(rewards)
    format_reward_1_answer_1 = sum(
        1 for r in rewards 
        if r["format_reward"] == 1.0 and r["answer_reward"] == 1.0
    )
    format_reward_1_answer_0 = sum(
        1 for r in rewards 
        if r["format_reward"] == 1.0 and r["answer_reward"] == 0.0
    )
    format_reward_0 = sum(1 for r in rewards if r["format_reward"] == 0.0)
    
    avg_total_reward = sum(r["reward"] for r in rewards) / total_examples
    avg_format_reward = sum(r["format_reward"] for r in rewards) / total_examples
    avg_answer_reward = sum(r["answer_reward"] for r in rewards) / total_examples
    
    metrics = {
        "total_examples": total_examples,
        "format_reward_1_answer_1": format_reward_1_answer_1,
        "format_reward_1_answer_0": format_reward_1_answer_0,
        "format_reward_0": format_reward_0,
        "accuracy": avg_answer_reward,
        "avg_total_reward": avg_total_reward,
        "avg_format_reward": avg_format_reward,
        "avg_answer_reward": avg_answer_reward,
    }
    
    results = {
        "generations": generations,
        "rewards": rewards,
        "metrics": metrics,
    }
    
    # Serialize results to disk if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                "generations": generations,
                "rewards": rewards,
                "metrics": metrics,
            }, f, indent=2)
        print(f"\nResults saved to {results_path}")
        
        # Save metrics separately
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
    
    return results


@app.command()
def main(
    data_path: str = typer.Option(
        "data/gsm8k/test.jsonl",
        help="Path to GSM8K test data (used as validation set)"
    ),
    prompt_path: str = typer.Option(
        "scripts/prompts/r1_zero.prompt",
        help="Path to r1_zero prompt template file"
    ),
    model_path: str = typer.Option(
        "Qwen/Qwen2.5-Math-1.5B",
        help="Path to model (local path or HuggingFace model ID)"
    ),
    output_dir: str = typer.Option(
        "results/math_baseline",
        help="Directory to save results"
    ),
    max_examples: int = typer.Option(
        None,
        help="Maximum number of examples to evaluate (None for all)"
    ),
    temperature: float = typer.Option(1.0, help="Sampling temperature"),
    top_p: float = typer.Option(1.0, help="Top-p sampling parameter"),
    max_tokens: int = typer.Option(1024, help="Maximum generation length"),
    gpu_memory_utilization: float = typer.Option(
        0.9, 
        help="GPU memory utilization for vLLM"
    ),
):
    """
    Evaluate Qwen 2.5 Math 1.5B zero-shot performance on GSM8K.
    
    Note: GSM8K doesn't have a separate validation set, so we use test.jsonl
    as the evaluation set (equivalent to MATH validation set in the original assignment).
    """
    print("=" * 80)
    print("GSM8K Zero-Shot Baseline Evaluation")
    print("=" * 80)
    
    # Load prompt template
    print(f"\nLoading r1_zero prompt template from {prompt_path}...")
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()
    print(f"Prompt template loaded.")
    
    # Load GSM8K data
    print(f"\nLoading GSM8K data from {data_path}...")
    examples = load_jsonl_data(data_path)
    
    if max_examples is not None:
        examples = examples[:max_examples]
        print(f"Limited to first {max_examples} examples")
    
    print(f"Loaded {len(examples)} examples")
    
    # Extract ground truth answers
    ground_truths = [extract_ground_truth_answer(ex["answer"]) for ex in examples]
    
    # Format prompts
    print("Formatting prompts...")
    prompts = format_prompts(examples, prompt_template)
    
    # Initialize vLLM model
    print(f"\nInitializing vLLM with model: {model_path}...")
    print("(This may take a few minutes on first run to download the model)")
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    print("Model loaded successfully!")
    
    # Set up sampling parameters as specified in the assignment
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    
    print(f"\nSampling parameters:")
    print(f"  - temperature: {temperature}")
    print(f"  - top_p: {top_p}")
    print(f"  - max_tokens: {max_tokens}")
    print(f"  - stop: ['</answer>']")
    
    # Evaluate model
    print("\n" + "=" * 80)
    print("Starting Evaluation")
    print("=" * 80)
    
    results = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_dir=output_dir,
    )
    
    # Print metrics
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"\nTotal examples: {results['metrics']['total_examples']}")
    print(f"\nCategory breakdown:")
    print(f"  (1) Correct (format=1, answer=1): {results['metrics']['format_reward_1_answer_1']}")
    print(f"  (2) Wrong answer (format=1, answer=0): {results['metrics']['format_reward_1_answer_0']}")
    print(f"  (3) Format error (format=0): {results['metrics']['format_reward_0']}")
    print(f"\nMetrics:")
    print(f"  - Accuracy: {results['metrics']['accuracy']:.4f} ({results['metrics']['accuracy']*100:.2f}%)")
    print(f"  - Average total reward: {results['metrics']['avg_total_reward']:.4f}")
    print(f"  - Average format reward: {results['metrics']['avg_format_reward']:.4f}")
    print(f"  - Average answer reward: {results['metrics']['avg_answer_reward']:.4f}")
    
    # Save full results with examples and prompts for analysis
    print(f"\nSaving detailed results for analysis...")
    full_results_path = os.path.join(output_dir, "full_results.json")
    with open(full_results_path, 'w') as f:
        json.dump({
            "examples": examples,
            "prompts": prompts,
            "ground_truths": ground_truths,
            "generations": results["generations"],
            "rewards": results["rewards"],
            "metrics": results["metrics"],
            "config": {
                "data_path": data_path,
                "model_path": model_path,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
        }, f, indent=2)
    print(f"Full results saved to {full_results_path}")
    
    # Print some example generations for inspection
    print("\n" + "=" * 80)
    print("Sample Generations (first 3 examples)")
    print("=" * 80)
    
    for i in range(min(3, len(examples))):
        print(f"\n--- Example {i+1} ---")
        print(f"Question: {examples[i]['question']}")
        print(f"Ground truth: {ground_truths[i]}")
        print(f"\nGeneration:\n{results['generations'][i]}")
        print(f"\nRewards: {results['rewards'][i]}")
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - evaluation_results.json  (generations and rewards)")
    print(f"  - metrics.json             (summary metrics)")
    print(f"  - full_results.json        (complete data for analysis)")


if __name__ == "__main__":
    app()
