#!/usr/bin/env python3
"""
Filter SFT dataset to only include examples that produce correct answers.

This script:
1. Loads the SFT dataset
2. Checks each response against the ground truth
3. Filters to only keep correct examples
4. Saves the filtered dataset
"""

import json
import argparse
from tqdm import tqdm

from scripts.drgrpo_grader import r1_zero_reward_fn


def extract_ground_truth_from_response(response: str) -> str:
    """
    Extract ground truth answer from SFT response.
    
    The response format is typically:
    "reasoning trace... <answer>ANSWER</answer>"
    
    We extract the answer from the <answer> tags.
    """
    if "<answer>" in response and "</answer>" in response:
        start = response.find("<answer>") + len("<answer>")
        end = response.find("</answer>")
        return response[start:end].strip()
    return ""


def filter_sft_dataset(input_path: str, output_path: str):
    """
    Filter SFT dataset to only include correct examples.
    
    Args:
        input_path: Path to input SFT JSONL file
        output_path: Path to save filtered JSONL file
    """
    print(f"Loading SFT dataset from {input_path}...")
    
    # Load all examples
    examples = []
    with open(input_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} examples")
    
    # Filter to correct examples
    correct_examples = []
    
    print("Filtering to correct examples...")
    for example in tqdm(examples):
        response = example["response"]
        
        # Extract ground truth from the response
        # Assuming the response contains the answer
        ground_truth = extract_ground_truth_from_response(response)
        
        if not ground_truth:
            # Skip if we can't extract ground truth
            continue
        
        # Check if response is correct using reward function
        reward_dict = r1_zero_reward_fn(response, ground_truth)
        
        # Keep only if answer is correct
        if reward_dict["answer_reward"] == 1.0:
            correct_examples.append(example)
    
    print(f"\nFiltering complete:")
    print(f"  Original examples: {len(examples)}")
    print(f"  Correct examples: {len(correct_examples)}")
    print(f"  Filtered out: {len(examples) - len(correct_examples)}")
    print(f"  Retention rate: {len(correct_examples) / len(examples):.2%}")
    
    # Save filtered dataset
    print(f"\nSaving filtered dataset to {output_path}...")
    with open(output_path, 'w') as f:
        for example in correct_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Filtered dataset saved with {len(correct_examples)} examples")


def main():
    parser = argparse.ArgumentParser(description="Filter SFT dataset to correct examples only")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/gsm8k/sft.jsonl",
        help="Path to input SFT JSONL file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/gsm8k/sft_correct.jsonl",
        help="Path to save filtered JSONL file"
    )
    
    args = parser.parse_args()
    
    filter_sft_dataset(args.input_path, args.output_path)


if __name__ == "__main__":
    main()

