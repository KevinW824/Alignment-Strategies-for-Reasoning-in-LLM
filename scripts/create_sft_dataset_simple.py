#!/usr/bin/env python3
"""
Create SFT dataset from GSM8K train data.
Simple standalone version with no external dependencies.
"""

import json
import os


def extract_reasoning_and_answer(gsm8k_answer):
    """Extract reasoning steps and final answer from GSM8K answer format."""
    if "####" in gsm8k_answer:
        parts = gsm8k_answer.split("####")
        reasoning = parts[0].strip()
        answer = parts[1].strip()
    else:
        reasoning = gsm8k_answer.strip()
        answer = ""
    return reasoning, answer


def create_sft_example(question, reasoning, answer, prompt_template):
    """Create an SFT training example in the r1_zero format."""
    # Create the prompt
    prompt = prompt_template.replace("{question}", question)
    
    # Create the response (reasoning + answer)
    response = f"{reasoning} </think> <answer>{answer}</answer>"
    
    return {
        "prompt": prompt,
        "response": response
    }


def main():
    # Configuration
    input_path = "data/gsm8k/train.jsonl"
    output_path = "data/gsm8k/sft.jsonl"
    prompt_path = "scripts/prompts/r1_zero.prompt"
    
    print("="*80)
    print("Creating SFT Dataset from GSM8K")
    print("="*80)
    
    # Load prompt template
    print(f"\nLoading prompt template from {prompt_path}...")
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()
    
    # Load GSM8K training data
    print(f"Loading GSM8K training data from {input_path}...")
    examples = []
    with open(input_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} training examples")
    
    # Create SFT examples
    print("\nCreating SFT examples...")
    sft_examples = []
    
    for i, example in enumerate(examples):
        question = example['question']
        gsm8k_answer = example['answer']
        
        # Extract reasoning and answer
        reasoning, answer = extract_reasoning_and_answer(gsm8k_answer)
        
        # Create SFT example
        sft_example = create_sft_example(question, reasoning, answer, prompt_template)
        sft_examples.append(sft_example)
        
        # Print first few examples for inspection
        if i < 3:
            print(f"\n--- Example {i+1} ---")
            print(f"Question: {question[:80]}...")
            print(f"Reasoning: {reasoning[:100]}...")
            print(f"Answer: {answer}")
            print(f"Prompt length: {len(sft_example['prompt'])} chars")
            print(f"Response length: {len(sft_example['response'])} chars")
    
    # Save to file
    print(f"\nSaving {len(sft_examples)} SFT examples to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for example in sft_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"✓ SFT dataset saved to {output_path}")
    
    # Print statistics
    print("\n" + "="*80)
    print("Statistics")
    print("="*80)
    print(f"Total SFT examples: {len(sft_examples)}")
    
    avg_prompt_len = sum(len(ex['prompt']) for ex in sft_examples) / len(sft_examples)
    avg_response_len = sum(len(ex['response']) for ex in sft_examples) / len(sft_examples)
    
    print(f"Average prompt length: {avg_prompt_len:.0f} characters")
    print(f"Average response length: {avg_response_len:.0f} characters")
    
    # Show a complete example
    print("\n" + "="*80)
    print("Sample Complete SFT Example")
    print("="*80)
    print("\nPrompt:")
    print(sft_examples[0]['prompt'])
    print("\nResponse:")
    print(sft_examples[0]['response'][:500] + "...")
    
    print("\n" + "="*80)
    print("✓ Done! You can now use this dataset for SFT training.")
    print("="*80)
    print(f"\nDataset location: {output_path}")
    print(f"Number of examples: {len(sft_examples)}")


if __name__ == "__main__":
    main()

