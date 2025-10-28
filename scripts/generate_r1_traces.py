#!/usr/bin/env python3
"""
Generate R1-style reasoning traces for GSM8K training data.

This script:
1. Loads GSM8K training questions
2. Uses a strong reasoning model (e.g., Qwen2.5-72B-Instruct) to generate detailed reasoning traces
3. Validates the generated answers against ground truth
4. Saves correct reasoning traces for SFT

Usage:
    python scripts/generate_r1_traces.py --model_name Qwen/Qwen2.5-72B-Instruct --num_examples 1000
"""

import json
import argparse
import re
from typing import List, Dict, Optional
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import wandb


def extract_ground_truth_answer(gsm8k_answer: str) -> str:
    """Extract the final numerical answer from GSM8K format."""
    if "####" in gsm8k_answer:
        answer = gsm8k_answer.split("####")[1].strip()
        # Remove commas and extra spaces
        answer = answer.replace(",", "").strip()
        return answer
    return ""


def extract_answer_from_response(response: str) -> str:
    """
    Extract the answer from model response.
    Looks for patterns like <answer>X</answer> or final numerical value.
    """
    # First try to find <answer> tags
    if "<answer>" in response and "</answer>" in response:
        start = response.find("<answer>") + len("<answer>")
        end = response.find("</answer>")
        answer = response[start:end].strip()
        # Clean up the answer
        answer = re.sub(r'[^\d\.\-]', '', answer)
        return answer
    
    # Fallback: look for "The answer is X" or similar patterns
    patterns = [
        r'(?:final answer|answer|result) is[:\s]+\$?([0-9,]+\.?[0-9]*)',
        r'therefore[,\s]+\$?([0-9,]+\.?[0-9]*)',
        r'= \$?([0-9,]+\.?[0-9]*)\s*$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            answer = match.group(1).replace(",", "").strip()
            return answer
    
    return ""


def is_correct_answer(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    try:
        pred_num = float(predicted.replace(",", "").strip())
        gt_num = float(ground_truth.replace(",", "").strip())
        return abs(pred_num - gt_num) < 0.01
    except (ValueError, AttributeError):
        return predicted.strip() == ground_truth.strip()


def create_r1_prompt(question: str) -> str:
    """
    Create a prompt that encourages detailed reasoning similar to R1.
    
    Note: The prompt template already contains "Assistant: <think>",
    so we only provide instruction on what comes after.
    """
    prompt = f"""You are an expert problem solver. Solve the following math problem step by step with detailed reasoning.

Think through the problem carefully:
- Break down the problem into steps
- Show your reasoning process
- Verify your logic as you go
- Provide the final answer

Format your response as:
<think>
[Your detailed step-by-step reasoning here]
</think>
<answer>[final answer only]</answer>

Problem: {question}

Solution:"""
    return prompt


def generate_reasoning_traces_vllm(
    questions: List[str],
    ground_truths: List[str],
    model_name: str,
    gpu_device: str = "cuda:0",
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Generate reasoning traces using vLLM for efficient batch inference.
    
    Args:
        questions: List of math questions
        ground_truths: List of ground truth answers
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-72B-Instruct")
        gpu_device: GPU device to use
        temperature: Sampling temperature (higher = more diverse)
        max_tokens: Maximum tokens to generate
    
    Returns:
        Tuple of (correct_examples, all_examples) for SFT and ExIt respectively
    """
    print(f"\nInitializing vLLM with {model_name}...")
    
    # Initialize vLLM
    llm = LLM(
        model=model_name,
        device=gpu_device,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
    )
    
    # Create prompts
    prompts = [create_r1_prompt(q) for q in questions]
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        stop=["</answer>", "\n\nProblem:", "User:", "Human:"],
    )
    
    print(f"Generating reasoning traces for {len(questions)} examples...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Process outputs and validate
    correct_examples = []
    all_examples = []
    correct_count = 0
    
    for i, output in enumerate(tqdm(outputs, desc="Validating traces")):
        # Log progress to wandb every 10 examples
        if i > 0 and i % 10 == 0:
            wandb.log({
                "generation/processed": i,
                "generation/correct": correct_count,
                "generation/accuracy": correct_count / i if i > 0 else 0,
            })
        response = output.outputs[0].text.strip()
        
        # Ensure response has proper closing tags
        # The prompt already contains <think>, so we need to close it with </think> before <answer>
        if "</think>" not in response:
            # If there's an <answer> tag, insert </think> before it
            if "<answer>" in response:
                response = response.replace("<answer>", "</think> <answer>")
            else:
                response += " </think>"
        
        if "<answer>" in response and "</answer>" not in response:
            response += "</answer>"
        
        # Extract answer and validate
        predicted_answer = extract_answer_from_response(response)
        is_correct = predicted_answer and is_correct_answer(predicted_answer, ground_truths[i])
        
        example = {
            "question": questions[i],
            "response": response,
            "ground_truth": ground_truths[i],
            "predicted": predicted_answer,
            "is_correct": is_correct,
        }
        
        all_examples.append(example)
        
        if is_correct:
            correct_count += 1
            correct_examples.append(example)
    
    print(f"\nGeneration complete:")
    print(f"  Total examples: {len(questions)}")
    print(f"  Correct answers: {correct_count}")
    print(f"  Incorrect answers: {len(questions) - correct_count}")
    print(f"  Accuracy: {correct_count / len(questions):.2%}")
    
    # Log final metrics
    wandb.log({
        "generation/total": len(questions),
        "generation/correct_final": correct_count,
        "generation/incorrect_final": len(questions) - correct_count,
        "generation/accuracy_final": correct_count / len(questions),
    })
    
    return correct_examples, all_examples


def generate_reasoning_traces_hf(
    questions: List[str],
    ground_truths: List[str],
    model_name: str,
    gpu_device: str = "cuda:0",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    batch_size: int = 4,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Generate reasoning traces using HuggingFace Transformers (fallback if vLLM doesn't work).
    
    Returns:
        Tuple of (correct_examples, all_examples) for SFT and ExIt respectively
    """
    print(f"\nLoading model {model_name} with HuggingFace Transformers...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=gpu_device,
        trust_remote_code=True,
    )
    model.eval()
    
    correct_examples = []
    all_examples = []
    correct_count = 0
    
    print(f"Generating reasoning traces for {len(questions)} examples...")
    
    for i in tqdm(range(0, len(questions), batch_size)):
        batch_questions = questions[i:i+batch_size]
        batch_ground_truths = ground_truths[i:i+batch_size]
        
        prompts = [create_r1_prompt(q) for q in batch_questions]
        
        # Tokenize
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        # Decode
        responses = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Validate each response
        for j, response in enumerate(responses):
            response = response.strip()
            
            # Ensure proper closing tags
            # The prompt already contains <think>, so we need to close it with </think> before <answer>
            if "</think>" not in response:
                # If there's an <answer> tag, insert </think> before it
                if "<answer>" in response:
                    response = response.replace("<answer>", "</think> <answer>")
                else:
                    response += " </think>"
            
            if "<answer>" in response and "</answer>" not in response:
                response += "</answer>"
            
            predicted_answer = extract_answer_from_response(response)
            is_correct = predicted_answer and is_correct_answer(predicted_answer, batch_ground_truths[j])
            
            example = {
                "question": batch_questions[j],
                "response": response,
                "ground_truth": batch_ground_truths[j],
                "predicted": predicted_answer,
                "is_correct": is_correct,
            }
            
            all_examples.append(example)
            
            if is_correct:
                correct_count += 1
                correct_examples.append(example)
    
    print(f"\nGeneration complete:")
    print(f"  Total examples: {len(questions)}")
    print(f"  Correct answers: {correct_count}")
    print(f"  Incorrect answers: {len(questions) - correct_count}")
    print(f"  Accuracy: {correct_count / len(questions):.2%}")
    
    return correct_examples, all_examples


def create_sft_dataset(
    correct_examples: List[Dict[str, str]],
    all_examples: List[Dict[str, str]],
    prompt_template_path: str,
    output_path_correct: str,
    output_path_all: str,
):
    """
    Create final SFT datasets in the required format.
    
    Saves two files:
    1. Correct-only dataset for standard SFT
    2. All examples (correct + incorrect) for Expert Iteration
    """
    with open(prompt_template_path, 'r') as f:
        prompt_template = f.read()
    
    # Create correct-only dataset
    sft_correct = []
    for example in correct_examples:
        prompt = prompt_template.replace("{question}", example["question"])
        sft_correct.append({
            "prompt": prompt,
            "response": example["response"]
        })
    
    # Create full dataset with labels
    sft_all = []
    for example in all_examples:
        prompt = prompt_template.replace("{question}", example["question"])
        sft_all.append({
            "prompt": prompt,
            "response": example["response"],
            "ground_truth": example["ground_truth"],
            "predicted": example["predicted"],
            "is_correct": example["is_correct"],
        })
    
    # Save correct-only dataset
    print(f"\nSaving {len(sft_correct)} CORRECT examples to {output_path_correct}...")
    with open(output_path_correct, 'w') as f:
        for example in sft_correct:
            f.write(json.dumps(example) + '\n')
    print(f"Correct-only SFT dataset saved!")
    
    # Save all examples dataset
    print(f"\nSaving {len(sft_all)} ALL examples (for ExIt) to {output_path_all}...")
    with open(output_path_all, 'w') as f:
        for example in sft_all:
            f.write(json.dumps(example) + '\n')
    print(f"Full dataset (with labels) saved!")


def main():
    parser = argparse.ArgumentParser(description="Generate R1-style reasoning traces for GSM8K")
    
    # Data arguments
    parser.add_argument("--input_path", type=str, default="data/gsm8k/train.jsonl",
                        help="Path to GSM8K training data")
    parser.add_argument("--output_path_correct", type=str, default="data/gsm8k/sft_r1_correct.jsonl",
                        help="Path to save correct-only SFT dataset")
    parser.add_argument("--output_path_all", type=str, default="data/gsm8k/sft_r1_all.jsonl",
                        help="Path to save all examples (for Expert Iteration)")
    parser.add_argument("--prompt_template_path", type=str, default="scripts/prompts/r1_zero.prompt",
                        help="Path to prompt template")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                        help="Model to use for generating reasoning traces")
    parser.add_argument("--use_vllm", action="store_true", default=True,
                        help="Use vLLM for faster inference")
    parser.add_argument("--gpu_device", type=str, default="cuda:0",
                        help="GPU device to use")
    
    # Generation arguments
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Number of examples to process (None = all)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum tokens to generate")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for HF generation")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Generating R1-style Reasoning Traces for SFT")
    print("="*80)
    
    # Initialize wandb
    wandb.init(
        project="r1-trace-generation",
        name=f"generate_{args.num_examples if args.num_examples else 'all'}_{args.model_name.split('/')[-1]}",
        config={
            "model": args.model_name,
            "num_examples": args.num_examples,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "use_vllm": args.use_vllm,
        }
    )
    
    # Load GSM8K training data
    print(f"\nLoading GSM8K data from {args.input_path}...")
    with open(args.input_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    if args.num_examples:
        data = data[:args.num_examples]
    
    print(f"Loaded {len(data)} examples")
    
    questions = [ex["question"] for ex in data]
    ground_truths = [extract_ground_truth_answer(ex["answer"]) for ex in data]
    
    # Generate reasoning traces
    if args.use_vllm:
        try:
            correct_examples, all_examples = generate_reasoning_traces_vllm(
                questions, ground_truths, args.model_name,
                args.gpu_device, args.temperature, args.max_tokens
            )
        except Exception as e:
            print(f"\nvLLM failed with error: {e}")
            print("Falling back to HuggingFace Transformers...")
            correct_examples, all_examples = generate_reasoning_traces_hf(
                questions, ground_truths, args.model_name,
                args.gpu_device, args.temperature, args.max_tokens, args.batch_size
            )
    else:
        correct_examples, all_examples = generate_reasoning_traces_hf(
            questions, ground_truths, args.model_name,
            args.gpu_device, args.temperature, args.max_tokens, args.batch_size
        )
    
    # Create final SFT datasets
    create_sft_dataset(
        correct_examples, all_examples, 
        args.prompt_template_path, 
        args.output_path_correct, 
        args.output_path_all
    )
    
    print("\n" + "="*80)
    print("Generation Complete!")
    print(f"Correct examples: {len(correct_examples)}")
    print(f"Total examples (incl. incorrect): {len(all_examples)}")
    print(f"Incorrect examples: {len(all_examples) - len(correct_examples)}")
    print(f"\nOutputs:")
    print(f"  Correct-only (for SFT): {args.output_path_correct}")
    print(f"  All examples (for ExIt): {args.output_path_all}")
    print("="*80)
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()

