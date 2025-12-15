import torch
import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    BatchEncoding,
)
from torch.utils.data import DataLoader
from dataclasses import dataclass
from tqdm import tqdm

from sft_dataset import SFTDataset, format_prompt
from vector_injector import ControlVectorInjector


@dataclass
class ValidationConfig:
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    val_data_path: str = "../data/gsm8k/test.jsonl"
    vector_file_path: str = (
        "../outputs/re/contrastive_pca_vectors_qwen2.5_math_1.5B.pth"
    )

    injection_layer: Optional[int] = (
        None  # If None, will be dynamically set to the middle layer of the model
    )

    alpha_start: float = -2.0
    alpha_end: float = 2.0

    batch_size: int = 8
    max_new_tokens: int = 512
    max_examples: Optional[int] = None
    output_dir: Optional[str] = None


def collate_fn_val(
    batch: List[Dict[str, str]], tokenizer: PreTrainedTokenizerFast
) -> Tuple[BatchEncoding, List[str], List[str]]:
    prompts = [format_prompt(item["question"]) for item in batch]
    responses = [item["answer"] for item in batch]
    questions = [item["question"] for item in batch]

    tokenized_prompts = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return tokenized_prompts, responses, questions


import re


def extract_answer_from_response(response: str) -> str:
    """
    Extract the final numeric answer from model output.
    Priority:
      1. Try last non-empty line (usually '#### X')
      2. Fallback: search for <answer>...</answer> or other patterns.
    Returns a clean numeric string or '' if not found.
    """
    if not response:
        return ""

    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    if not lines:
        return ""

    last_line = lines[-1]

    m = re.search(r"####\s*([\-]?\d*\.?\d+)$", last_line)
    if m:
        return m.group(1)
    m = re.search(r"([\-]?\d*\.?\d+)(?!.*\d)", last_line)
    if m:
        return m.group(1)

    # If the last line didn't contain a number, check for <answer> tags
    if "<answer>" in response and "</answer>" in response:
        start = response.find("<answer>") + len("<answer>")
        end = response.find("</answer>", start)
        ans = response[start:end].strip()
        ans = re.sub(r"[^\d\.\-]", "", ans)
        if ans:
            return ans

    # Fallback regexes (in case model didn't follow format)
    patterns = [
        r"####\s*([\-]?\d*\.?\d+)",
        r"(?:final answer|answer|result)\s*(?:is|:)\s*([\-]?\d*\.?\d+)",
        r"therefore[,\s]+\$?([\-]?\d*\.?\d+)",
        r"boxed\{([\-]?\d*\.?\d+)\}",
        r"=\s*\$?([\-]?\d*\.?\d+)\s*$",
    ]
    for p in patterns:
        m = re.search(p, response, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).replace(",", "")

    return ""


def plot_results(results_path: str, save_path: str, injection_layer: int):
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    with open(results_path, "r") as f:
        results = json.load(f)  # results = {"-1.00": 0.0061, "0.00": 0.0516, ...}

    sorted_alpha_strings = sorted(results.keys(), key=float)

    alphas_float = [float(key) for key in sorted_alpha_strings]

    accuracies = [results[key] for key in sorted_alpha_strings]

    best_alpha_str = max(results, key=results.get)
    best_acc = results[best_alpha_str]

    plt.figure(figsize=(10, 6))
    plt.plot(alphas_float, accuracies, marker="o", linestyle="-")

    baseline_acc = results.get("0.00", results.get("0.0", results.get("0", 0)))
    plt.axhline(
        y=baseline_acc,
        color="gray",
        linestyle="--",
        label=f"Baseline (α=0) Acc: {baseline_acc:.4f}",
    )

    plt.axvline(
        x=float(best_alpha_str),
        color="r",
        linestyle="--",
        label=f"Best α: {best_alpha_str} (Acc: {best_acc:.4f})",
    )

    plt.title(f"Accuracy vs. Alpha (Layer {injection_layer})")
    plt.xlabel("Alpha (α)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"\nPlot saved to {save_path}")
    print(f"Best alpha found: {best_alpha_str} with accuracy: {best_acc:.4f}")


def run_validation(config: ValidationConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        config.model_name,
        padding_side="left",
        torch_dtype=torch.float16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(config.model_name).to(
        device  # type: ignore
    )
    model.eval()

    # Dynamically set injection_layer to the middle layer if not specified
    if config.injection_layer is None:
        num_hidden_layers = model.config.num_hidden_layers
        config.injection_layer = num_hidden_layers // 2
        print(
            f"Injection layer not specified, setting to middle layer: {config.injection_layer}"
        )

    # Determine output directory
    if config.output_dir:
        output_dir = config.output_dir
    else:
        model_base_name = config.model_name.split("/")[-1]
        output_dir = os.path.join("outputs", model_base_name)
    os.makedirs(output_dir, exist_ok=True)

    results_save_path = os.path.join(output_dir, "validation_results.json")
    plot_save_path = os.path.join(output_dir, "alpha_accuracy_plot.png")

    try:
        control_vectors = torch.load(config.vector_file_path).to(device)
    except FileNotFoundError:
        print(f"Error: Control vector file not found at {config.vector_file_path}")
        return None, None, None

    dataset = SFTDataset(
        data_path=config.val_data_path, max_examples=config.max_examples
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=lambda batch: collate_fn_val(batch, tokenizer),
    )

    # Generate alphas with a fixed step of 0.1 and ensure 0.0 is included
    alphas_list = list(np.arange(config.alpha_start, config.alpha_end + 0.001, 0.1))
    if 0.0 not in alphas_list:
        alphas_list.append(0.0)
    alphas = np.array(sorted(list(set(alphas_list))))  # Remove duplicates and sort

    results = {}
    all_detailed_results = {}

    print(f"Starting validation on {len(dataset)} examples...")
    print(
        f"Testing {len(alphas)} alpha values from {config.alpha_start} to {config.alpha_end} with step 0.1"
    )
    print(f"Injecting into layer: {config.injection_layer}")

    if config.injection_layer is None:
        # This should not happen due to the logic above, but it satisfies the type checker
        raise ValueError("injection_layer cannot be None")

    for alpha in tqdm(alphas, desc="Evaluating Alphas"):
        alpha_val = round(float(alpha), 2)
        alpha_key = f"{alpha_val:.2f}"

        injector = ControlVectorInjector(
            model, control_vectors, alpha=alpha_val, layers=[config.injection_layer]
        )

        total_correct = 0
        total_samples = 0
        detailed_results_for_alpha = []

        with torch.inference_mode():
            for batch_inputs, batch_responses, batch_questions in tqdm(
                dataloader, desc=f"Alpha {alpha_val:.2f}", leave=False
            ):

                inputs = batch_inputs.to(device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

                generated_texts = tokenizer.batch_decode(
                    outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )

                for i, (gen_text, true_response) in enumerate(
                    zip(generated_texts, batch_responses)
                ):
                    pred_answer = extract_answer_from_response(gen_text)
                    true_answer = extract_answer_from_response(true_response)

                    is_correct = (
                        pred_answer is not None
                        and true_answer is not None
                        and pred_answer == true_answer
                    )

                    if is_correct:
                        total_correct += 1
                    total_samples += 1

                    detailed_results_for_alpha.append(
                        {
                            "question": batch_questions[i],
                            "generated_text": gen_text,
                            "predicted_answer": pred_answer,
                            "true_answer": true_answer,
                            "is_correct": is_correct,
                        }
                    )

        injector.remove()

        accuracy = total_correct / total_samples if total_samples > 0 else 0
        results[alpha_key] = accuracy
        all_detailed_results[alpha_key] = detailed_results_for_alpha

        print(
            f"Alpha: {alpha_val:.2f}, Correct: {total_correct}/{total_samples}, Accuracy: {accuracy:.4f}"
        )

    # Save aggregated accuracy results
    with open(results_save_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save detailed results
    detailed_results_save_path = os.path.join(
        output_dir, "detailed_validation_results.json"
    )
    with open(detailed_results_save_path, "w") as f:
        json.dump(all_detailed_results, f, indent=2)

    print(f"\nValidation complete. Results saved to {results_save_path}")
    print(f"Detailed results saved to {detailed_results_save_path}")
    return results_save_path, plot_save_path, config.injection_layer


if __name__ == "__main__":

    config = ValidationConfig()
    results_file, plot_file, injection_layer = run_validation(config)

    if results_file and plot_file and injection_layer is not None:
        plot_results(results_file, plot_file, injection_layer)
