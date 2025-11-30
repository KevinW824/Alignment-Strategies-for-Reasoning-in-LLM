import time
import torch
import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    BatchEncoding,
)
from torch.utils.data import DataLoader
from dataclasses import dataclass
from tqdm import tqdm
from functools import partial

from dataset import SFTDataset
from vector_injector import ControlVectorInjector

from math_grader import r1_zero_reward_fn

from utils import extract_ground_truth_answer, answer_only_reward_fn


@dataclass
class ValidationConfig:
    # model_name: str = "Qwen/Qwen3-1.7B"
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    val_data_path: str = "../data/gsm8k/test.jsonl"
    prompt_template_path: str = "../scripts/prompts/r1_zero.prompt"
    vector_file_path: str = (
        "../outputs/re/contrastive_pca_vectors_qwen2.5_math_with_format_1.5B.pth"
        # "../outputs/re/contrastive_pca_vectors_qwen3_with_format_1.7B.pth"
    )

    injection_layer: Optional[int] = (
        None  # If None, will be dynamically set to the middle layer of the model
    )

    alpha_start: float = -1
    alpha_end: float = 1
    alpha_step: float = 0.1

    batch_size: int = 16
    max_new_tokens: int = 512
    max_examples: Optional[int] = None
    output_dir: Optional[str] = None

    require_format: Optional[bool] = True


def format_prompt_with_template(question: str, template: str) -> str:
    return template.replace("{question}", question)


def collate_fn_val(
    batch: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizerFast,
    prompt_template: str,
) -> Tuple[BatchEncoding, List[str], List[str]]:
    prompts = [
        format_prompt_with_template(item["question"], prompt_template) for item in batch
    ]
    responses = [item["answer"] for item in batch]
    questions = [item["question"] for item in batch]

    tokenized_prompts = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return tokenized_prompts, responses, questions


def plot_results(results_path: str, save_path: str, injection_layer: int):
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    sorted_alpha_strings = sorted(results.keys(), key=float)
    alphas_float = [float(k) for k in sorted_alpha_strings]
    accuracies = [results[k]["accuracy"] for k in sorted_alpha_strings]

    best_alpha_str = max(results, key=lambda k: results[k]["accuracy"])
    best_acc = results[best_alpha_str]["accuracy"]

    plt.figure(figsize=(10, 6))
    plt.plot(alphas_float, accuracies, marker="o", linestyle="-")

    baseline_acc = results.get("0.00", {"accuracy": 0})["accuracy"]
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
        dtype=torch.float16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(config.model_name).to(device)  # type: ignore
    model.eval()

    # Dynamically set injection_layer
    if config.injection_layer is None:
        num_hidden_layers = model.config.num_hidden_layers
        config.injection_layer = num_hidden_layers // 2
        print(
            f"Injection layer not specified, setting to middle layer: {config.injection_layer}"
        )

    del model

    # Output directory
    output_dir = config.output_dir or os.path.join(
        "outputs", config.model_name.split("/")[-1]
    )
    os.makedirs(output_dir, exist_ok=True)
    results_save_path = os.path.join(output_dir, "validation_results.json")

    # Load control vectors
    try:
        control_vectors = torch.load(config.vector_file_path).to(device)
    except FileNotFoundError:
        print(f"Error: Control vector file not found at {config.vector_file_path}")
        return None, None, None

    # Load prompt template
    if os.path.exists(config.prompt_template_path):
        with open(config.prompt_template_path, "r") as f:
            prompt_template = f.read()
    else:
        print(
            f"Warning: Prompt template not found at {config.prompt_template_path}. Using default."
        )
        prompt_template = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
"""

    dataset = SFTDataset(
        data_path=config.val_data_path, max_examples=config.max_examples
    )

    # Use partial to pass prompt_template to collate_fn
    collate_fn = partial(
        collate_fn_val, tokenizer=tokenizer, prompt_template=prompt_template
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
    )

    alphas = np.arange(config.alpha_start, config.alpha_end + 1e-5, config.alpha_step)
    alphas = np.round(alphas, 2)

    results = {}
    all_detailed_results = {}

    print(f"\nStarting validation on {len(dataset)} examples...")
    print(
        f"Testing {len(alphas)} alpha values from {config.alpha_start} to {config.alpha_end}"
    )
    print(f"Injecting into layer: {config.injection_layer}\n")

    for alpha in tqdm(alphas, desc="Evaluating Alphas", dynamic_ncols=True):
        alpha_key = f"{alpha:.2f}"
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.float16).to(device)  # type: ignore
        model.eval()
        injector = ControlVectorInjector(
            model, control_vectors, alpha=float(alpha), layers=[config.injection_layer]  # type: ignore
        )

        total_correct = 0
        total_samples = 0
        formatted_count = 0
        detailed_results_for_alpha = []

        with torch.inference_mode():
            for step, (batch_inputs, batch_responses, batch_questions) in enumerate(
                dataloader, start=1
            ):
                inputs = batch_inputs.to(device)
                injector.register_hooks()

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    temperature=1.0,
                )

                generated_texts = tokenizer.batch_decode(
                    outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )

                for i, (gen_text, gt) in enumerate(
                    zip(generated_texts, batch_responses)
                ):
                    # Use the same extraction logic as sft evaluation
                    gt_extracted = extract_ground_truth_answer(gt)
                    if not gt_extracted:
                        # Fallback if extraction fails
                        gt_extracted = gt.strip()

                    reward_dict = (
                        r1_zero_reward_fn(gen_text, gt_extracted, fast=True)
                        if config.require_format
                        else answer_only_reward_fn(
                            gen_text.splitlines()[-1], gt_extracted
                        )
                    )

                    is_formatted = reward_dict.get("format_reward", 0.0) == 1.0
                    is_correct = reward_dict.get("answer_reward", 0.0) == 1.0

                    if is_formatted:
                        formatted_count += 1
                    if is_correct:
                        total_correct += 1

                    total_samples += 1

                    detailed_results_for_alpha.append(
                        {
                            "question": batch_questions[i],
                            "predicted_answer": gen_text,
                            "true_answer": gt_extracted,
                            "is_correct": is_correct,
                            "is_formatted": is_formatted,
                        }
                    )

                acc = total_correct / total_samples if total_samples > 0 else 0
                fmt_acc = formatted_count / total_samples if total_samples > 0 else 0
                elapsed = time.time() - start_time
                tqdm.write(
                    f"[α={alpha_key}] step {step}/{len(dataloader)} | "
                    f"acc={acc:.4f} | {total_correct}/{total_samples} | "
                    f"fmt={fmt_acc:.4f}"
                    f"elapsed={elapsed/60:.1f} min"
                )

                del outputs
                torch.cuda.empty_cache()

        injector.remove()
        del model

        # Final stats for this alpha
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        format_accuracy = formatted_count / total_samples if total_samples > 0 else 0
        elapsed_total = time.time() - start_time
        results[alpha_key] = {
            "accuracy": accuracy,
            "format_accuracy": format_accuracy,
        }
        all_detailed_results[alpha_key] = detailed_results_for_alpha

        tqdm.write(
            f"Finished α={alpha_key} | Accuracy={accuracy:.4f} | "
            f"Format={format_accuracy:.4f} ({formatted_count}/{total_samples}) "
            f"| Time={elapsed_total/60:.1f} min\n"
        )

    # Save results
    with open(results_save_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(output_dir, "detailed_validation_results.json"), "w") as f:
        json.dump(all_detailed_results, f, indent=2)

    print(f"\nValidation complete. Results saved to {results_save_path}")
    return (
        results_save_path,
        os.path.join(output_dir, "alpha_accuracy_plot.png"),
        config.injection_layer,
    )


if __name__ == "__main__":

    config = ValidationConfig()
    results_file, plot_file, injection_layer = run_validation(config)

    if results_file and plot_file and injection_layer is not None:
        plot_results(results_file, plot_file, injection_layer)
    # plot_results(
    #     results_path="outputs/Qwen2.5-Math-1.5B/validation_results.json",
    #     save_path="outputs/Qwen2.5-Math-1.5B/alpha_accuracy_plot.png",
    #     injection_layer=0,
    # )
