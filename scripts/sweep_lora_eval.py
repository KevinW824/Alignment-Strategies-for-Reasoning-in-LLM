# should serve vllm with lora enabled: vllm serve models/base/Qwen2.5-Math-1.5B --enable-lora --port xxxx
import os
import time
import requests
from pathlib import Path
import wandb
from tqdm import tqdm

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from math_baseline import load_jsonl_data, extract_ground_truth_answer
from drgrpo_grader import r1_zero_reward_fn

BASE_DIR = "outputs/lora_sweep"
VLLM_URL = "http://localhost:8300"
VAL_PATH = "data/gsm8k/test.jsonl"
PROMPT_TEMPLATE = "scripts/prompts/r1_zero.prompt"

BATCH_SIZE = 32

LORA_CONFIGS = [
    {"r": 1, "alpha": 4, "dropout": 0.05, "use_dora": False},
    # {"r": 4, "alpha": 16, "dropout": 0.05, "use_dora": False},
    # {"r": 16, "alpha": 32, "dropout": 0.05, "use_dora": False},
]


def run_name(cfg):
    base = f"r{cfg['r']}_a{cfg['alpha']}_d{str(cfg['dropout']).replace('.', 'p')}"
    return base + ("_dora" if cfg["use_dora"] else "")


def vllm_generate_batch(prompts, adapter_name):
    """Send a batch of prompts to vLLM with LoRA adapter enabled."""
    url = f"{VLLM_URL}/v1/completions"

    payload = {
        "model": adapter_name if adapter_name else "Qwen/Qwen2.5-Math-1.5B",
        "prompt": prompts,
        "max_tokens": 1024,
        "temperature": 1.0,
        "top_p": 1.0,
        "stop": ["</answer>"],
    }

    r = requests.post(url, json=payload)
    r.raise_for_status()

    # vLLM returns one choice per input
    return [c["text"] for c in r.json()["choices"]]


def unload_lora_from_vllm(name):
    url = f"{VLLM_URL}/v1/unload_lora_adapter"
    payload = {"lora_name": name}
    r = requests.post(url, json=payload)


def load_lora_to_vllm(name, path):
    try:
        unload_lora_from_vllm(name)
    except:
        pass
    print(f"[vLLM] Loading adapter {name}...")
    url = f"{VLLM_URL}/v1/load_lora_adapter"
    payload = {"lora_name": name, "lora_path": str(Path(path).absolute())}
    r = requests.post(url, json=payload)
    r.raise_for_status()


def evaluate_baseline(prompts, truths):
    print("\n" + "=" * 80)
    print("Running BASELINE evaluation (raw base model)")
    print("=" * 80)

    correct = 0
    fmt = 0
    N = len(prompts)

    for idx in tqdm(range(0, N, BATCH_SIZE), desc="Baseline"):
        batch_prompts = prompts[idx : idx + BATCH_SIZE]
        batch_truths = truths[idx : idx + BATCH_SIZE]

        batch_resps = vllm_generate_batch(batch_prompts, adapter_name=None)

        for resp, truth in zip(batch_resps, batch_truths):
            reward = r1_zero_reward_fn(f"{resp}</answer>", truth)
            correct += reward["answer_reward"]
            fmt += reward["format_reward"]

    acc = correct / N
    fmt_rate = fmt / N

    print(f"[BASELINE] acc={acc:.4f}, fmt={fmt_rate:.4f}")

    wandb.log(
        {
            "baseline_accuracy": acc,
            "baseline_format": fmt_rate,
        }
    )


def main():
    wandb.init(
        project="lora-sweep-vllm-eval", name=f"eval_{time.strftime('%Y%m%d_%H%M%S')}"
    )

    # Load evaluation set
    val_data = load_jsonl_data(VAL_PATH)
    template = Path(PROMPT_TEMPLATE).read_text()

    prompts = [template.replace("{question}", ex["question"]) for ex in val_data]
    truths = [extract_ground_truth_answer(ex["answer"]) for ex in val_data]

    evaluate_baseline(prompts, truths)

    N = len(prompts)
    print(f"Total eval samples = {N}")

    for cfg in LORA_CONFIGS:
        name = run_name(cfg)
        lora_dir = f"{BASE_DIR}/{name}/final"

        if not os.path.exists(lora_dir):
            print(f"[Skip] No folder: {lora_dir}")
            continue

        print("\n" + "=" * 80)
        print(f"Evaluating (batched): {name}")
        print("=" * 80)

        load_lora_to_vllm(name, lora_dir)

        correct = 0
        fmt = 0

        for idx in tqdm(range(0, N, BATCH_SIZE), desc=f"Eval {name}"):
            batch_prompts = prompts[idx : idx + BATCH_SIZE]
            batch_truths = truths[idx : idx + BATCH_SIZE]

            try:
                batch_responses = vllm_generate_batch(batch_prompts, name)
            except:
                time.sleep(1)
                batch_responses = vllm_generate_batch(batch_prompts, name)

            # Evaluate batch
            for resp, truth in zip(batch_responses, batch_truths):
                reward = r1_zero_reward_fn(f"{resp}</answer>", truth)
                correct += reward["answer_reward"]
                fmt += reward["format_reward"]

            # Every batch, log intermediate results
            processed = min(idx + BATCH_SIZE, N)
            wandb.log(
                {
                    "intermediate/accuracy": correct / processed,
                    "intermediate/format_rate": fmt / processed,
                    "intermediate/count": processed,
                    "model": name,
                },
                step=processed,
            )

        acc = correct / N
        fmt_rate = fmt / N

        print(f"[FINAL] acc={acc:.4f}, fmt={fmt_rate:.4f}")

        wandb.log(
            {
                "model": name,
                "rank": cfg["r"],
                "alpha": cfg["alpha"],
                "dropout": cfg["dropout"],
                "use_dora": cfg["use_dora"],
                "accuracy": acc,
                "format_rate": fmt_rate,
            }
        )


if __name__ == "__main__":
    main()
