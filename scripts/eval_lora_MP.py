import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------- paths & settings -----------------
MODEL_PATH = "Qwen/Qwen2.5-Math-1.5B"
SWEEP_DIR = Path("outputs/lora_sweep")
TEST_PATH = Path("data/gsm8k/test.jsonl")
OUT_PLOT = SWEEP_DIR / "gsm8k_lora_sweep_accuracy_bar.png"

MAX_SAMPLES = 50 # or set to None to eval all samples
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- helpers: answer parsing -----------------
def extract_gold(ans_str: str) -> str:
    """Get the gold final answer after '####'."""
    if "####" in ans_str:
        return ans_str.split("####")[-1].strip()
    return ans_str.strip()

num_re = re.compile(r"-?\d+\.?\d*")

def extract_pred(pred_str: str) -> str:
    """Grab the last number from the model output."""
    nums = num_re.findall(pred_str)
    return nums[-1] if nums else pred_str.strip()

def solve(model, tokenizer, question: str, max_new_tokens: int = 256) -> str:
    prompt = f"Q: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("A:", 1)[-1].strip()

# ----------------- LoRA config from folder name -----------------
def parse_lora_from_name(name: str):
    """
    Expect names like: r8_a16_d0p05 or r16_a32_d0p1
    Returns (r, alpha, dropout)
    """
    m = re.match(r"r(\d+)_a(\d+)_d0p(\d+)", name)
    if not m:
        # Fallback or skip
        return None
    r = int(m.group(1))
    alpha = int(m.group(2))
    d_str = m.group(3)  # '05' or '1'
    if len(d_str) == 1:
        dropout = int(d_str) / 10.0
    else:
        dropout = int(d_str) / 100.0
    return r, alpha, dropout

# ----------------- load test subset once -----------------
print(f"Loading first {MAX_SAMPLES} samples from {TEST_PATH} ...")
with open(TEST_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()[:MAX_SAMPLES]
examples = [json.loads(l) for l in lines]

# ----------------- tokenizer (shared) -----------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------- evaluate each LoRA run (sequential) -----------------
run_dirs = sorted(
    [p for p in SWEEP_DIR.iterdir() if p.is_dir() and p.name.startswith("r")]
)

final_acc = {}  # run name -> accuracy

for run_dir in run_dirs:
    name = run_dir.name
    final_adapter_dir = run_dir / "final"
    
    # Check if adapter config exists
    if not (final_adapter_dir / "adapter_config.json").exists():
        print(f"[WARN] Adapter not found in {final_adapter_dir}, skipping {name}")
        continue

    print(f"\n=== Evaluating LoRA run: {name} ===")
    
    # 1) load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    # 2) Load LoRA adapter using PeftModel
    print(f"Loading LoRA adapter from {final_adapter_dir} ...")
    model = PeftModel.from_pretrained(base_model, final_adapter_dir)
    
    model.to(device)
    model.eval()

    # 3) eval on examples
    correct = 0
    total = 0
    for ex in tqdm(examples, desc=f"{name} eval"):
        q = ex["question"]
        gold_full = ex["answer"]
        gold = extract_gold(gold_full)

        pred_full = solve(model, tokenizer, q)
        pred = extract_pred(pred_full)

        if pred == gold:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    final_acc[name] = acc
    print(f"{name}: {correct}/{total} = {acc * 100:.2f}%")

    # free GPU before next run
    del model, base_model
    torch.cuda.empty_cache()
    

# ----------------- bar chart of final accuracies -----------------
if not final_acc:
    print("No runs evaluated, nothing to plot.")
else:
    print("\nPlotting bar chart of final accuracies...")
    names = list(final_acc.keys())
    accs = [final_acc[n] * 100 for n in names]
    x = range(len(names))

    plt.figure()
    bars = plt.bar(x, accs)
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title(f"GSM8K Accuracy (first {MAX_SAMPLES} samples) – LoRA Sweep")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PLOT)
    print(f"Saved bar chart to {OUT_PLOT}")

    print("\nFinal accuracies:")
    for name, acc in final_acc.items():
        print(f"{name}: {acc * 100:.2f}%")