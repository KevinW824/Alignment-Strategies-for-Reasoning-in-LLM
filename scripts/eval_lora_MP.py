import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
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
        raise ValueError(f"Cannot parse LoRA config from folder name: {name}")
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
    adapter_bin = run_dir / "final" / "adapter_model.bin"
    if not adapter_bin.exists():
        print(f"[WARN] {adapter_bin} not found, skipping {name}")
        continue

    print(f"\n=== Evaluating LoRA run: {name} ===")
    r, alpha, dropout = parse_lora_from_name(name)
    print(f"LoRA config (from name): r={r}, alpha={alpha}, dropout={dropout}")

    # 1) load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    # 2) build LoRA config (matches adapter target modules)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    print("Wrapping base model with LoRA...")
    model = get_peft_model(base_model, lora_config)

    # 3) load adapter weights, FIXING key names
    print(f"Loading LoRA weights from {adapter_bin} ...")
    state = torch.load(adapter_bin, map_location="cpu")

    fixed_state = {}
    changed = 0
    for k, v in state.items():
        new_k = k
        if ".lora_A.weight" in new_k:
            new_k = new_k.replace(".lora_A.weight", ".lora_A.default.weight")
            changed += 1
        if ".lora_B.weight" in new_k:
            new_k = new_k.replace(".lora_B.weight", ".lora_B.default.weight")
            changed += 1
        fixed_state[new_k] = v

    print(f"Original keys: {len(state)}, after rename: {len(fixed_state)}, renamed entries: {changed}")

    missing, unexpected = model.load_state_dict(fixed_state, strict=False)
    print(f"Missing keys ({len(missing)}), first 5:", missing[:5])
    print(f"Unexpected keys ({len(unexpected)}), first 5:", unexpected[:5])

    model.to(device)
    model.eval()

    # 4) eval on examples
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
