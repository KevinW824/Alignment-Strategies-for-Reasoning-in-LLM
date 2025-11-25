import json, re, random, string, torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_dataset import format_prompt
from math_grader import answer_tag_reward_fn
from utils import extract_final

# MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
MODEL_ID = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)

gsm8k = load_dataset("openai/gsm8k", "main", split="train[:]")  # type: ignore
total = len(gsm8k)  # type: ignore
BATCH_SIZE = 64


def build_batch_prompts(batch):
    return [format_prompt(q.strip()) for q in batch["question"]]


positive, negative = [], []

total_correct = 0
total_samples = 0

for start in tqdm(range(0, total, BATCH_SIZE), desc="Evaluating GSM8K"):
    batch = gsm8k[start : start + BATCH_SIZE]  # type: ignore
    prompts = build_batch_prompts(batch)

    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, padding_side="left"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=512, temperature=0.2, do_sample=True
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for i, (q, gt_ans, gen_text) in enumerate(
        zip(batch["question"], batch["answer"], decoded)
    ):
        gt = extract_final(gt_ans)
        if not gt:
            continue
        metrics, reward = answer_tag_reward_fn(gen_text, gt, fast=True)

        is_correct = reward == 1.0
        total_samples += 1
        if is_correct:
            total_correct += 1
            positive.append(q)
        else:
            negative.append(q)

        acc = total_correct / total_samples if total_samples > 0 else 0.0
        tqdm.write(
            f"[{start+i}] GT={gt} | Reward={reward:.1f} | "
            f"{'✓' if is_correct else '✗'} | Acc={acc:.4f}"
        )

# Balance positive/negative sizes
while len(negative) < len(positive):
    negative.append("".join(random.choices(string.ascii_letters, k=75)))

out = {"positive_prompts": positive, "negative_prompts": negative}
with open("qwen3_17b_prompts_with_format.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"Saved {len(positive)} positive / {len(negative)} negative samples.")
print(
    f"📈 Final Accuracy: {total_correct}/{total_samples} = {total_correct / total_samples:.4f}"
)
