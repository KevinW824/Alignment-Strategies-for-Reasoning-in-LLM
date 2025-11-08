import json, re, random, string, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_dataset import format_prompt
from run_validation import extract_answer_from_response

from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)


def build_prompt(question: str) -> str:
    return format_prompt(question=question.strip())


def extract_final(text: str):
    m = re.search(r"####\s*([\d\.]+)", text)
    return m.group(1) if m else None


def get_answer(question: str) -> str:
    prompt = build_prompt(question)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=512, temperature=0.2, top_p=0.9, do_sample=False
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text


gsm8k = load_dataset("openai/gsm8k", "main", split="train[:]")

total = len(gsm8k)  # type: ignore[arg-type]

positive, negative = [], []
for i, item in tqdm(enumerate(gsm8k), total=total, desc="Processing GSM8K"):
    q, gt_ans = item["question"], extract_final(item["answer"])

    print("=" * 80)
    print(f"[{i}] Question:\n{q.strip()}\n")

    gen_text = get_answer(q)
    pred = extract_answer_from_response(gen_text)

    print(f"--- Model Output ---\n{gen_text.strip()}\n")
    print(f"GT={gt_ans} | PRED={pred} | {'✓ CORRECT' if pred==gt_ans else '✗ WRONG'}")

    if pred == gt_ans and pred is not None:
        positive.append(q)
    else:
        negative.append(q)

    if (i + 1) % 5 == 0 or i == total - 1:
        pct = (i + 1) / total * 100
        print(
            f"\n🔹 Progress: {i+1}/{total} ({pct:.2f}%) | "
            f"{len(positive)} positive | {len(negative)} negative\n"
        )
while len(negative) < len(positive) * 2:
    negative.append("".join(random.choices(string.ascii_letters, k=75)))

out = {"positive_prompts": positive, "negative_prompts": negative}
with open("qwen25_math15b_prompts.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(
    f"\n✅ Finished! Saved {len(positive)} positive / {len(negative)} negative prompts to qwen25_math15b_prompts.json."
)
