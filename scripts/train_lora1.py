#!/usr/bin/env python3
"""
LoRA Supervised Fine-Tuning (SFT) for math reasoning.

Standalone:
- Reads JSONL with {"prompt": ..., "response": ...}
- Builds prompt+response sequences, masks out prompt tokens in labels
- Trains ONLY LoRA adapter weights on top of a base (or SFT) model
"""

import os
import argparse
import json
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict



# =====================================================================
# Dataset
# =====================================================================

class SFTDataset(Dataset):
    """Dataset for SFT training with prompt-response pairs.

    Supports:
      - {"prompt": ..., "response": ...}
      - {"question": ..., "answer": ...}  (GSM8K style)
    """

    def __init__(self, data_path: str, max_examples: Optional[int] = None):
        self.examples: List[Dict[str, Any]] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))
        if max_examples is not None:
            self.examples = self.examples[:max_examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        ex = self.examples[idx]

        # If file already has prompt/response, just use them
        if "prompt" in ex and "response" in ex:
            prompt = ex["prompt"]
            response = ex["response"]
        # GSM8K-style: map question/answer -> prompt/response
        elif "question" in ex and "answer" in ex:
            prompt = ex["question"]
            response = ex["answer"]
        else:
            raise KeyError(
                f"Example missing expected keys. Got keys: {list(ex.keys())}"
            )

        return {"prompt": prompt, "response": response}



# =====================================================================
# Tokenization & collator
# =====================================================================

def tokenize_prompt_and_output(
    prompts: List[str],
    responses: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """
    Simple scheme:
      - input_ids = prompt_tokens + response_tokens (truncated / padded)
      - labels = input_ids but:
          * prompt tokens -> -100
          * padding tokens -> -100
      - attention_mask = 1 for real tokens, 0 for padding
    """
    input_batch = []
    label_batch = []
    attn_batch = []

    pad_id = tokenizer.pad_token_id

    for prompt, resp in zip(prompts, responses):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        resp_ids = tokenizer.encode(resp, add_special_tokens=False)

        ids = prompt_ids + resp_ids

        # Truncate
        ids = ids[:max_length]

        # Make labels
        labels = ids.copy()

        # Mask out prompt tokens in labels
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100

        # Attention mask
        attn = [1] * len(ids)

        # Pad
        pad_len = max_length - len(ids)
        if pad_len > 0:
            ids += [pad_id] * pad_len
            attn += [0] * pad_len
            labels += [-100] * pad_len

        input_batch.append(ids)
        label_batch.append(labels)
        attn_batch.append(attn)

    return {
        "input_ids": torch.tensor(input_batch, dtype=torch.long),
        "labels": torch.tensor(label_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attn_batch, dtype=torch.long),
    }


def collate_fn(batch: List[Dict[str, str]], tokenizer: PreTrainedTokenizer, max_length: int):
    prompts = [ex["prompt"] for ex in batch]
    responses = [ex["response"] for ex in batch]
    return tokenize_prompt_and_output(prompts, responses, tokenizer, max_length)


# =====================================================================
# LoRA SFT training
# =====================================================================

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA SFT training (standalone)")

    # Data / model paths
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Base or SFT model path (HF ID or local folder)",
    )
    parser.add_argument("--sft_data_path", type=str, default="data/gsm8k/sft_original.jsonl",
                        help="JSONL with 'prompt' and 'response'")
    parser.add_argument("--output_dir", type=str, default="checkpoints/lora_sft",
                        help="Directory for LoRA adapters")
    parser.add_argument("--max_train_examples", type=int, default=None)

    # Training
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)#可以高一点 或者10 epoch

    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=1024)

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj",
                        help="Comma-separated module name fragments")

    # System
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    return parser


def main():
    args = parse_args().parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("LoRA SFT Training (standalone)")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Data path:  {args.sft_data_path}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)

    # ---------------- Tokenizer & base model ----------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    # ---------------- Apply LoRA ----------------
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(base_model, lora_config)
    model.to(args.device)
    model.train()

    print("\nLoRA config:")
    print(f"  r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  target_modules={target_modules}")
    model.print_trainable_parameters()

    # ---------------- Data ----------------
    train_dataset = SFTDataset(args.sft_data_path, max_examples=args.max_train_examples)
    print(f"\nLoaded {len(train_dataset)} training examples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, args.max_length),
    )

    # ---------------- Optimizer ----------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    effective_bs = args.batch_size * args.gradient_accumulation_steps
    print("\nTraining config:")
    print(f"  epochs:        {args.num_epochs}")
    print(f"  batch size:    {args.batch_size}")
    print(f"  grad accum:    {args.gradient_accumulation_steps}")
    print(f"  eff batch:     {effective_bs}")
    print(f"  lr:            {args.learning_rate}")
    print(f"  max_grad_norm: {args.max_grad_norm}")
    print(f"  device:        {args.device}")
    print("=" * 80)

    # ---------------- Training loop ----------------
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.num_epochs):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'=' * 80}")

        epoch_loss = 0.0
        num_batches = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss  # HF handles ignore_index=-100

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            num_batches += 1

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = epoch_loss / max(1, num_batches)
                progress.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step})

        avg_epoch_loss = epoch_loss / max(1, num_batches)
        print(f"\nEpoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

        # Save LoRA adapter
        epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch + 1}")
        os.makedirs(epoch_dir, exist_ok=True)
        adapter_state = get_peft_model_state_dict(model)
        torch.save(adapter_state, os.path.join(epoch_dir, "adapter_model.bin"))
        tokenizer.save_pretrained(epoch_dir)
        print(f"Saved LoRA adapter to {epoch_dir}")

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    adapter_state = get_peft_model_state_dict(model)
    torch.save(adapter_state, os.path.join(final_dir, "adapter_model.bin"))
    tokenizer.save_pretrained(final_dir)
    print(f"\nFinal LoRA adapter saved to {final_dir}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
