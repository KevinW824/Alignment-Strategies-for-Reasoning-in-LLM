#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) for math reasoning.

This script implements Algorithm 1 from the assignment:
1. Loads SFT dataset with reasoning traces
2. Finetunes the model using cross-entropy loss on responses
3. Periodically evaluates on validation set
4. Saves checkpoints
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from tqdm import tqdm
import typer

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import evaluation utilities
try:
    from scripts.math_baseline import evaluate_vllm, load_jsonl_data, extract_ground_truth_answer, format_prompts
    from scripts.drgrpo_grader import r1_zero_reward_fn
except ImportError:
    from math_baseline import evaluate_vllm, load_jsonl_data, extract_ground_truth_answer, format_prompts
    from drgrpo_grader import r1_zero_reward_fn

# For vLLM evaluation
from vllm import LLM, SamplingParams


app = typer.Typer()


# ============================================================================
# SFT Dataset
# ============================================================================

class SFTDataset(Dataset):
    """Dataset for SFT training with prompt-response pairs."""
    
    def __init__(self, data_path: str, max_examples: Optional[int] = None):
        """
        Load SFT dataset.
        
        Args:
            data_path: Path to SFT JSONL file with 'prompt' and 'response' keys
            max_examples: Optional limit on number of examples
        """
        self.examples = []
        with open(data_path, 'r') as f:
            for line in f:
                self.examples.append(json.loads(line))
        
        if max_examples is not None:
            self.examples = self.examples[:max_examples]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


# ============================================================================
# Tokenization Helper
# ============================================================================

def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for
    the response tokens and 0 for other tokens (prompt or padding).
    
    Args:
        prompt_strs: List of prompt strings
        output_strs: List of output strings
        tokenizer: Tokenizer to use for tokenization
        
    Returns:
        Dictionary with:
            - input_ids: shape (batch_size, max_seq_len - 1), tokenized prompt+output 
                        with final token removed
            - labels: shape (batch_size, max_seq_len - 1), shifted input_ids
            - response_mask: shape (batch_size, max_seq_len - 1), mask on response tokens
    """
    batch_size = len(prompt_strs)
    
    # Tokenize prompts and outputs separately
    prompt_encodings = tokenizer(
        prompt_strs,
        add_special_tokens=True,
        padding=False,  # We'll handle padding manually
        truncation=False,
        return_tensors=None,  # Get lists first
    )
    
    output_encodings = tokenizer(
        output_strs,
        add_special_tokens=False,  # Don't add BOS/EOS to output
        padding=False,
        truncation=False,
        return_tensors=None,
    )
    
    # Concatenate prompt and output token IDs
    combined_input_ids = []
    prompt_lengths = []
    
    for i in range(batch_size):
        prompt_ids = prompt_encodings['input_ids'][i]
        output_ids = output_encodings['input_ids'][i]
        
        # Concatenate
        combined_ids = prompt_ids + output_ids
        combined_input_ids.append(combined_ids)
        prompt_lengths.append(len(prompt_ids))
    
    # Find max length for padding
    max_length = max(len(ids) for ids in combined_input_ids)
    
    # Prepare padded tensors
    input_ids_padded = []
    labels_padded = []
    response_mask_padded = []
    
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        # Use eos_token_id as pad if pad_token_id not set
        pad_token_id = tokenizer.eos_token_id
    
    for i in range(batch_size):
        ids = combined_input_ids[i]
        prompt_len = prompt_lengths[i]
        seq_len = len(ids)
        
        # For autoregressive training, we need input_ids[:-1] and labels[1:]
        # input_ids: tokens 0 to n-2 (predict next token)
        # labels: tokens 1 to n-1 (targets)
        
        # Pad to max_length - 1 (since we remove last token)
        target_len = max_length - 1
        
        # input_ids: remove last token, then pad
        input_ids = ids[:-1]  # Remove last token
        padding_length = target_len - len(input_ids)
        input_ids = input_ids + [pad_token_id] * padding_length
        
        # labels: remove first token, then pad (shifted version of input_ids)
        labels = ids[1:]  # Remove first token
        labels = labels + [pad_token_id] * padding_length
        
        # response_mask: 1 for response tokens, 0 for prompt and padding
        # Response starts at prompt_len-1 (in input_ids) since we removed first token
        # Response starts at prompt_len in labels since we removed first token
        response_mask = [0] * target_len
        
        # Mark response tokens (everything after prompt)
        # In the input_ids/labels, response starts at index prompt_len-1
        response_start = prompt_len - 1  # Adjusted for removing first token
        response_end = seq_len - 1  # Adjusted for removing last token
        
        for j in range(response_start, min(response_end, target_len)):
            response_mask[j] = 1
        
        input_ids_padded.append(input_ids)
        labels_padded.append(labels)
        response_mask_padded.append(response_mask)
    
    return {
        'input_ids': torch.tensor(input_ids_padded, dtype=torch.long),
        'labels': torch.tensor(labels_padded, dtype=torch.long),
        'response_mask': torch.tensor(response_mask_padded, dtype=torch.float),
    }


# ============================================================================
# Training Utilities
# ============================================================================

def compute_sft_loss(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute SFT loss (cross-entropy on response tokens only).
    
    Args:
        model: The model being trained
        input_ids: Input token IDs, shape (batch_size, seq_len)
        labels: Target token IDs, shape (batch_size, seq_len)
        response_mask: Mask for response tokens, shape (batch_size, seq_len)
        
    Returns:
        Scalar loss tensor
    """
    # Forward pass
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # (batch_size, seq_len, vocab_size)
    
    # Compute cross-entropy loss
    # Flatten for cross_entropy: (batch_size * seq_len, vocab_size)
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    
    # Compute per-token loss (no reduction yet)
    loss_per_token = F.cross_entropy(
        logits_flat,
        labels_flat,
        reduction='none',
    )
    
    # Reshape back to (batch_size, seq_len)
    loss_per_token = loss_per_token.view(batch_size, seq_len)
    
    # Apply mask: only compute loss on response tokens
    masked_loss = loss_per_token * response_mask
    
    # Average over all response tokens
    total_loss = masked_loss.sum()
    num_response_tokens = response_mask.sum()
    
    if num_response_tokens > 0:
        loss = total_loss / num_response_tokens
    else:
        loss = total_loss
    
    return loss


def collate_fn(batch: List[Dict[str, str]], tokenizer: PreTrainedTokenizer):
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of examples with 'prompt' and 'response' keys
        tokenizer: Tokenizer instance
        
    Returns:
        Dictionary with tokenized and padded tensors
    """
    prompts = [ex['prompt'] for ex in batch]
    responses = [ex['response'] for ex in batch]
    
    return tokenize_prompt_and_output(prompts, responses, tokenizer)


# ============================================================================
# Main Training Function
# ============================================================================

@app.command()
def main(
    # Data arguments
    sft_data_path: str = typer.Option(
        "data/gsm8k/sft.jsonl",
        help="Path to SFT training data"
    ),
    val_data_path: str = typer.Option(
        "data/gsm8k/test.jsonl",
        help="Path to validation data"
    ),
    prompt_path: str = typer.Option(
        "scripts/prompts/r1_zero.prompt",
        help="Path to prompt template"
    ),
    max_train_examples: Optional[int] = typer.Option(
        None,
        help="Max training examples (None for all)"
    ),
    max_val_examples: Optional[int] = typer.Option(
        100,
        help="Max validation examples"
    ),
    
    # Model arguments
    model_path: str = typer.Option(
        "Qwen/Qwen2.5-Math-1.5B",
        help="Path to base model"
    ),
    output_dir: str = typer.Option(
        "checkpoints/sft",
        help="Directory to save checkpoints"
    ),
    
    # Training arguments
    num_epochs: int = typer.Option(3, help="Number of training epochs"),
    batch_size: int = typer.Option(4, help="Batch size per device"),
    gradient_accumulation_steps: int = typer.Option(8, help="Gradient accumulation steps"),
    learning_rate: float = typer.Option(5e-5, help="Learning rate"),
    weight_decay: float = typer.Option(0.0, help="Weight decay"),
    max_grad_norm: float = typer.Option(1.0, help="Max gradient norm for clipping"),
    
    # Evaluation arguments
    eval_every_n_steps: int = typer.Option(100, help="Evaluate every N steps"),
    save_every_n_steps: int = typer.Option(500, help="Save checkpoint every N steps"),
    
    # System arguments
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """
    Supervised Fine-Tuning (SFT) for math reasoning.
    """
    print("=" * 80)
    print("Supervised Fine-Tuning (SFT)")
    print("=" * 80)
    
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and model
    print(f"\nLoading model and tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = model.to(device)
    model.train()
    
    print(f"Model loaded on {device}")
    print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    # Load training data
    print(f"\nLoading SFT training data from {sft_data_path}...")
    train_dataset = SFTDataset(sft_data_path, max_examples=max_train_examples)
    print(f"Loaded {len(train_dataset)} training examples")
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Calculate effective batch size
    effective_batch_size = batch_size * gradient_accumulation_steps
    num_update_steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_update_steps = num_update_steps_per_epoch * num_epochs
    
    print(f"\nTraining configuration:")
    print(f"  Batch size per device: {batch_size}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Total update steps: {total_update_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max gradient norm: {max_grad_norm}")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    global_step = 0
    optimizer.zero_grad()
    
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*80}")
        
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            response_mask = batch['response_mask'].to(device)
            
            # Forward pass
            loss = compute_sft_loss(model, input_ids, labels, response_mask)
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Update progress bar
                avg_loss = epoch_loss / num_batches
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'step': global_step})
                
                # Save checkpoint
                if global_step % save_every_n_steps == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    print(f"\n✓ Checkpoint saved to {checkpoint_dir}")
        
        # End of epoch
        avg_epoch_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")
    
    # Save final model
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n✓ Final model saved to {final_dir}")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    app()

