# SFT Implementation Guide

## ✅ What's Been Implemented

I've implemented a complete SFT (Supervised Fine-Tuning) training script in `scripts/sft.py` with all the key components from the assignment.

---

## 📦 Key Components

### 1. **tokenize_prompt_and_output()** (Lines 85-163)

This is the core helper function requested in the assignment.

**What it does:**
- Takes lists of prompt strings and response strings
- Tokenizes them separately, then concatenates
- Creates a `response_mask` to identify which tokens are in the response
- Returns properly formatted tensors for autoregressive training

**Returns:**
```python
{
    'input_ids': Tensor[batch_size, seq_len-1],      # Input tokens (predict next)
    'labels': Tensor[batch_size, seq_len-1],          # Target tokens (shifted by 1)
    'response_mask': Tensor[batch_size, seq_len-1],   # 1 for response, 0 for prompt/pad
}
```

**Key features:**
- Handles variable-length sequences with padding
- Creates proper mask so loss is only computed on response tokens (not prompts)
- Implements the shift needed for autoregressive training (labels = input_ids shifted by 1)

### 2. **compute_sft_loss()** (Lines 169-208)

Computes the SFT training loss.

**What it does:**
- Runs forward pass through model
- Computes cross-entropy loss per token
- Applies `response_mask` to only compute loss on response tokens
- Averages over response tokens only

**Why mask is important:**
- We don't want to train on predicting the prompt tokens
- Only the response (reasoning + answer) should contribute to loss

### 3. **SFTDataset** (Lines 47-69)

PyTorch Dataset class for loading SFT data.

- Loads JSONL file with `prompt` and `response` keys
- Supports limiting to N examples (useful for experiments)

### 4. **Training Loop with Gradient Accumulation** (Lines 380-459)

Implements Algorithm with all the bells and whistles:

**Gradient Accumulation:**
- Divides loss by `gradient_accumulation_steps`
- Accumulates gradients over multiple batches
- Only calls `optimizer.step()` every K batches
- This allows effective batch sizes larger than GPU memory permits

**Other features:**
- Gradient clipping (max_grad_norm=1.0)
- Checkpointing every N steps
- Progress tracking with tqdm
- Loss logging

### 5. **Model Loading** (Lines 351-368)

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,           # Save memory
    attn_implementation="flash_attention_2",  # Faster attention
)
```

---

### Run SFT Training

**Basic usage** (with default parameters):
```bash
uv run python scripts/sft.py
```

**With custom parameters:**
```bash
uv run python scripts/sft.py \
    --sft-data-path data/gsm8k/sft.jsonl \
    --val-data-path data/gsm8k/test.jsonl \
    --model-path Qwen/Qwen2.5-Math-1.5B \
    --output-dir checkpoints/sft \
    --num-epochs 3 \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --learning-rate 5e-5 \
    --max-grad-norm 1.0
```

**Quick test on subset:**
```bash
uv run python scripts/sft.py \
    --max-train-examples 100 \
    --num-epochs 1 \
    --batch-size 2 \
    --gradient-accumulation-steps 4
```

---

## ⚙️ Default Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 4 | Batch size per GPU |
| `gradient_accumulation_steps` | 8 | Accumulate over 8 batches |
| `num_epochs` | 3 | Number of training epochs |
| `learning_rate` | 5e-5 | Learning rate |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `save_every_n_steps` | 500 | Save checkpoint every 500 steps |

**Effective batch size** = `batch_size` × `gradient_accumulation_steps` = 4 × 8 = **32**

---

## 📊 What Gets Saved

```
checkpoints/sft/
├── checkpoint-500/      # Checkpoint at step 500
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
├── checkpoint-1000/     # Checkpoint at step 1000
│   └── ...
└── final/              # Final model after all epochs
    └── ...
```

Each checkpoint is a complete, loadable model that you can use for evaluation or resume training.

---

## 🔬 Technical Details

### Why Shift Input/Labels?

For autoregressive training:
- **Input**: `[token_0, token_1, token_2, ..., token_n-2]`
- **Target**: `[token_1, token_2, token_3, ..., token_n-1]`

At position `i`, the model sees tokens `0` to `i` and predicts token `i+1`.

### Why Mask Response Tokens?

Example prompt+response:
```
"Question: What is 2+2?\nAssistant: <think> 2 plus 2 equals 4 </think> <answer>4</answer>"
```

**Without mask**: Model trains on predicting the question too (wasteful, already knows it)
**With mask**: Only trains on predicting the reasoning and answer (what we care about)

### Memory Optimization

Three techniques keep memory usage low:
1. **bfloat16**: Half-precision (16-bit instead of 32-bit)
2. **FlashAttention-2**: Optimized attention implementation
3. **Gradient Accumulation**: Simulate large batches without loading them all at once

---

## 🎯 Expected Results

After SFT training, you should see:

| Metric | Before SFT | After SFT (Expected) |
|--------|------------|---------------------|
| Format Accuracy | 20% | 80-90% |
| Answer Accuracy | 2.4% | 15-20% |

The model should:
- ✅ Follow the `</think> <answer>X</answer>` format consistently
- ✅ Show improved math reasoning
- ✅ Get more answers correct

---

## 📝 Deliverables

There are several experiments we can do. With this implementation, you can:

1. **Vary dataset size**: Use `--max-train-examples 128/256/512/1024`
2. **Filter correct examples**: Modify dataset to only include correct answers
3. **Track validation accuracy**: Add evaluation code using `evaluate_vllm()` from baseline

---

## ✅ Next Steps

1. **Test tokenization**: Run `test_sft_tokenization.py`
2. **Quick test**: Run SFT on 100 examples for 1 epoch
3. **Full training**: Run on full dataset
4. **Evaluate**: Use `math_baseline.py` to evaluate the trained model
5. **Experiments**: Try different dataset sizes as per assignment

