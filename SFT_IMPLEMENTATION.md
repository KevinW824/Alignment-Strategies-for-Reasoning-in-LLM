# SFT Implementation Guide

## Tokenization Rules for `tokenize_prompt_and_output`

This document explains the correct implementation of the `tokenize_prompt_and_output` function for supervised fine-tuning.

### Function Signature

```python
def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, torch.Tensor]
```

### Implementation Requirements

#### 1. Tokenize Prompts and Outputs Separately

```python
# Tokenize prompts WITH special tokens (BOS)
prompt_encodings = tokenizer(
    prompt_strs,
    add_special_tokens=True,
    padding=False,
    truncation=False,
    return_tensors=None,
)

# Tokenize outputs WITHOUT special tokens
output_encodings = tokenizer(
    output_strs,
    add_special_tokens=False,  # Important!
    padding=False,
    truncation=False,
    return_tensors=None,
)
```

#### 2. Concatenate and Prepare Sequences

For each example:
1. Concatenate: `combined_ids = prompt_ids + output_ids`
2. Find max combined length across batch
3. Target length: `max_length - 1` (we remove one token for autoregressive training)

#### 3. Padding Logic (Critical!)

For sequences shorter than `max_length - 1`:

**`input_ids` padding:**
- Start with: `combined_ids[:-1]` (remove last token)
- Pad with: `[0] * (padding_length - 1) + [EOS_TOKEN_ID]`
- Format: `[actual_tokens...] + [0, 0, ..., 0, EOS]`

**`labels` padding:**
- Start with: `combined_ids[1:]` (remove first token, creating shifted version)
- Pad with: `[EOS_TOKEN_ID] * padding_length`
- Format: `[actual_tokens...] + [EOS, EOS, ..., EOS]`

**Example:**
```python
# Original combined: [9707, 11, 1879, 0, 9707, 11, 1879, 0]
# Max length - 1 = 9, current length = 8, needs 2 padding

# input_ids:  [9707, 11, 1879, 0, 9707, 11, 1879] + [0, 151643]
#              └─────── combined[:-1] ──────────┘   └─ padding ─┘

# labels:     [11, 1879, 0, 9707, 11, 1879, 0] + [151643, 151643]
#              └───────── combined[1:] ───────┘   └── padding ──┘
```

#### 4. Response Mask Construction

The `response_mask` indicates which tokens in `labels` are part of the response (vs. prompt or padding):

```python
response_mask = [0] * target_len  # Initialize all to 0

# Response starts after the prompt
response_start = prompt_len - 1  # Adjusted for shifted indexing

# Response ends before padding
response_end = min(len(combined_ids) - 1, target_len)

# Mark response tokens as 1
for j in range(response_start, response_end):
    response_mask[j] = 1
```

**Key insight:** After removing the first token to create `labels`, the response starts at index `(prompt_len - 1)` instead of `prompt_len`.

### Return Format

```python
{
    'input_ids': torch.Tensor,      # Shape: (batch_size, max_length - 1)
    'labels': torch.Tensor,          # Shape: (batch_size, max_length - 1)
    'response_mask': torch.Tensor,   # Shape: (batch_size, max_length - 1), dtype=float
}
```

### Complete Implementation

```python
def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, torch.Tensor]:
    batch_size = len(prompt_strs)
    
    # Tokenize separately
    prompt_encodings = tokenizer(
        prompt_strs,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_tensors=None,
    )
    
    output_encodings = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_tensors=None,
    )
    
    # Concatenate
    combined_input_ids = []
    prompt_lengths = []
    
    for i in range(batch_size):
        prompt_ids = prompt_encodings['input_ids'][i]
        output_ids = output_encodings['input_ids'][i]
        combined_ids = prompt_ids + output_ids
        combined_input_ids.append(combined_ids)
        prompt_lengths.append(len(prompt_ids))
    
    # Prepare padded tensors
    max_length = max(len(ids) for ids in combined_input_ids)
    target_len = max_length - 1
    
    input_ids_padded = []
    labels_padded = []
    response_mask_padded = []
    
    pad_token_id = 0
    
    for i in range(batch_size):
        ids = combined_input_ids[i]
        prompt_len = prompt_lengths[i]
        
        # Create input_ids and labels
        input_ids = ids[:-1]
        labels = ids[1:]
        
        # Compute padding needed
        padding_length = target_len - len(input_ids)
        
        if padding_length > 0:
            # Pad input_ids: [0, 0, ..., EOS]
            input_ids = input_ids + [pad_token_id] * (padding_length - 1) + [tokenizer.eos_token_id]
            
            # Pad labels: [EOS, EOS, ..., EOS]
            labels = labels + [tokenizer.eos_token_id] * padding_length
        
        # Create response mask
        response_mask = [0] * target_len
        response_start = prompt_len - 1
        response_end = min(len(ids) - 1, target_len)
        
        for j in range(response_start, response_end):
            response_mask[j] = 1
        
        input_ids_padded.append(input_ids)
        labels_padded.append(labels)
        response_mask_padded.append(response_mask)
    
    return {
        'input_ids': torch.tensor(input_ids_padded, dtype=torch.long),
        'labels': torch.tensor(labels_padded, dtype=torch.long),
        'response_mask': torch.tensor(response_mask_padded, dtype=torch.float),
    }
```

### Why These Rules?

1. **Autoregressive Training:** Language models predict the next token given previous context
   - `input_ids[t]` → model → predict `labels[t]`
   - This is why we shift: `input_ids = combined[:-1]`, `labels = combined[1:]`

2. **Padding Asymmetry:**
   - `input_ids` needs meaningful context, so we pad with `0` and end with `EOS`
   - `labels` represents targets, so padding tokens should predict `EOS`

3. **Response Mask:**
   - Loss should only be computed on response tokens
   - Prompt tokens: model shouldn't be penalized for "predicting" the given prompt
   - Padding tokens: no actual content to learn from

### Testing

Run the test with:
```bash
.venv/bin/pytest -k test_tokenize_prompt_and_output -v
```

Expected output: `PASSED`
