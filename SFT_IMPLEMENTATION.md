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

---

## Computing Per-Token Entropy

### Purpose

When doing RL, it's useful to track per-token entropies to monitor if the model's predictive distribution is becoming overconfident or underconfident. The entropy measures the uncertainty in the model's predictions.

### Entropy Definition

For a discrete distribution p(x) over vocabulary X:

```
H(p) = -∑_{x∈X} p(x) log p(x)
```

### Function Signature

```python
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the per-token entropy of next-token predictions.
    
    Args:
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
                containing unnormalized logits
    
    Returns:
        torch.Tensor of shape (batch_size, sequence_length)
        The entropy for each next-token prediction
    """
```

### Implementation

**Key Requirements:**
1. Convert logits to probabilities over the vocabulary
2. Compute entropy for each position in the sequence
3. Use numerically stable methods (avoid overflow/underflow)

**Numerically Stable Implementation:**

```python
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # Use log_softmax for numerical stability (avoids computing large exponentials)
    # Shape: (batch_size, sequence_length, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Compute probabilities from log probabilities
    # Shape: (batch_size, sequence_length, vocab_size)
    probs = torch.exp(log_probs)
    
    # Compute entropy: H(p) = -sum_x p(x) * log(p(x))
    # Shape: (batch_size, sequence_length)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    return entropy
```

### Why Numerical Stability Matters

**Unstable approach (avoid):**
```python
# BAD: Can cause overflow for large logits
probs = torch.softmax(logits, dim=-1)
log_probs = torch.log(probs)  # Can give -inf for very small probs
entropy = -torch.sum(probs * log_probs, dim=-1)
```

**Stable approach (use):**
```python
# GOOD: log_softmax handles large logits gracefully
log_probs = F.log_softmax(logits, dim=-1)  # Numerically stable
probs = torch.exp(log_probs)
entropy = -torch.sum(probs * log_probs, dim=-1)
```

The `log_softmax` function uses the logsumexp trick internally:
```
log_softmax(x_i) = x_i - log(∑_j exp(x_j))
                 = x_i - logsumexp(x)
```

where `logsumexp(x) = log(∑_j exp(x_j))` is computed stably by factoring out the maximum:
```
logsumexp(x) = max(x) + log(∑_j exp(x_j - max(x)))
```

### Interpretation

- **High entropy:** Model is uncertain (uniform distribution across vocabulary)
  - Maximum entropy for vocab size V: `log(V)`
  - Example: For vocab_size=50,000, max entropy ≈ 10.82

- **Low entropy:** Model is confident (peaked distribution)
  - Minimum entropy: 0 (deterministic prediction)
  - Example: p(token_A) = 1.0, all others = 0

### Usage in Training

Track entropy during training to:
1. Monitor model confidence over time
2. Detect overconfidence (entropy too low → overfitting)
3. Compare different training methods (SFT vs RL)
4. Debug training issues (entropy collapse, mode collapse)

Example logging:
```python
# During forward pass
outputs = model(input_ids=input_ids)
logits = outputs.logits

# Compute entropy
entropy = compute_entropy(logits)  # (batch_size, seq_len)

# Mask to response tokens only
masked_entropy = entropy * response_mask
avg_response_entropy = masked_entropy.sum() / response_mask.sum()

# Log
wandb.log({"train/avg_response_entropy": avg_response_entropy})
```

### Testing

Run the test with:
```bash
.venv/bin/pytest -k test_compute_entropy -v
```

Expected output: `PASSED`

---

## Getting Log-Probabilities from a Model

### Purpose

Obtaining log-probabilities from a model is a fundamental primitive needed for both SFT and RL. This function computes the conditional log-probability of each token given the previous tokens.

### Mathematical Definition

For a prefix x, an LM producing next-token logits f_θ(x) ∈ ℝ^|V|, and a label y ∈ V:

```
log p_θ(y | x) = log[softmax(f_θ(x))]_y
```

where [x]_y denotes the y-th element of vector x.

### Function Signature

```python
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Get per-token conditional log-probabilities from a causal LM.
    
    Args:
        model: HuggingFace model (on correct device, in inference mode if needed)
        input_ids: Shape (batch_size, sequence_length)
                  Concatenated prompt + response tokens
        labels: Shape (batch_size, sequence_length)
               Labels (shifted input_ids) from tokenization
        return_token_entropy: If True, also return per-token entropy
    
    Returns:
        dict with:
            "log_probs": Shape (batch_size, sequence_length)
                        Conditional log p_θ(x_t | x_<t)
            "token_entropy": Optional, shape (batch_size, sequence_length)
                           Per-token entropy (if return_token_entropy=True)
    """
```

### Implementation

**Key Steps:**

1. Get logits from model forward pass
2. Convert logits to log-probabilities using `log_softmax` (numerically stable)
3. Gather the log-probabilities corresponding to the actual labels
4. Optionally compute token entropy

**Complete Implementation:**

```python
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    # Forward pass through model
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)
    
    # Compute log probabilities using log_softmax (numerically stable)
    # Shape: (batch_size, sequence_length, vocab_size)
    log_probs_all = F.log_softmax(logits, dim=-1)
    
    # Gather the log probabilities for the actual labels
    # We need to select log_probs_all[b, t, labels[b, t]] for each (b, t)
    
    # Expand labels to match shape for gather
    # labels_expanded: (batch_size, sequence_length, 1)
    labels_expanded = labels.unsqueeze(-1)
    
    # Gather log probabilities for the labels along vocab dimension
    # Shape: (batch_size, sequence_length, 1)
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels_expanded)
    
    # Remove the extra dimension
    # Shape: (batch_size, sequence_length)
    log_probs = log_probs.squeeze(-1)
    
    # Prepare return dictionary
    result = {
        "log_probs": log_probs,
    }
    
    # Optionally compute and return token entropy
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        result["token_entropy"] = token_entropy
    
    return result
```

### Understanding `torch.gather`

The `torch.gather` operation selects specific elements from a tensor along a dimension:

```python
# For each batch and sequence position, we want:
# log_probs[b, t] = log_probs_all[b, t, labels[b, t]]

# Example with batch_size=2, seq_len=3, vocab_size=5
log_probs_all = [
    [[0.1, 0.2, 0.3, 0.4, 0.5],   # batch 0, position 0
     [0.2, 0.3, 0.4, 0.5, 0.6],   # batch 0, position 1
     [0.3, 0.4, 0.5, 0.6, 0.7]],  # batch 0, position 2
    [[...], [...], [...]]          # batch 1
]

labels = [
    [2, 4, 1],  # batch 0: token IDs
    [0, 3, 2]   # batch 1: token IDs
]

# After gather, we get:
log_probs = [
    [0.3, 0.6, 0.4],  # Selected: [b0,p0,tok2], [b0,p1,tok4], [b0,p2,tok1]
    [..., ..., ...]   # batch 1
]
```

### Important Notes

1. **No Masking**: The returned log-probs are NOT masked. Masking (using `response_mask`) happens in the training loop.

2. **Numerical Stability**: Always use `F.log_softmax()` instead of `torch.log(F.softmax())`:
   ```python
   # BAD: Can cause numerical issues
   probs = F.softmax(logits, dim=-1)
   log_probs = torch.log(probs)  # May give -inf
   
   # GOOD: Numerically stable
   log_probs = F.log_softmax(logits, dim=-1)
   ```

3. **Gradient Flow**: If `model` is in training mode and has gradients enabled, gradients will flow through this function back to the model parameters.

4. **Device Placement**: Ensure `input_ids` and `labels` are on the same device as the model.

### Usage Examples

**Example 1: SFT Training**
```python
# During training step
model.train()
outputs = get_response_log_probs(
    model=model,
    input_ids=batch['input_ids'],
    labels=batch['labels'],
    return_token_entropy=True,  # Monitor confidence
)

log_probs = outputs['log_probs']
token_entropy = outputs['token_entropy']

# Compute loss on response tokens only
loss = -log_probs * response_mask
loss = loss.sum() / response_mask.sum()

# Log average entropy on response tokens
avg_entropy = (token_entropy * response_mask).sum() / response_mask.sum()
```

**Example 2: RL Rollout Scoring**
```python
# Scoring rollouts (no gradients needed)
model.eval()
with torch.no_grad():
    outputs = get_response_log_probs(
        model=policy_model,
        input_ids=rollout_input_ids,
        labels=rollout_labels,
        return_token_entropy=False,
    )
    
    # Compute total log-prob for each sequence
    log_probs = outputs['log_probs']
    seq_log_probs = (log_probs * response_mask).sum(dim=-1)  # (batch_size,)
```

**Example 3: Computing KL Divergence (for RL)**
```python
# Get log-probs from policy and reference model
with torch.no_grad():
    ref_outputs = get_response_log_probs(
        model=ref_model,
        input_ids=input_ids,
        labels=labels,
    )
    ref_log_probs = ref_outputs['log_probs']

policy_outputs = get_response_log_probs(
    model=policy_model,
    input_ids=input_ids,
    labels=labels,
)
policy_log_probs = policy_outputs['log_probs']

# KL divergence: KL(policy || ref) = sum(policy * (log_policy - log_ref))
# For discrete case with tokens: exp(log_policy) * (log_policy - log_ref)
kl_div = torch.exp(policy_log_probs) * (policy_log_probs - ref_log_probs)
kl_div = (kl_div * response_mask).sum(dim=-1)  # Per-sequence KL
```

### Testing

Run the test with:
```bash
.venv/bin/pytest -k test_get_response_log_probs -v
```

Expected output: `PASSED`
