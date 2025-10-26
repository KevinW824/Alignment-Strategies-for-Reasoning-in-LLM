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

---

## Masked Normalization for SFT Loss

### Purpose

The SFT loss is the negative log-likelihood of the target output given the prompt. We need to:
1. Sum log-probabilities only over response tokens (exclude prompt and padding)
2. Normalize by a constant (e.g., number of response tokens, batch size)

The `masked_normalize` function is a general utility for this pattern.

### Function Signature

```python
def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float = 1.0,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Sum over tensor elements and normalize by a constant while respecting a mask.
    
    Args:
        tensor: The tensor to sum and normalize
        mask: Same shape as tensor; positions with 1 are included,
              positions with 0 are excluded
        normalize_constant: The constant to divide by (default: 1.0)
        dim: Dimension to sum along. If None, sum over all dimensions
    
    Returns:
        Normalized sum where masked elements (mask == 0) don't contribute
        - If dim is None: returns a scalar
        - If dim is specified: returns tensor with that dimension reduced
    """
```

### Implementation

```python
def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float = 1.0,
    dim: Optional[int] = None,
) -> torch.Tensor:
    # Apply mask: only keep elements where mask == 1
    masked_tensor = tensor * mask
    
    # Sum over specified dimension(s)
    if dim is None:
        # Sum over all dimensions (returns scalar)
        result = masked_tensor.sum()
    else:
        # Sum over specified dimension
        result = masked_tensor.sum(dim=dim)
    
    # Normalize by the constant
    result = result / normalize_constant
    
    return result
```

### Usage Examples

**Example 1: SFT Loss (scalar)**
```python
# Compute per-token log-probs
outputs = get_response_log_probs(model, input_ids, labels)
log_probs = outputs['log_probs']  # (batch_size, seq_len)

# SFT loss: negative log-likelihood on response tokens only
# Loss = -sum(log_probs * response_mask) / num_response_tokens
num_response_tokens = response_mask.sum()
loss = masked_normalize(
    tensor=-log_probs,
    mask=response_mask,
    normalize_constant=num_response_tokens,
    dim=None,  # Sum over all dimensions
)
# Returns a scalar loss
```

**Example 2: Per-Sequence Loss**
```python
# Compute loss for each sequence in the batch
per_seq_loss = masked_normalize(
    tensor=-log_probs,
    mask=response_mask,
    normalize_constant=1.0,
    dim=1,  # Sum over sequence dimension
)
# Returns shape: (batch_size,)
# Each element is the sum of -log_probs for that sequence

# Then average over batch
batch_loss = per_seq_loss.mean()
```

**Example 3: Normalize by Sequence Lengths**
```python
# Normalize each sequence by its own length
seq_lengths = response_mask.sum(dim=1, keepdim=True)  # (batch_size, 1)

# Compute per-token average for each sequence
per_seq_avg = masked_normalize(
    tensor=-log_probs,
    mask=response_mask,
    normalize_constant=1.0,
    dim=1,  # Sum over sequence
) / seq_lengths.squeeze()
# Returns shape: (batch_size,)
# Each element is the average -log_prob per token for that sequence
```

**Example 4: Different Dimensions**
```python
tensor = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])
mask = torch.tensor([
    [1.0, 1.0, 0.0],  # Include first 2 positions
    [1.0, 0.0, 0.0]   # Include only first position
])

# Sum all masked elements, normalize by 2
result = masked_normalize(tensor, mask, normalize_constant=2.0, dim=None)
# (1 + 2 + 4) / 2 = 3.5

# Sum along dimension 1 (columns), normalize by 2
result = masked_normalize(tensor, mask, normalize_constant=2.0, dim=1)
# [(1 + 2) / 2, (4) / 2] = [1.5, 2.0]

# Sum along dimension 0 (rows), normalize by 2
result = masked_normalize(tensor, mask, normalize_constant=2.0, dim=0)
# [(1 + 4) / 2, (2) / 2, (0) / 2] = [2.5, 1.0, 0.0]
```

### Common Normalization Strategies

**1. Normalize by total number of response tokens (Dr. GRPO style):**
```python
num_response_tokens = response_mask.sum()  # Scalar
loss = masked_normalize(-log_probs, response_mask, num_response_tokens, dim=None)
```

**2. Normalize by batch size:**
```python
batch_size = log_probs.shape[0]
loss = masked_normalize(-log_probs, response_mask, batch_size, dim=None)
```

**3. Normalize by average sequence length:**
```python
avg_seq_len = response_mask.sum() / response_mask.shape[0]
loss = masked_normalize(-log_probs, response_mask, avg_seq_len, dim=None)
```

**4. Per-sequence normalization, then batch average:**
```python
# First, get per-sequence sums
per_seq_sum = masked_normalize(-log_probs, response_mask, 1.0, dim=1)  # (batch_size,)

# Then normalize by sequence lengths
seq_lengths = response_mask.sum(dim=1)  # (batch_size,)
per_seq_avg = per_seq_sum / seq_lengths

# Average over batch
loss = per_seq_avg.mean()
```

### Key Points

1. **Masking happens before summation**: `masked_tensor = tensor * mask` ensures masked-out positions contribute 0 to the sum

2. **Flexible normalization**: The `normalize_constant` can be any value:
   - Total tokens: `response_mask.sum()`
   - Batch size: `tensor.shape[0]`
   - Custom constant: e.g., `1.0` for unnormalized sum

3. **Dimension flexibility**: 
   - `dim=None`: Returns scalar (sum over everything)
   - `dim=0`: Sum over batch dimension
   - `dim=1`: Sum over sequence dimension
   - `dim=-1`: Sum over last dimension

4. **Gradient flow**: Gradients flow through the masked sum and division, so this can be used in training

### Testing

Run the tests with:
```bash
.venv/bin/pytest -k test_masked_normalize -v
```

Expected output: Multiple tests should `PASS` for different dimensions

---

## SFT Microbatch Training Step

### Purpose

The SFT training loop processes data in minibatches, which are further split into microbatches when using gradient accumulation. This function implements a single microbatch update including:
1. Computing the negative log-likelihood loss
2. Masking to response tokens only
3. Scaling gradients for accumulation
4. Performing the backward pass

### The SFT Loss

The loss we minimize in SFT is the negative log-likelihood of the target output given the prompt:

```
L_SFT = -∑_{t∈response} log p_θ(token_t | tokens_{<t})
```

In practice:
```python
loss = -sum(log_probs * response_mask) / normalize_constant
```

### Function Signature

```python
def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a single SFT microbatch training step.
    
    Args:
        policy_log_probs: Shape (batch_size, sequence_length)
                         Per-token log-probabilities from policy
        response_mask: Shape (batch_size, sequence_length)
                      1 for response tokens, 0 for prompt/padding
        gradient_accumulation_steps: Number of microbatches per optimizer step
        normalize_constant: Constant to divide sum by (default: 1.0)
    
    Returns:
        tuple:
            loss: Scalar tensor - the scaled loss that was backward'd
            metadata: Dict with both scaled and unscaled loss, and other stats
    """
```

### Implementation

```python
def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute SFT loss: negative log-likelihood on response tokens
    loss = masked_normalize(
        tensor=-policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=None,  # Sum over all dimensions
    )
    
    # Scale loss for gradient accumulation
    # Gradients will be: grad_1/N + grad_2/N + ... + grad_N/N
    scaled_loss = loss / gradient_accumulation_steps
    
    # Backward pass (accumulates gradients)
    scaled_loss.backward()
    
    # Prepare metadata for logging
    metadata = {
        "scaled_loss": scaled_loss.detach(),
        "unscaled_loss": loss.detach(),
        "num_response_tokens": response_mask.sum().detach(),
    }
    
    # Return the scaled loss (what was actually backward'd)
    return scaled_loss.detach(), metadata
```

### Understanding Gradient Accumulation

When training with gradient accumulation, we split each minibatch into N microbatches:

**Without Gradient Accumulation (N=1):**
```python
# Process full batch
loss = compute_loss(batch)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

**With Gradient Accumulation (N>1):**
```python
optimizer.zero_grad()
for microbatch in split_into_N_microbatches(batch):
    loss = compute_loss(microbatch)
    scaled_loss = loss / N  # Scale before backward!
    scaled_loss.backward()  # Accumulates: grad += scaled_grad
optimizer.step()
optimizer.zero_grad()
```

**Why scale?** We want the final gradient to be the average:
```
final_grad = (grad_1 + grad_2 + ... + grad_N) / N
           = grad_1/N + grad_2/N + ... + grad_N/N
```

So we scale each loss by `1/N` before calling `.backward()`.

### Complete Training Loop Example

```python
# Setup
model = AutoModelForCausalLM.from_pretrained("model_name")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
gradient_accumulation_steps = 4

# Training loop
for minibatch in dataloader:
    optimizer.zero_grad()  # Zero gradients at start of minibatch
    
    # Split minibatch into microbatches
    microbatches = split_into_microbatches(minibatch, gradient_accumulation_steps)
    
    total_loss = 0.0
    for microbatch in microbatches:
        # Tokenize
        batch = tokenize_prompt_and_output(
            microbatch['prompts'],
            microbatch['outputs'],
            tokenizer,
        )
        
        # Get log-probs from model
        outputs = get_response_log_probs(
            model=model,
            input_ids=batch['input_ids'],
            labels=batch['labels'],
            return_token_entropy=False,
        )
        
        # Microbatch train step (computes loss, scales, and backprops)
        loss, metadata = sft_microbatch_train_step(
            policy_log_probs=outputs['log_probs'],
            response_mask=batch['response_mask'],
            gradient_accumulation_steps=gradient_accumulation_steps,
            normalize_constant=1.0,
        )
        
        total_loss += loss.item()
    
    # After all microbatches, take optimizer step
    optimizer.step()
    
    # Log average loss (note: loss is already scaled, so just average)
    avg_loss = total_loss / gradient_accumulation_steps
    print(f"Scaled Loss: {avg_loss:.4f}")
    
    # Or use unscaled loss from metadata for more interpretable logging
    # avg_unscaled_loss = sum(m['unscaled_loss'] for m in metadatas) / len(metadatas)
```

### Key Implementation Details

1. **Loss Scaling**: The loss is scaled by `1/gradient_accumulation_steps` BEFORE calling `.backward()`
   - This ensures gradients accumulate to the correct average

2. **Return Scaled Loss**: We return `scaled_loss.detach()` (the value that was backward'd)
   - This is consistent with the gradients that were computed
   - The metadata includes both `scaled_loss` and `unscaled_loss` for flexibility

3. **Detached Metadata**: All returned values are `.detach()`'ed
   - Prevents memory leaks from keeping computation graph in memory

4. **Flexible Normalization**: The `normalize_constant` parameter allows different normalization strategies:
   - `1.0`: Raw sum of negative log-probs
   - `num_response_tokens`: Average per-token loss
   - `batch_size`: Average per-sequence loss

### Normalization Strategies Comparison

```python
# Strategy 1: No normalization (sum)
loss, _ = sft_microbatch_train_step(
    policy_log_probs, response_mask,
    gradient_accumulation_steps=4,
    normalize_constant=1.0,
)

# Strategy 2: Normalize by total tokens (per-token average)
num_tokens = response_mask.sum()
loss, _ = sft_microbatch_train_step(
    policy_log_probs, response_mask,
    gradient_accumulation_steps=4,
    normalize_constant=num_tokens.item(),
)

# Strategy 3: Normalize by batch size
batch_size = policy_log_probs.shape[0]
loss, _ = sft_microbatch_train_step(
    policy_log_probs, response_mask,
    gradient_accumulation_steps=4,
    normalize_constant=batch_size,
)
```

### Testing

Run the test with:
```bash
.venv/bin/pytest -k test_sft_microbatch_train_step -v
```

Expected output: `PASSED`
