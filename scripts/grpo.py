#!/usr/bin/env python3
"""
Group Relative Policy Optimization (GRPO) Implementation

This script implements GRPO training for math reasoning, following the algorithm
from DeepSeekMath and DeepSeek-R1 papers.

GRPO is a group-relative policy optimization method that normalizes rewards
within groups of rollouts to reduce variance in policy gradient estimation.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Literal, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer

# Import utilities from sft.py
from scripts.sft import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    masked_normalize,
)


# ============================================================================
# Reward Normalization
# ============================================================================

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], Dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float = 1e-8,
    normalize_by_std: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    # Step 1: Compute raw rewards for all rollouts
    raw_rewards_list = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, ground_truth)
        raw_rewards_list.append(reward_dict["reward"])
    
    # Convert to tensor
    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    rollout_batch_size = len(raw_rewards_list)
    
    # Step 2: Reshape rewards into groups (n_groups, group_size)
    # rollout_batch_size = n_prompts * group_size
    n_groups = rollout_batch_size // group_size
    assert rollout_batch_size % group_size == 0, \
        f"rollout_batch_size ({rollout_batch_size}) must be divisible by group_size ({group_size})"
    
    # Reshape: (rollout_batch_size,) -> (n_groups, group_size)
    rewards_grouped = raw_rewards.view(n_groups, group_size)
    
    # Step 3: Compute group means
    group_means = rewards_grouped.mean(dim=1, keepdim=True)  # (n_groups, 1)
    
    # Step 4: Normalize rewards by subtracting group means (group-relative normalization)
    # This creates advantages: advantage = reward - group_mean
    advantages = rewards_grouped - group_means  # (n_groups, group_size)
    
    # Step 5: Optionally normalize by std if normalize_by_std=True
    if normalize_by_std:
        group_stds = rewards_grouped.std(dim=1, keepdim=True)  # (n_groups, 1)
        # Add epsilon to avoid division by zero
        group_stds = group_stds + advantage_eps
        advantages = advantages / group_stds
    
    # Flatten back to original shape: (n_groups, group_size) -> (rollout_batch_size,)
    normalized_rewards = advantages.view(rollout_batch_size)
    
    # Step 6: Compute metadata
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
        "min_reward": raw_rewards.min().item(),
        "max_reward": raw_rewards.max().item(),
        "mean_group_mean": group_means.mean().item(),
        "mean_normalized_reward": normalized_rewards.mean().item(),
        "std_normalized_reward": normalized_rewards.std().item(),
    }
    
    return normalized_rewards, raw_rewards, metadata
    


# ============================================================================
# Policy Gradient Loss Functions
# ============================================================================

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute policy gradient loss using either raw rewards or advantages.

    The naive policy gradient loss is:
        L = -sum(reward * log_prob) / normalize_constant

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    # Broadcast rewards/advantages from (batch_size, 1) to (batch_size, sequence_length)
    # PyTorch automatically broadcasts the dimension of size 1
    # Compute loss: -reward * log_prob (we minimize negative log-likelihood weighted by rewards)
    loss = -raw_rewards_or_advantages * policy_log_probs
    
    return loss


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the GRPO-Clip loss.

    GRPO-Clip uses PPO-style clipping on the importance sampling ratio:
        ratio = exp(log_prob - old_log_prob)
        clipped_ratio = clip(ratio, 1-cliprange, 1+cliprange)
        loss = -advantages * min(ratio * clipped_ratio, clipped_ratio)

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    # Step 1: Compute importance sampling ratio: exp(log_prob - old_log_prob)
    # ratio = π_θ(a|s) / π_θ_old(a|s) = exp(log π_θ(a|s) - log π_θ_old(a|s))
    ratio = torch.exp(policy_log_probs - old_log_probs)  # (batch_size, sequence_length)
    
    # Step 2: Compute clipped ratio bounds
    ratio_lower = 1.0 - cliprange
    ratio_upper = 1.0 + cliprange
    
    # Step 3: Compute PPO clipped loss
    # Standard PPO clipping: L^CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
    # We minimize the negative, so: loss = -min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
    clipped_ratio = torch.clamp(ratio, ratio_lower, ratio_upper)  # (batch_size, sequence_length)
    
    # Broadcast advantages from (batch_size, 1) to (batch_size, sequence_length)
    advantages_broadcast = advantages.expand_as(policy_log_probs)
    
    # Compute both unclipped and clipped objectives (before negation)
    # unclipped_objective = ratio * advantages
    # clipped_objective = clipped_ratio * advantages
    unclipped_objective = ratio * advantages_broadcast  # (batch_size, sequence_length)
    clipped_objective = clipped_ratio * advantages_broadcast  # (batch_size, sequence_length)
    
    # PPO clipping: take the minimum of unclipped and clipped objectives
    # This prevents large policy updates in either direction
    # When A > 0: min prevents ratio from being too large
    # When A < 0: min prevents ratio from being too small (since both are negative, min takes the less negative)
    objective = torch.min(unclipped_objective, clipped_objective)
    
    # Convert to loss (negate because we minimize loss, but want to maximize objective)
    loss = -objective
    
    # Step 4: Compute metadata (clip fraction - how often clipping occurs)
    # Clip occurs when ratio is outside [1-cliprange, 1+cliprange]
    clipped_mask = (ratio < ratio_lower) | (ratio > ratio_upper)
    clip_fraction = clipped_mask.float().mean()
    
    metadata = {
        "clip_fraction": clip_fraction,
        "mean_ratio": ratio.mean(),
        "min_ratio": ratio.min(),
        "max_ratio": ratio.max(),
    }
    
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    cliprange: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards is required for loss_type='no_baseline'")
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
    
    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("advantages is required for loss_type='reinforce_with_baseline'")
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
    
    elif loss_type == "grpo_clip":
        if advantages is None:
            raise ValueError("advantages is required for loss_type='grpo_clip'")
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for loss_type='grpo_clip'")
        if cliprange is None:
            raise ValueError("cliprange is required for loss_type='grpo_clip'")
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    return loss, metadata


# ============================================================================
# Utility Functions
# ============================================================================

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    masked = tensor * mask
    if dim is None:
        return masked.sum() / mask.sum()
    else:
        return masked.sum(dim=dim) / mask.sum(dim=dim)
    


# ============================================================================
# Training Step Functions
# ============================================================================

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    cliprange: Optional[float] = None,
    normalize_constant: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the policy gradient loss and backprop its gradients for a microbatch.

    This function:
    1. Computes the policy gradient loss based on loss_type
    2. Applies response_mask to only compute loss on response tokens
    3. Normalizes the loss appropriately
    4. Scales for gradient accumulation
    5. Performs backward pass

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        normalize_constant: float | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    # Step 1: Compute per-token policy gradient loss
    loss_per_token, loss_metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    # loss_per_token shape: (batch_size, sequence_length)
    
    # Step 2: Apply response_mask to only compute loss on response tokens
    # Use masked_mean to aggregate to scalar loss per example (dim=1)
    # This gives us the mean loss per example, considering only response tokens
    loss_per_example = masked_mean(loss_per_token, response_mask, dim=1)
    # loss_per_example shape: (batch_size,)
    
    # Step 3: Aggregate over batch dimension
    if normalize_constant is not None:
        # Sum over batch and normalize by constant (as in Dr. GRPO)
        loss = loss_per_example.sum() / normalize_constant
    else:
        # Average over batch dimension
        loss = loss_per_example.mean()
    # loss shape: scalar
    
    # Step 4: Scale by 1/gradient_accumulation_steps for gradient accumulation
    loss = loss / gradient_accumulation_steps
    
    # Step 5: Backward pass (accumulates gradients)
    loss.backward()
    
    # Step 6: Prepare metadata
    # Store unscaled loss for logging (before gradient accumulation adjustment)
    if normalize_constant is not None:
        unscaled_loss = loss_per_example.sum() / normalize_constant
    else:
        unscaled_loss = loss_per_example.mean()
    
    metadata = {
        "loss": loss.detach(),
        "unscaled_loss": unscaled_loss.detach(),
        "num_response_tokens": response_mask.sum().detach(),
        **loss_metadata,  # Include metadata from compute_policy_gradient_loss
    }
    
    # Step 7: Return detached loss and metadata
    return loss.detach(), metadata

