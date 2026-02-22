from typing import Callable, List, Tuple

import torch
from einops import rearrange
from torch.nn import functional as F
from transformers import PreTrainedModel
from typing_extensions import Literal


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool) -> Tuple[torch.Tensor, torch.Tensor, dict[str, float]]:

    """
    Compute group-normalized advantages for a batch of rollouts.
    Args:
        reward_fn: Reward function that takes a response and ground truth and returns a dict of rewards.
        rollout_responses: List of responses from the model (size is n_prompts_per_rollout_batch * group_size).
        repeated_ground_truths: List of ground truths, repeated `group_size` times (size is same as rollout_responses).
        group_size: Size of each group.
        advantage_eps: Epsilon to add to the denominator for numerical stability.
        normalize_by_std: Whether to normalize advantages by the standard deviation of the group.
    Returns:
        Tuple of (advantages, raw_rewards, stats)
    """

    # list of dict of (format_reward, answer_reward, reward) for each rollout
    rewards = [reward_fn(response, ground_truth) for response, ground_truth in zip(rollout_responses, repeated_ground_truths)]

    # tensor -> reshape from n_grp * group_size to (n_grp, group_size)
    raw_rewards = torch.tensor([r['reward'] for r in rewards])
    grp_rewards = rearrange(raw_rewards, '(n_grp group_size) -> n_grp group_size', group_size=group_size)

    # compute mean and std (n_grp,)
    mean = grp_rewards.mean(dim=-1)
    std = grp_rewards.std(dim=-1)

    # compute advantage
    grp_advantages = grp_rewards - mean.unsqueeze(-1)
    if normalize_by_std:
        grp_advantages = grp_advantages / (std.unsqueeze(-1) + advantage_eps)

    # reshape from (n_grp, group_size) -> n_grp * group_size
    advantages = rearrange(grp_advantages, 'n_grp group_size -> (n_grp group_size)', group_size=group_size)

    return advantages, raw_rewards, {'mean': mean, 'std': std, 'rewards': rewards}


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute the naive policy gradient loss.
    Args:
        raw_rewards_or_advantages: scalar reward/advantage per rollout (bs, 1).
        policy_log_probs: log probabilities of the tokens in the rollout (bs, seq_len).
    Returns:
        per token policy gradient loss (bs, seq_len).
    """
    # shape: (bs, 1) * (bs, seq_len) -> (bs, seq_len)
    return -(raw_rewards_or_advantages * policy_log_probs)


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
    clip: bool = True) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the GRPO clip loss.
    Args:
        advantages: Tensor of advantages (bs, 1).
        policy_log_probs: Tensor of policy log probabilities (bs, seq_len).
        old_log_probs: Tensor of old policy log probabilities (bs, seq_len).
        cliprange: Clip ratio.
        clip: Whether to clip the ratio.
    Returns:
        per token grpo clipped loss (bs, seq_len)
        metadata
    """
    ratios = torch.exp(policy_log_probs - old_log_probs)
    scores = ratios * advantages
    mean_ratio = ratios.mean()

    if not clip:
        return -scores, {"clip_fraction": 0.0, "mean_ratio": mean_ratio}

    # clip the ratios
    clipped_ratios = torch.clamp(ratios, 1.0 - cliprange, 1.0 + cliprange)
    clipped_scores = clipped_ratios * advantages

    # compute the clip fraction
    clip_fraction = (~torch.isclose(scores, clipped_scores)).float().mean()

    return -torch.min(scores, clipped_scores), {
        "clip_fraction": clip_fraction,
        "mean_ratio": mean_ratio,
    }


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    clip: bool = True) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the policy gradient loss.
    Args:
        policy_log_probs: Tensor of policy log probabilities (bs, seq_len).
        loss_type: Type of loss to compute.
        raw_rewards: Tensor of raw rewards (bs, 1).
        advantages: Tensor of advantages (bs, 1).
        old_log_probs: Tensor of old policy log probabilities (bs, seq_len).
        cliprange: Clip ratio.
        clip: Whether to clip the ratio.
    Returns:
        per token policy gradient loss (bs, seq_len)
        metadata
    """
    
    # assert loss_type
    assert loss_type in ["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"], f"Invalid loss type: {loss_type}"

    if loss_type == "no_baseline":
        assert raw_rewards is not None, f"raw_rewards is required for {loss_type} loss type"
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    
    assert advantages is not None, f"advantages is required for {loss_type} loss type"
    if loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    
    assert old_log_probs is not None, f"old_log_probs is required for {loss_type} loss type"
    if loss_type == "grpo_no_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange, clip)
    
    assert cliprange is not None, f"cliprange is required for {loss_type} loss type"
    if loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange, clip)

def masked_mean(
    tensor: torch.Tensor, 
    mask: torch.Tensor,
    dim: int | None = None
    ) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where
    mask == 1.
    Args:
        tensor: Tensor to compute the mean of.
        mask: Mask to consider only those elements where mask == 1.
        dim: Dimension along which to compute the mean.
    Returns:
        Mean of tensor along the given dimension, considering only those elements where mask == 1.
    """
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.
    """
    return (tensor * mask).sum(dim=dim) / normalize_constant

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    clip: bool = True,
    norm_mode: Literal["mean", "constant", "microbatch"] = "mean",
    norm_constant: float | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the GRPO microbatch train step.
    Args:
        policy_log_probs: Tensor of policy log probabilities (bs, seq_len).
        response_mask: Tensor of response mask (bs, seq_len).
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Type of loss to compute.
        raw_rewards: Tensor of raw rewards (bs, 1).
        advantages: Tensor of advantages (bs, 1).
        old_log_probs: Tensor of old policy log probabilities (bs, seq_len).
        cliprange: Clip ratio.
        clip: Whether to clip the ratio.
        norm_mode: Mode of normalization.
        norm_constant: Constant to normalize by.
    """

    # compute policy gradient loss
    policy_gradient_loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange, clip)

    # normalize
    if norm_mode == "mean":
        policy_gradient_loss = masked_mean(policy_gradient_loss, response_mask, dim=-1)
    elif norm_mode == "constant":
        assert norm_constant is not None, f"norm_constant is required for {norm_mode} norm mode"
        policy_gradient_loss = masked_normalize(policy_gradient_loss, response_mask, dim=-1, normalize_constant=norm_constant)
    elif norm_mode == "microbatch":
        # normalize by the longest response in the batch
        norm_constant = response_mask.sum(dim=-1).max().item()
        policy_gradient_loss = masked_normalize(policy_gradient_loss, response_mask, dim=-1, normalize_constant=norm_constant)
    else:
        raise ValueError(f"Invalid norm mode: {norm_mode}")
    
    # gradient accumulation
    policy_gradient_loss = policy_gradient_loss.mean() / gradient_accumulation_steps

    # backpropagate
    policy_gradient_loss.backward()

    return policy_gradient_loss, metadata


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
    """
    Get the conditional log-probs of the response given the prompt,
    and optionally the entropy of the next token predictions.
    Args:
        model: The model to get the log-probs from.
        input_ids: The input ids of the prompt.
        labels: The labels of the response.
        return_token_entropy: Whether to return the entropy of the next token predictions.
    Returns:
        A dictionary containing the log-probs of the response given the prompt and the entropy of the next token predictions.
    """
    response_dict = {}
    # get logits from the model
    logits = model(input_ids).logits
    # get the log-probs of the response given the prompt
    log_probs = F.log_softmax(logits, dim=-1) # (batch_size, sequence_length, vocab_size)
    # get the log-prob of the token that actually occurred there
    # labels: (batch_size, sequence_length) -> (batch_size, sequence_length, 1)
    response_dict["log_probs"] = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # (batch_size, sequence_length)
    
    if return_token_entropy:
        response_dict["token_entropy"] = compute_entropy(logits)
    
    return response_dict

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of the of the next-token predictions (i.e., entropy over the vocabulary dimension).
    entropy = -sum(p * log(p))
    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size) containing unnormalized logits.
    Returns:
        torch.Tensor Shape (batch_size, sequence_length): the entropy of the next-token predictions.
    """
    # Using logsumexp keeps everything in log-space, so nothing ever overflows or collapses to zero before you take the log.
    # If you do softmax -> log, tiny probs become 0 and log(0) blows up to -inf, destroying the entropy calculation.
    # See https://discuss.pytorch.org/t/justification-for-logsoftmax-being-better-than-log-softmax/140130
    with torch.no_grad():
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True) # (batch_size, sequence_length, vocab_size)
        return -torch.sum(torch.exp(log_probs) * log_probs, dim=-1) # (batch_size, sequence_length)