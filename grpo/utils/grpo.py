
import torch
from typing import Callable, List
from einops import rearrange

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

    # tensor -> reshape from n_grp, group_size: (n_grp X group_size)
    raw_rewards = torch.tensor([r['reward'] for r in rewards])
    grp_rewards = rearrange(raw_rewards, 'n_grp group_size -> (n_grp group_size)', group_size=group_size)

    # compute mean and std (n_grp,)
    mean = grp_rewards.mean(dim=-1) 
    std = grp_rewards.std(dim=-1)

    # compute advantage
    grp_advantages = grp_rewards - mean.unsqueeze(-1)
    if normalize_by_std:
        grp_advantages = grp_advantages / (std.unsqueeze(-1) + advantage_eps)

    # reshape from (n_grp X group_size) -> n_grp, group_size
    advantages = rearrange(grp_advantages, '(n_grp group_size) -> n_grp group_size')
    
    return advantages, raw_rewards, {'mean': mean, 'std': std}


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor
) -> torch.Tensor:
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
    clip: bool = True
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
    # compute ratio, torch.exp as it is log probs
    unclipped_ratio = torch.exp(policy_log_probs - old_log_probs)
    
    if not clip:
        return -(advantages * unclipped_ratio), {'clipped': False}
    
    # compute clipped ratio
    clipped_ratio = torch.clamp(unclipped_ratio, 1.0 - cliprange, 1.0 + cliprange)

    # compute loss
    adv_unclipped_ratio = advantages * unclipped_ratio
    adv_clipped_ratio = advantages * clipped_ratio
    loss = -torch.min(adv_unclipped_ratio, adv_clipped_ratio)
    
    # check if each token was clipped
    clipped = adv_clipped_ratio < adv_unclipped_ratio
    metadata = {
        'clipped': clipped,
    }
    
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    clip: bool = True
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
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
