
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