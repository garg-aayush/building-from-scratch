import random
from typing import Callable, List

from vllm import LLM, SamplingParams


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable,
    val_dataset: List[dict],
    prompt_template: str,
    sampling_params: SamplingParams,
    max_val_examples: int,
) -> dict:
    """
    Evaluate the model on a random subset of val_dataset using vLLM.

    Args:
        vllm_model: The vLLM model to use for generation.
        reward_fn: Reward function that takes (response, ground_truth) and returns a dict
                   with keys 'reward', 'format_reward', 'answer_reward'.
        val_dataset: List of dicts with 'problem' and 'answer' keys.
        prompt_template: Prompt template string with '{question}' placeholder.
        sampling_params: vLLM SamplingParams for generation.
        max_val_examples: Maximum number of examples to evaluate on.

    Returns:
        dict with keys:
            'mean_reward': float
            'mean_format_reward': float
            'mean_answer_reward': float
            'n_examples': int
    """
    n = min(max_val_examples, len(val_dataset))
    subset = random.sample(val_dataset, n)

    prompts = [prompt_template.replace("{question}", ex['problem']) for ex in subset]
    ground_truths = [ex['answer'] for ex in subset]

    outputs = vllm_model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]

    rewards = [reward_fn(response, gt) for response, gt in zip(responses, ground_truths)]

    mean_reward = sum(r['reward'] for r in rewards) / n
    mean_format_reward = sum(r['format_reward'] for r in rewards) / n
    mean_answer_reward = sum(r['answer_reward'] for r in rewards) / n

    return {
        'mean_reward': mean_reward,
        'mean_format_reward': mean_format_reward,
        'mean_answer_reward': mean_answer_reward,
        'n_examples': n,
    }
