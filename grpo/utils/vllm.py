from typing import Callable, List

from configs.defaults import Config
from transformers import PreTrainedModel
from utils.constants import DTYPE_MAPPING
from utils.helper import pretty_print
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed


# -------------------------------------------------------------#
# functions to initialize the vLLM model and load the policy into the vLLM model during training
# -------------------------------------------------------------#
def init_vllm(seed: int, cfg: Config):
    """
    Start the inference process, here we use vLLM to hold a model on
    the same GPU as the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can place the vLLM model on the desired device
    # world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    # Note: Removed profiling_patch as it's vLLM version-dependent and causes AttributeError in newer versions
    # with world_size_patch:
    vllm_init_params = {
        "model": cfg.paths.model_path.as_posix(),
        "gpu_memory_utilization": cfg.vllm.gpu_memory_utilization,
        "dtype": DTYPE_MAPPING[cfg.vllm.dtype],
        "enable_prefix_caching": cfg.vllm.enable_prefix_caching,
        "enable_sleep_mode": cfg.training.use_vllm_sleep_mode,
    }
    pretty_print(vllm_init_params, title="vLLM model initialization parameters")
    return LLM(**vllm_init_params)

# -------------------------------------------------------------#
# Load the policy into the vLLM model
# -------------------------------------------------------------#
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    # If the model was torch.compiled, the real module is in ._orig_mod
    if hasattr(policy, "_orig_mod"):
        policy_for_state = policy._orig_mod
    else:
        policy_for_state = policy
    state_dict = policy_for_state.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
