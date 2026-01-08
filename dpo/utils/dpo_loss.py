import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedModel

# TODO: 
# 1) 2 forward passes instead of 4 for better efficiency, use batching to, Batch it and do chosen/rejected in one pass per model (2 passes total) using padding.
# 2) Mask the prompt tokens in the loss calculation to ignore them (like in sft masking loss calculations)

def per_instance_dpo_loss(
    policy_model: PreTrainedModel, ref_model: PreTrainedModel, 
    tokenizer: AutoTokenizer, 
    prompt: str, chosen_response: str, rejected_response: str, prompt_template: str, 
    beta: float = 1.0) -> torch.Tensor:
    """
    Compute the DPO loss for a single instance.
    Assumes that both policy and reference models are present on the same device.
    """
    # combine the prompt and responses with the prompt template
    chosen_prompt = prompt_template.format(instruction=prompt, response=chosen_response)
    rejected_prompt = prompt_template.format(instruction=prompt, response=rejected_response)
    
    # tokenize the prompts
    chosen_toks = tokenizer(chosen_prompt, return_tensors="pt").input_ids.to(policy_model.device)
    rejected_toks = tokenizer(rejected_prompt, return_tensors="pt").input_ids.to(policy_model.device)
    
    # NLL -> multiply by num predicted tokens to get sum -> negate to get logp
    # pass both input and labels as labels triggers the internal cross-entropy loss calculation, otherwise it will return the logits
    chosen_len = chosen_toks.shape[-1] - 1
    rejected_len = rejected_toks.shape[-1] - 1
    # policy model
    chosen_logp_policy = -policy_model(input_ids=chosen_toks, labels=chosen_toks).loss * chosen_len
    rejected_logp_policy = -policy_model(input_ids=rejected_toks, labels=rejected_toks).loss * rejected_len
    # reference model
    with torch.no_grad():
        chosen_logp_ref = -ref_model(input_ids=chosen_toks, labels=chosen_toks).loss * chosen_len
        rejected_logp_ref = -ref_model(input_ids=rejected_toks, labels=rejected_toks).loss * rejected_len
    
    loss = -F.logsigmoid(beta * (chosen_logp_policy - chosen_logp_ref - rejected_logp_policy + rejected_logp_ref))
    
    return loss