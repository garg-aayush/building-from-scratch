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
    beta: float = 1.0,
    max_seq_len: int = 1024) -> torch.Tensor:
    """
    Compute the DPO loss for a single instance.
    Policy and reference models can be on different devices.
    """
    # combine the prompt and responses with the prompt template
    chosen_prompt = prompt_template.format(instruction=prompt, response=chosen_response)
    rejected_prompt = prompt_template.format(instruction=prompt, response=rejected_response)
    
    # tokenize the prompts
    chosen_toks = tokenizer(chosen_prompt, return_tensors="pt", truncation=True, max_length=max_seq_len).input_ids
    rejected_toks = tokenizer(rejected_prompt, return_tensors="pt", truncation=True, max_length=max_seq_len).input_ids
    
    # NLL -> multiply by num predicted tokens to get sum -> negate to get logp
    # pass both input and labels as labels triggers the internal cross-entropy loss calculation, otherwise it will return the logits
    chosen_len = chosen_toks.shape[-1] - 1
    rejected_len = rejected_toks.shape[-1] - 1
    
    # policy model (on policy device)
    chosen_toks_policy = chosen_toks.to(policy_model.device)
    rejected_toks_policy = rejected_toks.to(policy_model.device)
    chosen_logp_policy = -policy_model(input_ids=chosen_toks_policy, labels=chosen_toks_policy).loss * chosen_len
    rejected_logp_policy = -policy_model(input_ids=rejected_toks_policy, labels=rejected_toks_policy).loss * rejected_len
    
    # reference model (on ref device)
    with torch.no_grad():
        chosen_toks_ref = chosen_toks.to(ref_model.device)
        rejected_toks_ref = rejected_toks.to(ref_model.device)
        chosen_logp_ref = -ref_model(input_ids=chosen_toks_ref, labels=chosen_toks_ref).loss * chosen_len
        rejected_logp_ref = -ref_model(input_ids=rejected_toks_ref, labels=rejected_toks_ref).loss * rejected_len
        # move ref log probs to policy device for loss computation
        chosen_logp_ref = chosen_logp_ref.to(policy_model.device)
        rejected_logp_ref = rejected_logp_ref.to(policy_model.device)
    
    loss = -F.logsigmoid(beta * (chosen_logp_policy - chosen_logp_ref - rejected_logp_policy + rejected_logp_ref))
    
    return loss


@torch.no_grad()
def dpo_pref_accuracy(
    policy_model: PreTrainedModel,
    tokenizer: AutoTokenizer, 
    prompt: str, chosen_response: str, rejected_response: str, prompt_template: str,
    max_seq_len: int = 1024) -> bool:
    """
    Compute the DPO preference accuracy for a single instance.
    """
    
    # combine the prompt and responses with the prompt template
    chosen_prompt = prompt_template.format(instruction=prompt, response=chosen_response)
    rejected_prompt = prompt_template.format(instruction=prompt, response=rejected_response)
    
    # tokenize the prompts
    chosen_toks = tokenizer(chosen_prompt, return_tensors="pt", truncation=True, max_length=max_seq_len).input_ids.to(policy_model.device)
    rejected_toks = tokenizer(rejected_prompt, return_tensors="pt", truncation=True, max_length=max_seq_len).input_ids.to(policy_model.device)
    
    # logp
    chosen_logp_policy = -policy_model(input_ids=chosen_toks, labels=chosen_toks).loss * (chosen_toks.shape[-1] - 1)
    rejected_logp_policy = -policy_model(input_ids=rejected_toks, labels=rejected_toks).loss * (rejected_toks.shape[-1] - 1)
    
    return (chosen_logp_policy > rejected_logp_policy).item()