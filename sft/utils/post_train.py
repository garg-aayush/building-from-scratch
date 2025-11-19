from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedModel


def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str], tokenizer: AutoTokenizer) -> dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for the prompt and padding tokens.
    Args:
        prompt_strs (List[str]): List of prompt strings.
        output_strs (List[str]): List of output strings.
        tokenizer (AutoTokenizer): Tokenizer to use for tokenization.
    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (num_prompts, max_prompt_output_token_len - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (num_prompts, max_prompt_output_token_len - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (num_prompts, max_prompt_output_token_len - 1):
                a mask on the response tokens in `labels`.
    """
    # check if the number of prompts and outputs are the same
    num_prompts = len(prompt_strs)
    assert num_prompts == len(output_strs), "Number of prompts and outputs must be the same"
    
    # tokenize the prompts and outputs -> dict: input_ids, attention_mask
    prompt_tokens, output_tokens = map(lambda x: tokenizer(x, return_tensors=None, padding=False, truncation=False)["input_ids"], [prompt_strs, output_strs])
    
    # get the max length of the prompt and output tokens
    max_prompt_output_token_len = max([len(p_tokens) + len(o_tokens) for p_tokens, o_tokens in zip(prompt_tokens, output_tokens)])
    
    # create the input_ids, labels, response_mask tensors
    input_ids, labels, response_mask = map(lambda x: torch.zeros((num_prompts, max_prompt_output_token_len - 1), dtype=x), 
                                           [torch.long, torch.long, torch.bool]
                                           )
    
    for i, (p_tokens, o_tokens) in enumerate(zip(prompt_tokens, output_tokens)):
        comb_tokens = torch.tensor(p_tokens + o_tokens)
        concat_len = len(comb_tokens)
        
        # no padding on left: 0, right: max_prompt_output_token_len - concat_len
        padded_comb_tokens = F.pad(comb_tokens, (0, max_prompt_output_token_len - concat_len), 'constant', value=tokenizer.pad_token_id)
        
        input_ids[i] = padded_comb_tokens[:-1]
        labels[i] = padded_comb_tokens[1:]
        
        # true only for the labels: start from the first output token to the last output token
        response_mask[i,(len(p_tokens)-1):(concat_len-1)] = True
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }
    

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


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
    """
    Get the conditional log-probs of the response given the prompt,
    and optionally the entropy of the next token predictions.
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

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.
    """
    # sum over the given dimension only for the elements with mask value 1
    return torch.sum(tensor * mask, dim=dim) / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
    per_token_loss: bool = False,
) -> torch.Tensor:
    """Train a single microbatch of data."""
    # dim=-1 means sum over the last dimension (sequence_length)
    # calculate loss (sum over the sequence dimension) and divide by the gradient accumulation steps
    # mean() is used to average over the batch dimension
    
    
    response_lengths = response_mask.sum(dim=-1)
    if per_token_loss:
        # Per-token loss: sum over sequence and divide by response length for each example -> mean over the batch dimension
        loss = -masked_normalize(policy_log_probs, response_mask, dim=-1, normalize_constant=response_lengths).mean()
    else:
        # normalize by a constant (sum over the sequence dimension) -> mean over the batch dimension
        loss = -masked_normalize(policy_log_probs, response_mask, dim=-1, normalize_constant=normalize_constant).mean()
    
    # adjust the loss by the gradient accumulation steps
    scaled_loss = loss / gradient_accumulation_steps
    
    # backprop the loss
    scaled_loss.backward()
    
    return scaled_loss, {"loss": loss.item(),
                         "scaled_loss": scaled_loss.item(),
                        }

def sft_eval_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    normalize_constant: float = 1.0,
    per_token_loss: bool = False,
) -> torch.Tensor:
    """Evaluate a single microbatch of data."""
    # dim=-1 means sum over the last dimension (sequence_length)
    # calculate loss (sum over the sequence dimension) and divide by the gradient accumulation steps
    # mean() is used to average over the batch dimension
    
    response_lengths = response_mask.sum(dim=-1)
    if per_token_loss:
        # Per-token loss: sum over sequence and divide by response length for each example -> mean over the batch dimension
        loss = -masked_normalize(policy_log_probs, response_mask, dim=-1, normalize_constant=response_lengths).mean()
    else:
        # normalize by a constant (sum over the sequence dimension) -> mean over the batch dimension
        loss = -masked_normalize(policy_log_probs, response_mask, dim=-1, normalize_constant=normalize_constant).mean()
    
    return loss