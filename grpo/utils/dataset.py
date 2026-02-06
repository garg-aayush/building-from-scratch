import json
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


# -------------------------------------------------------------#
# functions to load the dataset
# -------------------------------------------------------------#
def load_dataset(data_file: str, data_type: str='train', prompt_template: str=None):
    with open(data_file, 'r') as f:
        if data_type in ['train', 'val']:
            data = [json.loads(line) for line in f]
            data = [{'problem': item['problem'], 'answer': item['answer']} for item in data]
            return data
        elif data_type == 'prompt':
            return f.read()
        else:
            raise ValueError(f"Invalid data type: {data_type}")

# -------------------------------------------------------------#
# functions to tokenize the prompt and output
# -------------------------------------------------------------#
def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str], tokenizer: AutoTokenizer) -> dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for the prompt and padding tokens.
    Args:
        prompt_strs (List[str]): List of prompt strings.
        output_strs (List[str]): List of output strings.
        tokenizer (AutoTokenizer): Tokenizer to use for tokenization.
    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (num_prompts, max_len - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (num_prompts, max_len - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (num_prompts, max_len - 1):
                a mask on the response tokens in `labels`.
    """
    # check if the number of prompts and outputs are the same
    bz = len(prompt_strs)
    assert bz == len(output_strs), "Number of prompts and outputs must be the same"
    
    # tokenize the prompts and outputs -> dict: input_ids, attention_mask
    prompt_tokens, output_tokens = map(lambda x: tokenizer(x, return_tensors=None, padding=False, truncation=False)["input_ids"], [prompt_strs, output_strs])
    
    # get the max length of the prompt and output tokens
    max_len = max([len(p_tokens) + len(o_tokens) for p_tokens, o_tokens in zip(prompt_tokens, output_tokens)])
    
    # create the input_ids, labels, response_mask tensors
    input_ids, labels, response_mask = map(lambda x: torch.zeros((bz, max_len - 1), dtype=x), [torch.long, torch.long, torch.bool])
    
    for i, (p_tokens, o_tokens) in enumerate(zip(prompt_tokens, output_tokens)):
        comb_tokens = torch.tensor(p_tokens + o_tokens)
        concat_len = len(comb_tokens)
        
        # no padding on left: 0, right: max_len - concat_len
        padded_comb_tokens = F.pad(comb_tokens, (0, max_len - concat_len), 'constant', value=tokenizer.pad_token_id)
        
        input_ids[i] = padded_comb_tokens[:-1]
        labels[i] = padded_comb_tokens[1:]
        
        # true only for the labels: start from the first output token to the last output token
        response_mask[i,(len(p_tokens)-1):(concat_len-1)] = True
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }