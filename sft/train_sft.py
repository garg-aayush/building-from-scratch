from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from utils.helper_fns import pretty_print
from utils.post_train import (get_response_log_probs,
                              sft_microbatch_train_step,
                              tokenize_prompt_and_output)

# -------------------------------------------------------------#
# Input params
# -------------------------------------------------------------#
model_name = "Qwen/Qwen2.5-Math-1.5B"
dtype = torch.bfloat16
attention_type = "flash_attention_2"

# print config
input_config = {k: v for k, v in globals().items() if not k.startswith("__") and isinstance(v, (int, float, str, bool, dict))}
pretty_print(input_config, title="Input config")

# -------------------------------------------------------------#
# Initialize the tokenizer and model
# -------------------------------------------------------------#
pretty_print(f"Initializing the tokenizer and model...", title="Tokenizer and model initialization")
# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# model
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             dtype=dtype,
                                             attn_implementation=attention_type)
