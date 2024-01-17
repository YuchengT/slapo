

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
import torch
import deepspeed
import time
from deepspeed.accelerator import get_accelerator
import os

model = "tiiuae/falcon-40b"

tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
model = deepspeed.init_inference(model, mp_size=world_size, replace_with_kernel_inject=True)

input_prompt = [
"Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:"
]
input_tokens = tokenizer.batch_encode_plus(input_prompt, return_tensors="pt",)
token_num = input_tokens['input_ids'].size(-1)
for t in input_tokens:
    if torch.is_tensor(input_tokens[t]):
        input_tokens[t] = input_tokens[t].to(get_accelerator().current_device_name())
input_tokens.pop('token_type_ids')
sequences = model.generate(**input_tokens, min_length=200, max_length=300, do_sample=True)

if torch.distributed.get_rank() == 0:
    print(f"Result: {tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]}")
