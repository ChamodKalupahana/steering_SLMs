from dotenv import load_dotenv
load_dotenv()

from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoTokenizer
from jaxtyping import Float
from torch import Tensor

import matplotlib.pyplot as plt
import einops

MODEL_NAME = "google/gemma-3-270m-it"
MAX_NEW_TOKENS = 100

model = TransformerBridge.boot_transformers(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

messages_dict = [
    {"role" : "system", "content" : "You are harmless assistant"},
    {"role" : "user", "content" : "hello! write a story about a dragon and a castle, make it more dramatic please!" * 5},
]

input = tokenizer.apply_chat_template(messages_dict,
                              tokenize=False,
                              add_generation_prompt=True)

LAYER_IDX = 12

def attn_hook_fn(attn_scores : Float[Tensor, "batch n_heads query_seq_length key_seq_len"], hook):
    HEAD_IDX = 0
    attn_scores[:,HEAD_IDX,:,-1] = -1e6
    return attn_scores

def resid_post_hook_fn(resid_post : Float[Tensor, "batch d_model seq_len"], hook):
    to_plot = einops.reduce(resid_post, "batch seq_len d_model -> seq_len d_model", "max")
    plt.imshow(to_plot)
    plt.show()
    return resid_post
    

fwd_hooks=[
    # (f"blocks.{LAYER_IDX}.attn.hook_pattern", attn_hook_fn)
    (f"blocks.{LAYER_IDX}.hook_resid_pre", resid_post_hook_fn)
]
with model.hooks(fwd_hooks=fwd_hooks):
    output = model.generate(input, max_new_tokens=MAX_NEW_TOKENS, skip_special_tokens=False, temperature=0)
print(output)