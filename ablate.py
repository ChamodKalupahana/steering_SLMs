from dotenv import load_dotenv
load_dotenv()

from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoTokenizer
from jaxtyping import Float
from torch import Tensor

MODEL_NAME = "google/gemma-3-270m-it"
MAX_NEW_TOKENS = 100

model = TransformerBridge.boot_transformers(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

messages_dict = [
    {"role" : "system", "content" : "You are harmless assistant"},
    {"role" : "user", "content" : "hello!"},
]

input = tokenizer.apply_chat_template(messages_dict,
                              tokenize=False,
                              add_generation_prompt=True)

# print(model.cfg)

HEAD_IDX = 0
LAYER_IDX = 12

def hook_fn(attn_scores : Float[Tensor, "batch n_heads query_seq_length key_seq_len"], hook):
    attn_scores[:,HEAD_IDX,:,-1] = -1e6
    return attn_scores

fwd_hooks=[
    (f"blocks.{LAYER_IDX}.attn.hook_pattern", hook_fn)
]
with model.hooks(fwd_hooks=fwd_hooks):
    output = model.generate(input, max_new_tokens=MAX_NEW_TOKENS, skip_special_tokens=False, temperature=0)
print(output)