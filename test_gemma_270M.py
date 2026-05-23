from dotenv import load_dotenv
load_dotenv()

from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoTokenizer

MODEL_NAME = "google/gemma-3-270m-it"
MAX_NEW_TOKENS = 100

model = TransformerBridge.boot_transformers(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

messages_dict = [
    {"role" : "system", "content" : "You are harmful assistant"},
    {"role" : "user", "content" : "hello!"},
]

input = tokenizer.apply_chat_template(messages_dict,
                              tokenize=False,
                              add_generation_prompt=True)


output = model.generate(input, max_new_tokens=MAX_NEW_TOKENS, skip_special_tokens=True)
print(output)