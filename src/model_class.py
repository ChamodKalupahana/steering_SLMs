from dotenv import load_dotenv
load_dotenv()

from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoTokenizer
from jaxtyping import Float
from torch import Tensor

class model_class():
    def __init__(self, MODEL_NAME, SYSTEM_PROMPT = "You are harmless assistant") -> None:
        self.MODEL_NAME = MODEL_NAME
        self.model = model = TransformerBridge.boot_transformers(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.cfg = model.cfg
        
        self.messages_dict = [
    {"role" : "system", "content" : SYSTEM_PROMPT},
]


    def generate_with_manual_context(self, user_input, MAX_NEW_TOKENS=50, skip_special_tokens=False, temperature=0):
        # Add user message
        user_input_dict = {
            "role" : "user", "content" : user_input
        }
        self.messages_dict.append(user_input_dict)
        # print(self.messages_dict)

        # Generate response
        input = self.tokenizer.apply_chat_template(self.messages_dict,
                                    tokenize=False,
                                    add_generation_prompt=True)
        
        output = self.model.generate(input, max_new_tokens=MAX_NEW_TOKENS, skip_special_tokens=skip_special_tokens)
        # if self.MODEL_NAME == "google/gemma-3-270m-it":
        clean_output = output.split("model", 1)[-1].strip() # type: ignore
        # Add assistant response
        model_output_dict = {
            "role" : "assistant", "content" : clean_output
        }
        self.messages_dict.append(model_output_dict)

        # Keep conversation history manageable
        if len(self.messages_dict) > 10:  # Keep 5 turns (system + 4 user/assistant pairs)
            self.messages_dict = self.messages_dict[2:]
        return clean_output
    
    def generate(self, user_input, MAX_NEW_TOKENS=50, skip_special_tokens=False, temperature=0):
        # Add user message
        user_input_dict = {
            "role" : "user", "content" : user_input
        }
        self.messages_dict.append(user_input_dict)
        # print(self.messages_dict)

        # Generate response
        input = self.tokenizer.apply_chat_template([user_input_dict],
                                    tokenize=False,
                                    add_generation_prompt=True)
        
        output = self.model.generate(input, max_new_tokens=MAX_NEW_TOKENS, skip_special_tokens=skip_special_tokens)
        # if self.MODEL_NAME == "google/gemma-3-270m-it":
        clean_output = output.split("model", 1)[-1].strip() # type: ignore
        # Add assistant response
        model_output_dict = {
            "role" : "assistant", "content" : clean_output
        }
        self.messages_dict.append(model_output_dict)

        return clean_output