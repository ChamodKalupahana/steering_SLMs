from dotenv import load_dotenv
load_dotenv()

from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoTokenizer

from src.model_class import model_class

MODEL_NAME = "google/gemma-3-270m-it"
MAX_NEW_TOKENS = 100

model = model_class(MODEL_NAME)
output = "write a story"
for i in range(5): 
    output = model.generate("this sucks bro, try something different", MAX_NEW_TOKENS)
    print(output)
