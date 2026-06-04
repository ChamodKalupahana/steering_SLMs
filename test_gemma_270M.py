from dotenv import load_dotenv
load_dotenv()

from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoTokenizer

from src.model_class import model_class

MODEL_NAME = "google/gemma-3-270m-it"
MAX_NEW_TOKENS = 100

model = model_class(MODEL_NAME)
input = "write a story about a dragon and a castle"
for i in range(5): 
    output = model.generate(input, MAX_NEW_TOKENS, temperature=1)
    input = "make it more dramatic please"
    print(output)
