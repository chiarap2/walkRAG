import pandas as pd
from RAG_system.utils import *
from QUAG import QUAG
from RAG_system.RAG import RAG
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

data_path = '../../../raid/mamendola/SSTD/'
cache_dir = '../../../raid/mamendola/'
encoder_id = 'Snowflake/snowflake-arctic-embed-l-v2.0'

# Load LLM model
print('Loading LLM model...')
llm_model_id = "meta-llama/Llama-3.1-8B-Instruct"
llm_tokenizer, llm_model = load_llm(llm_model_id, cache_dir)

# Init classes
print('Loading RAG model...')
rag = RAG(data_path, cache_dir, encoder_id, llm_tokenizer, llm_model)
qwag = QUAG(rag, llm_tokenizer, llm_model)

# Start interactive loop
print("\n--- Welcome to the WalkRAG assistant ---")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Your query: ")
    if user_input.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break
    response = qwag.handle_query(user_input)
    """
    try:
        response = qwag.handle_query(user_input)
    except Exception as e:
        response = f"[ERROR] {e}"
    """

    print("\n[Response]\n\t", response, "\n")