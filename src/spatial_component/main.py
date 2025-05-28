import pandas as pd
from RAG_system.utils import *
from QUAG import QUAG
from RAG_system.RAG import RAG

data_path = '../../../../raid/mamendola/SSTD/'
cache_dir = '../../../../raid/mamendola/'
encoder_id = 'Snowflake/snowflake-arctic-embed-l-v2.0'

# Load LLM model
print('Loading LLM model...')
llm_model_id = "meta-llama/Llama-3.1-8B-Instruct"
llm_tokenizer, llm_model = load_llm(llm_model_id, cache_dir)

# Init classes
print('Loading RAG model...')
rag = RAG(data_path, cache_dir, encoder_id, llm_tokenizer, llm_model)
qwag = QUAG(rag, llm_tokenizer, llm_model)

"""# Example
query_info = "How did the Obélisque de Louxor come to be installed in Paris?"
response = qwag.handle_query(query_info)
print("\\n[Info Query Response]\\n", response)"""

"""query_spatial = "Give me a nice walking route from the Eiffel Tower to the Louvre with good air quality."
json_path = "./output/best_routes/12345.json"  # Replace with actual file path
response = qwag.handle_query(query_spatial, json_path=json_path)
print("\\n[Spatial Query Response]\\n", response)"""

# Start interactive loop
print("\n--- Welcome to the WalkRAG assistant ---")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Your query: ")
    if user_input.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    try:
        response = qwag.handle_query(user_input, json_path=None)
    except Exception as e:
        response = f"[ERROR] {e}"

    print("\n[Response]\n\t", response, "\n")