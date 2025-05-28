import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM

def embed_passages_snowflake(queries, model,tokenizer, max_length=512):
    query_prefix = 'query: '    
    tokenizer.pad_token = tokenizer.eos_token
    queries_with_prefix = ["{}{}".format(query_prefix, i) for i in queries]
    query_tokens = tokenizer(queries_with_prefix, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
    query_tokens = {k: v.to('cuda') for k, v in query_tokens.items()}
    with torch.no_grad():
        query_embeddings = model(**query_tokens)[0][:, 0]
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    return query_embeddings.cpu().numpy()

def search_docs(query, query_encoder, tokenizer, index, top_k):
    
    query_embeddings = embed_passages_snowflake([query], query_encoder, tokenizer, max_length=512)
    query_embeddings = np.asarray(query_embeddings, dtype='float32').reshape(1, -1)
    distances, indices = index.search(query_embeddings, top_k)

    return indices

def get_corpus(indices, index_id, id_corpus):
    docs = []
    for idx in indices[0]:
        id_ = index_id.get(idx)
        doc = id_corpus.get(id_)
        docs.append(doc)
    return docs

def query_llm(prompt, instruction, tokenizer, model, max_new_tokens=100, temperature=0.7, do_sample=True):
        
    messages = [
    {"role": "system", "content": instruction},
    {"role": "user", "content": prompt},
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [tokenizer.eos_token_id]
    terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=do_sample,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
        #top_p=0.1,
    )

    response = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=True)
    
    return response

def load_faiss_index(index_path):
    """Load a FAISS index from a file."""
    print(f"Loading FAISS index from: {index_path}")
    index = faiss.read_index(index_path)
    print(f"Index loaded successfully with {index.ntotal} vectors.")
    return index

def load_llm(model_name, cache_dir):
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #n_gpus = torch.cuda.device_count()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model