import os
import json
from .utils import *
import pandas as pd
from transformers import AutoModel

class RAG:
    
    def __init__(self, data_path, cache_dir, encoder_id, llm_tokenizer, llm_model):

        self.encoder = self.load_encoder(cache_dir, encoder_id)
        self.tokenizer = self.load_tokenizer(cache_dir, encoder_id)
        self.index = False#self.load_index(data_path)
        self.llm_tokenizer = llm_tokenizer
        self.llm_model = llm_model
        self.index_id, self.id_corpus = False, False#self.load_corpus(data_path)
        
    @staticmethod
    def load_index(data_path):
        index = load_faiss_index(data_path + '/indexes/ivf/snowflake_ivf_6216.faiss')
        return index
    
    @staticmethod
    def load_corpus(data_path):
        corpus = pd.read_csv(data_path + 'data/CAST2019collection.tsv', sep='\\t')
        id_mapping = pd.read_csv(data_path + 'data/CAST2019_ID_Mapping.tsv', sep='\\t')
        index_id = dict(zip(id_mapping.index, id_mapping.id))
        id_corpus = dict(zip(corpus.id, corpus.text))
        
        return index_id, id_corpus
    
    @staticmethod
    def load_tokenizer(cache_dir, encoder_id):
        encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_id, device_map='auto', cache_dir=cache_dir)
        return encoder_tokenizer
    
    @staticmethod
    def load_encoder(cache_dir, encoder_id):    
        encoder = AutoModel.from_pretrained(encoder_id, device_map='auto', cache_dir=cache_dir, add_pooling_layer=False)
        return encoder

    def handle_information_request(self, query):
        
        prompt_ir = "Provide a complete and accurate answer based on the background information above and your own knowledge. Do not mention the background source explicitly."
        
        instruction_ir = "You are a helpful assistant that answers questions clearly and accurately, using background information provided when helpful. Do not refer to 'passages' or any retrieval process. Just answer naturally, as if you know the information."
        
        indices = search_docs(query, self.encoder, self.tokenizer, self.index, top_k=5)
        docs = get_corpus(indices, self.index_id, self.id_corpus)
        
        prompt = prompt_ir + f"Background Information:\n" + "\n".join(docs) + f"\n\nQuestion: {query}\n\n{prompt_ir}"
        
        response = query_llm(prompt, instruction_ir, self.llm_tokenizer, self.llm_model, temperature=0.7, max_new_tokens=1000)
        
        return response

    def handle_spatial_request(self, query, json_path):
        instruction = """
            You are a route provider. Your task is to produce a detailed, step-by-step description of an itinerary specified in a JSON.

            The JSON contains:
            - length_m: total path length in kilometers
            - walkability_score: 0–1 (0 not walkable, 1 very walkable)
            - air_quality: “Good”, “Fair”, “Moderate”, “Poor”, or “Very Poor”
            - disability_friendly: “yes” or “no”
            - segments: a JSON of navigation steps, each with:
                - segment_id (unique number)
                - instruction (e.g., “Continue”, “Turn left”, etc.)
                - poi: points of interest by category (tourism, leisure, natural)

            Your output must be one single paragraph that:

            1. Opens with a general summary of:
                - route length (in kilometers)
                - qualitative walkability: report "low walkable" if the score is lower than 0.4, "moderate" if the score is between 0.4 and 0.7, or "excellent" if the score is higher than 0.7
                - air quality rating
                - whether it is disability-friendly.

            2. For each segment_id, in the order they appear in the JSON, you have to report the instruction value together with some of the POI listed in the relative segment, if any. If there are no POI, provide only the instruction value.

            3. Report always the instruction value

            Example JSON:
            {
                "segments": [
                                {
                                    "segment_id": 0,
                                    "instruction": "Turn right",
                                    "poi": {
                                        "tourism": {
                                            "attraction": [
                                                "Champ de Mars"
                                            ]
                                        }
                                    }
                                },
                                {
                                    "segment_id": 1,
                                    "instruction": "Turn left",
                                    "poi": {                    
                                        "leisure": {
                                            "park": [
                                                "Jardin Japonais"
                                            ]
                                        }
                                    }
                                },
            }
            {
                                    "segment_id": 2,
                                    "instruction": "Continue",
                                    "poi": {
                                        "natural": {
                                            "tree": 1
                                        }
                                    }
                                },
            }
            }

            Example description segments:
            Turn right with nearby attractions including Champ de Mars, turn left and walk near to Jardin Japonais and then continue straight.

        """
        
        with open(json_path, "rt") as f:
            file_json = json.load(f)
            
        prompt = json.dumps(file_json)
        response = query_llm(prompt, instruction, self.llm_tokenizer, self.llm_model, max_new_tokens=2000, temperature=0.3)
        
        return response
