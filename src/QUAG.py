from RAG_system.utils import query_llm
import re
from spatial_component.main import spatialComponent

class QUAG:
    def __init__(self, rag, tokenizer, model):
        self.rag = rag
        self.tokenizer = tokenizer
        self.model = model


    def extract_class(text):
        match = re.search(r'Class:\s*(Spatial Request|Information Request)', text)
        return match.group(1) if match else None

    def classify_intent(self, query):
        instruction = """
            You are a classifier. Your task is to determine the type of user prompt based on its content. For each prompt, classify it into one of the following two categories and follow the corresponding output format:

            Categories:

            1. Spatial Request – The user is asking for directions, routes, travel paths, or any spatial navigation between locations.

            Output format:
            Class: Spatial Request  
            From: [fill]  
            To: [fill]  
            Walkability Indicators: [list the relevant indicators mentioned or implied: [sidewalk availability | air quality index | green areas]. If none are mentioned, return "none"]  
            POI Categories: [list of point-of-interest macro-categories the user refers to, if any: [bar | biergarten | cafe | fast_food | pub | restaurant | arts_centre | cinema | fountain | nightclub | bench | drinking_water | dance | garden | shop]. If none are mentioned, return "none"]

            2. Information Request – The user is asking for factual, descriptive, or explanatory information about a location, person, event, object, or concept, without requesting a route or spatial navigation.  
            Output format:
            Class: Information Request  
            Prompt: [original user prompt]
        """
        
        return query_llm(query, instruction, self.tokenizer, self.model, temperature=0.7, max_new_tokens=1000)

    def handle_query(self, query):
        classification = self.classify_intent(query)
        if "Spatial Request" in classification:
            
            place_from = re.search(r'From:\s*(.*)\\n', classification)
            place_to = re.search(r'To:\s*(.*)\\n', classification)
            
            walkability_indicators = re.search(r'Walkability Indicators:\s*\[(.*)\]', classification)
            poi_categories = re.search(r'POI Categories:\s*\[(.*)\]', classification)
            
            json_path = spatialComponent(place_A=place_from.group(1).strip() if place_from else None,
                                         place_B=place_to.group(1).strip() if place_to else None,
                                         walkability_indicators=walkability_indicators.group(1).strip() if walkability_indicators else None,
                                         poi_categories=poi_categories.group(1).strip() if poi_categories else None)
            if not json_path:
                return "No valid route found or required file missing."
            
            return self.rag.handle_spatial_request(query, json_path=json_path)
        elif "Information Request" in classification:
            return self.rag.handle_information_request(query)
        else:
            return "Intent could not be classified or required file missing."