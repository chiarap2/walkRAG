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
            Walkability Indicators: [list the relevant indicators mentioned or implied: [sidewalk availability | air quality index | green areas | disability friendly]. If none are mentioned, return "none"]  
            POI Categories: [list of point-of-interest macro-categories the user refers to, if any: [bar | biergarten | cafe | fast_food | pub | restaurant | arts_centre | cinema | fountain | nightclub | bench | drinking_water | dance | garden | shop]. If none are mentioned, return "none"]

            2. Information Request – The user is asking for factual, descriptive, or explanatory information about a location, person, event, object, or concept, without requesting a route or spatial navigation.  
            Output format:
            Class: Information Request  
            Prompt: [original user prompt]
        """
        
        return query_llm(query, instruction, self.tokenizer, self.model, temperature=0.7, max_new_tokens=1000)

    
    def parse_field(a):
        return None if a.lower() == 'none' else [v for v in a.split(',')]

    def handle_query(self, query):
        classification = self.classify_intent(query)
        print(type(classification), classification)
        if "Spatial Request" in classification:
            
            match = re.search(
                r'From:\s*(.+?)\s*To:\s*(.+?)\s*Walkability Indicators:\s*(.+?)\s*POI Categories:\s*(.+)',
                classification,
                re.DOTALL
            )

            if match:
                from_location = match.group(1).strip()
                to_location = match.group(2).strip()
                walkability_indicators = match.group(3).strip()
                poi_categories = match.group(4).strip()
                
            walkability_indicators = None if walkability_indicators.lower() == 'none' else [v for v in walkability_indicators.split(', ')]
            poi_categories = None if poi_categories.lower() == 'none' else [v for v in poi_categories.split(', ')]
            
            print(from_location, to_location, walkability_indicators, poi_categories)
            
            json_path = spatialComponent(
                place_A=from_location,
                place_B=to_location,
                indicators_preference=walkability_indicators,
                pois_user=poi_categories
            )
            
            if not json_path:
                return "No valid route found or required file missing."
            
            return self.rag.handle_spatial_request(query, json_path=json_path)
        elif "Information Request" in classification:
            return self.rag.handle_information_request(query)
        else:
            return "Intent could not be classified or required file missing."