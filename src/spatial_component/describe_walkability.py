import requests
import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from geopy.geocoders import Nominatim
from routingpy import Valhalla, Graphhopper, ORS
from shapely.geometry import LineString
import polyline


def compute_walkability_score(counts: dict, weights: dict, max_count: int) -> float:
    """
    Compute a normalized weighted sum of indicator counts between 0 and 1.
    counts: dict with keys matching weights, values are counts
    weights: normalized dict (values in [0,1] summing to 1)
    max_count: maximum count for each indicator
    """
    
    score = 0.0

    for key in counts:    
        score += counts[key] * weights[key]
    
    score /= max_count * sum(weights.values())
        
    return score


def aggregate_segment_pois_by_type(seg_df, detailed_categories=["tourism", "natural", "leisure", "landuse"]):
    
    # check if detailed_categories are in the dataframe
    categories = [cat for cat in detailed_categories if cat in seg_df.columns]
   
    poi_info = {}

    # Process detailed categories:
    for cat in categories:
        # Initialize a dictionary for this category
        cat_info = {}
        # Select rows where the category is not null
        df_cat = seg_df[seg_df[cat].notnull()]
        if not df_cat.empty:
            for _, row in df_cat.iterrows():
                poi_type = row[cat]
                # Determine if there is a valid name for the POI.
                poi_name = row["name"] if "name" in row and pd.notnull(row["name"]) else None
                # If not seen before, initialize
                if poi_type not in cat_info:
                    if poi_name is not None:
                        cat_info[poi_type] = [poi_name]
                    else:
                        cat_info[poi_type] = 1
                else:
                    # If a name is provided and we already have a count (int) then convert to a list.
                    if poi_name is not None:
                        if isinstance(cat_info[poi_type], int):
                            cat_info[poi_type] = [poi_name]
                        else:
                            # Append the name if not already included
                            if poi_name not in cat_info[poi_type]:
                                cat_info[poi_type].append(poi_name)
                    else:
                        if isinstance(cat_info[poi_type], int):
                            cat_info[poi_type] += 1
                        # If already a list, we assume names are already present.
            poi_info[cat] = cat_info
            
    return poi_info