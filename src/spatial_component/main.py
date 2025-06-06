from spatial_component.routing import routing_graphhopper
import requests
import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from geopy.geocoders import Nominatim
from shapely.geometry import LineString, box
import polyline
import json
from spatial_component.enrichment import pois, add_pois_areas_to_gdf, pollution_for_edge
from spatial_component.describe_walkability import compute_walkability_score, aggregate_segment_pois_by_type

# ----- CONFIGURATION -----
OWM_API_KEY = "86f3b5265ec1d4daf3434cad93141670"  # OpenWeatherMap API key
GRAPHHOPPER_API = "3d3fdeee-0b60-4240-a648-1b0c0175b719"  # GraphHopper API key
# Lists for category matching
GREEN_TAGS = {
    'leisure': {'park', 'nature_reserve', 'garden'},
    'landuse': {'forest', 'meadow', 'grass'},
    'natural': {'tree', 'waterway'}
}
PED_FRIENDLY_TAGS = {
    'footway': {'path', 'sidewalk', 'footway', 'pedestrian'},
    'landuse': {'pedestrian'},
    'highway': {'footway', 'path', 'pedestrian'}
}
DISABILITY_TAGS = {'yes', 'limited', 'designated'}
MAX_COUNT = 5  # Maximum count for each indicator

def spatialComponent(place_A, place_B, indicators_preference=None, pois_user=None):
    # ----- INITIALIZATION -----
    # Define two places and a city name
    # place_A = "Leaning tower, Pisa, Italy" # this is the starting point given by the prompt
    # place_B = "Piazza Vittorio Emanuele II, Pisa, Italy" # this is the destination given by the prompt
    # city = "Pisa"
    geolocator = Nominatim(user_agent="my_app", timeout=10)
    locA = geolocator.geocode(place_A)
    locB = geolocator.geocode(place_B)
    latA, lonA = locA.latitude, locA.longitude
    latB, lonB = locB.latitude, locB.longitude
    
    # Weights for walkability score indicators
    all_keys = ['green areas', 'sidewalk availability', 'disability friendly', 'air quality index']
    RAW_WEIGHTS = dict.fromkeys(all_keys, 1/4)
    
    if indicators_preference is None or len(indicators_preference) == 4:
        # Default weights
        RAW_WEIGHTS = dict.fromkeys(all_keys, 1/4)
    else:
        prefs = set(indicators_preference) & set(all_keys)
        n = len(prefs)
        
        if n == 3:
            # Three indicators selected
            RAW_WEIGHTS = {}
            for key in all_keys:
                if key in prefs:
                    RAW_WEIGHTS[key] = 0.30
                else:
                    RAW_WEIGHTS[key] = 0.10
            

        if n == 2:
            # Two indicators selected
            RAW_WEIGHTS = {}
            for key in all_keys:
                if key in prefs:
                    RAW_WEIGHTS[key] = 0.40
                else:
                    RAW_WEIGHTS[key] = 0.10

        elif n == 1:
            # One indicator selected
            RAW_WEIGHTS = {}
            for key in all_keys:
                if key in prefs:
                    RAW_WEIGHTS[key] = 0.70
                else:
                    RAW_WEIGHTS[key] = 0.10
                     
    # Create a bbox around the start and end points

    bbox = box(lonA, latA, lonB, latB)
    bbox = bbox.buffer(0.01)  # buffer by 0.01 degrees (approx 1 km)
    bbox_bounds = bbox.bounds  # get the bounding box of the buffered area
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")
    
    # ----- GET THE ROUTES -----
    routes = routing_graphhopper(lonA, latA, lonB, latB, mode='foot', graphhopper_api_key=GRAPHHOPPER_API)
    
    # ----- DOWNLOAD WALKABILITY INDICATORS -----
    walkability_indicators = {
        "leisure": ["park", "nature_reserve", "garden"],
        "landuse": ["forest", "meadow", "grass", "pedestrian"],
        "footway": ["path", "sidewalk", "footway", "pedestrian"],
        "highway": ["footway", "path", "pedestrian"],
        "natural": ["tree", "waterway"],
        "wheelchair": ["yes", "limited", "designated"]
    }
    
    pois_list = {
            "attraction": "all",
            "tourism": ['museum', 'attraction', 'hotel', 'artwork', 'viewpoint', 'gallery']
        }
    
    if pois_user is not None:
        pois_list.update(pois_user)
    
    # Merge the two dictionaries
    walkability_indicators.update(pois_list)

    walkability_indicators_gdf = pois(bbox, walkability_indicators)
    routes_gdf = add_pois_areas_to_gdf(routes, walkability_indicators_gdf, distance=100, distance_col='walkable_distance')
    # use index as segment id
    routes_gdf['segment_id'] = routes_gdf.index
    routes_gdf = routes_gdf.to_crs("EPSG:4326")
    routes_gdf = routes_gdf.set_geometry('geometry')
    
    # drop duplicates
    cols = ['geometry', 'segment_id', 'name']
    for key in walkability_indicators.keys():
        cols.append(key)
    cols = [col for col in cols if col in routes_gdf.columns]
    routes_gdf = routes_gdf.drop_duplicates(subset=cols)
    """
    for key in walkability_indicators.keys():
        if key in routes_gdf.columns:
            routes_gdf = routes_gdf[routes_gdf[key].isin(walkability_indicators[key]) | routes_gdf[key].isnull()]
    """
    #----- GET BEST ROUTE -----
    routes_grouped = routes_gdf.groupby("route_id")

    routes_summary = []
    
    for route_id, group in routes_grouped:

        aggregated_geom = group.unary_union
        agg_gdf = gpd.GeoDataFrame(geometry=[aggregated_geom], crs=group.crs)
        centroid = agg_gdf.geometry.centroid.iloc[0]
        centroid_latlon = gpd.GeoSeries([centroid], crs=group.crs).to_crs("EPSG:4326").iloc[0]
        centroid_lat, centroid_lon = centroid_latlon.y, centroid_latlon.x
        # Calculate the route length in meters.
        route_length = agg_gdf.to_crs("EPSG:3857").geometry.length.iloc[0]

        # Retrieve air quality and weather data using the centroid coordinates.
        aqi = pollution_for_edge(centroid_lat, centroid_lon, OWM_API_KEY)
        # compute the inverse of the aqi to get a score
        aqi_inverse = 6 - aqi
        
        # Calculate the walkability score for the route.
        sum_counts = {'green':0, 'pedestrian_friendly':0, 'disability_friendly':0, 'air_quality':aqi_inverse}
        n_segments = 0
        
        # Process each segment
        segments_info = []
        for seg_id, seg_group in group.groupby('segment_id'):
            n_segments += 1
            # Count indicators per segment
            c_green = sum(seg_group[seg_group[key].isin(vals)].shape[0]
                          for key, vals in GREEN_TAGS.items())
            c_ped = sum(seg_group[seg_group[key].isin(vals)].shape[0]
                        for key, vals in PED_FRIENDLY_TAGS.items())
            c_dis = seg_group[seg_group['wheelchair'].isin(DISABILITY_TAGS)].shape[0]
            # Cap each at MAX_COUNT
            c_green = min(c_green, MAX_COUNT)
            c_ped = min(c_ped, MAX_COUNT)
            c_dis = min(c_dis, MAX_COUNT)
            # Accumulate
            sum_counts['green'] += c_green
            sum_counts['pedestrian_friendly'] += c_ped
            sum_counts['disability_friendly'] += c_dis

        
            # Use the first row's instruction as the segment instruction (default to "N/A" if missing).
            instruction = seg_group["instruction"].iloc[0] if "instruction" in seg_group.columns and pd.notna(seg_group["instruction"].iloc[0]) else "N/A"
            categories_list = list(pois_list)
            categories_list = categories_list + ["leisure", "natural"]
            
            poi_details = aggregate_segment_pois_by_type(seg_group, detailed_categories=categories_list)
            segments_info.append({
                "segment_id": seg_id,
                "instruction": instruction,
                "poi": poi_details
            })
      
        # Calculate the mean counts for each indicator.
        default_keys = ['green', 'pedestrian_friendly', 'disability_friendly', 'air_quality']
        mean_counts = {k: (sum_counts[k] / n_segments if n_segments else 0) for k in default_keys}
        
        aqi_mapping = {
                1: "Good",
                2: "Fair",
                3: "Moderate",
                4: "Poor",
                5: "Very Poor"
            }
        aqi = aqi_mapping.get(aqi, "Unknown")
        
        # Calculate the walkability score.
        walk_score = compute_walkability_score(mean_counts, RAW_WEIGHTS, MAX_COUNT)
        
        # Build the route dictionary.
        route_dict = {
            "route_id": route_id,
            "from": place_A,
            "to": place_B,
            "length_m": route_length,
            "walkability_score": walk_score,
            "air_quality": aqi,
            "disability_friendly": "yes" if sum_counts['disability_friendly'] > 0 else "no",
            "segments": segments_info
        }
        routes_summary.append(route_dict)

        # Select the route with the highest walkability score.
        best_route = max(routes_summary, key=lambda x: x["walkability_score"])
        
        # Save the best route to a JSON file.
        
        # remove everything after the first comma
        place_A = place_A.split(",")[0]
        place_B = place_B.split(",")[0]
        
        file_path = f"../../output/best_routes/best_route_from_{place_A}_to_{place_B}.json"
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(best_route, f, ensure_ascii=False, indent=4)

        print("File best_route.json saved successfully.")
        
        return file_path
        