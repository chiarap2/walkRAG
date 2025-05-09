import requests
import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from geopy.geocoders import Nominatim
from routingpy import Valhalla, Graphhopper, ORS
from shapely.geometry import LineString
import polyline

def pois(polygon, tags):
    """
    Get the green areas within a polygon using OSMnx.
    """
    # Get the green areas within the polygon
    gdf = ox.geometries_from_polygon(polygon, tags=tags)

    cols = [col for col in gdf.columns if col in tags.keys()]
    cols.append('name')
    cols.append('geometry')
    
    if 'name' not in gdf.columns:
        gdf['name'] = None
    
    gdf = gdf[cols]
    gdf = gdf.reset_index()
    
    return gdf  

def add_pois_areas_to_gdf(gdf, pois_gdf, distance=100, distance_col='green_dist'):
    """
    Add green areas to a GeoDataFrame.
    """

    gdf = gdf.to_crs("EPSG:3857")
    pois_gdf = pois_gdf.to_crs("EPSG:3857")
    gdf['buffer'] = gdf.geometry.buffer(distance)
    routes_gdf = gdf.set_geometry('buffer').sjoin(pois_gdf, how='left', predicate='intersects')
    routes_gdf = routes_gdf.to_crs("EPSG:4326")
    
    return routes_gdf

def get_air_pollution(lat, lon, api_key):
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {'lat': lat, 'lon': lon, 'appid': api_key}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return None

def extract_aqi(pollution_data):
    try:
        return pollution_data['list'][0]['main']['aqi']
    except Exception:
        return None

def pollution_for_edge(lat, lon, api_key):
    # if edge_geom is None:
    #     return None
    # Use the midpoint of the edge geometry
    # midpoint = edge_geom.interpolate(0.5, normalized=True)
    # lat, lon = midpoint.y, midpoint.x
    # lon, lat = edge_geom.y, edge_geom.x
    data = get_air_pollution(lat, lon, api_key)
    return extract_aqi(data)

def get_weather(city, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    if response.status_code == 200:
        return data
    else:
        print("Error:", data.get("message", "Unable to fetch weather data"))
        return None
