import requests
import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from geopy.geocoders import Nominatim
from routingpy import Valhalla, Graphhopper, ORS
from shapely.geometry import LineString
import polyline

def routing_graphhopper(lonStart, latStart, lonEnd, latEnd, mode='foot', graphhopper_api_key=None):
    """
    Get the alternative routes between two points using Graphhopper.
    """
    client = Graphhopper(base_url='https://graphhopper.com/api/1', api_key=graphhopper_api_key)
    routes = client.directions(
        locations=[[lonStart, latStart], [lonEnd, latEnd]],
        profile=mode,
        instructions=True,
        algorithm='alternative_route',
        alternative_route_max_paths=3,
        alternative_route_max_weight_factor=1.4,
        alternative_route_max_share_factor=0.5
    )
    
    # Extract the routes from the response
    segments = []

    for l, route_option in enumerate(routes.raw['paths']):
        # decoding geometry
        geometry = polyline.decode(route_option['points'], precision=5)
        
        instructions_mapping = {}
        if 'instructions' in route_option:
            for inst in route_option['instructions']:
                start, end = inst['interval']
                # for idx in range(start, end):
                instructions_mapping[start] = inst['text']
        
        for i in range(len(geometry) - 1):
            segment = LineString([
                (geometry[i][1], geometry[i][0]),  
                (geometry[i + 1][1], geometry[i + 1][0])
            ])

            inst_text = instructions_mapping.get(i, None)
            segments.append({
                'route_id': l,
                'geometry': segment,
                'instruction': inst_text
            })
        if len(geometry) > 2:
            # Add the last segment
            segments.append({
                'route_id': l,
                'geometry': LineString([
                    (geometry[-2][1], geometry[-2][0]),  
                    (geometry[-1][1], geometry[-1][0])
                ]),
                'instruction': route_option['instructions'][-1]['text'] if route_option['instructions'] else None
            })
        segments.append({
            'route_id': l,
            'geometry': LineString([
                (geometry[-2][1], geometry[-2][0]),  
                (geometry[-1][1], geometry[-1][0])
            ]),
            'instruction': route_option['instructions'][-1]['text'] if route_option['instructions'] else None
        })
    
    gdf = gpd.GeoDataFrame(segments, geometry='geometry', crs="EPSG:4326")
    
    return gdf

def bufferize_routes(routes_gdf, buffer_size=100):
    """
    Buffer the routes by a given size.
    """
    buffer_geom = routes_gdf.dissolve(by='route_id')

    buffer_geom = buffer_geom.to_crs("EPSG:3857")  
    buffer_geom['buffer'] = buffer_geom.geometry.buffer(buffer_size)  # Buffer 100m
    buffer_geom = buffer_geom.to_crs("EPSG:4326")  # Convert back to WGS84
    buffer_geom = buffer_geom.reset_index()
    buffer_geom = buffer_geom[['buffer']]
    buffer_geom.rename(columns={'buffer': 'geometry'}, inplace=True)
    buffer_geom = gpd.GeoDataFrame(buffer_geom)
    buffer_geom.to_crs("EPSG:4326", inplace=True)
    
    return buffer_geom