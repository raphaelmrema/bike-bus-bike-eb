# enhanced_osrm_gmaps_multimodal_planner.py
# Complete OSRM Multi-Route + Google Maps Transit Bike-Bus-Bike Planner
# Version 4.0 - Multi-route bike options + Google Maps transit integration

import os
import json
import http.server
import socketserver
import urllib.parse
import tempfile
import threading
import webbrowser
import shutil
import math
import requests
import datetime
import zipfile
import pandas as pd
import io
import socket
import time
import random
from functools import partial
from typing import List, Dict, Optional

# Try importing required packages
try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    import polyline
except ImportError as e:
    print(f"Installing missing packages: {e}")
    import subprocess
    packages = ['geopandas', 'shapely', 'polyline']
    for pkg in packages:
        try:
            subprocess.check_call(['pip', 'install', pkg])
        except:
            print(f"Failed to install {pkg}")
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    import polyline

# =============================================================================
# CONFIGURATION
# =============================================================================

# Google Maps API Key - REPLACE WITH YOUR KEY
GOOGLE_API_KEY = "AIzaSyB7Khy5ec8OotFSO-4Eckjpqot6BxOLWBo"

# OSRM Configuration
OSRM_SERVER = "http://router.project-osrm.org"
BIKE_SPEED_MPH = 11
USE_OSRM_DURATION = True
REQUEST_EDGE_ANNOTATIONS = True

# File paths for shapefiles (optional - will work without them)
ROADS_SHAPEFILE = r"D:\Users\n01621754\OneDrive - University of North Florida\Desktop\WSDOT RESEARCH\GIS WORKS\roads.shp"
TRANSIT_STOPS_SHAPEFILE = r"D:\Users\n01621754\OneDrive - University of North Florida\Desktop\WSDOT RESEARCH\GIS WORKS\transit_stops.shp"

# Temporary workspace
TEMP_WORKSPACE = os.path.join(tempfile.gettempdir(), "enhanced_multimodal_app")
if not os.path.exists(TEMP_WORKSPACE):
    os.makedirs(TEMP_WORKSPACE)

# =============================================================================
# ENHANCED GTFS MANAGER (For local transit data enhancement)
# =============================================================================

class EnhancedGTFSManager:
    """Enhanced GTFS manager with real-time capabilities"""
    
    def __init__(self):
        self.gtfs_data = {}
        self.stops_df = None
        self.routes_df = None
        self.stop_times_df = None
        self.trips_df = None
        self.is_loaded = False
        self.last_update = None
        self.realtime_cache = {}
        self.cache_duration = 30  # seconds
        
    def load_gtfs_data(self, gtfs_urls=None):
        """Load GTFS data with enhanced error handling"""
        if gtfs_urls is None:
            gtfs_urls = [
                "https://ride.jtafla.com/gtfs-archive/gtfs.zip",
                "https://schedules.jtafla.com/schedulesgtfs/download",
                "https://openmobilitydata.org/p/jacksonville-transportation-authority/331/latest/download",
            ]
        
        print("Loading GTFS data for transit enhancement...")
        
        for i, url in enumerate(gtfs_urls):
            try:
                print(f"Trying GTFS source {i+1}: {url}")
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/zip, application/octet-stream, */*',
                }
                
                response = requests.get(url, timeout=30, headers=headers, verify=False, stream=True)
                response.raise_for_status()
                
                content = response.content
                if not content.startswith(b'PK'):
                    continue
                
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    files_to_load = {
                        'stops.txt': 'stops_df',
                        'routes.txt': 'routes_df', 
                        'trips.txt': 'trips_df',
                        'stop_times.txt': 'stop_times_df',
                    }
                    
                    loaded_files = 0
                    for filename, attr_name in files_to_load.items():
                        if filename in z.namelist():
                            try:
                                with z.open(filename) as f:
                                    df = pd.read_csv(f, low_memory=False)
                                    setattr(self, attr_name, df)
                                    self.gtfs_data[filename] = df
                                    loaded_files += 1
                            except:
                                pass
                    
                    if loaded_files >= 3:
                        self.is_loaded = True
                        self.last_update = datetime.datetime.now()
                        print(f"GTFS data loaded successfully")
                        return True
                        
            except Exception as e:
                continue
        
        print("Could not load GTFS data - using Google Maps only")
        return False
    
    def find_nearby_stops(self, lat, lon, radius_km=0.5):
        """Find GTFS stops near coordinates"""
        if not self.is_loaded or self.stops_df is None:
            return []
        
        try:
            stops = self.stops_df.copy()
            stops['distance_km'] = ((stops['stop_lat'] - lat)**2 + (stops['stop_lon'] - lon)**2)**0.5 * 111
            nearby = stops[stops['distance_km'] <= radius_km].sort_values('distance_km')
            return nearby[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'distance_km']].head(5).to_dict('records')
        except:
            return []

# Global GTFS manager
gtfs_manager = EnhancedGTFSManager()

# =============================================================================
# MULTI-ROUTE OSRM BIKE TOOL
# =============================================================================

class MultiRouteOSRMTool:
    def __init__(self):
        self.roads_gdf = None
        self.transit_stops_gdf = None
        self.bike_speed_mph = BIKE_SPEED_MPH
        print("Loading shapefiles...")
        self.load_data()

    def load_data(self):
        """Load shapefiles if they exist"""
        try:
            if os.path.exists(ROADS_SHAPEFILE):
                self.roads_gdf = gpd.read_file(ROADS_SHAPEFILE)
                print(f"Loaded roads: {len(self.roads_gdf)} features")
            
            if os.path.exists(TRANSIT_STOPS_SHAPEFILE):
                self.transit_stops_gdf = gpd.read_file(TRANSIT_STOPS_SHAPEFILE)
                print(f"Loaded transit stops: {len(self.transit_stops_gdf)} features")
        except Exception as e:
            print(f"Note: Shapefiles not loaded - {e}")

    def generate_alternative_points(self, start_coords, end_coords, num_points=4):
        """Generate intermediate waypoints for alternative routes"""
        try:
            start_lon, start_lat = start_coords
            end_lon, end_lat = end_coords

            center_lon = (start_lon + end_lon) / 2
            center_lat = (start_lat + end_lat) / 2
            distance = math.sqrt((end_lon - start_lon)**2 + (end_lat - start_lat)**2)
            variation_radius = distance * 0.3

            waypoint_sets = []
            for i in range(num_points):
                angle = (i * 2 * math.pi / num_points) + random.uniform(-0.3, 0.3)
                radius = variation_radius * random.uniform(0.5, 1.0)
                waypoint_lon = center_lon + radius * math.cos(angle)
                waypoint_lat = center_lat + radius * math.sin(angle)
                waypoint_sets.append([waypoint_lon, waypoint_lat])

            return waypoint_sets
        except:
            return []

    def create_bike_route_osrm(self, start_coords, end_coords, waypoints=None, route_name="Bike Route"):
        """Create a bike route via OSRM"""
        try:
            coords_list = [f"{start_coords[0]},{start_coords[1]}"]
            if waypoints:
                for wp in waypoints:
                    coords_list.append(f"{wp[0]},{wp[1]}")
            coords_list.append(f"{end_coords[0]},{end_coords[1]}")
            coords = ";".join(coords_list)

            url = f"{OSRM_SERVER}/route/v1/cycling/{coords}"
            params = {
                "overview": "full",
                "geometries": "polyline",
                "steps": "true",
                "alternatives": "false",
            }
            if REQUEST_EDGE_ANNOTATIONS:
                params["annotations"] = "duration"

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("code") != "Ok" or not data.get("routes"):
                return None

            route = data["routes"][0]

            # Extract geometry
            geometry_polyline = route["geometry"]
            coords_latlon = polyline.decode(geometry_polyline)
            route_geometry = {
                "type": "LineString",
                "coordinates": [[lon, lat] for (lat, lon) in coords_latlon],
            }

            # Distance and duration
            distance_meters = float(route.get("distance", 0.0))
            distance_miles = distance_meters * 0.000621371

            osrm_duration_sec = route.get("duration", None)
            if USE_OSRM_DURATION and isinstance(osrm_duration_sec, (int, float)) and osrm_duration_sec > 0:
                travel_time_minutes = float(osrm_duration_sec) / 60.0
                osrm_duration_used = True
            else:
                travel_time_minutes = (distance_miles / self.bike_speed_mph) * 60.0
                osrm_duration_used = False

            route_data = {
                "name": route_name,
                "length_feet": distance_meters * 3.28084,
                "length_miles": distance_miles,
                "travel_time_minutes": travel_time_minutes,
                "travel_time_formatted": self.format_time_duration(travel_time_minutes),
                "osrm_duration_used": osrm_duration_used,
                "geometry": route_geometry,
                "waypoints": waypoints or [],
            }

            # Analyze with shapefiles if available
            if self.roads_gdf is not None:
                segments, overall_score, facility_stats = self.analyze_route_segments(route_geometry)
                route_data.update({
                    "segments": segments,
                    "overall_score": overall_score,
                    "facility_stats": facility_stats,
                })
            else:
                route_data.update({
                    "segments": [],
                    "overall_score": 70,  # Default score when no analysis available
                    "facility_stats": {
                        "MIXED": {
                            "length_miles": distance_miles,
                            "percentage": 100,
                            "avg_score": 70,
                        }
                    },
                })

            return route_data

        except Exception as e:
            print(f"Error creating OSRM route: {e}")
            return None

    def create_multiple_routes(self, start_coords, end_coords, num_routes=4):
        """Create multiple alternative bike routes"""
        routes = []

        # Direct route first
        direct_route = self.create_bike_route_osrm(start_coords, end_coords, None, "Direct Route")
        if direct_route:
            direct_route['route_type'] = 'direct'
            routes.append(direct_route)

        # Generate alternative routes
        waypoint_sets = self.generate_alternative_points(start_coords, end_coords, num_routes - 1)
        route_names = ["Scenic Route", "Alternative Route", "Extended Route", "Northern Route", "Southern Route", "Eastern Route"]

        for i, waypoints in enumerate(waypoint_sets):
            route_name = route_names[i] if i < len(route_names) else f"Route {i + 2}"
            alt_route = self.create_bike_route_osrm(start_coords, end_coords, [waypoints], route_name)
            if alt_route:
                alt_route['route_type'] = 'alternative'
                routes.append(alt_route)

        # Sort by score then distance
        routes.sort(key=lambda r: (-r.get('overall_score', 0), r.get('length_miles', 999)))
        return routes

    def analyze_route_segments(self, route_geometry):
        """Analyze route with shapefile data"""
        try:
            if not route_geometry or not route_geometry.get('coordinates'):
                return [], 50, {"MIXED": {"length_miles": 0.5, "percentage": 100, "avg_score": 50}}

            route_coords = route_geometry['coordinates']
            route_line = LineString([(coord[0], coord[1]) for coord in route_coords])
            route_gdf = gpd.GeoDataFrame([1], geometry=[route_line], crs='EPSG:4326')

            if self.roads_gdf.crs != route_gdf.crs:
                route_gdf = route_gdf.to_crs(self.roads_gdf.crs)

            buffer_size = 0.0001 if route_gdf.crs.is_geographic else 15
            route_buffer = route_gdf.buffer(buffer_size)

            intersecting_roads = gpd.sjoin(
                self.roads_gdf,
                gpd.GeoDataFrame([1], geometry=route_buffer, crs=route_gdf.crs),
                predicate='intersects'
            )

            if intersecting_roads.empty:
                return [], 50, {"MIXED": {"length_miles": route_line.length/5280, "percentage": 100, "avg_score": 50}}

            # Find facility field
            facility_field = None
            possible_fields = ['facility_type', 'FACILITY_TYPE', 'bike_facility', 'BIKE_FACILITY']
            for field in possible_fields:
                if field in intersecting_roads.columns:
                    facility_field = field
                    break

            if not facility_field:
                return [], 50, {"MIXED": {"length_miles": route_line.length/5280, "percentage": 100, "avg_score": 50}}

            # Analyze facilities
            facility_stats = {}
            total_length = 0

            for _, row in intersecting_roads.iterrows():
                facility_type = row.get(facility_field, "NO BIKELANE")
                if pd.isna(facility_type):
                    facility_type = "NO BIKELANE"

                length = row.geometry.length if hasattr(row.geometry, 'length') else 100
                total_length += length

                if facility_type not in facility_stats:
                    facility_stats[facility_type] = {
                        'length_feet': 0,
                        'length_miles': 0,
                        'count': 0,
                        'avg_score': self.get_facility_score(facility_type),
                        'percentage': 0
                    }

                facility_stats[facility_type]['length_feet'] += length
                facility_stats[facility_type]['length_miles'] += length / 5280
                facility_stats[facility_type]['count'] += 1

            # Calculate percentages and overall score
            overall_score = 0
            for facility_type, stats in facility_stats.items():
                if total_length > 0:
                    stats['percentage'] = (stats['length_feet'] / total_length) * 100
                    overall_score += stats['avg_score'] * (stats['percentage'] / 100)

            return [], round(overall_score, 1), facility_stats

        except:
            return [], 50, {"MIXED": {"length_miles": 0.5, "percentage": 100, "avg_score": 50}}

    def get_facility_score(self, facility_type):
        """Assign scores to different bicycle facility types"""
        scoring = {
            'PROTECTED_BIKE_LANE': 90,
            'BUFFERED_BIKE_LANE': 85,
            'BIKE_LANE': 75,
            'SHARED_LANE': 60,
            'BIKE_PATH': 95,
            'MIXED_USE_PATH': 85,
            'NO BIKELANE': 30,
            'ARTERIAL': 25,
            'HIGHWAY': 10
        }
        return scoring.get(facility_type.upper(), 50)

    def format_time_duration(self, minutes):
        """Format time duration"""
        if minutes < 1:
            return "< 1 min"
        elif minutes < 60:
            return f"{int(minutes)} min"
        else:
            hours = int(minutes // 60)
            mins = int(minutes % 60)
            return f"{hours}h {mins}m" if mins > 0 else f"{hours}h"

# Global OSRM tool instance
osrm_tool = MultiRouteOSRMTool()

# =============================================================================
# GOOGLE MAPS TRANSIT FUNCTIONS
# =============================================================================

def decode_polyline(polyline_str):
    """Decode Google polyline string to coordinates"""
    try:
        index = 0
        lat = 0
        lng = 0
        coordinates = []
        
        while index < len(polyline_str):
            # Decode latitude
            result = 1
            shift = 0
            while True:
                b = ord(polyline_str[index]) - 63 - 1
                index += 1
                result += b << shift
                shift += 5
                if b < 0x1f:
                    break
            lat += (~result >> 1) if (result & 1) != 0 else (result >> 1)
            
            # Decode longitude
            result = 1
            shift = 0
            while True:
                b = ord(polyline_str[index]) - 63 - 1
                index += 1
                result += b << shift
                shift += 5
                if b < 0x1f:
                    break
            lng += (~result >> 1) if (result & 1) != 0 else (result >> 1)
            
            coordinates.append([lat / 1e5, lng / 1e5])
        
        return coordinates
    except:
        return []

def get_transit_routes_google(origin, destination, api_key, departure_time="now"):
    """Get transit routes using Google Maps API"""
    try:
        print(f"Getting Google Maps transit routes: '{origin}' -> '{destination}'")
        
        url = "https://maps.googleapis.com/maps/api/directions/json"
        
        if departure_time == "now":
            departure_timestamp = int(datetime.datetime.now().timestamp())
        else:
            departure_timestamp = departure_time
        
        params = {
            'origin': origin,
            'destination': destination,
            'mode': 'transit',
            'departure_time': departure_timestamp,
            'alternatives': 'true',
            'transit_mode': 'bus|subway|train|tram',
            'transit_routing_preference': 'fewer_transfers',
            'key': api_key
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') != 'OK':
            error_msg = data.get('error_message', f"API Status: {data.get('status')}")
            return {"error": error_msg}
        
        routes = []
        if 'routes' in data and data['routes']:
            for idx, route_data in enumerate(data['routes']):
                route = parse_google_transit_route(route_data, idx)
                if route:
                    routes.append(route)
        
        if not routes:
            return {"error": "No transit routes found between these locations"}
        
        return {
            "routes": routes,
            "service": "Google Maps Transit",
            "total_routes": len(routes),
            "last_update": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error getting Google Maps transit routes: {e}")
        return {"error": str(e)}

def parse_google_transit_route(route_data, route_index):
    """Parse a single Google transit route"""
    try:
        legs = route_data.get('legs', [])
        if not legs:
            return None
        
        leg = legs[0]
        
        duration_seconds = leg['duration']['value']
        duration_minutes = round(duration_seconds / 60, 1)
        duration_text = leg['duration']['text']
        
        distance_meters = leg['distance']['value']
        distance_km = round(distance_meters / 1000, 2)
        distance_miles = round(distance_km * 0.621371, 2)
        
        departure_time = leg.get('departure_time', {})
        arrival_time = leg.get('arrival_time', {})
        
        overview_polyline = route_data.get('overview_polyline', {}).get('points', '')
        route_geometry = []
        if overview_polyline:
            decoded_coords = decode_polyline(overview_polyline)
            if decoded_coords:
                route_geometry = [[coord[1], coord[0]] for coord in decoded_coords]
        
        steps = []
        transit_lines = []
        total_fare = 0
        
        for step_idx, step in enumerate(leg.get('steps', [])):
            parsed_step = parse_transit_step(step, step_idx)
            if parsed_step:
                steps.append(parsed_step)
                
                if parsed_step['travel_mode'] == 'TRANSIT':
                    if parsed_step.get('transit_line'):
                        transit_lines.append(parsed_step['transit_line'])
                    if parsed_step.get('fare_value'):
                        total_fare += parsed_step['fare_value']
        
        transit_steps = [s for s in steps if s['travel_mode'] == 'TRANSIT']
        transfers = max(0, len(transit_steps) - 1)
        
        walking_steps = [s for s in steps if s['travel_mode'] == 'WALKING']
        total_walking_meters = sum(s['distance_meters'] for s in walking_steps)
        total_walking_miles = round(total_walking_meters * 0.000621371, 2)
        
        route_name = f"Transit Route {route_index + 1}"
        if transfers > 0:
            route_name += f" ({transfers} transfer{'s' if transfers != 1 else ''})"
        
        return {
            "route_number": route_index + 1,
            "name": route_name,
            "description": f"Transit route with {transfers} transfer{'s' if transfers != 1 else ''}",
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_minutes,
            "duration_text": duration_text,
            "distance_meters": distance_meters,
            "distance_km": distance_km,
            "distance_miles": distance_miles,
            "departure_time": departure_time.get('text', 'Unknown'),
            "arrival_time": arrival_time.get('text', 'Unknown'),
            "transfers": transfers,
            "walking_distance_miles": total_walking_miles,
            "total_fare": total_fare if total_fare > 0 else None,
            "transit_lines": list(set(transit_lines)),
            "steps": steps,
            "geometry": {
                "type": "LineString",
                "coordinates": route_geometry
            },
            "service": "Google Maps Transit"
        }
        
    except Exception as e:
        print(f"Error parsing Google transit route: {e}")
        return None

def parse_transit_step(step, step_index):
    """Parse individual transit step"""
    try:
        travel_mode = step.get('travel_mode', 'UNKNOWN')
        
        # Clean HTML from instructions
        instruction = step.get('html_instructions', '')
        import re
        instruction = re.sub(r'<[^>]+>', '', instruction)
        
        duration_seconds = step['duration']['value']
        duration_minutes = round(duration_seconds / 60, 1)
        duration_text = step['duration']['text']
        
        distance_meters = step['distance']['value']
        distance_km = round(distance_meters / 1000, 2)
        distance_miles = round(distance_km * 0.621371, 2)
        
        step_data = {
            "step_number": step_index + 1,
            "travel_mode": travel_mode,
            "instruction": instruction,
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_minutes,
            "duration_text": duration_text,
            "distance_meters": distance_meters,
            "distance_km": distance_km,
            "distance_miles": distance_miles
        }
        
        if travel_mode == 'TRANSIT' and 'transit_details' in step:
            transit = step['transit_details']
            line = transit.get('line', {})
            vehicle = line.get('vehicle', {})
            agencies = line.get('agencies', [])
            
            step_data.update({
                "transit_line": line.get('short_name', line.get('name', 'Unknown Line')),
                "transit_line_color": line.get('color', '#1f8dd6'),
                "transit_vehicle_type": vehicle.get('type', 'BUS'),
                "transit_vehicle_name": vehicle.get('name', 'Bus'),
                "transit_agency": agencies[0].get('name', 'Transit Agency') if agencies else 'Transit Agency'
            })
            
            departure_stop = transit.get('departure_stop', {})
            arrival_stop = transit.get('arrival_stop', {})
            
            step_data.update({
                "departure_stop_name": departure_stop.get('name', 'Unknown Stop'),
                "departure_stop_location": departure_stop.get('location', {}),
                "arrival_stop_name": arrival_stop.get('name', 'Unknown Stop'),
                "arrival_stop_location": arrival_stop.get('location', {}),
                "num_stops": transit.get('num_stops', 0)
            })
            
            departure_time = transit.get('departure_time', {})
            arrival_time = transit.get('arrival_time', {})
            
            step_data.update({
                "scheduled_departure": departure_time.get('text', ''),
                "scheduled_arrival": arrival_time.get('text', ''),
                "departure_timestamp": departure_time.get('value', 0),
                "arrival_timestamp": arrival_time.get('value', 0)
            })
            
            step_data["headsign"] = transit.get('headsign', '')
            
            if 'fare' in line:
                fare = line['fare']
                step_data.update({
                    "fare_text": fare.get('text', ''),
                    "fare_value": fare.get('value', 0),
                    "fare_currency": fare.get('currency', 'USD')
                })
        
        return step_data
        
    except Exception as e:
        print(f"Error parsing step: {e}")
        return None

# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def find_nearby_bus_stops_simple(point_coords, max_stops=3):
    """Find nearby bus stops using simple distance calculation"""
    # This is a simplified version - in production you'd use real transit stop data
    base_coords = point_coords
    stops = []
    
    offsets = [0.003, 0.006, 0.009]  # Different distances
    names = ["Transit Hub", "Main Station", "Local Stop"]
    
    for i in range(min(max_stops, 3)):
        offset = offsets[i] if i < len(offsets) else 0.003 * (i + 1)
        name = names[i] if i < len(names) else f"Stop {i+1}"
        
        stops.append({
            "id": f"stop_{i+1}",
            "name": name,
            "x": base_coords[0] + offset,
            "y": base_coords[1] + offset,
            "display_x": base_coords[0] + offset,
            "display_y": base_coords[1] + offset,
            "distance_meters": (i + 1) * 400,
        })
    
    return stops

def analyze_enhanced_multimodal_routes(start_point, end_point, departure_time="now", num_bike_routes=4):
    """Main analysis function combining multi-route bikes + Google Maps transit"""
    try:
        print(f"Starting enhanced multimodal analysis: {start_point} to {end_point}")
        
        routes = []
        
        # Step 1: Create multiple bike route alternatives using OSRM
        print("Creating multiple bike route alternatives...")
        bike_routes = osrm_tool.create_multiple_routes(start_point, end_point, num_bike_routes)
        
        # Add bike routes to results
        colors = ["#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#27ae60", "#e67e22"]
        for i, bike_route in enumerate(bike_routes):
            route_color = colors[i % len(colors)]
            
            formatted_route = {
                "id": len(routes) + 1,
                "name": bike_route['name'],
                "type": "direct_bike",
                "route_type": bike_route.get('route_type', 'alternative'),
                "summary": {
                    "total_time_minutes": bike_route['travel_time_minutes'],
                    "total_time_formatted": bike_route['travel_time_formatted'],
                    "total_distance_miles": bike_route['length_miles'],
                    "bike_distance_miles": bike_route['length_miles'],
                    "transit_distance_miles": 0,
                    "bike_percentage": 100,
                    "average_bike_score": bike_route.get('overall_score', 70),
                    "transfers": 0,
                    "departure_time": "Immediate",
                    "arrival_time":
