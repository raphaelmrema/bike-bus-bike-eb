# complete_enhanced_osrm_otp_planner.py
# Complete Enhanced OSRM + OpenTripPlanner Bike-Bus-Bike Route Planner
# Version 2.0 - Full Featured

import os
import logging
import requests
import datetime
import math
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Install polyline if needed
try:
    import polyline
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "polyline"])
    import polyline

# =============================================================================
# ENHANCED CONFIGURATION
# =============================================================================

OSRM_SERVER = os.getenv("OSRM_SERVER", "http://router.project-osrm.org")
OTP_SERVER = os.getenv("OTP_SERVER", "http://otp.prod.obanyc.com/otp")
OTP_ROUTER_ID = os.getenv("OTP_ROUTER_ID", "default")
BIKE_SPEED_MPH = 11

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for working OTP configuration
WORKING_OTP_SERVER = None
WORKING_OTP_ROUTER = None
SYSTEM_STATUS = {"otp_working": False, "last_check": None}

# Extended potential OTP servers to test
POTENTIAL_OTP_SERVERS = [
    {"name": "JTA Jacksonville", "server": "http://otp.jtafla.com/otp", "router": "jta"},
    {"name": "JTA Alternative", "server": "https://otp.jtafla.com/otp", "router": "default"},
    {"name": "Florida DOT", "server": "http://otp.fdot.gov/otp", "router": "florida"},
    {"name": "NYC Demo", "server": "http://otp.prod.obanyc.com/otp", "router": "default"},
    {"name": "TriMet Portland", "server": "http://maps.trimet.org/otp", "router": "default"},
    {"name": "OpenTripPlanner Demo", "server": "http://otp-demo.opentripplanner.org/otp", "router": "default"}
]

# =============================================================================
# FASTAPI APP SETUP
# =============================================================================

app = FastAPI(
    title="Enhanced OSRM + OTP Bike-Bus-Bike Planner", 
    version="2.0.0",
    description="Advanced multimodal transportation planning with enhanced features"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# ENHANCED UTILITY FUNCTIONS
# =============================================================================

def format_time_duration(minutes):
    """Enhanced time duration formatting"""
    if minutes < 1:
        return "< 1 min"
    elif minutes < 60:
        return f"{int(minutes)} min"
    else:
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours}h {mins}m" if mins > 0 else f"{hours}h"

def parse_otp_time(time_milliseconds):
    """Enhanced OTP time parsing with error handling"""
    try:
        dt = datetime.datetime.fromtimestamp(time_milliseconds / 1000)
        return dt.strftime("%H:%M")
    except (ValueError, TypeError, OSError):
        return "Unknown"

def calculate_distance_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers"""
    try:
        R = 6371  # Earth's radius in kilometers
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2) * math.sin(dlon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    except:
        return 0

# =============================================================================
# ENHANCED OTP SERVER TESTING
# =============================================================================

def test_otp_server_enhanced(server_url, router_id):
    """Enhanced OTP server testing with better error handling"""
    try:
        logger.info(f"Testing OTP server: {server_url} with router: {router_id}")
        
        # Test routers endpoint first
        routers_url = f"{server_url}/routers"
        response = requests.get(routers_url, timeout=15)
        
        if response.status_code != 200:
            return False, f"Server not accessible (HTTP {response.status_code})"
        
        # Check if our router exists in the list
        try:
            routers_data = response.json()
            router_ids = [r.get('id', '') for r in routers_data]
            if router_id not in router_ids and router_ids:
                # Try the first available router if default doesn't exist
                router_id = router_ids[0]
                logger.info(f"Router '{router_id}' not found, trying '{router_ids[0]}'")
        except:
            pass
        
        # Test plan endpoint with various test coordinates
        plan_url = f"{server_url}/routers/{router_id}/plan"
        test_coordinates = [
            ('30.3322,-81.6557', '30.3400,-81.6600'),  # Jacksonville
            ('40.7128,-74.0060', '40.7589,-73.9851'),  # NYC
            ('45.5152,-122.6784', '45.5051,-122.6750'), # Portland
            ('47.6062,-122.3321', '47.6205,-122.3493')  # Seattle
        ]
        
        for origin, destination in test_coordinates:
            test_params = {
                'fromPlace': origin,
                'toPlace': destination,
                'mode': 'TRANSIT,WALK',
                'date': datetime.datetime.now().strftime("%m-%d-%Y"),
                'time': '10:00AM',
                'maxWalkDistance': 1000
            }
            
            try:
                plan_response = requests.get(plan_url, params=test_params, timeout=20)
                
                if plan_response.status_code == 200:
                    plan_data = plan_response.json()
                    if 'plan' in plan_data and plan_data['plan'].get('itineraries'):
                        global WORKING_OTP_SERVER, WORKING_OTP_ROUTER
                        WORKING_OTP_SERVER = server_url
                        WORKING_OTP_ROUTER = router_id
                        return True, f"Server working with {len(plan_data['plan']['itineraries'])} routes"
                    elif 'error' in plan_data:
                        error_msg = plan_data['error'].get('msg', 'Unknown error')
                        if 'VertexNotFoundException' in error_msg:
                            continue  # Try next coordinate set
                        else:
                            return False, f"Plan error: {error_msg}"
                else:
                    continue  # Try next coordinate set
                    
            except Exception as e:
                continue  # Try next coordinate set
        
        return False, "No working coordinate sets found"
        
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def find_working_otp_server_enhanced():
    """Enhanced OTP server discovery"""
    global WORKING_OTP_SERVER, WORKING_OTP_ROUTER, SYSTEM_STATUS
    
    logger.info("Starting enhanced OTP server discovery...")
    
    # Try environment variables first
    env_server = os.getenv("OTP_SERVER")
    env_router = os.getenv("OTP_ROUTER_ID")
    
    if env_server and env_router:
        is_working, message = test_otp_server_enhanced(env_server, env_router)
        if is_working:
            WORKING_OTP_SERVER = env_server
            WORKING_OTP_ROUTER = env_router
            SYSTEM_STATUS = {"otp_working": True, "last_check": datetime.datetime.now(), "server": env_server}
            logger.info(f"Environment OTP server working: {env_server}")
            return env_server, env_router
    
    # Test potential servers
    for server_config in POTENTIAL_OTP_SERVERS:
        server = server_config["server"]
        router = server_config["router"]
        name = server_config["name"]
        
        logger.info(f"Testing {name}: {server}")
        is_working, message = test_otp_server_enhanced(server, router)
        
        if is_working:
            WORKING_OTP_SERVER = server
            WORKING_OTP_ROUTER = router
            SYSTEM_STATUS = {"otp_working": True, "last_check": datetime.datetime.now(), "server": server, "name": name}
            logger.info(f"Found working OTP server: {name} ({server})")
            return server, router
        else:
            logger.info(f"{name} failed: {message}")
    
    SYSTEM_STATUS = {"otp_working": False, "last_check": datetime.datetime.now()}
    logger.warning("No working OTP servers found")
    return None, None

# =============================================================================
# ENHANCED OSRM BICYCLE ROUTING
# =============================================================================

def calculate_bike_route_osrm_enhanced(start_coords, end_coords, route_name="Bike Route"):
    """Enhanced OSRM bicycle routing with better error handling and metrics"""
    try:
        coords = f"{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
        url = f"{OSRM_SERVER}/route/v1/cycling/{coords}"
        
        params = {
            "overview": "full",
            "geometries": "polyline",
            "steps": "true",
            "annotations": "speed,duration,distance"
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("code") != "Ok" or not data.get("routes"):
            return None
        
        route = data["routes"][0]
        
        # Extract geometry
        geometry_polyline = route["geometry"]
        coords_latlon = polyline.decode(geometry_polyline)
        route_geometry = [[lon, lat] for (lat, lon) in coords_latlon]
        
        # Extract distance and duration
        distance_meters = float(route.get("distance", 0.0))
        distance_miles = distance_meters * 0.000621371
        
        duration_seconds = route.get("duration", 0)
        if duration_seconds > 0:
            duration_minutes = duration_seconds / 60.0
        else:
            duration_minutes = (distance_miles / BIKE_SPEED_MPH) * 60.0
        
        # Calculate enhanced bike score based on route characteristics
        bike_score = calculate_enhanced_bike_score(route, distance_meters)
        
        # Extract route segments for visualization
        segments = extract_route_segments(route, route_geometry)
        
        return {
            "name": route_name,
            "length_miles": distance_miles,
            "length_feet": distance_meters * 3.28084,
            "travel_time_minutes": duration_minutes,
            "travel_time_formatted": format_time_duration(duration_minutes),
            "geometry": {
                "type": "LineString",
                "coordinates": route_geometry
            },
            "overall_score": bike_score,
            "segments": segments,
            "facility_stats": generate_facility_stats(segments),
            "osrm_enhanced": True
        }
        
    except Exception as e:
        logger.error(f"Error calculating enhanced bike route: {e}")
        return None

def calculate_enhanced_bike_score(route, distance_meters):
    """Calculate enhanced bike safety score"""
    try:
        base_score = 70  # Base OSRM cycling score
        
        # Adjust based on route characteristics
        legs = route.get('legs', [])
        if legs:
            leg = legs[0]
            steps = leg.get('steps', [])
            
            # Penalize routes with many turns (complexity)
            if len(steps) > 20:
                base_score -= min(10, (len(steps) - 20) * 0.5)
            
            # Bonus for longer routes (likely on better infrastructure)
            if distance_meters > 5000:  # > 5km
                base_score += 5
            
            # Analyze road types from steps
            highway_count = 0
            residential_count = 0
            
            for step in steps:
                name = step.get('name', '').lower()
                if any(term in name for term in ['highway', 'freeway', 'interstate']):
                    highway_count += 1
                elif any(term in name for term in ['residential', 'local', 'neighborhood']):
                    residential_count += 1
            
            # Adjust score based on road types
            if highway_count > len(steps) * 0.3:  # > 30% highway
                base_score -= 15
            if residential_count > len(steps) * 0.5:  # > 50% residential
                base_score += 10
        
        return max(20, min(100, int(base_score)))
        
    except Exception as e:
        logger.error(f"Error calculating bike score: {e}")
        return 70

def extract_route_segments(route, route_geometry):
    """Extract route segments for enhanced visualization"""
    try:
        segments = []
        legs = route.get('legs', [])
        
        if not legs:
            return []
        
        leg = legs[0]
        steps = leg.get('steps', [])
        
        coord_index = 0
        for step in steps:
            step_distance = step.get('distance', 0)
            step_duration = step.get('duration', 0)
            step_name = step.get('name', 'Unnamed road')
            
            # Estimate coordinates for this step
            coords_needed = max(2, int(len(route_geometry) * (step_distance / leg.get('distance', 1))))
            step_coords = route_geometry[coord_index:coord_index + coords_needed]
            coord_index += coords_needed - 1  # Overlap by 1
            
            if len(step_coords) >= 2:
                # Determine facility type based on road name and characteristics
                facility_type = classify_road_facility(step_name, step_distance)
                
                segments.append({
                    "facility_type": facility_type,
                    "bike_score": get_facility_score(facility_type),
                    "total_score": get_facility_score(facility_type),
                    "shape_length_feet": step_distance * 3.28084,
                    "shape_length_miles": step_distance * 0.000621371,
                    "geometry": {
                        "type": "LineString",
                        "coordinates": step_coords
                    },
                    "road_name": step_name,
                    "duration_seconds": step_duration
                })
        
        return segments
        
    except Exception as e:
        logger.error(f"Error extracting segments: {e}")
        return []

def classify_road_facility(road_name, distance):
    """Classify road facility type based on name and characteristics"""
    name_lower = road_name.lower()
    
    if any(term in name_lower for term in ['bike', 'cycle', 'path', 'trail']):
        if 'protected' in name_lower or 'separated' in name_lower:
            return "PROTECTED BIKELANE"
        else:
            return "SHARED USE PATH"
    elif any(term in name_lower for term in ['highway', 'freeway', 'interstate', 'arterial']):
        return "NO BIKELANE"
    elif any(term in name_lower for term in ['residential', 'local', 'neighborhood']):
        return "SHARED LANE"
    elif distance > 1000:  # Long segments likely on major roads
        return "UNBUFFERED BIKELANE"
    else:
        return "SHARED LANE"

def get_facility_score(facility_type):
    """Get safety score for facility type"""
    scores = {
        "PROTECTED BIKELANE": 95,
        "BUFFERED BIKELANE": 85,
        "UNBUFFERED BIKELANE": 70,
        "SHARED USE PATH": 90,
        "SHARED LANE": 50,
        "NO BIKELANE": 25
    }
    return scores.get(facility_type, 50)

def generate_facility_stats(segments):
    """Generate facility statistics"""
    if not segments:
        return {"NO BIKELANE": {"length_feet": 1000, "length_miles": 0.19, "count": 1, "avg_score": 25, "percentage": 100}}
    
    facility_stats = {}
    total_length = sum(seg['shape_length_feet'] for seg in segments)
    
    for segment in segments:
        facility_type = segment['facility_type']
        if facility_type not in facility_stats:
            facility_stats[facility_type] = {
                'length_feet': 0,
                'length_miles': 0,
                'count': 0,
                'total_score_sum': 0,
                'avg_score': 0,
                'percentage': 0
            }
        
        facility_stats[facility_type]['length_feet'] += segment['shape_length_feet']
        facility_stats[facility_type]['length_miles'] += segment['shape_length_miles']
        facility_stats[facility_type]['count'] += 1
        facility_stats[facility_type]['total_score_sum'] += segment['bike_score']
    
    # Calculate averages and percentages
    for facility_type, stats in facility_stats.items():
        if stats['count'] > 0:
            stats['avg_score'] = stats['total_score_sum'] / stats['count']
        if total_length > 0:
            stats['percentage'] = (stats['length_feet'] / total_length) * 100
    
    return facility_stats

# =============================================================================
# ENHANCED OTP TRANSIT ROUTING
# =============================================================================

def get_transit_routes_otp_enhanced(origin_coords, destination_coords, departure_time="now"):
    """Enhanced OTP transit routing with better error handling"""
    global WORKING_OTP_SERVER, WORKING_OTP_ROUTER
    
    if not WORKING_OTP_SERVER:
        server, router = find_working_otp_server_enhanced()
        if not server:
            return {"error": "No working OTP server found"}
    
    try:
        # Prepare time parameters
        if departure_time == "now":
            time_str = datetime.datetime.now().strftime("%I:%M%p")
            date_str = datetime.datetime.now().strftime("%m-%d-%Y")
        else:
            try:
                dt = datetime.datetime.fromtimestamp(int(departure_time))
                time_str = dt.strftime("%I:%M%p")
                date_str = dt.strftime("%m-%d-%Y")
            except:
                time_str = datetime.datetime.now().strftime("%I:%M%p")
                date_str = datetime.datetime.now().strftime("%m-%d-%Y")
        
        url = f"{WORKING_OTP_SERVER}/routers/{WORKING_OTP_ROUTER}/plan"
        
        params = {
            'fromPlace': f"{origin_coords[1]},{origin_coords[0]}",
            'toPlace': f"{destination_coords[1]},{destination_coords[0]}",
            'time': time_str,
            'date': date_str,
            'mode': 'TRANSIT,WALK',
            'maxWalkDistance': 1200,
            'numItineraries': 5,  # Request more options
            'walkReluctance': 2,
            'transferPenalty': 300
        }
        
        logger.info(f"OTP request: {url} with params: {params}")
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'error' in data:
            error_msg = data['error'].get('msg', 'OTP error')
            logger.error(f"OTP API error: {error_msg}")
            return {"error": error_msg}
        
        if 'plan' not in data or 'itineraries' not in data['plan']:
            logger.warning("No transit routes found in OTP response")
            return {"error": "No transit routes found"}
        
        routes = []
        for idx, itinerary in enumerate(data['plan']['itineraries']):
            route = parse_otp_itinerary_enhanced(itinerary, idx)
            if route:
                routes.append(route)
        
        logger.info(f"Successfully parsed {len(routes)} transit routes")
        
        return {
            "routes": routes,
            "service": "OpenTripPlanner Enhanced",
            "total_routes": len(routes),
            "otp_server": WORKING_OTP_SERVER
        }
        
    except Exception as e:
        logger.error(f"Enhanced OTP error: {e}")
        return {"error": str(e)}

def parse_otp_itinerary_enhanced(itinerary, route_index):
    """Enhanced OTP itinerary parsing with more detailed information"""
    try:
        duration_seconds = itinerary.get('duration', 0)
        duration_minutes = duration_seconds / 60.0
        
        legs = itinerary.get('legs', [])
        steps = []
        transit_lines = []
        transfers = 0
        total_distance = 0
        route_geometry = []
        walking_distance = 0
        
        for leg_idx, leg in enumerate(legs):
            step = parse_otp_leg_enhanced(leg, leg_idx)
            if step:
                steps.append(step)
                
                leg_distance = leg.get('distance', 0) * 0.000621371
                total_distance += leg_distance
                
                if step['travel_mode'] == 'TRANSIT':
                    if step.get('transit_line'):
                        transit_lines.append(step['transit_line'])
                    transfers += 1
                elif step['travel_mode'] == 'WALKING':
                    walking_distance += leg_distance
                
                # Add enhanced geometry
                if leg.get('legGeometry', {}).get('points'):
                    try:
                        leg_coords = polyline.decode(leg['legGeometry']['points'])
                        leg_coords_geojson = [[lon, lat] for (lat, lon) in leg_coords]
                        route_geometry.extend(leg_coords_geojson)
                    except:
                        pass
        
        transfers = max(0, transfers - 1)
        
        start_time = itinerary.get('startTime', 0)
        end_time = itinerary.get('endTime', 0)
        
        route_name = f"Transit Route {route_index + 1}"
        if transfers > 0:
            route_name += f" ({transfers} transfer{'s' if transfers != 1 else ''})"
        
        return {
            "route_number": route_index + 1,
            "name": route_name,
            "description": f"Enhanced OTP route with {transfers} transfer{'s' if transfers != 1 else ''}",
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_minutes,
            "duration_text": format_time_duration(duration_minutes),
            "distance_miles": total_distance,
            "walking_distance_miles": walking_distance,
            "departure_time": parse_otp_time(start_time),
            "arrival_time": parse_otp_time(end_time),
            "transfers": transfers,
            "transit_lines": list(set(transit_lines)),
            "route_geometry": route_geometry,
            "steps": steps,
            "service": "OpenTripPlanner Enhanced",
            "otp_enhanced": True
        }
        
    except Exception as e:
        logger.error(f"Error parsing enhanced itinerary: {e}")
        return None

def parse_otp_leg_enhanced(leg, leg_index):
    """Enhanced OTP leg parsing with detailed transit information"""
    try:
        mode = leg.get('mode', 'UNKNOWN')
        
        if mode == 'WALK':
            travel_mode = 'WALKING'
        elif mode in ['BUS', 'SUBWAY', 'RAIL', 'TRAM', 'FERRY']:
            travel_mode = 'TRANSIT'
        else:
            travel_mode = mode
        
        duration_seconds = leg.get('duration', 0)
        duration_minutes = duration_seconds / 60.0
        
        distance_meters = leg.get('distance', 0)
        distance_miles = distance_meters * 0.000621371
        
        from_place = leg.get('from', {})
        to_place = leg.get('to', {})
        from_name = from_place.get('name', 'Unknown')
        to_name = to_place.get('name', 'Unknown')
        
        if travel_mode == 'WALKING':
            instruction = f"Walk from {from_name} to {to_name}"
        else:
            route_name = leg.get('route', leg.get('routeShortName', 'Transit'))
            instruction = f"Take {route_name} from {from_name} to {to_name}"
        
        step_data = {
            "step_number": leg_index + 1,
            "travel_mode": travel_mode,
            "instruction": instruction,
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_minutes,
            "duration_text": format_time_duration(duration_minutes),
            "distance_meters": distance_meters,
            "distance_miles": distance_miles,
            "from_place": from_name,
            "to_place": to_name
        }
        
        if travel_mode == 'TRANSIT':
            route_short_name = leg.get('routeShortName', leg.get('route', 'Unknown'))
            route_long_name = leg.get('routeLongName', '')
            
            step_data.update({
                "transit_line": route_short_name,
                "transit_line_full": route_long_name,
                "transit_vehicle_type": mode,
                "transit_agency": leg.get('agencyName', 'Transit Agency'),
                "route_color": leg.get('routeColor', '#1f8dd6')
            })
            
            # Enhanced stop information
            step_data.update({
                "departure_stop_name": from_name,
                "departure_stop_location": {
                    "lat": from_place.get('lat', 0),
                    "lng": from_place.get('lon', 0)
                },
                "arrival_stop_name": to_name,
                "arrival_stop_location": {
                    "lat": to_place.get('lat', 0),
                    "lng": to_place.get('lon', 0)
                },
                "departure_stop_id": from_place.get('stopId', ''),
                "arrival_stop_id": to_place.get('stopId', '')
            })
            
            # Enhanced timing information
            start_time = leg.get('startTime', 0)
            end_time = leg.get('endTime', 0)
            
            step_data.update({
                "scheduled_departure": parse_otp_time(start_time),
                "scheduled_arrival": parse_otp_time(end_time),
                "departure_timestamp": start_time,
                "arrival_timestamp": end_time,
                "headsign": leg.get('headsign', ''),
                "trip_id": leg.get('tripId', '')
            })
            
            # Real-time information if available
            if leg.get('realTime'):
                step_data.update({
                    "realtime_departure": parse_otp_time(leg.get('departureDelay', 0) + start_time),
                    "realtime_arrival": parse_otp_time(leg.get('arrivalDelay', 0) + end_time),
                    "departure_delay": leg.get('departureDelay', 0) / 60,  # Convert to minutes
                    "arrival_delay": leg.get('arrivalDelay', 0) / 60,
                    "is_realtime": True
                })
        
        elif travel_mode == 'WALKING':
            step_data.update({
                "walking_type": "transit_connection" if leg_index > 0 else "access",
                "is_transfer_walk": leg_index > 0,
                "walking_steps": leg.get('steps', [])
            })
        
        return step_data
        
    except Exception as e:
        logger.error(f"Error parsing enhanced leg: {e}")
        return None

def find_nearby_bus_stops_enhanced(point_coords, max_stops=3):
    """Enhanced nearby bus stops finding with better error handling"""
    global WORKING_OTP_SERVER, WORKING_OTP_ROUTER
    
    if not WORKING_OTP_SERVER:
        return create_fallback_stops(point_coords, max_stops)
    
    try:
        url = f"{WORKING_OTP_SERVER}/routers/{WORKING_OTP_ROUTER}/index/stops"
        params = {
            'lat': point_coords[1],
            'lon': point_coords[0],
            'radius': 1000  # Increased radius
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            stops_data = response.json()
            
            formatted_stops = []
            for stop in stops_data:
                stop_lat = stop.get('lat', point_coords[1])
                stop_lon = stop.get('lon', point_coords[0])
                
                # Calculate actual distance
                distance_km = calculate_distance_km(point_coords[1], point_coords[0], stop_lat, stop_lon)
                
                formatted_stops.append({
                    "id": stop.get('id', f"stop_{len(formatted_stops)}"),
                    "name": stop.get('name', f"Bus Stop {len(formatted_stops) + 1}"),
                    "x": stop_lon,
                    "y": stop_lat,
                    "display_x": stop_lon,
                    "display_y": stop_lat,
                    "distance_meters": distance_km * 1000,
                    "stop_code": stop.get('code', ''),
                    "wheelchair_accessible": stop.get('wheelchairAccessible', False)
                })
            
            # Sort by distance and return closest
            formatted_stops.sort(key=lambda x: x['distance_meters'])
            return formatted_stops[:max_stops] if formatted_stops else create_fallback_stops(point_coords, max_stops)
        
        return create_fallback_stops(point_coords, max_stops)
        
    except Exception as e:
        logger.error(f"Error finding enhanced stops: {e}")
        return create_fallback_stops(point_coords, max_stops)

def create_fallback_stops(point_coords, max_stops):
    """Create fallback bus stops when OTP is not available"""
    fallback_stops = []
    offsets = [0.003, 0.006, 0.009]  # Different distances
    
    for i in range(min(max_stops, 3)):
        offset = offsets[i] if i < len(offsets) else 0.003 * (i + 1)
        fallback_stops.append({
            "id": f"fallback_stop_{i+1}",
            "name": f"Bus Stop {i+1}",
            "x": point_coords[0] + offset,
            "y": point_coords[1] + offset,
            "display_x": point_coords[0] + offset,
            "display_y": point_coords[1] + offset,
            "distance_meters": (i + 1) * 400,
            "stop_code": f"STOP{i+1}",
            "wheelchair_accessible": True
        })
    
    return fallback_stops

# =============================================================================
# ENHANCED MAIN ROUTE ANALYSIS
# =============================================================================

def analyze_bike_bus_bike_routes_enhanced(start_point, end_point, departure_time="now"):
    """Enhanced main analysis function with comprehensive multimodal routing"""
    try:
        logger.info(f"Starting enhanced analysis: {start_point} to {end_point}")
        
        # Find nearby bus stops with enhanced logic
        start_bus_stops = find_nearby_bus_stops_enhanced(start_point, max_stops=3)
        end_bus_stops = find_nearby_bus_stops_enhanced(end_point, max_stops=3)
        
        routes = []
        analysis_type = "enhanced_multimodal"
        
        # Enhanced fallback decision logic
        should_fallback = False
        fallback_reason = ""
        
        if start_bus_stops and end_bus_stops:
            start_stop = start_bus_stops[0]
            end_stop = end_bus_stops[0]
            
            # Calculate distances more accurately
            dist1 = calculate_distance_km(start_point[1], start_point[0], 
                                        start_stop['display_y'], start_stop['display_x']) * 1000
            dist2 = calculate_distance_km(end_point[1], end_point[0], 
                                        end_stop['display_y'], end_stop['display_x']) * 1000
            
            if dist1 < 400 and dist2 < 400:
                should_fallback = True
                fallback_reason = f"Both bike segments very short ({dist1:.0f}m + {dist2:.0f}m)"
            elif not SYSTEM_STATUS.get('otp_working', False):
                should_fallback = True
                fallback_reason = "OTP server unavailable - using direct routing"
        
        # Enhanced transit fallback
        if should_fallback:
            logger.info(f"Using enhanced transit fallback: {fallback_reason}")
            transit_result = get_transit_routes_otp_enhanced(start_point, end_point, departure_time)
            
            if transit_result.get('routes'):
                for i, transit_route in enumerate(transit_result['routes']):
                    enhanced_route = create_enhanced_route_object(
                        route_id=i + 1,
                        name=f"Transit Option {i + 1}",
                        route_type="transit_fallback",
                        legs=[{
                            "type": "transit",
                            "name": f"Transit Route {i + 1}",
                            "description": "Enhanced direct transit route",
                            "route": transit_route,
                            "color": "#2196f3",
                            "order": 1
                        }],
                        fallback_reason=fallback_reason
                    )
                    routes.append(enhanced_route)
        
        # Enhanced bike-bus-bike routes
        if start_bus_stops and end_bus_stops and not should_fallback and len(start_bus_stops) > 0 and len(end_bus_stops) > 0:
            start_bus_stop = start_bus_stops[0]
            end_bus_stop = end_bus_stops[0]
            
            # Ensure different stops
            if start_bus_stop["id"] == end_bus_stop["id"] and len(end_bus_stops) > 1:
                end_bus_stop = end_bus_stops[1]
            
            if start_bus_stop["id"] != end_bus_stop["id"]:
                # Create enhanced bike legs
                bike_leg_1 = calculate_bike_route_osrm_enhanced(
                    start_point,
                    [start_bus_stop["display_x"], start_bus_stop["display_y"]],
                    "Bike to Transit"
                )
                
                bike_leg_2 = calculate_bike_route_osrm_enhanced(
                    [end_bus_stop["display_x"], end_bus_stop["display_y"]],
                    end_point,
                    "Transit to Destination"
                )
                
                if bike_leg_1 and bike_leg_2:
                    # Enhanced transit between stops
                    start_coords = (start_bus_stop['display_x'], start_bus_stop['display_y'])
                    end_coords = (end_bus_stop['display_x'], end_bus_stop['display_y'])
                    
                    transit_result = get_transit_routes_otp_enhanced(start_coords, end_coords, departure_time)
                    
                    if transit_result.get('routes'):
                        for i, transit_route in enumerate(transit_result['routes']):
                            enhanced_route = create_enhanced_multimodal_route(
                                route_id=len(routes) + 1,
                                route_index=i,
                                bike_leg_1=bike_leg_1,
                                transit_route=transit_route,
                                bike_leg_2=bike_leg_2,
                                start_stop=start_bus_stop,
                                end_stop=end_bus_stop
                            )
                            routes.append(enhanced_route)
        
        # Enhanced direct bike route
        direct_bike_route = calculate_bike_route_osrm_enhanced(start_point, end_point, "Direct Bike Route")
        
        if direct_bike_route:
            enhanced_direct = create_enhanced_route_object(
                route_id=len(routes) + 1,
                name="Direct Bike Route",
                route_type="direct_bike",
                legs=[{
                    "type": "bike",
                    "name": "Direct Bike Route",
                    "description": "Complete enhanced bike route from start to destination",
                    "route": direct_bike_route,
                    "color": "#e74c3c",
                    "order": 1
                }]
            )
            routes.append(enhanced_direct)
        
        if not routes:
            raise HTTPException(status_code=400, detail="No enhanced routes found")
        
        # Enhanced sorting and statistics
        routes.sort(key=lambda x: x['summary']['total_time_minutes'])
        
        # Calculate enhanced statistics
        statistics = calculate_enhanced_statistics(routes)
        
        return {
            "success": True,
            "analysis_type": analysis_type,
            "fallback_used": should_fallback,
            "fallback_reason": fallback_reason if should_fallback else None,
            "routes": routes,
            "statistics": statistics,
            "bus_stops": {
                "start_stops": start_bus_stops,
                "end_stops": end_bus_stops,
                "selected_start": start_bus_stops[0] if start_bus_stops else None,
                "selected_end": end_bus_stops[0] if end_bus_stops else None
            },
            "system_info": SYSTEM_STATUS,
            "otp_server": WORKING_OTP_SERVER,
            "analysis_timestamp": datetime.datetime.now().isoformat(),
            "enhanced_features": True
        }
        
    except Exception as e:
        logger.error(f"Enhanced analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def create_enhanced_route_object(route_id, name, route_type, legs, fallback_reason=None):
    """Create enhanced route object with comprehensive metrics"""
    try:
        # Calculate totals
        total_time = sum(leg['route'].get('travel_time_minutes', leg['route'].get('duration_minutes', 0)) for leg in legs)
        total_bike_miles = sum(leg['route'].get('length_miles', 0) for leg in legs if leg['type'] == 'bike')
        total_transit_miles = sum(leg['route'].get('distance_miles', 0) for leg in legs if leg['type'] == 'transit')
        total_miles = total_bike_miles + total_transit_miles
        
        # Calculate enhanced bike score
        bike_legs = [leg for leg in legs if leg['type'] == 'bike']
        if bike_legs:
            total_bike_length = sum(leg['route'].get('length_feet', 0) for leg in bike_legs)
            if total_bike_length > 0:
                weighted_score = sum(
                    (leg['route'].get('overall_score', 70) * leg['route'].get('length_feet', 0))
                    for leg in bike_legs
                ) / total_bike_length
            else:
                weighted_score = 70
        else:
            weighted_score = 0
        
        # Count transfers
        transit_legs = [leg for leg in legs if leg['type'] == 'transit']
        transfers = sum(leg['route'].get('transfers', 0) for leg in transit_legs)
        
        enhanced_route = {
            "id": route_id,
            "name": name,
            "type": route_type,
            "summary": {
                "total_time_minutes": round(total_time, 1),
                "total_time_formatted": format_time_duration(total_time),
                "total_distance_miles": round(total_miles, 2),
                "bike_distance_miles": round(total_bike_miles, 2),
                "transit_distance_miles": round(total_transit_miles, 2),
                "bike_percentage": round((total_bike_miles / total_miles) * 100, 1) if total_miles > 0 else 0,
                "average_bike_score": round(weighted_score, 1),
                "transfers": transfers,
                "total_fare": 0,  # Could be enhanced with fare data
                "departure_time": legs[0]['route'].get('departure_time', 'Immediate') if legs else 'Unknown',
                "arrival_time": legs[-1]['route'].get('arrival_time', 'Flexible') if legs else 'Unknown'
            },
            "legs": legs,
            "enhanced": True
        }
        
        if fallback_reason:
            enhanced_route["fallback_reason"] = fallback_reason
            
        return enhanced_route
        
    except Exception as e:
        logger.error(f"Error creating enhanced route object: {e}")
        return None

def create_enhanced_multimodal_route(route_id, route_index, bike_leg_1, transit_route, bike_leg_2, start_stop, end_stop):
    """Create enhanced multimodal route with all details"""
    try:
        # Calculate total time with realistic transfer penalties
        total_time = (
            bike_leg_1['travel_time_minutes'] + 
            transit_route['duration_minutes'] + 
            bike_leg_2['travel_time_minutes'] + 
            5  # Transfer time
        )
        
        # Add transit route stop information
        enhanced_transit_route = transit_route.copy()
        enhanced_transit_route['start_stop'] = start_stop
        enhanced_transit_route['end_stop'] = end_stop
        
        return create_enhanced_route_object(
            route_id=route_id,
            name=f"Bike-Bus-Bike Option {route_index + 1}",
            route_type="bike_bus_bike",
            legs=[
                {
                    "type": "bike",
                    "name": "Bike to Transit",
                    "description": f"Enhanced bike route to {start_stop['name']}",
                    "route": bike_leg_1,
                    "color": "#27ae60",
                    "order": 1
                },
                {
                    "type": "transit",
                    "name": "Transit",
                    "description": f"Enhanced transit from {start_stop['name']} to {end_stop['name']}",
                    "route": enhanced_transit_route,
                    "color": "#3498db",
                    "order": 2
                },
                {
                    "type": "bike",
                    "name": "Transit to Destination",
                    "description": f"Enhanced bike route from {end_stop['name']} to destination",
                    "route": bike_leg_2,
                    "color": "#27ae60",
                    "order": 3
                }
            ]
        )
        
    except Exception as e:
        logger.error(f"Error creating enhanced multimodal route: {e}")
        return None

def calculate_enhanced_statistics(routes):
    """Calculate comprehensive route statistics"""
    try:
        if not routes:
            return {}
            
        # Find best options
        fastest_route = min(routes, key=lambda x: x['summary']['total_time_minutes'])
        shortest_route = min(routes, key=lambda x: x['summary']['total_distance_miles'])
        highest_score_route = max(routes, key=lambda x: x['summary']['average_bike_score'])
        
        # Calculate type breakdowns
        type_counts = {}
        for route in routes:
            route_type = route['type']
            if route_type not in type_counts:
                type_counts[route_type] = 0
            type_counts[route_type] += 1
        
        return {
            "total_options": len(routes),
            "route_types": type_counts,
            "fastest_option": {
                "name": fastest_route['name'],
                "time": fastest_route['summary']['total_time_formatted'],
                "type": fastest_route['type']
            },
            "shortest_option": {
                "name": shortest_route['name'],
                "distance": f"{shortest_route['summary']['total_distance_miles']:.1f} miles",
                "type": shortest_route['type']
            },
            "highest_bike_score": {
                "name": highest_score_route['name'],
                "score": highest_score_route['summary']['average_bike_score'],
                "type": highest_score_route['type']
            },
            "averages": {
                "time_minutes": round(sum(r['summary']['total_time_minutes'] for r in routes) / len(routes), 1),
                "distance_miles": round(sum(r['summary']['total_distance_miles'] for r in routes) / len(routes), 2),
                "bike_score": round(sum(r['summary']['average_bike_score'] for r in routes) / len(routes), 1)
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating enhanced statistics: {e}")
        return {"total_options": len(routes) if routes else 0}

# =============================================================================
# ENHANCED FASTAPI ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def get_enhanced_ui():
    """Serve the enhanced UI with all features"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced OSRM + OTP Route Planner</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <style>
            body { margin: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .header { background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 20px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
            .header h1 { margin: 0; font-size: 2.5em; font-weight: 300; display: flex; align-items: center; justify-content: center; gap: 15px; }
            .tech-badge { background: linear-gradient(45deg, #e74c3c, #f39c12); color: white; padding: 6px 12px; border-radius: 20px; font-size: 0.7em; font-weight: bold; }
            .container { display: flex; height: calc(100vh - 120px); max-width: 1600px; margin: 0 auto; background: white; border-radius: 15px 15px 0 0; overflow: hidden; box-shadow: 0 -8px 30px rgba(0,0,0,0.2); }
            #map { flex: 2.5; }
            .sidebar { flex: 1; max-width: 450px; padding: 25px; background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%); overflow-y: auto; border-left: 1px solid #dee2e6; }
            
            .system-status { padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center; font-weight: 500; font-size: 0.9em; }
            .status-good { background: linear-gradient(135deg, #d4edda, #c3e6cb); color: #155724; border-left: 5px solid #28a745; }
            .status-limited { background: linear-gradient(135deg, #fff3cd, #ffeaa7); color: #856404; border-left: 5px solid #f1c40f; }
            
            .controls { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; font-size: 1em; }
            select, input { width: 100%; padding: 12px 15px; border: 2px solid #e1e5e9; border-radius: 8px; font-size: 1em; transition: all 0.3s ease; background: white; box-sizing: border-box; }
            select:focus, input:focus { outline: none; border-color: #3498db; box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1); }
            
            button { background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 15px 25px; border: none; border-radius: 8px; cursor: pointer; font-size: 1.1em; font-weight: 600; width: 100%; margin-bottom: 10px; transition: all 0.3s ease; }
            button:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(52, 152, 219, 0.3); }
            button:disabled { background: linear-gradient(135deg, #bdc3c7, #95a5a6); cursor: not-allowed; transform: none; box-shadow: none; }
            .btn-clear { background: linear-gradient(135deg, #e74c3c, #c0392b); }
            
            .coordinates { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #6c757d; font-size: 14px; }
            .spinner { border: 4px solid rgba(52, 152, 219, 0.2); width: 40px; height: 40px; border-radius: 50%; border-left-color: #3498db; animation: spin 1s linear infinite; margin: 20px auto; display: none; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            
            .route-card { background: white; border-radius: 12px; padding: 25px; margin-bottom: 20px; box-shadow: 0 6px 20px rgba(0,0,0,0.1); border: 2px solid #e9ecef; cursor: pointer; transition: all 0.3s ease; }
            .route-card:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,0,0,0.15); }
            .route-card.selected { border-color: #3498db; box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1); }
            
            .route-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
            .route-name { font-weight: 700; color: #2c3e50; font-size: 1.3em; }
            .route-type-badge { padding: 6px 12px; border-radius: 15px; font-size: 0.8em; font-weight: 600; }
            .badge-bike-bus-bike { background: linear-gradient(135deg, #e74c3c, #f39c12); color: white; }
            .badge-direct-bike { background: linear-gradient(135deg, #27ae60, #2ecc71); color: white; }
            .badge-transit-fallback { background: linear-gradient(135deg, #2196f3, #1976d2); color: white; }
            
            .route-summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 15px; margin-bottom: 20px; }
            .summary-item { text-align: center; padding: 12px; background: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef; }
            .summary-value { font-weight: bold; color: #3498db; font-size: 1.2em; display: block; }
            .summary-label { font-size: 0.8em; color: #6c757d; text-transform: uppercase; margin-top: 4px; }
            
            .legs-container { margin-top: 20px; }
            .leg-item { margin: 15px 0; padding: 20px; border-radius: 10px; position: relative; overflow: hidden; }
            .leg-bike { background: linear-gradient(135deg, #e8f5e8, #d4edda); border-left: 5px solid #27ae60; }
            .leg-transit { background: linear-gradient(135deg, #e3f2fd, #bbdefb); border-left: 5px solid #2196f3; }
            
            .leg-header { display: flex; align-items: center; margin-bottom: 15px; }
            .leg-icon { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 18px; margin-right: 15px; font-weight: bold; }
            .leg-icon.bike { background: linear-gradient(135deg, #27ae60, #2ecc71); color: white; }
            .leg-icon.transit { background: linear-gradient(135deg, #3498db, #5dade2); color: white; }
            
            .leg-title { font-weight: 600; color: #2c3e50; font-size: 1.1em; }
            .leg-description { color: #6c757d; font-size: 0.9em; margin-top: 2px; }
            
            .leg-metrics { display: flex; gap: 20px; margin: 10px 0; flex-wrap: wrap; }
            .leg-metric { background: rgba(255, 255, 255, 0.8); padding: 8px 12px; border-radius: 15px; font-size: 0.85em; }
            .leg-metric strong { color: #2c3e50; }
            
            .comparison-summary { background: linear-gradient(135deg, #fff, #f8f9fa); border: 2px solid #e9ecef; border-radius: 12px; padding: 20px; margin-bottom: 25px; }
            .comparison-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }
            .comparison-item { text-align: center; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            .comparison-label { font-size: 0.8em; color: #6c757d; text-transform: uppercase; margin-bottom: 8px; }
            .comparison-value { font-weight: bold; color: #2c3e50; font-size: 1.1em; }
            .comparison-type { font-size: 0.75em; color: #3498db; margin-top: 4px; }
            
            .fallback-notice { background: linear-gradient(135deg, #e3f2fd, #bbdefb); color: #1565c0; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #2196f3; font-size: 0.9em; }
            .otp-server-info { background: linear-gradient(135deg, #e8f5e8, #d4edda); color: #155724; padding: 10px; border-radius: 6px; margin: 10px 0; font-size: 12px; }
            
            .error { background: linear-gradient(135deg, #ffebee, #ffcdd2); color: #c62828; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #f44336; }
            .hidden { display: none; }
            
            @media (max-width: 1200px) {
                .container { flex-direction: column; height: auto; min-height: calc(100vh - 120px); }
                #map { height: 50vh; min-height: 400px; }
                .sidebar { max-width: none; flex: none; }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>
                🚴‍♂️🚌🚴‍♀️ Enhanced Bike-Bus-Bike Planner
                <span class="tech-badge">OSRM + OTP</span>
            </h1>
            <p>Open-source multimodal transportation planning with OSRM bicycle routing + OpenTripPlanner transit</p>
        </div>
        
        <div class="container">
            <div id="map"></div>
            
            <div class="sidebar">
                <div class="system-status" id="status">
                    System: Loading...
                </div>
                
                <div class="controls">
                    <div class="form-group">
                        <label for="departureTime">🕐 Departure Time:</label>
                        <select id="departureTime">
                            <option value="now">Leave Now</option>
                            <option value="custom">Custom Time</option>
                        </select>
                    </div>
                    
                    <div class="form-group hidden" id="customTimeGroup">
                        <label for="customTime">Select Time:</label>
                        <input type="datetime-local" id="customTime">
                    </div>
                    
                    <button id="findRoutesBtn" disabled>🔍 Find Enhanced Routes</button>
                    <button class="btn-clear" onclick="clearAll()">🗑️ Clear Map</button>
                    
                    <div class="coordinates">
                        <div><strong>Start:</strong> <span id="startCoords">Click map to select</span></div>
                        <div><strong>End:</strong> <span id="endCoords">Click map to select</span></div>
                    </div>
                </div>
                
                <div class="spinner" id="spinner"></div>
                <div id="results"></div>
            </div>
        </div>

        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
            let map, startPoint = null, endPoint = null, startMarker = null, endMarker = null;
            let routeLayersGroup, currentRoutes = [], clickCount = 0;
            let systemInfo = {};
            
            // Facility colors matching backend
            const facilityColors = {
                'BUFFERED BIKELANE': '#85c766', 'PROTECTED BIKELANE': '#27ae60', 
                'UNBUFFERED BIKELANE': '#3498db', 'SHARED LANE': '#f1c40f',
                'SHARED USE PATH': '#9b59b6', 'NO BIKELANE': '#e74c3c'
            };
            
            function initMap() {
                map = L.map('map').setView([30.3322, -81.6557], 12);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors'
                }).addTo(map);
                routeLayersGroup = L.layerGroup().addTo(map);
                map.on('click', handleMapClick);
                loadSystemStatus();
            }
            
            async function loadSystemStatus() {
                try {
                    const response = await fetch('/api/health');
                    systemInfo = await response.json();
                    const statusDiv = document.getElementById('status');
                    
                    if (systemInfo.otp_working) {
                        statusDiv.className = 'system-status status-good';
                        statusDiv.innerHTML = `
                            <div>🟢 System: OSRM + OTP Connected</div>
                            <div style="font-size: 0.8em; margin-top: 5px; opacity: 0.8;">
                                OSRM: ✅ | OTP: ✅ (${systemInfo.otp_server ? 'Connected' : 'Testing...'})
                            </div>
                        `;
                    } else {
                        statusDiv.className = 'system-status status-limited';
                        statusDiv.innerHTML = `
                            <div>🟡 System: OSRM Only (Limited Transit)</div>
                            <div style="font-size: 0.8em; margin-top: 5px; opacity: 0.8;">
                                OSRM: ✅ | OTP: ❌ (No working server found)
                            </div>
                        `;
                    }
                } catch (error) {
                    const statusDiv = document.getElementById('status');
                    statusDiv.className = 'system-status status-limited';
                    statusDiv.innerHTML = '⚠️ System: Status unknown - Connection issues';
                }
            }
            
            function handleMapClick(e) {
                const lat = e.latlng.lat, lng = e.latlng.lng;
                
                if (clickCount === 0) {
                    if (startMarker) map.removeLayer(startMarker);
                    startMarker = L.marker([lat, lng], {
                        icon: L.icon({
                            iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
                            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                            iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41]
                        })
                    }).addTo(map);
                    startMarker.bindTooltip("Start Point", {permanent: true, direction: 'top'}).openTooltip();
                    
                    startPoint = [lng, lat];
                    document.getElementById('startCoords').textContent = lat.toFixed(5) + ', ' + lng.toFixed(5);
                    clickCount = 1;
                } else if (clickCount === 1) {
                    if (endMarker) map.removeLayer(endMarker);
                    endMarker = L.marker([lat, lng], {
                        icon: L.icon({
                            iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
                            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                            iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41]
                        })
                    }).addTo(map);
                    endMarker.bindTooltip("End Point", {permanent: true, direction: 'top'}).openTooltip();
                    
                    endPoint = [lng, lat];
                    document.getElementById('endCoords').textContent = lat.toFixed(5) + ', ' + lng.toFixed(5);
                    document.getElementById('findRoutesBtn').disabled = false;
                    clickCount = 2;
                } else {
                    clearAll();
                    handleMapClick(e);
                }
            }
            
            function clearAll() {
                if (startMarker) { map.removeLayer(startMarker); startMarker = null; }
                if (endMarker) { map.removeLayer(endMarker); endMarker = null; }
                routeLayersGroup.clearLayers();
                startPoint = null; endPoint = null; clickCount = 0;
                document.getElementById('startCoords').textContent = 'Click map to select';
                document.getElementById('endCoords').textContent = 'Click map to select';
                document.getElementById('findRoutesBtn').disabled = true;
                document.getElementById('results').innerHTML = '';
                currentRoutes = [];
            }
            
            function showSpinner(show) {
                document.getElementById('spinner').style.display = show ? 'block' : 'none';
                document.getElementById('findRoutesBtn').disabled = show;
                document.getElementById('findRoutesBtn').innerHTML = show ? '⏳ Analyzing Enhanced Routes...' : '🔍 Find Enhanced Routes';
            }
            
            async function findRoutes() {
                if (!startPoint || !endPoint) return;
                
                showSpinner(true);
                routeLayersGroup.clearLayers();
                
                try {
                    const departureTime = document.getElementById('departureTime').value === 'custom' 
                        ? Math.floor(new Date(document.getElementById('customTime').value).getTime() / 1000)
                        : 'now';
                    
                    const params = new URLSearchParams({
                        start_lon: startPoint[0], start_lat: startPoint[1],
                        end_lon: endPoint[0], end_lat: endPoint[1],
                        departure_time: departureTime
                    });
                    
                    const response = await fetch('/api/analyze?' + params);
                    const data = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(data.detail || 'Analysis failed');
                    }
                    
                    currentRoutes = data.routes;
                    displayEnhancedResults(data);
                    
                } catch (error) {
                    document.getElementById('results').innerHTML = 
                        '<div class="error"><h4>⚠️ Analysis Error</h4><p>' + error.message + '</p></div>';
                } finally {
                    showSpinner(false);
                }
            }
            
            function displayEnhancedResults(data) {
                let html = '';
                
                // Show fallback notice if used
                if (data.fallback_used) {
                    html += `<div class="fallback-notice">
                        <strong>🔄 Smart Fallback Activated</strong><br>
                        ${data.fallback_reason || 'Using optimized routing strategy'}
                    </div>`;
                }
                
                // Show OTP server info if available
                if (data.otp_server) {
                    html += `<div class="otp-server-info">
                        <strong>🌐 OTP Server:</strong> ${data.otp_server.replace('http://', '').replace('https://', '')}
                    </div>`;
                }
                
                // Create comparison summary
                html += '<div class="comparison-summary">';
                html += '<h3>🛣️ Enhanced Route Analysis Results</h3>';
                html += `<div class="comparison-grid">`;
                html += `<div class="comparison-item">`;
                html += `<div class="comparison-label">Total Options</div>`;
                html += `<div class="comparison-value">${data.routes.length}</div>`;
                html += `</div>`;
                
                // Find fastest and shortest routes
                const fastestRoute = data.routes.reduce((prev, curr) => 
                    prev.summary.total_time_minutes < curr.summary.total_time_minutes ? prev : curr
                );
                const shortestRoute = data.routes.reduce((prev, curr) => 
                    prev.summary.total_distance_miles < curr.summary.total_distance_miles ? prev : curr
                );
                
                html += `<div class="comparison-item">`;
                html += `<div class="comparison-label">Fastest Route</div>`;
                html += `<div class="comparison-value">${fastestRoute.summary.total_time_formatted}</div>`;
                html += `<div class="comparison-type">${fastestRoute.type === 'bike_bus_bike' ? 'Multimodal' : fastestRoute.type === 'direct_bike' ? 'Direct Bike' : 'Transit'}</div>`;
                html += `</div>`;
                
                html += `<div class="comparison-item">`;
                html += `<div class="comparison-label">Shortest Route</div>`;
                html += `<div class="comparison-value">${shortestRoute.summary.total_distance_miles.toFixed(1)} mi</div>`;
                html += `<div class="comparison-type">${shortestRoute.type === 'bike_bus_bike' ? 'Multimodal' : shortestRoute.type === 'direct_bike' ? 'Direct Bike' : 'Transit'}</div>`;
                html += `</div>`;
                
                html += `</div></div>`;
                
                // Display each route with enhanced cards
                data.routes.forEach((route, index) => {
                    html += createEnhancedRouteCard(route, index);
                });
                
                document.getElementById('results').innerHTML = html;
                if (currentRoutes.length > 0) selectRoute(0);
            }
            
            function createEnhancedRouteCard(route, index) {
                let routeClass, routeIcon;
                
                if (route.type === 'bike_bus_bike') {
                    routeClass = 'badge-bike-bus-bike';
                    routeIcon = '🚴‍♂️🚌🚴‍♀️';
                } else if (route.type === 'transit_fallback') {
                    routeClass = 'badge-transit-fallback';
                    routeIcon = '🚶‍♂️🚌🚶‍♀️';
                } else {
                    routeClass = 'badge-direct-bike';
                    routeIcon = '🚴‍♂️🚴‍♀️';
                }
                
                let html = `
                    <div class="route-card" onclick="selectRoute(${index})" id="route${index}">
                        <div class="route-header">
                            <div class="route-name">${routeIcon} ${route.name}</div>
                            <span class="route-type-badge ${routeClass}">
                                ${route.type === 'bike_bus_bike' ? 'MULTIMODAL' : 
                                  route.type === 'transit_fallback' ? 'TRANSIT' : 'DIRECT BIKE'}
                            </span>
                        </div>
                        
                        <!-- Enhanced Route Summary Metrics -->
                        <div class="route-summary">
                            <div class="summary-item">
                                <span class="summary-value">${route.summary.total_time_formatted}</span>
                                <span class="summary-label">Total Time</span>
                            </div>
                            <div class="summary-item">
                                <span class="summary-value">${route.summary.total_distance_miles.toFixed(1)} mi</span>
                                <span class="summary-label">Distance</span>
                            </div>
                            <div class="summary-item">
                                <span class="summary-value">${route.summary.bike_distance_miles.toFixed(1)} mi</span>
                                <span class="summary-label">Bike Distance</span>
                            </div>
                            <div class="summary-item">
                                <span class="summary-value">${route.summary.transfers || 0}</span>
                                <span class="summary-label">Transfers</span>
                            </div>
                            <div class="summary-item">
                                <span class="summary-value">${Math.round(route.summary.average_bike_score || 0)}</span>
                                <span class="summary-label">Bike Score</span>
                            </div>
                        </div>
                `;
                
                // Add enhanced legs breakdown
                html += `<div class="legs-container">`;
                html += `<h4 style="margin: 15px 0 10px 0; color: #2c3e50;">🔍 Enhanced Route Breakdown</h4>`;
                
                route.legs.forEach((leg, legIndex) => {
                    html += createEnhancedLegCard(leg, legIndex);
                });
                
                html += `</div>`;
                
                // Add fallback reason if applicable
                if (route.fallback_reason) {
                    html += `<div style="background: #e3f2fd; padding: 10px; border-radius: 6px; margin-top: 10px; font-size: 0.9em;">
                        <strong>💡 Smart Routing:</strong> ${route.fallback_reason}
                    </div>`;
                }
                
                html += `
                        <div style="margin-top: 15px; padding: 10px; background: #e8f4fd; border-radius: 6px; text-align: center;">
                            <strong>💡 Click to view this route on the map</strong>
                        </div>
                    </div>
                `;
                
                return html;
            }
            
            function createEnhancedLegCard(leg, legIndex) {
                const legType = leg.type;
                const legIcon = legType === 'bike' ? '🚴‍♂️' : '🚌';
                const legClass = legType === 'bike' ? 'leg-bike' : 'leg-transit';
                const legIconClass = legType === 'bike' ? 'bike' : 'transit';
                
                let html = `
                    <div class="leg-item ${legClass}">
                        <div class="leg-header">
                            <div class="leg-icon ${legIconClass}">${legIcon}</div>
                            <div>
                                <div class="leg-title">${leg.name}</div>
                                <div class="leg-description">${leg.description}</div>
                            </div>
                        </div>
                        
                        <div class="leg-metrics">
                            <div class="leg-metric">
                                <strong>Distance:</strong> ${(leg.route.length_miles || leg.route.distance_miles || 0).toFixed(2)} miles
                            </div>
                            <div class="leg-metric">
                                <strong>Time:</strong> ${leg.route.travel_time_formatted || leg.route.duration_text || 'N/A'}
                            </div>
                `;
                
                // Add enhanced type-specific metrics
                if (legType === 'bike') {
                    html += `
                            <div class="leg-metric">
                                <strong>OSRM Score:</strong> ${leg.route.overall_score || 70}/100
                            </div>
                            <div class="leg-metric">
                                <strong>Source:</strong> OSRM Enhanced
                            </div>
                    `;
                } else if (legType === 'transit') {
                    html += `
                            <div class="leg-metric">
                                <strong>Lines:</strong> ${leg.route.transit_lines ? leg.route.transit_lines.join(', ') : 'Transit'}
                            </div>
                            <div class="leg-metric">
                                <strong>Source:</strong> ${systemInfo.otp_working ? 'OpenTripPlanner' : 'Limited'}
                            </div>
                    `;
                }
                
                html += `    </div>
                    </div>
                `;
                return html;
            }
            
            function selectRoute(index) {
                document.querySelectorAll('.route-card').forEach(card => card.classList.remove('selected'));
                document.getElementById('route' + index).classList.add('selected');
                
                routeLayersGroup.clearLayers();
                const route = currentRoutes[index];
                
                // Enhanced visualization with different colors for different leg types
                const legColors = { 'bike': '#27ae60', 'transit': '#3498db' };
                let allCoords = [];
                
                route.legs.forEach((leg, legIndex) => {
                    if (leg.route.geometry && leg.route.geometry.coordinates && leg.route.geometry.coordinates.length > 0) {
                        const coords = leg.route.geometry.coordinates.map(coord => [coord[1], coord[0]]);
                        const color = legColors[leg.type] || '#95a5a6';
                        
                        if (leg.type === 'bike' && leg.route.segments && leg.route.segments.length > 0) {
                            // Draw colored segments for bike routes based on facility type
                            leg.route.segments.forEach((segment, segmentIndex) => {
                                if (segment.geometry && segment.geometry.coordinates && segment.geometry.coordinates.length > 0) {
                                    const facilityColor = facilityColors[segment.facility_type] || '#95a5a6';
                                    
                                    const segmentLatLngs = segment.geometry.coordinates.map(coord => 
                                        L.latLng(coord[1], coord[0])
                                    );
                                    
                                    const segmentLine = L.polyline(segmentLatLngs, {
                                        color: facilityColor,
                                        weight: 8,
                                        opacity: 0.8
                                    }).addTo(routeLayersGroup);
                                    
                                    segmentLine.bindPopup(`
                                        <div style="font-family: 'Segoe UI', sans-serif;">
                                            <h4 style="margin: 0 0 10px 0; color: ${facilityColor};">
                                                ${formatFacilityType(segment.facility_type)}
                                            </h4>
                                            <p style="margin: 5px 0;"><strong>Length:</strong> ${segment.shape_length_miles.toFixed(3)} miles</p>
                                            <p style="margin: 5px 0;"><strong>Score:</strong> ${segment.bike_score || 0}/100</p>
                                            <p style="margin: 5px 0;"><strong>Road:</strong> ${segment.road_name || 'Unknown'}</p>
                                        </div>
                                    `);
                                }
                            });
                        } else {
                            // Draw simple line for transit routes or bike routes without segments
                            const routeLine = L.polyline(coords, {
                                color: color,
                                weight: leg.type === 'bike' ? 6 : 8,
                                opacity: 0.8,
                                dashArray: leg.type === 'transit' ? '10, 5' : null
                            }).addTo(routeLayersGroup);
                            
                            // Enhanced popup with more information
                            routeLine.bindPopup(`
                                <div style="font-family: 'Segoe UI', sans-serif;">
                                    <h4 style="margin: 0 0 10px 0; color: ${color};">
                                        ${leg.type === 'bike' ? '🚴‍♂️' : '🚌'} ${leg.name}
                                    </h4>
                                    <p style="margin: 5px 0;"><strong>Distance:</strong> ${(leg.route.length_miles || leg.route.distance_miles || 0).toFixed(2)} miles</p>
                                    <p style="margin: 5px 0;"><strong>Time:</strong> ${leg.route.travel_time_formatted || leg.route.duration_text || 'N/A'}</p>
                                    ${leg.type === 'bike' ? 
                                        `<p style="margin: 5px 0;"><strong>Bike Score:</strong> ${leg.route.overall_score || 70}</p>
                                         <p style="margin: 5px 0;"><strong>Source:</strong> OSRM Cycling Network</p>` : ''}
                                    ${leg.type === 'transit' && leg.route.transit_lines ? 
                                        `<p style="margin: 5px 0;"><strong>Lines:</strong> ${leg.route.transit_lines.join(', ')}</p>
                                         <p style="margin: 5px 0;"><strong>Source:</strong> OpenTripPlanner</p>` : ''}
                                </div>
                            `);
                        }
                        
                        allCoords = allCoords.concat(coords);
                        
                        // Add bus stop markers for transit legs
                        if (leg.type === 'transit') {
                            addBusStopsForTransitLeg(leg, legIndex);
                        }
                    }
                });
                
                // Fit map to show the entire route
                try {
                    if (allCoords.length > 0) {
                        const bounds = L.latLngBounds(allCoords);
                        map.fitBounds(bounds, { padding: [20, 20] });
                    }
                } catch (e) {
                    console.warn('Could not fit map bounds:', e);
                }
            }
            
            function addBusStopsForTransitLeg(leg, legIndex) {
                // Add bus stop markers based on route step information
                if (leg.route.steps) {
                    leg.route.steps.forEach(step => {
                        if (step.travel_mode === 'TRANSIT') {
                            // Add departure stop
                            if (step.departure_stop_location && step.departure_stop_location.lat) {
                                const stopIcon = L.divIcon({
                                    html: '<div style="width: 20px; height: 20px; background: #3498db; border: 3px solid white; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 10px; font-weight: bold;">🚌</div>',
                                    className: 'bus-stop-icon',
                                    iconSize: [26, 26],
                                    iconAnchor: [13, 13]
                                });
                                
                                L.marker([step.departure_stop_location.lat, step.departure_stop_location.lng], {
                                    icon: stopIcon
                                }).addTo(routeLayersGroup)
                                  .bindPopup(`
                                    <div style="font-family: 'Segoe UI', sans-serif;">
                                        <h5 style="margin: 0 0 8px 0; color: #3498db;">🚏 ${step.departure_stop_name}</h5>
                                        <p style="margin: 2px 0;"><strong>Departure:</strong> ${step.scheduled_departure}</p>
                                        <p style="margin: 2px 0;"><strong>Line:</strong> ${step.transit_line}</p>
                                        <p style="margin: 2px 0;"><strong>Agency:</strong> ${step.transit_agency}</p>
                                    </div>
                                  `);
                            }
                            
                            // Add arrival stop
                            if (step.arrival_stop_location && step.arrival_stop_location.lat) {
                                const stopIcon = L.divIcon({
                                    html: '<div style="width: 20px; height: 20px; background: #e74c3c; border: 3px solid white; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 10px; font-weight: bold;">🚌</div>',
                                    className: 'bus-stop-icon',
                                    iconSize: [26, 26],
                                    iconAnchor: [13, 13]
                                });
                                
                                L.marker([step.arrival_stop_location.lat, step.arrival_stop_location.lng], {
                                    icon: stopIcon
                                }).addTo(routeLayersGroup)
                                  .bindPopup(`
                                    <div style="font-family: 'Segoe UI', sans-serif;">
                                        <h5 style="margin: 0 0 8px 0; color: #e74c3c;">🚏 ${step.arrival_stop_name}</h5>
                                        <p style="margin: 2px 0;"><strong>Arrival:</strong> ${step.scheduled_arrival}</p>
                                        <p style="margin: 2px 0;"><strong>Line:</strong> ${step.transit_line}</p>
                                        <p style="margin: 2px 0;"><strong>Agency:</strong> ${step.transit_agency}</p>
                                    </div>
                                  `);
                            }
                        }
                    });
                }
            }
            
            function formatFacilityType(facilityType) {
                return facilityType.toLowerCase()
                    .split(' ')
                    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                    .join(' ')
                    .replace('Bikelane', 'Bike Lane')
                    .replace('Bikeway', 'Bikeway');
            }
            
            // Setup event listeners
            document.getElementById('departureTime').addEventListener('change', function() {
                const customTimeGroup = document.getElementById('customTimeGroup');
                if (this.value === 'custom') {
                    customTimeGroup.classList.remove('hidden');
                    const now = new Date();
                    now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
                    document.getElementById('customTime').value = now.toISOString().slice(0, 16);
                } else {
                    customTimeGroup.classList.add('hidden');
                }
            });
            
            document.getElementById('findRoutesBtn').addEventListener('click', findRoutes);
            
            // Initialize the application
            initMap();
        </script>
    </body>
    </html>
    '''

@app.get("/api/health")
async def enhanced_health_check():
    """Enhanced health check endpoint"""
    global WORKING_OTP_SERVER, SYSTEM_STATUS
    
    if not WORKING_OTP_SERVER:
        find_working_otp_server_enhanced()
    
    return {
        "status": "healthy",
        "service": "Enhanced OSRM + OTP Route Planner",
        "version": "2.0.0",
        "osrm_server": OSRM_SERVER,
        "otp_server": WORKING_OTP_SERVER,
        "otp_router": WORKING_OTP_ROUTER,
        "otp_working": bool(WORKING_OTP_SERVER),
        "system_status": SYSTEM_STATUS,
        "features": [
            "Enhanced bike routing with OSRM",
            "Advanced transit planning with OTP", 
            "Multimodal bike-bus-bike analysis",
            "Real-time transit information",
            "Enhanced route visualization",
            "Comprehensive safety scoring",
            "Smart fallback routing",
            "Facility type classification"
        ],
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/api/analyze")
async def analyze_enhanced_routes(
    start_lon: float = Query(..., description="Start longitude"),
    start_lat: float = Query(..., description="Start latitude"),
    end_lon: float = Query(..., description="End longitude"),
    end_lat: float = Query(..., description="End latitude"),
    departure_time: str = Query("now", description="Departure time")
):
    """Enhanced route analysis endpoint"""
    
    # Validate coordinates
    if not (-180 <= start_lon <= 180 and -90 <= start_lat <= 90):
        raise HTTPException(status_code=400, detail="Invalid start coordinates")
    if not (-180 <= end_lon <= 180 and -90 <= end_lat <= 90):
        raise HTTPException(status_code=400, detail="Invalid end coordinates")
    
    try:
        logger.info(f"Enhanced analysis request: ({start_lat},{start_lon}) to ({end_lat},{end_lon})")
        
        result = analyze_bike_bus_bike_routes_enhanced(
            [start_lon, start_lat],
            [end_lon, end_lat],
            departure_time
        )
        
        logger.info(f"Enhanced analysis completed: {len(result.get('routes', []))} routes found")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def enhanced_startup_event():
    """Enhanced initialization on startup"""
    logger.info("Starting Enhanced OSRM + OTP API...")
    
    # Test OTP servers on startup
    working_server, working_router = find_working_otp_server_enhanced()
    
    if working_server:
        logger.info(f"✅ Found working OTP server: {working_server}")
        logger.info(f"✅ Using router: {working_router}")
    else:
        logger.warning("⚠️  No working OTP servers found - transit features limited")
    
    logger.info("🚀 Enhanced API ready with full feature set")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting Enhanced OSRM + OTP server on port {port}")
    logger.info(f"OSRM Server: {OSRM_SERVER}")
    logger.info(f"Features: Enhanced routing, safety scoring, real-time transit")
    
    uvicorn.run(
        "complete_enhanced_osrm_otp_planner:app",  # Update this to match your filename
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
