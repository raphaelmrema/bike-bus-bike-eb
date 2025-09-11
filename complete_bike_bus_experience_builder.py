# complete_bike_bus_experience_builder_otp.py
# Complete Bike-Bus-Bike Route Planner with OSRM + OpenTripPlanner for ArcGIS Experience Builder
# Uses OSRM for bicycle routing and OpenTripPlanner for transit routing

import os
import json
import logging
import tempfile
import requests
import datetime
import zipfile
import pandas as pd
import io
import math
import re
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Try importing polyline for OSRM geometry decoding
try:
    import polyline
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'polyline'])
    import polyline

# =============================================================================
# CONFIGURATION FOR EXPERIENCE BUILDER DEPLOYMENT WITH OSRM + OTP
# =============================================================================

# Environment variables for cloud deployment
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", 
    "https://experience.arcgis.com,https://*.maps.arcgis.com,http://localhost:*,https://*.railway.app").split(",")

# OSRM Configuration for bicycle routing
OSRM_SERVER = os.getenv("OSRM_SERVER", "http://router.project-osrm.org")
USE_OSRM_DURATION = True

# OpenTripPlanner Configuration for transit routing
OTP_SERVER = os.getenv("OTP_SERVER", "http://otp.prod.obanyc.com/otp")  # NYC OTP instance as example
OTP_ROUTER_ID = os.getenv("OTP_ROUTER_ID", "default")  # Router ID for your region

# Jacksonville specific OTP server (if available)
# OTP_SERVER = "http://your-jta-otp-server.com/otp"  # Replace with actual JTA OTP server
# OTP_ROUTER_ID = "jta"

# Bicycle Configuration
BIKE_SPEED_MPH = 11
BIKE_SPEED_FEET_PER_SECOND = BIKE_SPEED_MPH * 5280 / 3600

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# FASTAPI APP SETUP WITH EXPERIENCE BUILDER CORS
# =============================================================================

app = FastAPI(
    title="OSRM + OpenTripPlanner Bike-Bus-Bike Route Planner API",
    description="Advanced multimodal transportation planning API with OSRM bicycle routing and OpenTripPlanner transit routing for ArcGIS Experience Builder",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware configured for Experience Builder
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# ENHANCED GTFS MANAGER (SIMPLIFIED - OTP HANDLES MOST GTFS OPERATIONS)
# =============================================================================

class SimpleGTFSManager:
    """Simplified GTFS manager since OTP handles most GTFS operations"""
    
    def __init__(self):
        self.is_loaded = False
        self.last_update = None
        
    def load_gtfs_data(self):
        """Simple check - OTP server handles GTFS data"""
        try:
            # Test OTP server connection
            response = requests.get(f"{OTP_SERVER}/routers", timeout=10)
            if response.status_code == 200:
                self.is_loaded = True
                self.last_update = datetime.datetime.now()
                logger.info("✅ OTP server accessible - GTFS data available through OTP")
                return True
            else:
                logger.warning(f"⚠️ OTP server returned status {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"⚠️ Could not connect to OTP server: {e}")
            return False
    
    def find_nearby_stops(self, lat, lon, radius_meters=500):
        """Find nearby stops using OTP index API"""
        try:
            url = f"{OTP_SERVER}/routers/{OTP_ROUTER_ID}/index/stops"
            params = {
                'lat': lat,
                'lon': lon,
                'radius': radius_meters
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                stops_data = response.json()
                
                formatted_stops = []
                for stop in stops_data[:5]:  # Limit to 5 nearest
                    formatted_stops.append({
                        'stop_id': stop.get('id', ''),
                        'stop_name': stop.get('name', 'Unknown Stop'),
                        'stop_lat': stop.get('lat', lat),
                        'stop_lon': stop.get('lon', lon),
                        'distance_km': 0  # OTP doesn't return distance, but we could calculate
                    })
                
                return formatted_stops
            else:
                logger.warning(f"OTP stops query failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error finding stops via OTP: {e}")
            return []

# Global GTFS manager
gtfs_manager = SimpleGTFSManager()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_bike_time_minutes(distance_feet):
    """Calculate bicycle travel time in minutes given distance in feet"""
    if distance_feet <= 0:
        return 0
    return (distance_feet / BIKE_SPEED_FEET_PER_SECOND) / 60

def format_time_duration(minutes):
    """Format time duration in a user-friendly way"""
    if minutes < 1:
        return "< 1 min"
    elif minutes < 60:
        return f"{int(minutes)} min"
    else:
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        if mins == 0:
            return f"{hours}h"
        else:
            return f"{hours}h {mins}m"

def parse_otp_time(time_seconds):
    """Parse OTP time from seconds since epoch to readable format"""
    try:
        dt = datetime.datetime.fromtimestamp(time_seconds / 1000)  # OTP uses milliseconds
        return dt.strftime("%H:%M")
    except:
        return "Unknown"

# =============================================================================
# OSRM BICYCLE ROUTING FUNCTIONS (SAME AS BEFORE)
# =============================================================================

def calculate_bike_route_osrm(start_coords, end_coords, waypoints=None, route_name="Bike Route"):
    """Create a bike route using OSRM"""
    try:
        logger.info(f"🚴‍♂️ Creating OSRM bike route: {route_name} from {start_coords} to {end_coords}")
        
        # Build coordinates string for OSRM: "lon,lat;lon,lat;..."
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
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("code") != "Ok" or not data.get("routes"):
            logger.warning(f"No OSRM route found for {route_name}")
            return None
        
        route = data["routes"][0]
        
        # Extract geometry
        geometry_polyline = route["geometry"]
        coords_latlon = polyline.decode(geometry_polyline)  # [(lat, lon), ...]
        route_geometry = [[lon, lat] for (lat, lon) in coords_latlon]  # Convert to [lon, lat] for GeoJSON
        
        # Extract distance and duration
        distance_meters = float(route.get("distance", 0.0))
        distance_miles = distance_meters * 0.000621371
        distance_feet = distance_meters * 3.28084
        
        # Use OSRM's duration if available, otherwise fallback to speed calculation
        osrm_duration_sec = route.get("duration", None)
        if USE_OSRM_DURATION and isinstance(osrm_duration_sec, (int, float)) and osrm_duration_sec > 0:
            duration_minutes = float(osrm_duration_sec) / 60.0
            osrm_time_used = True
        else:
            duration_minutes = (distance_miles / BIKE_SPEED_MPH) * 60.0
            osrm_time_used = False
        
        # Simulate bike safety analysis (since we don't have ArcGIS)
        overall_score = simulate_bike_safety_score(distance_miles, route_geometry)
        facility_stats = simulate_facility_stats(distance_miles)
        
        return {
            "name": route_name,
            "length_feet": distance_feet,
            "length_miles": distance_miles,
            "travel_time_minutes": duration_minutes,
            "travel_time_formatted": format_time_duration(duration_minutes),
            "osrm_time_used": osrm_time_used,
            "geometry": {
                "type": "LineString",
                "coordinates": route_geometry
            },
            "segments": [],  # Simplified - no detailed segment analysis
            "overall_score": overall_score,
            "facility_stats": facility_stats,
            "waypoints": waypoints or []
        }
        
    except Exception as e:
        logger.error(f"Error calculating OSRM bike route: {e}")
        return None

def simulate_bike_safety_score(distance_miles, route_geometry):
    """Simulate bike safety score (replaces ArcGIS LTS analysis)"""
    base_score = 65
    
    if distance_miles < 1:
        base_score += 10
    elif distance_miles > 5:
        base_score -= 10
    
    if route_geometry and len(route_geometry) > 20:
        base_score += 5
    
    import random
    variation = random.randint(-15, 15)
    final_score = max(0, min(100, base_score + variation))
    
    return round(final_score, 1)

def simulate_facility_stats(distance_miles):
    """Simulate bicycle facility statistics"""
    import random
    
    facility_types = [
        "PROTECTED BIKELANE",
        "BUFFERED BIKELANE", 
        "UNBUFFERED BIKELANE",
        "SHARED LANE",
        "NO BIKELANE"
    ]
    
    total_feet = distance_miles * 5280
    stats = {}
    
    remaining_distance = total_feet
    for i, facility_type in enumerate(facility_types):
        if i == len(facility_types) - 1:
            facility_feet = remaining_distance
        else:
            if facility_type == "NO BIKELANE":
                percentage = random.uniform(0.3, 0.6)
            elif facility_type == "PROTECTED BIKELANE":
                percentage = random.uniform(0.05, 0.2)
            else:
                percentage = random.uniform(0.1, 0.3)
            
            facility_feet = min(remaining_distance * percentage, remaining_distance)
        
        remaining_distance -= facility_feet
        
        if facility_feet > 0:
            stats[facility_type] = {
                'length_feet': facility_feet,
                'length_miles': facility_feet / 5280,
                'count': 1,
                'avg_score': random.randint(20, 90),
                'percentage': (facility_feet / total_feet) * 100
            }
        
        if remaining_distance <= 0:
            break
    
    return stats

# =============================================================================
# OPENTRIPPLANNER TRANSIT ROUTING FUNCTIONS (REPLACES GOOGLE MAPS)
# =============================================================================

def get_transit_routes_otp(origin_coords: Tuple[float, float], 
                          destination_coords: Tuple[float, float], 
                          departure_time: str = "now") -> Dict:
    """Get transit routes using OpenTripPlanner instead of Google Maps"""
    try:
        logger.info(f"🚌 Getting OTP transit routes: {origin_coords} → {destination_coords}")
        
        # Prepare time parameter
        if departure_time == "now":
            time_str = datetime.datetime.now().strftime("%I:%M%p")
            date_str = datetime.datetime.now().strftime("%m-%d-%Y")
        else:
            try:
                if isinstance(departure_time, (int, float)):
                    dt = datetime.datetime.fromtimestamp(int(departure_time))
                else:
                    dt = datetime.datetime.fromisoformat(departure_time.replace('Z', '+00:00'))
                time_str = dt.strftime("%I:%M%p")
                date_str = dt.strftime("%m-%d-%Y")
            except:
                time_str = datetime.datetime.now().strftime("%I:%M%p")
                date_str = datetime.datetime.now().strftime("%m-%d-%Y")
        
        # OTP plan endpoint
        url = f"{OTP_SERVER}/routers/{OTP_ROUTER_ID}/plan"
        
        params = {
            'fromPlace': f"{origin_coords[1]},{origin_coords[0]}",  # lat,lon format
            'toPlace': f"{destination_coords[1]},{destination_coords[0]}",
            'time': time_str,
            'date': date_str,
            'mode': 'TRANSIT,WALK',  # Transit + walking
            'maxWalkDistance': 1000,  # meters
            'arriveBy': 'false',  # depart at time
            'numItineraries': 3,  # number of alternative routes
            'optimize': 'TRANSFERS',  # can be QUICK, SAFE, FLAT, GREENWAYS, TRANSFERS
            'walkSpeed': 1.34,  # m/s (3 mph walking speed)
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for errors
        if 'error' in data:
            error_msg = data['error'].get('msg', 'Unknown OTP error')
            logger.error(f"OTP API error: {error_msg}")
            return {"error": error_msg}
        
        # Check for plan
        if 'plan' not in data or 'itineraries' not in data['plan']:
            logger.warning("No transit itineraries found")
            return {"error": "No transit routes found between these locations"}
        
        routes = []
        itineraries = data['plan']['itineraries']
        
        for idx, itinerary in enumerate(itineraries):
            route = parse_otp_itinerary(itinerary, idx)
            if route:
                routes.append(route)
        
        if not routes:
            return {"error": "No valid transit routes could be parsed"}
        
        return {
            "routes": routes,
            "service": "OpenTripPlanner",
            "total_routes": len(routes),
            "otp_server": OTP_SERVER,
            "router_id": OTP_ROUTER_ID,
            "last_update": datetime.datetime.now().isoformat()
        }
        
    except requests.RequestException as e:
        logger.error(f"OTP API error: {e}")
        raise HTTPException(status_code=500, detail=f"Transit API error: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting OTP transit routes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def parse_otp_itinerary(itinerary: Dict, route_index: int) -> Optional[Dict]:
    """Parse a single OTP itinerary into our standard format"""
    try:
        # Basic route info
        duration_seconds = itinerary.get('duration', 0)
        duration_minutes = round(duration_seconds / 60, 1)
        duration_text = format_time_duration(duration_minutes)
        
        # Calculate distance from legs
        total_distance = 0
        walking_distance = 0
        
        # Parse legs
        legs = itinerary.get('legs', [])
        steps = []
        transit_lines = []
        total_fare = 0
        transfers = 0
        
        route_geometry = []
        
        for leg_idx, leg in enumerate(legs):
            step = parse_otp_leg(leg, leg_idx)
            if step:
                steps.append(step)
                
                # Accumulate distance
                leg_distance = leg.get('distance', 0) * 0.000621371  # meters to miles
                total_distance += leg_distance
                
                if step['travel_mode'] == 'WALK':
                    walking_distance += leg_distance
                elif step['travel_mode'] == 'TRANSIT':
                    if step.get('transit_line'):
                        transit_lines.append(step['transit_line'])
                    transfers += 1
                
                # Add geometry
                if leg.get('legGeometry', {}).get('points'):
                    leg_coords = decode_otp_polyline(leg['legGeometry']['points'])
                    route_geometry.extend(leg_coords)
        
        # Adjust transfer count (first transit leg is not a transfer)
        transfers = max(0, transfers - 1)
        
        # Times
        start_time = itinerary.get('startTime', 0)
        end_time = itinerary.get('endTime', 0)
        
        departure_time = parse_otp_time(start_time)
        arrival_time = parse_otp_time(end_time)
        
        route_name = f"Transit Route {route_index + 1}"
        if transfers > 0:
            route_name += f" ({transfers} transfer{'s' if transfers != 1 else ''})"
        
        return {
            "route_number": route_index + 1,
            "name": route_name,
            "description": f"OTP transit route with {transfers} transfer{'s' if transfers != 1 else ''}",
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_minutes,
            "duration_text": duration_text,
            "distance_meters": total_distance * 1609.34,  # miles to meters
            "distance_km": total_distance * 1.60934,  # miles to km
            "distance_miles": total_distance,
            "departure_time": departure_time,
            "departure_timestamp": start_time,
            "arrival_time": arrival_time,
            "arrival_timestamp": end_time,
            "transfers": transfers,
            "walking_distance_miles": walking_distance,
            "walking_distance_meters": walking_distance * 1609.34,
            "total_fare": total_fare,  # OTP can provide fare info if configured
            "fare_currency": "USD",
            "transit_lines": list(set(transit_lines)),
            "route_geometry": route_geometry,
            "steps": steps,
            "service": "OpenTripPlanner"
        }
        
    except Exception as e:
        logger.error(f"Error parsing OTP itinerary: {e}")
        return None

def parse_otp_leg(leg: Dict, leg_index: int) -> Optional[Dict]:
    """Parse an individual leg from OTP itinerary"""
    try:
        mode = leg.get('mode', 'UNKNOWN')
        
        # Map OTP modes to our standard
        if mode == 'WALK':
            travel_mode = 'WALKING'
        elif mode in ['BUS', 'SUBWAY', 'RAIL', 'TRAM', 'FERRY']:
            travel_mode = 'TRANSIT'
        else:
            travel_mode = mode
        
        # Basic leg info
        duration_seconds = leg.get('duration', 0)
        duration_minutes = round(duration_seconds / 60, 1)
        duration_text = format_time_duration(duration_minutes)
        
        distance_meters = leg.get('distance', 0)
        distance_km = round(distance_meters / 1000, 2)
        distance_miles = round(distance_km * 0.621371, 2)
        
        # Instructions
        from_place = leg.get('from', {}).get('name', '')
        to_place = leg.get('to', {}).get('name', '')
        
        if travel_mode == 'WALKING':
            instruction = f"Walk from {from_place} to {to_place}"
        else:
            route_name = leg.get('route', 'Unknown Route')
            instruction = f"Take {route_name} from {from_place} to {to_place}"
        
        step_data = {
            "step_number": leg_index + 1,
            "travel_mode": travel_mode,
            "instruction": instruction,
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_minutes,
            "duration_text": duration_text,
            "distance_meters": distance_meters,
            "distance_km": distance_km,
            "distance_miles": distance_miles
        }
        
        # Transit-specific details
        if travel_mode == 'TRANSIT':
            route_short_name = leg.get('routeShortName', leg.get('route', 'Unknown'))
            route_long_name = leg.get('routeLongName', '')
            agency_name = leg.get('agencyName', 'Transit Agency')
            
            step_data.update({
                "transit_line": route_short_name,
                "transit_line_color": leg.get('routeColor', '1f8dd6'),
                "transit_vehicle_type": mode,
                "transit_vehicle_name": mode.title(),
                "transit_agency": agency_name,
                "transit_route_long_name": route_long_name
            })
            
            # Stop information
            from_stop = leg.get('from', {})
            to_stop = leg.get('to', {})
            
            step_data.update({
                "departure_stop_name": from_stop.get('name', 'Unknown Stop'),
                "departure_stop_location": {
                    "lat": from_stop.get('lat', 0),
                    "lng": from_stop.get('lon', 0)
                },
                "arrival_stop_name": to_stop.get('name', 'Unknown Stop'),
                "arrival_stop_location": {
                    "lat": to_stop.get('lat', 0),
                    "lng": to_stop.get('lon', 0)
                }
            })
            
            # Timing
            start_time = leg.get('startTime', 0)
            end_time = leg.get('endTime', 0)
            
            step_data.update({
                "scheduled_departure": parse_otp_time(start_time),
                "scheduled_arrival": parse_otp_time(end_time),
                "departure_timestamp": start_time,
                "arrival_timestamp": end_time
            })
            
            # Trip headsign
            step_data["headsign"] = leg.get('headsign', '')
            
            # Real-time info if available
            if 'realTime' in leg and leg['realTime']:
                step_data["real_time"] = True
                if 'departureDelay' in leg:
                    delay_seconds = leg['departureDelay']
                    step_data["delay_seconds"] = delay_seconds
                    step_data["delay_minutes"] = round(delay_seconds / 60, 1)
        
        return step_data
        
    except Exception as e:
        logger.error(f"Error parsing OTP leg: {e}")
        return None

def decode_otp_polyline(encoded_polyline: str) -> List[List[float]]:
    """Decode OTP polyline to coordinates"""
    try:
        # OTP uses the same polyline encoding as Google
        coords_latlon = polyline.decode(encoded_polyline)  # [(lat, lon), ...]
        return [[lon, lat] for (lat, lon) in coords_latlon]  # Convert to [lon, lat] for GeoJSON
    except Exception as e:
        logger.error(f"Error decoding OTP polyline: {e}")
        return []

def find_nearby_bus_stops_otp(point_coords, max_stops=3):
    """Find nearby bus stops using OTP"""
    try:
        logger.info(f"🚌 Finding nearest bus stops to {point_coords} via OTP")
        
        # Use GTFS manager which now uses OTP
        nearby_stops = gtfs_manager.find_nearby_stops(point_coords[1], point_coords[0], radius_meters=800)
        
        formatted_stops = []
        for stop in nearby_stops[:max_stops]:
            formatted_stops.append({
                "id": stop['stop_id'],
                "name": stop['stop_name'],
                "x": stop['stop_lon'],
                "y": stop['stop_lat'],
                "display_x": stop['stop_lon'],
                "display_y": stop['stop_lat'],
                "distance_meters": 0  # Would need to calculate if needed
            })
        
        logger.info(f"✅ Found {len(formatted_stops)} nearby bus stops via OTP")
        return formatted_stops
        
    except Exception as e:
        logger.error(f"Error finding bus stops via OTP: {e}")
        return []

# =============================================================================
# MAIN ROUTE ANALYSIS ENGINE WITH OSRM + OTP
# =============================================================================

def analyze_complete_bike_bus_bike_routes(start_point, end_point, departure_time="now"):
    """Main function to analyze complete bike-bus-bike routing options using OSRM + OTP"""
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"OSRM + OTP BIKE-BUS-BIKE ROUTE ANALYSIS")
        logger.info(f"{'='*60}")
        logger.info(f"Start: {start_point}")
        logger.info(f"End: {end_point}")
        logger.info(f"Departure: {departure_time}")
        logger.info(f"Routing: OSRM (bikes) + OpenTripPlanner (transit)")
        logger.info(f"{'='*60}")
        
        # Step 1: Find nearest bus stops to start and end points
        logger.info("\n🚌 STEP 1: Finding nearest bus stops via OTP...")
        start_bus_stops = find_nearby_bus_stops_otp(start_point, max_stops=2)
        end_bus_stops = find_nearby_bus_stops_otp(end_point, max_stops=2)
        
        if not start_bus_stops:
            logger.warning("No bus stops found near start point")
        
        if not end_bus_stops:
            logger.warning("No bus stops found near end point")
        
        # Check for transit fallback (very short bike segments)
        should_fallback = should_use_transit_fallback(start_point, end_point, start_bus_stops, end_bus_stops)
        
        routes = []
        
        if should_fallback:
            logger.info("\n🔄 SMART FALLBACK: Using OTP transit-only routing")
            logger.info("   Reason: Both bike segments are very short (< 400m)")
            
            # Direct transit routing using OTP
            try:
                transit_result = get_transit_routes_otp(start_point, end_point, departure_time)
                
                if transit_result.get('routes'):
                    for i, transit_route in enumerate(transit_result['routes']):
                        routes.append({
                            "id": i + 1,
                            "name": f"OTP Transit Option {i + 1}",
                            "type": "transit_fallback",
                            "summary": {
                                "total_time_minutes": transit_route['duration_minutes'],
                                "total_time_formatted": transit_route['duration_text'],
                                "total_distance_miles": transit_route['distance_miles'],
                                "bike_distance_miles": 0,
                                "transit_distance_miles": transit_route['distance_miles'],
                                "bike_percentage": 0,
                                "average_bike_score": 0,
                                "transfers": transit_route.get('transfers', 0),
                                "total_fare": transit_route.get('total_fare'),
                                "departure_time": transit_route.get('departure_time', 'Unknown'),
                                "arrival_time": transit_route.get('arrival_time', 'Unknown')
                            },
                            "legs": [{
                                "type": "transit",
                                "name": f"OTP Transit Route {i + 1}",
                                "description": f"Direct transit via OTP with {transit_route.get('transfers', 0)} transfer{'s' if transit_route.get('transfers', 0) != 1 else ''}",
                                "route": transit_route,
                                "color": "#2196f3",
                                "order": 1
                            }],
                            "fallback_reason": "Both bike segments < 400m - Transit more practical"
                        })
            except Exception as e:
                logger.warning(f"OTP transit fallback failed: {e}")
        
        # Always try to create bike-bus-bike routes if we have bus stops
        if start_bus_stops and end_bus_stops and len(start_bus_stops) > 0 and len(end_bus_stops) > 0:
            logger.info("\n🚴‍♂️ STEP 2: Creating OSRM bicycle route legs...")
            
            start_bus_stop = start_bus_stops[0]
            end_bus_stop = end_bus_stops[0]
            
            # Ensure different bus stops
            if start_bus_stop["id"] == end_bus_stop["id"] and len(start_bus_stops) > 1:
                end_bus_stop = start_bus_stops[1]
            elif start_bus_stop["id"] == end_bus_stop["id"] and len(end_bus_stops) > 1:
                end_bus_stop = end_bus_stops[1]
