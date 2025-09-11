# complete_osrm_otp_bike_bus_planner.py
# Complete Bike-Bus-Bike Route Planner with OSRM + OpenTripPlanner
# Fixed and complete version for ArcGIS Experience Builder

import os
import json
import logging
import requests
import datetime
import math
import re
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Install polyline if not available
try:
    import polyline
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "polyline"])
    import polyline

# =============================================================================
# CONFIGURATION
# =============================================================================

# Environment variables for deployment
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", 
    "https://experience.arcgis.com,https://*.maps.arcgis.com,http://localhost:*,https://*.railway.app").split(",")

# OSRM Configuration for bicycle routing
OSRM_SERVER = os.getenv("OSRM_SERVER", "http://router.project-osrm.org")
USE_OSRM_DURATION = True

# OpenTripPlanner Configuration for transit routing
OTP_SERVER = os.getenv("OTP_SERVER", "http://otp.prod.obanyc.com/otp")
OTP_ROUTER_ID = os.getenv("OTP_ROUTER_ID", "default")

# Bicycle Configuration
BIKE_SPEED_MPH = 11

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# FASTAPI APP SETUP
# =============================================================================

app = FastAPI(
    title="OSRM + OpenTripPlanner Bike-Bus-Bike Route Planner",
    description="Multimodal transportation planning with OSRM bicycle routing and OpenTripPlanner transit routing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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

def parse_otp_time(time_milliseconds):
    """Parse OTP time from milliseconds since epoch to readable format"""
    try:
        dt = datetime.datetime.fromtimestamp(time_milliseconds / 1000)
        return dt.strftime("%H:%M")
    except:
        return "Unknown"

# =============================================================================
# OSRM BICYCLE ROUTING
# =============================================================================

def calculate_bike_route_osrm(start_coords, end_coords, route_name="Bike Route"):
    """Create a bike route using OSRM"""
    try:
        logger.info(f"Creating OSRM bike route: {route_name}")
        
        coords = f"{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
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
        coords_latlon = polyline.decode(geometry_polyline)
        route_geometry = [[lon, lat] for (lat, lon) in coords_latlon]
        
        # Extract distance and duration
        distance_meters = float(route.get("distance", 0.0))
        distance_miles = distance_meters * 0.000621371
        
        osrm_duration_sec = route.get("duration", None)
        if USE_OSRM_DURATION and osrm_duration_sec:
            duration_minutes = float(osrm_duration_sec) / 60.0
            osrm_time_used = True
        else:
            duration_minutes = (distance_miles / BIKE_SPEED_MPH) * 60.0
            osrm_time_used = False
        
        # Simulate safety score
        overall_score = 70 + (10 if distance_miles < 2 else -5)
        
        return {
            "name": route_name,
            "length_miles": distance_miles,
            "travel_time_minutes": duration_minutes,
            "travel_time_formatted": format_time_duration(duration_minutes),
            "osrm_time_used": osrm_time_used,
            "geometry": {
                "type": "LineString",
                "coordinates": route_geometry
            },
            "overall_score": max(0, min(100, overall_score))
        }
        
    except Exception as e:
        logger.error(f"Error calculating OSRM bike route: {e}")
        return None

# =============================================================================
# OPENTRIPPLANNER TRANSIT ROUTING
# =============================================================================

def get_transit_routes_otp(origin_coords: Tuple[float, float], 
                          destination_coords: Tuple[float, float], 
                          departure_time: str = "now") -> Dict:
    """Get transit routes using OpenTripPlanner"""
    try:
        logger.info(f"Getting OTP transit routes: {origin_coords} -> {destination_coords}")
        
        # Prepare time parameter
        if departure_time == "now":
            time_str = datetime.datetime.now().strftime("%I:%M%p")
            date_str = datetime.datetime.now().strftime("%m-%d-%Y")
        else:
            try:
                if isinstance(departure_time, (int, float)):
                    dt = datetime.datetime.fromtimestamp(int(departure_time))
                else:
                    dt = datetime.datetime.now()
                time_str = dt.strftime("%I:%M%p")
                date_str = dt.strftime("%m-%d-%Y")
            except:
                time_str = datetime.datetime.now().strftime("%I:%M%p")
                date_str = datetime.datetime.now().strftime("%m-%d-%Y")
        
        url = f"{OTP_SERVER}/routers/{OTP_ROUTER_ID}/plan"
        
        params = {
            'fromPlace': f"{origin_coords[1]},{origin_coords[0]}",
            'toPlace': f"{destination_coords[1]},{destination_coords[0]}",
            'time': time_str,
            'date': date_str,
            'mode': 'TRANSIT,WALK',
            'maxWalkDistance': 1000,
            'arriveBy': 'false',
            'numItineraries': 3,
            'optimize': 'TRANSFERS',
            'walkSpeed': 1.34,
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'error' in data:
            error_msg = data['error'].get('msg', 'Unknown OTP error')
            logger.error(f"OTP API error: {error_msg}")
            return {"error": error_msg}
        
        if 'plan' not in data or 'itineraries' not in data['plan']:
            return {"error": "No transit routes found"}
        
        routes = []
        itineraries = data['plan']['itineraries']
        
        for idx, itinerary in enumerate(itineraries):
            route = parse_otp_itinerary(itinerary, idx)
            if route:
                routes.append(route)
        
        if not routes:
            return {"error": "No valid transit routes found"}
        
        return {
            "routes": routes,
            "service": "OpenTripPlanner",
            "total_routes": len(routes)
        }
        
    except Exception as e:
        logger.error(f"Error getting OTP transit routes: {e}")
        return {"error": str(e)}

def parse_otp_itinerary(itinerary: Dict, route_index: int) -> Optional[Dict]:
    """Parse a single OTP itinerary"""
    try:
        duration_seconds = itinerary.get('duration', 0)
        duration_minutes = round(duration_seconds / 60, 1)
        duration_text = format_time_duration(duration_minutes)
        
        legs = itinerary.get('legs', [])
        steps = []
        transit_lines = []
        transfers = 0
        total_distance = 0
        walking_distance = 0
        route_geometry = []
        
        for leg_idx, leg in enumerate(legs):
            step = parse_otp_leg(leg, leg_idx)
            if step:
                steps.append(step)
                
                leg_distance = leg.get('distance', 0) * 0.000621371  # meters to miles
                total_distance += leg_distance
                
                if step['travel_mode'] == 'WALKING':
                    walking_distance += leg_distance
                elif step['travel_mode'] == 'TRANSIT':
                    if step.get('transit_line'):
                        transit_lines.append(step['transit_line'])
                    transfers += 1
                
                # Add geometry
                if leg.get('legGeometry', {}).get('points'):
                    leg_coords = decode_otp_polyline(leg['legGeometry']['points'])
                    route_geometry.extend(leg_coords)
        
        transfers = max(0, transfers - 1)
        
        start_time = itinerary.get('startTime', 0)
        end_time = itinerary.get('endTime', 0)
        
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
            "distance_miles": total_distance,
            "distance_meters": total_distance * 1609.34,
            "departure_time": parse_otp_time(start_time),
            "arrival_time": parse_otp_time(end_time),
            "departure_timestamp": start_time,
            "arrival_timestamp": end_time,
            "transfers": transfers,
            "walking_distance_miles": walking_distance,
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
        
        if mode == 'WALK':
            travel_mode = 'WALKING'
        elif mode in ['BUS', 'SUBWAY', 'RAIL', 'TRAM', 'FERRY']:
            travel_mode = 'TRANSIT'
        else:
            travel_mode = mode
        
        duration_seconds = leg.get('duration', 0)
        duration_minutes = round(duration_seconds / 60, 1)
        duration_text = format_time_duration(duration_minutes)
        
        distance_meters = leg.get('distance', 0)
        distance_miles = round(distance_meters * 0.000621371, 2)
        
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
            "distance_miles": distance_miles
        }
        
        if travel_mode == 'TRANSIT':
            route_short_name = leg.get('routeShortName', leg.get('route', 'Unknown'))
            
            step_data.update({
                "transit_line": route_short_name,
                "transit_line_color": leg.get('routeColor', '1f8dd6'),
                "transit_vehicle_type": mode,
                "transit_agency": leg.get('agencyName', 'Transit Agency')
            })
            
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
            
            start_time = leg.get('startTime', 0)
            end_time = leg.get('endTime', 0)
            
            step_data.update({
                "scheduled_departure": parse_otp_time(start_time),
                "scheduled_arrival": parse_otp_time(end_time),
                "departure_timestamp": start_time,
                "arrival_timestamp": end_time,
                "headsign": leg.get('headsign', '')
            })
        
        return step_data
        
    except Exception as e:
        logger.error(f"Error parsing OTP leg: {e}")
        return None

def decode_otp_polyline(encoded_polyline: str) -> List[List[float]]:
    """Decode OTP polyline to coordinates"""
    try:
        coords_latlon = polyline.decode(encoded_polyline)
        return [[lon, lat] for (lat, lon) in coords_latlon]
    except Exception as e:
        logger.error(f"Error decoding OTP polyline: {e}")
        return []

def find_nearby_bus_stops_otp(point_coords, max_stops=3):
    """Find nearby bus stops using OTP"""
    try:
        url = f"{OTP_SERVER}/routers/{OTP_ROUTER_ID}/index/stops"
        params = {
            'lat': point_coords[1],
            'lon': point_coords[0],
            'radius': 800
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            stops_data = response.json()
            
            formatted_stops = []
            for stop in stops_data[:max_stops]:
                formatted_stops.append({
                    "id": stop.get('id', ''),
                    "name": stop.get('name', 'Unknown Stop'),
                    "x": stop.get('lon', point_coords[0]),
                    "y": stop.get('lat', point_coords[1]),
                    "display_x": stop.get('lon', point_coords[0]),
                    "display_y": stop.get('lat', point_coords[1]),
                    "distance_meters": 0
                })
            
            return formatted_stops
        else:
            return []
            
    except Exception as e:
        logger.error(f"Error finding bus stops: {e}")
        return []

# =============================================================================
# MAIN ROUTE ANALYSIS
# =============================================================================

def should_use_transit_fallback(start_point, end_point, start_bus_stops, end_bus_stops, distance_threshold_meters=400):
    """Check if both bike legs would be short enough to warrant transit-only fallback"""
    try:
        if not start_bus_stops or not end_bus_stops:
            return False
            
        start_bus_stop = start_bus_stops[0]
        end_bus_stop = end_bus_stops[0]
        
        dx1 = start_point[0] - start_bus_stop['display_x']
        dy1 = start_point[1] - start_bus_stop['display_y']
        bike_leg_1_distance = math.sqrt(dx1*dx1 + dy1*dy1) * 111000
        
        dx2 = end_point[0] - end_bus_stop['display_x'] 
        dy2 = end_point[1] - end_bus_stop['display_y']
        bike_leg_2_distance = math.sqrt(dx2*dx2 + dy2*dy2) * 111000
        
        return (bike_leg_1_distance < distance_threshold_meters and 
                bike_leg_2_distance < distance_threshold_meters)
        
    except Exception as e:
        logger.error(f"Error in fallback check: {e}")
        return False

def analyze_complete_bike_bus_bike_routes(start_point, end_point, departure_time="now"):
    """Main function to analyze bike-bus-bike routes using OSRM + OTP"""
    try:
        logger.info("Starting OSRM + OTP bike-bus-bike analysis")
        
        # Find nearby bus stops
        start_bus_stops = find_nearby_bus_stops_otp(start_point, max_stops=2)
        end_bus_stops = find_nearby_bus_stops_otp(end_point, max_stops=2)
        
        routes = []
        
        # Check for transit fallback
        should_fallback = should_use_transit_fallback(start_point, end_point, start_bus_stops, end_bus_stops)
        
        if should_fallback:
            logger.info("Using transit fallback")
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
                                "total_fare": 0,
                                "departure_time": transit_route.get('departure_time', 'Unknown'),
                                "arrival_time": transit_route.get('arrival_time', 'Unknown')
                            },
                            "legs": [{
                                "type": "transit",
                                "name": f"OTP Transit Route {i + 1}",
                                "description": f"Direct transit via OTP",
                                "route": transit_route,
                                "color": "#2196f3",
                                "order": 1
                            }],
                            "fallback_reason": "Both bike segments < 400m"
                        })
            except Exception as e:
                logger.warning(f"OTP transit fallback failed: {e}")
        
        # Create bike-bus-bike routes if we have bus stops
        if start_bus_stops and end_bus_stops:
            start_bus_stop = start_bus_stops[0]
            end_bus_stop = end_bus_stops[0]
            
            # Ensure different bus stops
            if start_bus_stop["id"] == end_bus_stop["id"] and len(start_bus_stops) > 1:
                end_bus_stop = start_bus_stops[1]
            elif start_bus_stop["id"] == end_bus_stop["id"] and len(end_bus_stops) > 1:
                end_bus_stop = end_bus_stops[1]
            
            if start_bus_stop["id"] != end_bus_stop["id"]:
                # Create bike legs
                bike_leg_1 = calculate_bike_route_osrm(
                    start_point, 
                    [start_bus_stop["display_x"], start_bus_stop["display_y"]],
                    "Start to Bus Stop"
                )
                
                bike_leg_2 = calculate_bike_route_osrm(
                    [end_bus_stop["display_x"], end_bus_stop["display_y"]], 
                    end_point,
                    "Bus Stop to End"
                )
                
                if bike_leg_1 and bike_leg_2:
                    # Get transit between stops
                    start_stop_coords = (start_bus_stop['display_x'], start_bus_stop['display_y'])
                    end_stop_coords = (end_bus_stop['display_x'], end_bus_stop['display_y'])
                    
                    try:
                        transit_result = get_transit_routes_otp(start_stop_coords, end_stop_coords, departure_time)
                        
                        if transit_result.get('routes'):
                            for i, transit_route in enumerate(transit_result['routes']):
                                bike_time_1 = bike_leg_1['travel_time_minutes']
                                bike_time_2 = bike_leg_2['travel_time_minutes']
                                transit_time = transit_route['duration_minutes']
                                total_time = bike_time_1 + transit_time + bike_time_2 + 5
                                
                                total_bike_miles = bike_leg_1['length_miles'] + bike_leg_2['length_miles']
                                total_transit_miles = transit_route['distance_miles']
                                total_miles = total_bike_miles + total_transit_miles
                                
                                weighted_score = ((bike_leg_1['overall_score'] * bike_leg_1['length_miles']) +
                                                (bike_leg_2['overall_score'] * bike_leg_2['length_miles'])) / total_bike_miles if total_bike_miles > 0 else 0
                                
                                enhanced_transit_route = transit_route.copy()
                                enhanced_transit_route['start_stop'] = start_bus_stop
                                enhanced_transit_route['end_stop'] = end_bus_stop
                                
                                routes.append({
                                    "id": len(routes) + 1,
                                    "name": f"OSRM+OTP Bike-Bus-Bike Option {i + 1}",
                                    "type": "bike_bus_bike",
                                    "summary": {
                                        "total_time_minutes": round(total_time, 1),
                                        "total_time_formatted": format_time_duration(total_time),
                                        "total_distance_miles": round(total_miles, 2),
                                        "bike_distance_miles": round(total_bike_miles, 2),
                                        "transit_distance_miles": round(total_transit_miles, 2),
                                        "bike_percentage": round((total_bike_miles / total_miles) * 100, 1) if total_miles > 0 else 0,
                                        "average_bike_score": round(weighted_score, 1),
                                        "transfers": transit_route.get('transfers', 0),
                                        "total_fare": 0,
                                        "departure_time": transit_route.get('departure_time', 'Unknown'),
                                        "arrival_time": transit_route.get('arrival_time', 'Unknown')
                                    },
                                    "legs": [
                                        {
                                            "type": "bike",
                                            "name": "OSRM Bike to Bus Stop",
                                            "description": f"OSRM bike route to {start_bus_stop['name']}",
                                            "route": bike_leg_1,
                                            "color": "#27ae60",
                                            "order": 1
                                        },
                                        {
                                            "type": "transit",
                                            "name": f"OTP Transit",
                                            "description": f"OTP transit from {start_bus_stop['name']} to {end_bus_stop['name']}",
                                            "route": enhanced_transit_route,
                                            "color": "#3498db",
                                            "order": 2
                                        },
                                        {
                                            "type": "bike",
                                            "name": "OSRM Bus Stop to Destination",
                                            "description": f"OSRM bike route from {end_bus_stop['name']}",
                                            "route": bike_leg_2,
                                            "color": "#27ae60",
                                            "order": 3
                                        }
                                    ]
                                })
                    
                    except Exception as e:
                        logger.warning(f"OTP transit routing failed: {e}")
        
        # Add direct bike route
        direct_bike_route = calculate_bike_route_osrm(start_point, end_point, "Direct OSRM Bike Route")
        
        if direct_bike_route:
            routes.append({
                "id": len(routes) + 1,
                "name": "Direct OSRM Bike Route",
                "type": "direct_bike",
                "summary": {
                    "total_time_minutes": direct_bike_route['travel_time_minutes'],
                    "total_time_formatted": direct_bike_route['travel_time_formatted'],
                    "total_distance_miles": direct_bike_route['length_miles'],
                    "bike_distance_miles": direct_bike_route['length_miles'],
                    "transit_distance_miles": 0,
                    "bike_percentage": 100,
                    "average_bike_score": direct_bike_route['overall_score'],
                    "transfers": 0,
                    "total_fare": 0,
                    "departure_time": "Immediate",
                    "arrival_time": "Flexible"
                },
                "legs": [{
                    "type": "bike",
                    "name": "Direct OSRM Bike Route",
                    "description": "Complete OSRM bike route",
                    "route": direct_bike_route,
                    "color": "#e74c3c",
                    "order": 1
                }]
            })
        
        if not routes:
            raise HTTPException(status_code=400, detail="No routes found")
        
        # Sort by time
        routes.sort(key=lambda x: x['summary']['total_time_minutes'])
        
        return {
            "success": True,
            "analysis_type": "osrm_otp_enhanced",
            "fallback_used": should_fallback,
            "routing_engine": "OSRM + OpenTripPlanner",
            "routes": routes,
            "bus_stops": {
                "start_stops": start_bus_stops,
                "end_stops": end_bus_stops
            },
            "statistics": {
                "total_options": len(routes),
                "bike_bus_bike_options": len([r for r in routes if r['type'] == 'bike_bus_bike']),
                "direct_bike_options": len([r for r in routes if r['type'] == 'direct_bike']),
                "transit_fallback_options": len([r for r in routes if r['type'] == 'transit_fallback']),
                "fastest_option": routes[0]['name'] if routes else None,
                "fastest_time": routes[0]['summary']['total_time_formatted'] if routes else None
            },
            "bike_speed_mph": BIKE_SPEED_MPH,
            "osrm_server": OSRM_SERVER,
            "otp_server": OTP_SERVER,
            "analysis_timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# FASTAPI ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve the UI"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OSRM + OTP Bike-Bus-Bike Planner</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <style>
            body { margin: 0; font-family: Arial, sans-serif; }
            .header { background: #2c3e50; color: white; padding: 15px; text-align: center; }
            .container { display: flex; height: calc(100vh - 60px); }
            #map { flex: 2; }
            .sidebar { flex: 1; max-width: 400px; padding: 20px; background: #f8f9fa; overflow-y: auto; }
            .controls { margin-bottom: 20px; }
            label { display: block; margin: 10px 0 5px 0; font-weight: bold; }
            select, input { width: 100%; padding: 8px; margin-bottom: 10px; }
            button { width: 100%; padding: 12px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; margin: 5px 0; }
            button:disabled { background: #bdc3c7; cursor: not-allowed; }
            button:hover:not(:disabled) { background: #2980b9; }
            .btn-clear { background: #e74c3c; }
            .btn-clear:hover { background: #c0392b; }
            .route-card { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; cursor: pointer; }
            .route-card:hover { border-color: #3498db; }
            .route-card.selected { border-color: #3498db; background: #f8f9ff; }
            .route-header { font-weight: bold; margin-bottom: 10px; }
            .route-summary { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 10px 0; }
            .summary-item { text-align: center; padding: 8px; background: #f8f9fa; border-radius: 4px; }
            .summary-value { font-weight: bold; color: #3498db; }
            .summary-label { font-size: 12px; color: #666; }
            .coordinates { background: #e9ecef; padding: 10px; border-radius: 4px; margin: 10px 0; font-size: 14px; }
            .error { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 4px; margin: 10px 0; }
            .spinner { border: 3px solid #f3f3f3; border-top: 3px solid #3498db; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 20px auto; display: none; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .hidden { display: none; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>OSRM + OpenTripPlanner Bike-Bus-Bike Route Planner</h1>
            <p>Advanced multimodal transportation planning</p>
        </div>
        
        <div class="container">
            <div id="map"></div>
            
            <div class="sidebar">
                <div class="controls">
                    <label for="departureTime">Departure Time:</label>
                    <select id="departureTime">
                        <option value="now">Leave Now</option>
                        <option value="custom">Custom Time</option>
                    </select>
                    
                    <div id="customTimeGroup" class="hidden">
                        <label for="customTime">Select Time:</label>
                        <input type="datetime-local" id="customTime">
                    </div>
                    
                    <button id="findRoutesBtn" disabled>Find Routes</button>
                    <button class="btn-clear" onclick="clearAll()">Clear Map</button>
                    
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
            
            function initMap() {
                map = L.map('map').setView([30.3322, -81.6557], 12);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors'
                }).addTo(map);
                routeLayersGroup = L.layerGroup().addTo(map);
                map.on('click', handleMapClick);
            }
            
            function handleMapClick(e) {
                const lat = e.latlng.lat, lng = e.latlng.lng;
                
                if (clickCount === 0) {
                    if (startMarker) map.removeLayer(startMarker);
                    startMarker = L.marker([lat, lng]).addTo(map);
                    startMarker.bindPopup("Start Point").openPopup();
                    startPoint = [lng, lat];
                    document.getElementById('startCoords').textContent = lat.toFixed(5) + ', ' + lng.toFixed(5);
                    clickCount = 1;
                } else if (clickCount === 1) {
                    if (endMarker) map.removeLayer(endMarker);
                    endMarker = L.marker([lat, lng]).addTo(map);
                    endMarker.bindPopup("End Point").openPopup();
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
                document.getElementById('findRoutesBtn').innerHTML = show ? 'Analyzing routes...' : 'Find Routes';
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
                    displayResults(data);
                    
                } catch (error) {
                    document.getElementById('results').innerHTML = 
                        '<div class="error">Error: ' + error.message + '</div>';
                } finally {
                    showSpinner(false);
                }
            }
            
            function displayResults(data) {
                let html = '<h3>Found ' + data.routes.length + ' Route Options</h3>';
                
                if (data.fallback_used) {
                    html += '<div style="background: #e3f2fd; padding: 10px; border-radius: 4px; margin: 10px 0;">';
                    html += '<strong>Smart Routing:</strong> Using optimized transit routes via OpenTripPlanner.';
                    html += '</div>';
                }
                
                data.routes.forEach((route, index) => {
                    html += '<div class="route-card" onclick="selectRoute(' + index + ')" id="route' + index + '">';
                    html += '<div class="route-header">' + route.name + '</div>';
                    
                    html += '<div class="route-summary">';
                    html += '<div class="summary-item">';
                    html += '<div class="summary-value">' + route.summary.total_time_formatted + '</div>';
                    html += '<div class="summary-label">Total Time</div>';
                    html += '</div>';
                    html += '<div class="summary-item">';
                    html += '<div class="summary-value">' + route.summary.total_distance_miles.toFixed(1) + ' mi</div>';
                    html += '<div class="summary-label">Distance</div>';
                    html += '</div>';
                    html += '</div>';
                    
                    html += '<div style="font-size: 14px; margin-top: 10px;">';
                    route.legs.forEach(leg => {
                        html += '<div>' + leg.name + ': ' + (leg.route.length_miles || leg.route.distance_miles || 0).toFixed(1) + ' mi</div>';
                    });
                    html += '</div>';
                    
                    html += '</div>';
                });
                
                document.getElementById('results').innerHTML = html;
                if (currentRoutes.length > 0) selectRoute(0);
            }
            
            function selectRoute(index) {
                document.querySelectorAll('.route-card').forEach(card => card.classList.remove('selected'));
                document.getElementById('route' + index).classList.add('selected');
                
                routeLayersGroup.clearLayers();
                const route = currentRoutes[index];
                
                const colors = { 'bike': '#27ae60', 'transit': '#3498db' };
                
                route.legs.forEach(leg => {
                    if (leg.route.geometry && leg.route.geometry.coordinates) {
                        const coords = leg.route.geometry.coordinates.map(coord => [coord[1], coord[0]]);
                        const color = colors[leg.type] || '#95a5a6';
                        
                        const polyline = L.polyline(coords, {
                            color: color,
                            weight: 5,
                            opacity: 0.8,
                            dashArray: leg.type === 'transit' ? '10, 5' : null
                        }).addTo(routeLayersGroup);
                        
                        polyline.bindPopup(
                            '<strong>' + leg.name + '</strong><br>' +
                            'Distance: ' + (leg.route.length_miles || leg.route.distance_miles || 0).toFixed(2) + ' miles<br>' +
                            'Time: ' + (leg.route.travel_time_formatted || leg.route.duration_text || 'N/A')
                        );
                    }
                });
                
                // Add bus stop markers for bike-bus-bike routes
                if (route.type === 'bike_bus_bike') {
                    const transitLeg = route.legs.find(leg => leg.type === 'transit');
                    if (transitLeg && transitLeg.route.steps) {
                        transitLeg.route.steps.forEach(step => {
                            if (step.travel_mode === 'TRANSIT') {
                                if (step.departure_stop_location) {
                                    L.marker([step.departure_stop_location.lat, step.departure_stop_location.lng])
                                     .addTo(routeLayersGroup)
                                     .bindPopup('Departure: ' + step.departure_stop_name);
                                }
                                if (step.arrival_stop_location) {
                                    L.marker([step.arrival_stop_location.lat, step.arrival_stop_location.lng])
                                     .addTo(routeLayersGroup)
                                     .bindPopup('Arrival: ' + step.arrival_stop_name);
                                }
                            }
                        });
                    }
                }
                
                // Fit map to route
                try {
                    if (routeLayersGroup.getLayers().length > 0) {
                        map.fitBounds(routeLayersGroup.getBounds(), { padding: [20, 20] });
                    }
                } catch (e) {
                    console.warn('Could not fit map bounds:', e);
                }
            }
            
            // Event listeners
            document.getElementById('departureTime').addEventListener('change', function() {
                const customGroup = document.getElementById('customTimeGroup');
                if (this.value === 'custom') {
                    customGroup.classList.remove('hidden');
                    const now = new Date();
                    now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
                    document.getElementById('customTime').value = now.toISOString().slice(0, 16);
                } else {
                    customGroup.classList.add('hidden');
                }
            });
            
            document.getElementById('findRoutesBtn').addEventListener('click', findRoutes);
            
            initMap();
        </script>
    </body>
    </html>
    """

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "OSRM + OpenTripPlanner Bike-Bus-Bike Route Planner",
        "version": "1.0.0",
        "routing_engines": {
            "bicycle": "OSRM",
            "transit": "OpenTripPlanner"
        },
        "osrm_server": OSRM_SERVER,
        "otp_server": OTP_SERVER,
        "otp_router_id": OTP_ROUTER_ID,
        "bike_speed_mph": BIKE_SPEED_MPH,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/api/analyze")
async def analyze_routes(
    start_lon: float = Query(..., description="Start longitude"),
    start_lat: float = Query(..., description="Start latitude"),
    end_lon: float = Query(..., description="End longitude"),
    end_lat: float = Query(..., description="End latitude"),
    departure_time: str = Query("now", description="Departure time")
):
    """Analyze bike-bus-bike routes using OSRM + OTP"""
    
    if not (-180 <= start_lon <= 180 and -90 <= start_lat <= 90):
        raise HTTPException(status_code=400, detail="Invalid start coordinates")
    if not (-180 <= end_lon <= 180 and -90 <= end_lat <= 90):
        raise HTTPException(status_code=400, detail="Invalid end coordinates")
    
    try:
        result = analyze_complete_bike_bus_bike_routes(
            [start_lon, start_lat],
            [end_lon, end_lat],
            departure_time
        )
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stops")
async def get_nearby_stops(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    radius_meters: int = Query(500, description="Search radius in meters")
):
    """Get nearby transit stops via OTP"""
    try:
        stops = find_nearby_bus_stops_otp([lon, lat], max_stops=10)
        return {
            "stops": stops,
            "count": len(stops),
            "center": {"lat": lat, "lon": lon},
            "radius_meters": radius_meters,
            "source": "OpenTripPlanner"
        }
    except Exception as e:
        logger.error(f"Error getting stops: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# STARTUP EVENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    logger.info("Starting OSRM + OpenTripPlanner Bike-Bus-Bike API...")
    
    # Test OTP connection
    try:
        response = requests.get(f"{OTP_SERVER}/routers", timeout=10)
        if response.status_code == 200:
            logger.info("OpenTripPlanner connection established")
        else:
            logger.warning("OpenTripPlanner connection failed")
    except Exception as e:
        logger.warning(f"OTP initialization failed: {e}")
    
    logger.info("API ready for Experience Builder integration")
    logger.info(f"OSRM Server: {OSRM_SERVER}")
    logger.info(f"OTP Server: {OTP_SERVER}")
    logger.info(f"OTP Router: {OTP_ROUTER_ID}")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting server on port {port}")
    logger.info(f"OSRM Server: {OSRM_SERVER}")
    logger.info(f"OTP Server: {OTP_SERVER}")
    logger.info(f"OTP Router: {OTP_ROUTER_ID}")
    
    uvicorn.run(
        "complete_osrm_otp_bike_bus_planner:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
