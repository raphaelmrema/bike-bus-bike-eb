# complete_fixed_osrm_otp_planner.py
# Complete and corrected OSRM + OpenTripPlanner Bike-Bus-Bike Route Planner

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
# CONFIGURATION
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

# Potential OTP servers to test
POTENTIAL_OTP_SERVERS = [
    {"name": "JTA OTP", "server": "http://otp.jtafla.com/otp", "router": "jta"},
    {"name": "Florida DOT", "server": "http://otp.fdot.gov/otp", "router": "florida"},
    {"name": "NYC Demo", "server": "http://otp.prod.obanyc.com/otp", "router": "default"}
]

# =============================================================================
# FASTAPI APP SETUP
# =============================================================================

app = FastAPI(title="OSRM + OTP Bike-Bus-Bike Planner", version="1.0.0")

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
    """Format time duration"""
    if minutes < 1:
        return "< 1 min"
    elif minutes < 60:
        return f"{int(minutes)} min"
    else:
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours}h {mins}m" if mins > 0 else f"{hours}h"

def parse_otp_time(time_milliseconds):
    """Parse OTP time"""
    try:
        dt = datetime.datetime.fromtimestamp(time_milliseconds / 1000)
        return dt.strftime("%H:%M")
    except:
        return "Unknown"

# =============================================================================
# OTP SERVER TESTING
# =============================================================================

def test_otp_server(server_url, router_id):
    """Test if an OTP server works"""
    try:
        # Test routers endpoint
        routers_url = f"{server_url}/routers"
        response = requests.get(routers_url, timeout=10)
        
        if response.status_code != 200:
            return False, f"Server not accessible (HTTP {response.status_code})"
        
        # Test plan endpoint with Jacksonville coordinates
        plan_url = f"{server_url}/routers/{router_id}/plan"
        test_params = {
            'fromPlace': '30.3322,-81.6557',
            'toPlace': '30.3400,-81.6600',
            'mode': 'TRANSIT,WALK',
            'date': datetime.datetime.now().strftime("%m-%d-%Y"),
            'time': '10:00AM'
        }
        
        plan_response = requests.get(plan_url, params=test_params, timeout=15)
        
        if plan_response.status_code == 200:
            plan_data = plan_response.json()
            if 'plan' in plan_data:
                return True, "Server working"
            elif 'error' in plan_data:
                error_msg = plan_data['error'].get('msg', 'Unknown error')
                if 'VertexNotFoundException' in error_msg:
                    return True, "Server working (coordinates outside coverage)"
                else:
                    return False, f"Plan error: {error_msg}"
        
        return False, f"Plan request failed (HTTP {plan_response.status_code})"
        
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def find_working_otp_server():
    """Find a working OTP server"""
    global WORKING_OTP_SERVER, WORKING_OTP_ROUTER
    
    # Try environment variables first
    env_server = os.getenv("OTP_SERVER")
    env_router = os.getenv("OTP_ROUTER_ID")
    
    if env_server and env_router:
        is_working, message = test_otp_server(env_server, env_router)
        if is_working:
            WORKING_OTP_SERVER = env_server
            WORKING_OTP_ROUTER = env_router
            return env_server, env_router
    
    # Test potential servers
    for server_config in POTENTIAL_OTP_SERVERS:
        server = server_config["server"]
        router = server_config["router"]
        
        is_working, message = test_otp_server(server, router)
        
        if is_working:
            WORKING_OTP_SERVER = server
            WORKING_OTP_ROUTER = router
            return server, router
    
    return None, None

# =============================================================================
# OSRM BICYCLE ROUTING
# =============================================================================

def calculate_bike_route_osrm(start_coords, end_coords, route_name="Bike Route"):
    """Create bike route using OSRM"""
    try:
        coords = f"{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
        url = f"{OSRM_SERVER}/route/v1/cycling/{coords}"
        
        params = {
            "overview": "full",
            "geometries": "polyline",
            "steps": "true"
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
        
        return {
            "name": route_name,
            "length_miles": distance_miles,
            "travel_time_minutes": duration_minutes,
            "travel_time_formatted": format_time_duration(duration_minutes),
            "geometry": {
                "type": "LineString",
                "coordinates": route_geometry
            },
            "overall_score": 70
        }
        
    except Exception as e:
        logger.error(f"Error calculating bike route: {e}")
        return None

# =============================================================================
# OTP TRANSIT ROUTING
# =============================================================================

def get_transit_routes_otp(origin_coords, destination_coords, departure_time="now"):
    """Get transit routes using OTP"""
    global WORKING_OTP_SERVER, WORKING_OTP_ROUTER
    
    if not WORKING_OTP_SERVER:
        server, router = find_working_otp_server()
        if not server:
            return {"error": "No working OTP server found"}
        WORKING_OTP_SERVER = server
        WORKING_OTP_ROUTER = router
    
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
            'numItineraries': 3
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'error' in data:
            return {"error": data['error'].get('msg', 'OTP error')}
        
        if 'plan' not in data or 'itineraries' not in data['plan']:
            return {"error": "No transit routes found"}
        
        routes = []
        for idx, itinerary in enumerate(data['plan']['itineraries']):
            route = parse_otp_itinerary(itinerary, idx)
            if route:
                routes.append(route)
        
        return {
            "routes": routes,
            "service": "OpenTripPlanner",
            "total_routes": len(routes)
        }
        
    except Exception as e:
        logger.error(f"OTP error: {e}")
        return {"error": str(e)}

def parse_otp_itinerary(itinerary, route_index):
    """Parse OTP itinerary"""
    try:
        duration_seconds = itinerary.get('duration', 0)
        duration_minutes = duration_seconds / 60.0
        
        legs = itinerary.get('legs', [])
        steps = []
        transit_lines = []
        transfers = 0
        total_distance = 0
        route_geometry = []
        
        for leg_idx, leg in enumerate(legs):
            step = parse_otp_leg(leg, leg_idx)
            if step:
                steps.append(step)
                
                leg_distance = leg.get('distance', 0) * 0.000621371
                total_distance += leg_distance
                
                if step['travel_mode'] == 'TRANSIT':
                    if step.get('transit_line'):
                        transit_lines.append(step['transit_line'])
                    transfers += 1
                
                # Add geometry
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
            "description": f"OTP transit route with {transfers} transfer{'s' if transfers != 1 else ''}",
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_minutes,
            "duration_text": format_time_duration(duration_minutes),
            "distance_miles": total_distance,
            "departure_time": parse_otp_time(start_time),
            "arrival_time": parse_otp_time(end_time),
            "transfers": transfers,
            "transit_lines": list(set(transit_lines)),
            "route_geometry": route_geometry,
            "steps": steps,
            "service": "OpenTripPlanner"
        }
        
    except Exception as e:
        logger.error(f"Error parsing itinerary: {e}")
        return None

def parse_otp_leg(leg, leg_index):
    """Parse OTP leg"""
    try:
        mode = leg.get('mode', 'UNKNOWN')
        
        if mode == 'WALK':
            travel_mode = 'WALKING'
        elif mode in ['BUS', 'SUBWAY', 'RAIL', 'TRAM']:
            travel_mode = 'TRANSIT'
        else:
            travel_mode = mode
        
        duration_seconds = leg.get('duration', 0)
        duration_minutes = duration_seconds / 60.0
        
        distance_meters = leg.get('distance', 0)
        distance_miles = distance_meters * 0.000621371
        
        from_place = leg.get('from', {}).get('name', '')
        to_place = leg.get('to', {}).get('name', '')
        
        if travel_mode == 'WALKING':
            instruction = f"Walk from {from_place} to {to_place}"
        else:
            route_name = leg.get('route', 'Transit')
            instruction = f"Take {route_name} from {from_place} to {to_place}"
        
        step_data = {
            "step_number": leg_index + 1,
            "travel_mode": travel_mode,
            "instruction": instruction,
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_minutes,
            "duration_text": format_time_duration(duration_minutes),
            "distance_meters": distance_meters,
            "distance_miles": distance_miles
        }
        
        if travel_mode == 'TRANSIT':
            route_short_name = leg.get('routeShortName', leg.get('route', 'Unknown'))
            
            step_data.update({
                "transit_line": route_short_name,
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
                "headsign": leg.get('headsign', '')
            })
        
        return step_data
        
    except Exception as e:
        logger.error(f"Error parsing leg: {e}")
        return None

def find_nearby_bus_stops(point_coords, max_stops=3):
    """Find nearby bus stops"""
    global WORKING_OTP_SERVER, WORKING_OTP_ROUTER
    
    if not WORKING_OTP_SERVER:
        # Create fallback stops
        fallback_stops = []
        for i in range(max_stops):
            offset = 0.005 * (i + 1)
            fallback_stops.append({
                "id": f"stop_{i+1}",
                "name": f"Bus Stop {i+1}",
                "x": point_coords[0] + offset,
                "y": point_coords[1] + offset,
                "display_x": point_coords[0] + offset,
                "display_y": point_coords[1] + offset,
                "distance_meters": (i + 1) * 500
            })
        return fallback_stops
    
    try:
        url = f"{WORKING_OTP_SERVER}/routers/{WORKING_OTP_ROUTER}/index/stops"
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
            
            return formatted_stops if formatted_stops else []
        
        return []
        
    except Exception as e:
        logger.error(f"Error finding stops: {e}")
        return []

# =============================================================================
# MAIN ROUTE ANALYSIS
# =============================================================================

def analyze_bike_bus_bike_routes(start_point, end_point, departure_time="now"):
    """Main analysis function"""
    try:
        # Find nearby bus stops
        start_bus_stops = find_nearby_bus_stops(start_point, max_stops=2)
        end_bus_stops = find_nearby_bus_stops(end_point, max_stops=2)
        
        routes = []
        
        # Check for transit fallback
        should_fallback = False
        if start_bus_stops and end_bus_stops:
            start_stop = start_bus_stops[0]
            end_stop = end_bus_stops[0]
            
            # Calculate distances
            dx1 = start_point[0] - start_stop['display_x']
            dy1 = start_point[1] - start_stop['display_y']
            dist1 = math.sqrt(dx1*dx1 + dy1*dy1) * 111000
            
            dx2 = end_point[0] - end_stop['display_x']
            dy2 = end_point[1] - end_stop['display_y']
            dist2 = math.sqrt(dx2*dx2 + dy2*dy2) * 111000
            
            should_fallback = dist1 < 400 and dist2 < 400
        
        # Transit fallback
        if should_fallback:
            transit_result = get_transit_routes_otp(start_point, end_point, departure_time)
            
            if transit_result.get('routes'):
                for i, transit_route in enumerate(transit_result['routes']):
                    routes.append({
                        "id": i + 1,
                        "name": f"Transit Option {i + 1}",
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
                            "name": f"Transit Route {i + 1}",
                            "description": "Direct transit route",
                            "route": transit_route,
                            "color": "#2196f3",
                            "order": 1
                        }]
                    })
        
        # Bike-bus-bike routes
        if start_bus_stops and end_bus_stops and not should_fallback:
            start_bus_stop = start_bus_stops[0]
            end_bus_stop = end_bus_stops[0]
            
            if start_bus_stop["id"] != end_bus_stop["id"]:
                # Create bike legs
                bike_leg_1 = calculate_bike_route_osrm(
                    start_point,
                    [start_bus_stop["display_x"], start_bus_stop["display_y"]]
                )
                
                bike_leg_2 = calculate_bike_route_osrm(
                    [end_bus_stop["display_x"], end_bus_stop["display_y"]],
                    end_point
                )
                
                if bike_leg_1 and bike_leg_2:
                    # Get transit between stops
                    start_coords = (start_bus_stop['display_x'], start_bus_stop['display_y'])
                    end_coords = (end_bus_stop['display_x'], end_bus_stop['display_y'])
                    
                    transit_result = get_transit_routes_otp(start_coords, end_coords, departure_time)
                    
                    if transit_result.get('routes'):
                        for i, transit_route in enumerate(transit_result['routes']):
                            total_time = (bike_leg_1['travel_time_minutes'] + 
                                        transit_route['duration_minutes'] + 
                                        bike_leg_2['travel_time_minutes'] + 5)
                            
                            total_bike_miles = bike_leg_1['length_miles'] + bike_leg_2['length_miles']
                            total_transit_miles = transit_route['distance_miles']
                            total_miles = total_bike_miles + total_transit_miles
                            
                            routes.append({
                                "id": len(routes) + 1,
                                "name": f"Bike-Bus-Bike Option {i + 1}",
                                "type": "bike_bus_bike",
                                "summary": {
                                    "total_time_minutes": round(total_time, 1),
                                    "total_time_formatted": format_time_duration(total_time),
                                    "total_distance_miles": round(total_miles, 2),
                                    "bike_distance_miles": round(total_bike_miles, 2),
                                    "transit_distance_miles": round(total_transit_miles, 2),
                                    "bike_percentage": round((total_bike_miles / total_miles) * 100, 1) if total_miles > 0 else 0,
                                    "average_bike_score": 70,
                                    "transfers": transit_route.get('transfers', 0),
                                    "total_fare": 0,
                                    "departure_time": transit_route.get('departure_time', 'Unknown'),
                                    "arrival_time": transit_route.get('arrival_time', 'Unknown')
                                },
                                "legs": [
                                    {
                                        "type": "bike",
                                        "name": "Bike to Bus Stop",
                                        "description": f"Bike to {start_bus_stop['name']}",
                                        "route": bike_leg_1,
                                        "color": "#27ae60",
                                        "order": 1
                                    },
                                    {
                                        "type": "transit",
                                        "name": "Transit",
                                        "description": f"Transit from {start_bus_stop['name']} to {end_bus_stop['name']}",
                                        "route": transit_route,
                                        "color": "#3498db",
                                        "order": 2
                                    },
                                    {
                                        "type": "bike",
                                        "name": "Bus Stop to Destination",
                                        "description": f"Bike from {end_bus_stop['name']}",
                                        "route": bike_leg_2,
                                        "color": "#27ae60",
                                        "order": 3
                                    }
                                ]
                            })
        
        # Direct bike route
        direct_bike_route = calculate_bike_route_osrm(start_point, end_point)
        
        if direct_bike_route:
            routes.append({
                "id": len(routes) + 1,
                "name": "Direct Bike Route",
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
                    "name": "Direct Bike Route",
                    "description": "Complete bike route",
                    "route": direct_bike_route,
                    "color": "#e74c3c",
                    "order": 1
                }]
            })
        
        if not routes:
            raise HTTPException(status_code=400, detail="No routes found")
        
        routes.sort(key=lambda x: x['summary']['total_time_minutes'])
        
        return {
            "success": True,
            "analysis_type": "osrm_otp",
            "fallback_used": should_fallback,
            "routes": routes,
            "statistics": {
                "total_options": len(routes),
                "fastest_option": routes[0]['name'] if routes else None,
                "fastest_time": routes[0]['summary']['total_time_formatted'] if routes else None
            },
            "otp_server": WORKING_OTP_SERVER,
            "analysis_timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
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
        <title>OSRM + OTP Route Planner</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <style>
            body { margin: 0; font-family: Arial, sans-serif; }
            .header { background: #2c3e50; color: white; padding: 15px; text-align: center; }
            .container { display: flex; height: calc(100vh - 60px); }
            #map { flex: 2; }
            .sidebar { flex: 1; max-width: 400px; padding: 20px; background: #f8f9fa; overflow-y: auto; }
            .status { background: #d4edda; color: #155724; padding: 10px; border-radius: 4px; margin-bottom: 15px; }
            .controls { margin-bottom: 20px; }
            label { display: block; margin: 10px 0 5px 0; font-weight: bold; }
            select, input { width: 100%; padding: 8px; margin-bottom: 10px; border: 2px solid #ddd; border-radius: 4px; }
            button { width: 100%; padding: 12px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; margin: 5px 0; }
            button:disabled { background: #bdc3c7; cursor: not-allowed; }
            button:hover:not(:disabled) { background: #2980b9; }
            .btn-clear { background: #e74c3c; }
            .route-card { background: white; border: 2px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; cursor: pointer; }
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
            <h1>OSRM + OpenTripPlanner Route Planner</h1>
            <p>Bike-Bus-Bike multimodal transportation planning</p>
        </div>
        
        <div class="container">
            <div id="map"></div>
            
            <div class="sidebar">
                <div class="status" id="status">
                    System: OSRM + OTP Ready
                </div>
                
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
                loadSystemStatus();
            }
            
            async function loadSystemStatus() {
                try {
                    const response = await fetch('/api/health');
                    const status = await response.json();
                    const statusDiv = document.getElementById('status');
                    if (status.otp_working) {
                        statusDiv.innerHTML = 'System: OSRM + OTP Connected';
                        statusDiv.style.background = '#d4edda';
                    } else {
                        statusDiv.innerHTML = 'System: OSRM Only (OTP unavailable)';
                        statusDiv.style.background = '#fff3cd';
                    }
                } catch (error) {
                    document.getElementById('status').innerHTML = 'System: Status unknown';
                }
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
                document.getElementById('findRoutesBtn').innerHTML = show ? 'Analyzing...' : 'Find Routes';
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
                
                if (data.otp_server) {
                    html += '<div style="background: #e3f2fd; padding: 8px; border-radius: 4px; margin: 10px 0; font-size: 12px;">';
                    html += 'Using OTP: ' + data.otp_server;
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
                    
                    html += '<div style="font-size: 14px;">';
                    route.legs.forEach(leg => {
                        html += '<div>' + leg.name + '</div>';
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
                        
                        L.polyline(coords, {
                            color: color,
                            weight: 5,
                            opacity: 0.8,
                            dashArray: leg.type === 'transit' ? '10, 5' : null
                        }).addTo(routeLayersGroup);
                    }
                });
                
                try {
                    if (routeLayersGroup.getLayers().length > 0) {
                        map.fitBounds(routeLayersGroup.getBounds(), { padding: [20, 20] });
                    }
                } catch (e) {
                    console.warn('Could not fit map bounds');
                }
            }
            
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
    global WORKING_OTP_SERVER
    
    if not WORKING_OTP_SERVER:
        find_working_otp_server()
    
    return {
        "status": "healthy",
        "service": "OSRM + OTP Route Planner",
        "osrm_server": OSRM_SERVER,
        "otp_server": WORKING_OTP_SERVER,
        "otp_working": bool(WORKING_OTP_SERVER),
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/api/analyze")
async def analyze_routes(
    start_lon: float = Query(...),
    start_lat: float = Query(...),
    end_lon: float = Query(...),
    end_lat: float = Query(...),
    departure_time: str = Query("now")
):
    """Analyze routes"""
    
    if not (-180 <= start_lon <= 180 and -90 <= start_lat <= 90):
        raise HTTPException(status_code=400, detail="Invalid start coordinates")
    if not (-180 <= end_lon <= 180 and -90 <= end_lat <= 90):
        raise HTTPException(status_code=400, detail="Invalid end coordinates")
    
    try:
        result = analyze_bike_bus_bike_routes(
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

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting OSRM + OTP API...")
    
    # Test OTP servers
    working_server, working_router = find_working_otp_server()
    
    if working_server:
        logger.info(f"Found working OTP server: {working_server}")
    else:
        logger.warning("No working OTP servers found")
    
    logger.info("API ready")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting server on port {port}")
    logger.info(f"OSRM Server: {OSRM_SERVER}")
    
    uvicorn.run(
        "complete_fixed_osrm_otp_planner:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
