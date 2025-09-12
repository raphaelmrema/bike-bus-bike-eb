# app.py - Railway Deployment Ready
# OSRM + Google Maps Multi-Route Bike-Bus-Bike Planner

import os
import json
import math
import requests
import datetime
import random
from typing import List, Dict, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
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

# Environment variables for Railway
GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "YOUR_GOOGLE_MAPS_API_KEY_HERE")
OSRM_SERVER = os.getenv("OSRM_SERVER", "http://router.project-osrm.org")
BIKE_SPEED_MPH = int(os.getenv("BIKE_SPEED_MPH", "11"))
PORT = int(os.getenv("PORT", "8000"))

# =============================================================================
# FASTAPI APP SETUP
# =============================================================================

app = FastAPI(
    title="OSRM + Google Maps Route Planner",
    version="1.0.0",
    description="Multi-route bike and transit planning tool"
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
        return f"{hours}h {mins}m" if mins > 0 else f"{hours}h"

def is_valid_google_api_key():
    """Check if Google Maps API key is valid"""
    invalid_keys = [
        "YOUR_GOOGLE_MAPS_API_KEY_HERE",
        "AIzaSyBmGvPmWyKlR5DAtOu8vrmO0Cd8N1y4KF8",  # Demo key
        "",
        None
    ]
    return GOOGLE_API_KEY and GOOGLE_API_KEY not in invalid_keys and len(GOOGLE_API_KEY) > 30

def decode_polyline(polyline_str):
    """Decode Google polyline string to coordinates"""
    try:
        decoded = polyline.decode(polyline_str)
        return [[lon, lat] for (lat, lon) in decoded]
    except:
        return []

# =============================================================================
# OSRM BICYCLE ROUTING
# =============================================================================

def generate_alternative_waypoints(start_coords, end_coords, num_alternatives=3):
    """Generate waypoints for alternative routes"""
    try:
        start_lon, start_lat = start_coords
        end_lon, end_lat = end_coords

        center_lon = (start_lon + end_lon) / 2
        center_lat = (start_lat + end_lat) / 2
        distance = math.sqrt((end_lon - start_lon)**2 + (end_lat - start_lat)**2)
        variation_radius = distance * 0.3

        waypoint_sets = []
        for i in range(num_alternatives):
            angle = (i * 2 * math.pi / num_alternatives) + random.uniform(-0.5, 0.5)
            radius = variation_radius * random.uniform(0.4, 1.2)
            waypoint_lon = center_lon + radius * math.cos(angle)
            waypoint_lat = center_lat + radius * math.sin(angle)
            waypoint_sets.append([waypoint_lon, waypoint_lat])

        return waypoint_sets
    except:
        return []

def create_osrm_bike_route(start_coords, end_coords, waypoints=None, route_name="Bike Route"):
    """Create bike route using OSRM"""
    try:
        # Build coordinates string
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
            return None

        route = data["routes"][0]

        # Extract geometry
        geometry_polyline = route["geometry"]
        route_geometry = {
            "type": "LineString",
            "coordinates": decode_polyline(geometry_polyline)
        }

        # Distance and duration
        distance_meters = float(route.get("distance", 0.0))
        distance_miles = distance_meters * 0.000621371

        # Use OSRM duration if available, otherwise calculate from speed
        osrm_duration_sec = route.get("duration")
        if osrm_duration_sec and osrm_duration_sec > 0:
            travel_time_minutes = float(osrm_duration_sec) / 60.0
        else:
            travel_time_minutes = (distance_miles / BIKE_SPEED_MPH) * 60.0

        # Simple scoring based on route characteristics
        base_score = 70
        if distance_miles > 5:
            base_score += 5  # Longer routes often use better infrastructure
        if waypoints:
            base_score += 10  # Alternative routes might avoid main roads

        return {
            "name": route_name,
            "length_feet": distance_meters * 3.28084,
            "length_miles": distance_miles,
            "travel_time_minutes": travel_time_minutes,
            "travel_time_formatted": format_time_duration(travel_time_minutes),
            "geometry": route_geometry,
            "overall_score": min(100, max(20, base_score)),
            "waypoints": waypoints or [],
        }

    except Exception as e:
        print(f"Error creating OSRM route: {e}")
        return None

def create_multiple_bike_routes(start_coords, end_coords, num_routes=4):
    """Create multiple alternative bike routes"""
    routes = []

    # Direct route first
    direct_route = create_osrm_bike_route(start_coords, end_coords, None, "Direct Route")
    if direct_route:
        direct_route['route_type'] = 'direct'
        routes.append(direct_route)

    # Generate alternative routes
    if num_routes > 1:
        waypoint_sets = generate_alternative_waypoints(start_coords, end_coords, num_routes - 1)
        route_names = ["Scenic Route", "Alternative Route", "Extended Route", "Northern Route", "Southern Route"]

        for i, waypoints in enumerate(waypoint_sets):
            route_name = route_names[i] if i < len(route_names) else f"Route {i + 2}"
            alt_route = create_osrm_bike_route(start_coords, end_coords, [waypoints], route_name)
            if alt_route:
                alt_route['route_type'] = 'alternative'
                routes.append(alt_route)

    # Sort by score then distance
    routes.sort(key=lambda r: (-r.get('overall_score', 0), r.get('length_miles', 999)))
    return routes

# =============================================================================
# GOOGLE MAPS TRANSIT ROUTING
# =============================================================================

def get_google_transit_routes(origin, destination, departure_time="now"):
    """Get transit routes using Google Maps API"""
    try:
        if not is_valid_google_api_key():
            return {"error": "Google Maps API key not configured"}

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
            'key': GOOGLE_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'OK':
            return {"error": data.get('error_message', f"API Status: {data.get('status')}")}
        
        routes = []
        if 'routes' in data and data['routes']:
            for idx, route_data in enumerate(data['routes']):
                route = parse_google_transit_route(route_data, idx)
                if route:
                    routes.append(route)
        
        return {"routes": routes, "service": "Google Maps"}
        
    except Exception as e:
        return {"error": str(e)}

def parse_google_transit_route(route_data, route_index):
    """Parse Google transit route"""
    try:
        legs = route_data.get('legs', [])
        if not legs:
            return None
        
        leg = legs[0]
        
        duration_seconds = leg['duration']['value']
        duration_minutes = round(duration_seconds / 60, 1)
        
        distance_meters = leg['distance']['value']
        distance_miles = round(distance_meters * 0.000621371, 2)
        
        departure_time = leg.get('departure_time', {})
        arrival_time = leg.get('arrival_time', {})
        
        # Extract route geometry
        overview_polyline = route_data.get('overview_polyline', {}).get('points', '')
        route_geometry = {
            "type": "LineString",
            "coordinates": decode_polyline(overview_polyline) if overview_polyline else []
        }
        
        # Parse steps for transit details
        steps = []
        transit_lines = []
        
        for step_idx, step in enumerate(leg.get('steps', [])):
            parsed_step = parse_transit_step(step, step_idx)
            if parsed_step:
                steps.append(parsed_step)
                if parsed_step['travel_mode'] == 'TRANSIT':
                    if parsed_step.get('transit_line'):
                        transit_lines.append(parsed_step['transit_line'])
        
        # Count transfers
        transit_steps = [s for s in steps if s['travel_mode'] == 'TRANSIT']
        transfers = max(0, len(transit_steps) - 1)
        
        return {
            "route_number": route_index + 1,
            "name": f"Transit Route {route_index + 1}",
            "duration_minutes": duration_minutes,
            "duration_text": format_time_duration(duration_minutes),
            "distance_miles": distance_miles,
            "departure_time": departure_time.get('text', 'Unknown'),
            "arrival_time": arrival_time.get('text', 'Unknown'),
            "transfers": transfers,
            "transit_lines": list(set(transit_lines)),
            "steps": steps,
            "geometry": route_geometry,
        }
        
    except:
        return None

def parse_transit_step(step, step_index):
    """Parse individual transit step"""
    try:
        travel_mode = step.get('travel_mode', 'UNKNOWN')
        
        # Clean HTML instructions
        instruction = step.get('html_instructions', '')
        import re
        instruction = re.sub(r'<[^>]+>', '', instruction)
        
        step_data = {
            "step_number": step_index + 1,
            "travel_mode": travel_mode,
            "instruction": instruction,
            "duration_text": step['duration']['text'],
            "distance_miles": round(step['distance']['value'] * 0.000621371, 2)
        }
        
        if travel_mode == 'TRANSIT' and 'transit_details' in step:
            transit = step['transit_details']
            line = transit.get('line', {})
            
            step_data.update({
                "transit_line": line.get('short_name', line.get('name', 'Transit')),
                "departure_stop_name": transit.get('departure_stop', {}).get('name', 'Stop'),
                "arrival_stop_name": transit.get('arrival_stop', {}).get('name', 'Stop'),
                "scheduled_departure": transit.get('departure_time', {}).get('text', ''),
                "scheduled_arrival": transit.get('arrival_time', {}).get('text', ''),
            })
        
        return step_data
    except:
        return None

# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def create_simple_bus_stops(point_coords, max_stops=2):
    """Create simple mock bus stops for demonstration"""
    stops = []
    offsets = [0.004, 0.008, 0.012]
    names = ["Transit Center", "Main Station", "Local Stop"]
    
    for i in range(min(max_stops, 3)):
        offset = offsets[i]
        stops.append({
            "id": f"stop_{i+1}",
            "name": names[i],
            "display_x": point_coords[0] + offset,
            "display_y": point_coords[1] + offset,
            "distance_meters": (i + 1) * 500,
        })
    
    return stops

def analyze_multimodal_routes(start_point, end_point, departure_time="now", num_bike_routes=4):
    """Main analysis function"""
    try:
        routes = []
        
        # Create multiple bike routes
        bike_routes = create_multiple_bike_routes(start_point, end_point, num_bike_routes)
        
        # Add bike routes to results
        colors = ["#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#27ae60", "#e67e22"]
        for i, bike_route in enumerate(bike_routes):
            route_color = colors[i % len(colors)]
            
            route = {
                "id": len(routes) + 1,
                "name": bike_route['name'],
                "type": "direct_bike",
                "summary": {
                    "total_time_minutes": bike_route['travel_time_minutes'],
                    "total_time_formatted": bike_route['travel_time_formatted'],
                    "total_distance_miles": bike_route['length_miles'],
                    "bike_distance_miles": bike_route['length_miles'],
                    "bike_score": bike_route.get('overall_score', 70),
                    "transfers": 0,
                },
                "legs": [{
                    "type": "bike",
                    "name": bike_route['name'],
                    "route": bike_route,
                    "color": route_color
                }],
                "color": route_color
            }
            routes.append(route)
        
        # Add transit routes if Google Maps API is available
        if is_valid_google_api_key():
            # Create mock bus stops
            start_stops = create_simple_bus_stops(start_point, 2)
            end_stops = create_simple_bus_stops(end_point, 2)
            
            # Get direct transit routes
            origin_str = f"{start_point[1]},{start_point[0]}"
            dest_str = f"{end_point[1]},{end_point[0]}"
            
            transit_result = get_google_transit_routes(origin_str, dest_str, departure_time)
            
            if transit_result.get('routes'):
                for i, transit_route in enumerate(transit_result['routes']):
                    route = {
                        "id": len(routes) + 1,
                        "name": f"Transit Route {i + 1}",
                        "type": "transit_only",
                        "summary": {
                            "total_time_minutes": transit_route['duration_minutes'],
                            "total_time_formatted": transit_route['duration_text'],
                            "total_distance_miles": transit_route['distance_miles'],
                            "bike_distance_miles": 0,
                            "bike_score": 0,
                            "transfers": transit_route.get('transfers', 0),
                        },
                        "legs": [{
                            "type": "transit",
                            "name": f"Transit Route {i + 1}",
                            "route": transit_route,
                            "color": "#2196f3"
                        }]
                    }
                    routes.append(route)
        
        # Sort routes by total time
        routes.sort(key=lambda x: x['summary']['total_time_minutes'])
        
        # Calculate statistics
        bike_only = len([r for r in routes if r['type'] == 'direct_bike'])
        transit_only = len([r for r in routes if r['type'] == 'transit_only'])
        
        return {
            "success": True,
            "routes": routes,
            "statistics": {
                "total_options": len(routes),
                "bike_only_options": bike_only,
                "transit_only_options": transit_only,
                "fastest_time": routes[0]['summary']['total_time_formatted'] if routes else None
            },
            "google_maps_enabled": is_valid_google_api_key(),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# FASTAPI ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def get_home():
    """Serve the main UI"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Route Bike-Transit Planner</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
        <style>
            body { margin: 0; font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .header { background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 20px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
            .header h1 { margin: 0; font-size: 2.2em; font-weight: 300; }
            .container { display: flex; height: calc(100vh - 100px); max-width: 1400px; margin: 0 auto; background: white; border-radius: 10px 10px 0 0; overflow: hidden; box-shadow: 0 -5px 20px rgba(0,0,0,0.1); }
            #map { flex: 2.5; }
            .sidebar { flex: 1; max-width: 400px; padding: 20px; background: #f8f9fa; overflow-y: auto; border-left: 1px solid #dee2e6; }
            
            .status { padding: 12px; border-radius: 8px; margin-bottom: 15px; text-align: center; font-weight: 500; font-size: 0.9em; }
            .status-good { background: linear-gradient(135deg, #d4edda, #c3e6cb); color: #155724; }
            .status-limited { background: linear-gradient(135deg, #fff3cd, #ffeaa7); color: #856404; }
            
            .controls { background: white; padding: 15px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: 600; color: #333; }
            select { width: 100%; padding: 8px 12px; border: 2px solid #e1e5e9; border-radius: 6px; font-size: 1em; }
            select:focus { outline: none; border-color: #3498db; }
            
            button { background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 12px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 1em; font-weight: 600; width: 100%; margin: 5px 0; }
            button:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3); }
            button:disabled { background: #bdc3c7; cursor: not-allowed; }
            .btn-clear { background: linear-gradient(135deg, #e74c3c, #c0392b); }
            
            .coordinates { background: #f8f9fa; padding: 12px; border-radius: 6px; margin: 10px 0; border-left: 4px solid #6c757d; font-size: 13px; }
            .spinner { border: 3px solid rgba(52, 152, 219, 0.2); width: 30px; height: 30px; border-radius: 50%; border-left-color: #3498db; animation: spin 1s linear infinite; margin: 15px auto; display: none; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            
            .route-toggles { margin-bottom: 15px; }
            .route-toggle { cursor: pointer; margin: 5px 0; padding: 10px; background: #e9ecef; border-radius: 6px; display: flex; align-items: center; transition: all 0.2s; }
            .route-toggle.active { background: #007cba; color: white; }
            .route-toggle:hover { transform: translateY(-1px); }
            .route-color { width: 16px; height: 16px; border-radius: 3px; margin-right: 10px; }
            
            .route-card { background: white; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 3px 10px rgba(0,0,0,0.1); border: 2px solid #e9ecef; }
            .route-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
            .route-name { font-weight: 700; color: #2c3e50; font-size: 1.1em; }
            .route-type { padding: 4px 8px; border-radius: 12px; font-size: 0.75em; font-weight: 600; }
            .type-bike { background: #27ae60; color: white; }
            .type-transit { background: #2196f3; color: white; }
            
            .route-stats { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 10px; }
            .stat { text-align: center; padding: 8px; background: #f8f9fa; border-radius: 4px; }
            .stat-value { font-weight: bold; color: #3498db; display: block; }
            .stat-label { font-size: 0.75em; color: #6c757d; text-transform: uppercase; }
            
            .error { background: linear-gradient(135deg, #ffebee, #ffcdd2); color: #c62828; padding: 15px; border-radius: 8px; margin: 15px 0; }
            .hidden { display: none; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Multi-Route Bike-Transit Planner</h1>
            <p>OSRM bicycle routing + Google Maps transit integration</p>
        </div>
        
        <div class="container">
            <div id="map"></div>
            
            <div class="sidebar">
                <div class="status" id="status">System: Loading...</div>
                
                <div class="controls">
                    <div class="form-group">
                        <label for="routeCount">Number of Bike Routes:</label>
                        <select id="routeCount">
                            <option value="3">3 Routes</option>
                            <option value="4" selected>4 Routes</option>
                            <option value="5">5 Routes</option>
                        </select>
                    </div>
                    
                    <button id="findRoutesBtn" disabled onclick="findRoutes()">Find Routes</button>
                    <button class="btn-clear" onclick="clearAll()">Clear Map</button>
                    
                    <div class="coordinates">
                        <div><strong>Start:</strong> <span id="startCoords">Click map to select</span></div>
                        <div><strong>End:</strong> <span id="endCoords">Click map to select</span></div>
                    </div>
                </div>
                
                <div class="spinner" id="spinner"></div>
                <div id="routeToggles" class="route-toggles"></div>
                <div id="results"></div>
            </div>
        </div>

        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
            let map, startPoint = null, endPoint = null, startMarker = null, endMarker = null;
            let routeLayers = new Map();
            let currentRoutes = [];

            function initMap() {
                map = L.map('map').setView([30.3322, -81.6557], 12);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
                map.on('click', handleMapClick);
                loadSystemStatus();
            }

            async function loadSystemStatus() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    const statusDiv = document.getElementById('status');
                    
                    if (data.google_maps_enabled) {
                        statusDiv.className = 'status status-good';
                        statusDiv.innerHTML = 'System: OSRM + Google Maps Ready';
                    } else {
                        statusDiv.className = 'status status-limited';
                        statusDiv.innerHTML = 'System: OSRM Only (Bike routes available)';
                    }
                } catch (error) {
                    document.getElementById('status').innerHTML = 'System: Connection issues';
                }
            }

            function handleMapClick(e) {
                if (!startPoint) {
                    startPoint = e.latlng;
                    startMarker = L.marker(startPoint, {
                        icon: L.icon({
                            iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
                            iconSize: [25, 41], iconAnchor: [12, 41]
                        })
                    }).addTo(map).bindTooltip("Start", {permanent: true, direction: 'top'});
                    updateCoords();
                } else if (!endPoint) {
                    endPoint = e.latlng;
                    endMarker = L.marker(endPoint, {
                        icon: L.icon({
                            iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
                            iconSize: [25, 41], iconAnchor: [12, 41]
                        })
                    }).addTo(map).bindTooltip("End", {permanent: true, direction: 'top'});
                    updateCoords();
                } else {
                    clearAll();
                    handleMapClick(e);
                }
            }

            function updateCoords() {
                document.getElementById('startCoords').textContent = 
                    startPoint ? `${startPoint.lat.toFixed(4)}, ${startPoint.lng.toFixed(4)}` : 'Click map to select';
                document.getElementById('endCoords').textContent = 
                    endPoint ? `${endPoint.lat.toFixed(4)}, ${endPoint.lng.toFixed(4)}` : 'Click map to select';
                document.getElementById('findRoutesBtn').disabled = !(startPoint && endPoint);
            }

            function clearAll() {
                startPoint = null; endPoint = null;
                if (startMarker) map.removeLayer(startMarker);
                if (endMarker) map.removeLayer(endMarker);
                clearAllRoutes();
                document.getElementById('results').innerHTML = '';
                document.getElementById('routeToggles').innerHTML = '';
                updateCoords();
            }

            function clearAllRoutes() {
                routeLayers.forEach(layer => map.removeLayer(layer));
                routeLayers.clear();
            }

            function toggleRoute(routeId) {
                const layer = routeLayers.get(routeId);
                if (layer) {
                    if (map.hasLayer(layer)) {
                        map.removeLayer(layer);
                    } else {
                        map.addLayer(layer);
                    }
                    updateToggleButtons();
                }
            }

            function updateToggleButtons() {
                document.querySelectorAll('.route-toggle').forEach(toggle => {
                    const routeId = parseInt(toggle.dataset.routeId);
                    const layer = routeLayers.get(routeId);
                    toggle.classList.toggle('active', layer && map.hasLayer(layer));
                });
            }

            function showSpinner(show) {
                document.getElementById('spinner').style.display = show ? 'block' : 'none';
                document.getElementById('findRoutesBtn').disabled = show;
                document.getElementById('findRoutesBtn').innerHTML = show ? 'Finding Routes...' : 'Find Routes';
            }

            async function findRoutes() {
                if (!startPoint || !endPoint) return;

                const routeCount = document.getElementById('routeCount').value;
                showSpinner(true);
                clearAllRoutes();

                try {
                    const params = new URLSearchParams({
                        start_lon: startPoint.lng,
                        start_lat: startPoint.lat,
                        end_lon: endPoint.lng,
                        end_lat: endPoint.lat,
                        num_routes: routeCount
                    });

                    const response = await fetch(`/api/analyze?${params}`);
                    const data = await response.json();

                    if (data.success) {
                        displayResults(data);
                    } else {
                        document.getElementById('results').innerHTML = `<div class="error">Error: ${data.error}</div>`;
                    }
                } catch (error) {
                    document.getElementById('results').innerHTML = `<div class="error">Error: ${error.message}</div>`;
                }

                showSpinner(false);
            }

            function displayResults(data) {
                currentRoutes = data.routes;
                clearAllRoutes();

                // Create route toggles
                let toggleHtml = '<h4>Route Visibility</h4>';
                data.routes.forEach((route) => {
                    const routeColor = route.color || '#3498db';
                    toggleHtml += `<div class="route-toggle active" data-route-id="${route.id}" onclick="toggleRoute(${route.id})">
                        <span class="route-color" style="background-color: ${routeColor};"></span>
                        <span>${route.name}</span>
                    </div>`;
                });
                document.getElementById('routeToggles').innerHTML = toggleHtml;

                // Display route cards
                let html = `<h4>Found ${data.routes.length} Routes</h4>`;
                data.routes.forEach((route) => {
                    html += createRouteCard(route);
                });
                document.getElementById('results').innerHTML = html;

                // Add routes to map
                data.routes.forEach((route) => {
                    addRouteToMap(route);
                });

                // Fit map bounds
                if (routeLayers.size > 0) {
                    const group = new L.FeatureGroup(Array.from(routeLayers.values()));
                    map.fitBounds(group.getBounds().pad(0.1));
                }
            }

            function createRouteCard(route) {
                const routeTypeClass = route.type === 'direct_bike' ? 'type-bike' : 'type-transit';
                const routeTypeText = route.type === 'direct_bike' ? 'BIKE' : 'TRANSIT';
                
                return `
                    <div class="route-card">
                        <div class="route-header">
                            <div class="route-name">${route.name}</div>
                            <span class="route-type ${routeTypeClass}">${routeTypeText}</span>
                        </div>
                        <div class="route-stats">
                            <div class="stat">
                                <span class="stat-value">${route.summary.total_time_formatted}</span>
                                <span class="stat-label">Time</span>
                            </div>
                            <div class="stat">
                                <span class="stat-value">${route.summary.total_distance_miles.toFixed(1)}</span>
                                <span class="stat-label">Miles</span>
                            </div>
                            <div class="stat">
                                <span class="stat-value">${route.summary.bike_score || 0}</span>
                                <span class="stat-label">Score</span>
                            </div>
                        </div>
                    </div>
                `;
            }

            function addRouteToMap(route) {
                const routeColor = route.color || '#3498db';
                
                route.legs.forEach((leg) => {
                    if (leg.route.geometry && leg.route.geometry.coordinates.length > 0) {
                        const coords = leg.route.geometry.coordinates.map(coord => [coord[1], coord[0]]);
                        const weight = leg.type === 'bike' ? 4 : 6;
                        const dashArray = leg.type === 'transit' ? '10, 5' : null;
                        
                        const routeLine = L.polyline(coords, {
                            color: leg.color || routeColor,
                            weight: weight,
                            opacity: 0.8,
                            dashArray: dashArray
                        });
                        
                        routeLine.bindPopup(`
                            <div>
                                <h4>${leg.name}</h4>
                                <p><strong>Distance:</strong> ${(leg.route.length_miles || leg.route.distance_miles || 0).toFixed(2)} miles</p>
                                <p><strong>Time:</strong> ${leg.route.travel_time_formatted || leg.route.duration_text || 'N/A'}</p>
                            </div>
                        `);
                        
                        if (!routeLayers.has(route.id)) {
                            routeLayers.set(route.id, L.layerGroup());
                        }
                        routeLayers.get(route.id).addLayer(routeLine);
                    }
                });
                
                if (routeLayers.has(route.id)) {
                    routeLayers.get(route.id).addTo(map);
                }
            }

            // Initialize
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
        "service": "Multi-Route Bike-Transit Planner",
        "version": "1.0.0",
        "google_maps_enabled": is_valid_google_api_key(),
        "osrm_server": OSRM_SERVER,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/api/analyze")
async def analyze_routes(
    start_lon: float = Query(...),
    start_lat: float = Query(...),
    end_lon: float = Query(...),
    end_lat: float = Query(...),
    num_routes: int = Query(4, ge=3, le=6),
    departure_time: str = Query("now")
):
    """Main route analysis endpoint"""
    
    # Validate coordinates
    if not (-180 <= start_lon <= 180 and -90 <= start_lat <= 90):
        raise HTTPException(status_code=400, detail="Invalid start coordinates")
    if not (-180 <= end_lon <= 180 and -90 <= end_lat <= 90):
        raise HTTPException(status_code=400, detail="Invalid end coordinates")
    
    try:
        result = analyze_multimodal_routes(
            [start_lon, start_lat],
            [end_lon, end_lat],
            departure_time,
            num_routes
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health endpoint alias
@app.get("/health")
async def health_alias():
    return await health_check()

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PORT,
        reload=False
    )
