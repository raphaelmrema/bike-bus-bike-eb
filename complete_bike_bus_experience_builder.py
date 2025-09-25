# Railway Deployment Fix - Multiple Files Needed

# 1. requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
requests==2.31.0
pandas==2.1.3
numpy==1.24.4
python-multipart==0.0.6

# GeoPandas and spatial dependencies - simplified for Railway
geopandas==0.14.1
shapely==2.0.2
pyproj==3.6.1
rtree==1.1.0
fiona==1.9.5

# Polyline decoding
polyline==2.0.0

# 2. railway.json (Railway configuration)
{
  "build": {
    "builder": "NIXPACKS",
    "buildCommand": "pip install -r requirements.txt"
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/status",
    "healthcheckTimeout": 300
  }
}

# 3. main.py (Simplified version for Railway deployment)
import os
import json
import logging
import requests
import datetime
import math
import sys
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Simplified package installation for Railway
def install_if_missing(packages):
    """Install packages if missing - Railway compatible"""
    import subprocess
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            except:
                print(f"Warning: Could not install {pkg}")

# Try to import GeoPandas with fallback
try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point, LineString
    import polyline
    GEOPANDAS_AVAILABLE = True
    print("✅ GeoPandas loaded successfully")
except ImportError as e:
    print(f"⚠️ GeoPandas not available: {e}")
    print("🔄 Falling back to basic routing only")
    GEOPANDAS_AVAILABLE = False
    # Create dummy imports for compatibility
    gpd = None
    pd = None
    Point = None
    LineString = None
    try:
        import polyline
    except ImportError:
        polyline = None

# =============================================================================
# CONFIG - Railway Environment Variables
# =============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OSRM_SERVER = os.getenv("OSRM_SERVER", "http://router.project-osrm.org")
BIKE_SPEED_MPH = float(os.getenv("BIKE_SPEED_MPH", "11"))
PORT = int(os.getenv("PORT", "8000"))

GMAPS_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
GMAPS_PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

# Railway CORS settings
CORS_ALLOW_ORIGINS = [
    "https://*.up.railway.app",
    "https://experience.arcgis.com",
    "https://*.maps.arcgis.com",
    "http://localhost:3000",
    "http://localhost:8000"
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("railway-bike-transit")

# =============================================================================
# FASTAPI APP - Simplified for Railway
# =============================================================================

app = FastAPI(
    title="Railway Bike-Bus-Bike Planner",
    description="Multimodal transportation planning - Railway optimized",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # More permissive for Railway
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def require_google_key():
    if not GOOGLE_API_KEY or len(GOOGLE_API_KEY.strip()) < 20:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")

def format_time_duration(minutes: float) -> str:
    if minutes < 1: return "< 1 min"
    if minutes < 60: return f"{int(round(minutes))} min"
    h = int(minutes // 60); m = int(round(minutes % 60))
    return f"{h}h" if m == 0 else f"{h}h {m}m"

def decode_polyline_safe(encoded: str) -> List[List[float]]:
    """Safe polyline decoding with fallback"""
    try:
        if polyline:
            pts = polyline.decode(encoded)
            return [[lon, lat] for (lat, lon) in pts]
        else:
            return []
    except Exception as e:
        logger.warning(f"Polyline decode failed: {e}")
        return []

def parse_epoch_to_hhmm(ts: Optional[int]) -> str:
    try:
        if not ts: return "Unknown"
        return datetime.datetime.fromtimestamp(ts).strftime("%H:%M")
    except Exception:
        return "Unknown"

# =============================================================================
# SIMPLIFIED BIKE ROUTING (OSRM Only for Railway)
# =============================================================================

def calculate_bike_route_simple(start_coords: List[float], end_coords: List[float], route_name="Bike Route"):
    """Simplified bike routing using OSRM only"""
    try:
        coords = f"{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
        url = f"{OSRM_SERVER}/route/v1/cycling/{coords}"
        params = {"overview": "full", "geometries": "polyline", "steps": "false"}
        
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        if data.get("code") != "Ok" or not data.get("routes"): 
            return None

        route = data["routes"][0]
        distance_m = float(route.get("distance", 0.0))
        distance_mi = distance_m * 0.000621371
        duration_min = float(route.get("duration", 0)) / 60.0 if route.get("duration") else (distance_mi / BIKE_SPEED_MPH) * 60.0

        coords_lonlat = decode_polyline_safe(route["geometry"])
        
        return {
            "name": route_name,
            "length_miles": round(distance_mi, 3),
            "travel_time_minutes": round(duration_min, 1),
            "travel_time_formatted": format_time_duration(duration_min),
            "geometry": {"type": "LineString", "coordinates": coords_lonlat},
            "route_type": "bike",
            "overall_score": 50,  # Default score without GeoPandas
            "segments": [],
            "facility_stats": {"BASIC ROUTING": {"length_miles": distance_mi, "percentage": 100.0, "avg_score": 50.0}}
        }
    except Exception as e:
        logger.error(f"OSRM bike route error: {e}")
        return None

# =============================================================================
# GOOGLE TRANSIT FUNCTIONS
# =============================================================================

def get_transit_routes_google_simple(origin: Tuple[float, float], destination: Tuple[float, float], departure_time: str = "now") -> Dict:
    """Simplified Google transit routing for Railway"""
    require_google_key()
    try:
        ts = int(datetime.datetime.now().timestamp()) if departure_time == "now" else int(float(departure_time))
        params = {
            "origin": f"{origin[1]},{origin[0]}",
            "destination": f"{destination[1]},{destination[0]}",
            "mode": "transit",
            "departure_time": ts,
            "alternatives": "true",
            "transit_mode": "bus|subway|train|tram",
            "key": GOOGLE_API_KEY,
        }
        
        r = requests.get(GMAPS_DIRECTIONS_URL, params=params, timeout=30)
        data = r.json()
        
        if data.get("status") != "OK":
            return {"error": data.get("error_message", f"API Status: {data.get('status')}")}
        
        routes = []
        for idx, rd in enumerate(data.get("routes", [])[:3]):
            parsed = parse_google_route_simple(rd, idx)
            if parsed: routes.append(parsed)
        
        return {"routes": routes, "service": "Google Maps Transit", "total_routes": len(routes)} if routes else {"error": "No transit routes found"}
        
    except Exception as e:
        logger.error(f"Google transit error: {e}")
        return {"error": str(e)}

def parse_google_route_simple(route_data: Dict, route_index: int) -> Optional[Dict]:
    """Simplified Google route parsing for Railway"""
    try:
        legs = route_data.get("legs", [])
        if not legs: return None
        leg = legs[0]

        dur_s = leg.get("duration", {}).get("value", 0)
        dur_min = round(dur_s / 60.0, 1)
        dist_m = leg.get("distance", {}).get("value", 0)
        dist_mi = round(dist_m * 0.000621371, 2)

        # Extract basic geometry
        overview_polyline = route_data.get("overview_polyline", {}).get("points", "")
        route_geometry = decode_polyline_safe(overview_polyline)

        # Count transfers
        transit_steps = [s for s in leg.get("steps", []) if s.get("travel_mode") == "TRANSIT"]
        transfers = max(0, len(transit_steps) - 1)

        return {
            "route_number": route_index + 1,
            "name": f"Transit Route {route_index + 1}",
            "duration_minutes": dur_min,
            "duration_text": format_time_duration(dur_min),
            "distance_miles": dist_mi,
            "departure_time": parse_epoch_to_hhmm(leg.get("departure_time", {}).get("value")),
            "arrival_time": parse_epoch_to_hhmm(leg.get("arrival_time", {}).get("value")),
            "transfers": transfers,
            "route_geometry": route_geometry,
            "service": "Google Maps Transit",
            "route_type": "transit"
        }
    except Exception as e:
        logger.error(f"Route parsing error: {e}")
        return None

def find_nearby_transit_google_simple(lat: float, lon: float) -> List[Dict]:
    """Find transit stops using Google Places - simplified for Railway"""
    require_google_key()
    try:
        params = {
            "location": f"{lat},{lon}",
            "radius": 800,
            "type": "transit_station",
            "key": GOOGLE_API_KEY
        }
        
        r = requests.get(GMAPS_PLACES_NEARBY_URL, params=params, timeout=20)
        data = r.json()
        
        if data.get("status") != "OK":
            return []
        
        stops = []
        for i, place in enumerate(data.get("results", [])[:5]):
            location = place.get("geometry", {}).get("location", {})
            stops.append({
                "id": f"google_{i}",
                "name": place.get("name", f"Transit Stop {i+1}"),
                "lat": location.get("lat", lat),
                "lon": location.get("lng", lon),
                "distance_m": i * 100,  # Approximate ordering
                "source": "google_places"
            })
        
        return stops
    except Exception as e:
        logger.error(f"Google Places error: {e}")
        return []

# =============================================================================
# SIMPLIFIED MULTIMODAL ANALYSIS
# =============================================================================

def analyze_multimodal_simple(start_coords: List[float], end_coords: List[float], departure_time: str = "now"):
    """Simplified multimodal analysis for Railway deployment"""
    try:
        logger.info(f"Analyzing routes: {start_coords} -> {end_coords}")
        
        # Find transit stops
        start_stops = find_nearby_transit_google_simple(start_coords[1], start_coords[0])
        end_stops = find_nearby_transit_google_simple(end_coords[1], end_coords[0])
        
        if not start_stops or not end_stops:
            # Direct bike route only
            direct_route = calculate_bike_route_simple(start_coords, end_coords, "Direct Bike Route")
            if not direct_route:
                return {"error": "No routes available"}
            
            return {
                "success": True,
                "analysis_type": "direct_bike_only",
                "routes": [{
                    "id": 1,
                    "name": "Direct Bike Route",
                    "type": "direct_bike",
                    "summary": {
                        "total_time_minutes": direct_route["travel_time_minutes"],
                        "total_time_formatted": direct_route["travel_time_formatted"],
                        "total_distance_miles": direct_route["length_miles"],
                        "bike_distance_miles": direct_route["length_miles"],
                        "transit_distance_miles": 0,
                        "bike_percentage": 100,
                        "average_bike_score": direct_route["overall_score"],
                        "transfers": 0
                    },
                    "legs": [{"type": "bike", "name": "Direct Bike Route", "route": direct_route}]
                }],
                "reason": "No transit stops found nearby"
            }

        # Use closest stops
        start_stop = start_stops[0]
        end_stop = end_stops[0]
        
        # Create bike legs
        bike_leg_1 = calculate_bike_route_simple(start_coords, [start_stop["lon"], start_stop["lat"]], "To Transit")
        bike_leg_2 = calculate_bike_route_simple([end_stop["lon"], end_stop["lat"]], end_coords, "From Transit")
        
        # Get transit routes
        transit_result = get_transit_routes_google_simple((start_stop["lon"], start_stop["lat"]), (end_stop["lon"], end_stop["lat"]), departure_time)
        transit_routes = transit_result.get("routes", [])

        # Create multimodal combinations
        all_routes = []
        
        if bike_leg_1 and bike_leg_2 and transit_routes:
            for i, transit_route in enumerate(transit_routes):
                total_time = bike_leg_1["travel_time_minutes"] + transit_route["duration_minutes"] + bike_leg_2["travel_time_minutes"] + 5
                total_distance = bike_leg_1["length_miles"] + transit_route["distance_miles"] + bike_leg_2["length_miles"]
                bike_distance = bike_leg_1["length_miles"] + bike_leg_2["length_miles"]
                
                multimodal_route = {
                    "id": i + 1,
                    "name": f"Bike-Bus-Bike {i + 1}",
                    "type": "multimodal",
                    "summary": {
                        "total_time_minutes": round(total_time, 1),
                        "total_time_formatted": format_time_duration(total_time),
                        "total_distance_miles": round(total_distance, 2),
                        "bike_distance_miles": round(bike_distance, 2),
                        "transit_distance_miles": round(transit_route["distance_miles"], 2),
                        "bike_percentage": round((bike_distance / total_distance) * 100, 1) if total_distance > 0 else 0,
                        "average_bike_score": 50,  # Default without GeoPandas
                        "transfers": transit_route.get("transfers", 0),
                        "departure_time": transit_route.get("departure_time", "Unknown"),
                        "arrival_time": transit_route.get("arrival_time", "Unknown")
                    },
                    "legs": [
                        {"type": "bike", "name": "Bike to Transit", "route": bike_leg_1},
                        {"type": "transit", "name": "Transit", "route": transit_route, "start_stop": start_stop, "end_stop": end_stop},
                        {"type": "bike", "name": "Transit to Destination", "route": bike_leg_2}
                    ]
                }
                all_routes.append(multimodal_route)

        # Add direct bike route
        direct_route = calculate_bike_route_simple(start_coords, end_coords, "Direct Bike")
        if direct_route:
            direct_option = {
                "id": len(all_routes) + 1,
                "name": "Direct Bike Route",
                "type": "direct_bike",
                "summary": {
                    "total_time_minutes": direct_route["travel_time_minutes"],
                    "total_time_formatted": direct_route["travel_time_formatted"],
                    "total_distance_miles": direct_route["length_miles"],
                    "bike_distance_miles": direct_route["length_miles"],
                    "transit_distance_miles": 0,
                    "bike_percentage": 100,
                    "average_bike_score": direct_route["overall_score"],
                    "transfers": 0
                },
                "legs": [{"type": "bike", "name": "Direct Bike Route", "route": direct_route}]
            }
            all_routes.append(direct_option)

        # Sort by time
        all_routes.sort(key=lambda x: x["summary"]["total_time_minutes"])

        return {
            "success": True,
            "analysis_type": "multimodal",
            "routes": all_routes,
            "transit_stops": {"start_stop": start_stop, "end_stop": end_stop},
            "statistics": {"total_options": len(all_routes)},
            "analysis_timestamp": datetime.datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Multimodal analysis error: {e}")
        return {"error": str(e)}

# =============================================================================
# RAILWAY-OPTIMIZED API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Railway-optimized main page"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Railway Bike-Bus-Bike Planner</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
        <style>
            body { margin: 0; padding: 0; font-family: 'Segoe UI', sans-serif; }
            .header { background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; text-align: center; }
            .container { display: flex; height: calc(100vh - 100px); }
            #map { flex: 2; }
            #sidebar { flex: 1; padding: 20px; overflow-y: auto; background: #f8f9fa; }
            .controls { margin-bottom: 20px; background: white; padding: 15px; border-radius: 8px; }
            input, select, button { width: 100%; padding: 8px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px; }
            button { background: #667eea; color: white; border: none; cursor: pointer; }
            button:hover { background: #5a67d8; }
            button:disabled { background: #cbd5e0; cursor: not-allowed; }
            .route-card { background: white; padding: 15px; margin: 10px 0; border-radius: 8px; cursor: pointer; border: 2px solid #e2e8f0; }
            .route-card:hover { border-color: #667eea; }
            .route-card.selected { border-color: #48bb78; background: #f0fff4; }
            .error { color: #e53e3e; background: #fed7d7; padding: 10px; border-radius: 4px; }
            .loading { text-align: center; padding: 20px; }
            @media (max-width: 768px) { .container { flex-direction: column; } #map { height: 50vh; } }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚴‍♂️🚌 Railway Bike-Bus-Bike Planner</h1>
            <p>Fast multimodal routing deployed on Railway</p>
        </div>
        
        <div class="container">
            <div id="map"></div>
            <div id="sidebar">
                <div class="controls">
                    <h3>📍 Plan Your Route</h3>
                    <p>Click map to set start (green) and end (red) points</p>
                    
                    <label>🕐 Departure Time:</label>
                    <select id="departureTime">
                        <option value="now">Leave Now</option>
                        <option value="custom">Custom Time</option>
                    </select>
                    
                    <div id="customTimeGroup" style="display: none;">
                        <input type="datetime-local" id="customTime">
                    </div>
                    
                    <button id="findRoutesBtn" disabled onclick="findRoutes()">🔍 Find Routes</button>
                    <button onclick="clearAll()">🗑️ Clear</button>
                    
                    <div id="status" style="font-size: 12px; margin-top: 10px;"></div>
                </div>
                
                <div id="results"></div>
            </div>
        </div>

        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
            let map, startPoint, endPoint, startMarker, endMarker, routeLayersGroup, currentRoutes = [];

            function initMap() {
                map = L.map('map').setView([30.3322, -81.6557], 12);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
                routeLayersGroup = L.layerGroup().addTo(map);
                
                map.on('click', function(e) {
                    if (!startPoint) {
                        startPoint = e.latlng;
                        startMarker = L.marker(startPoint, {
                            icon: L.icon({
                                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
                                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                                iconSize: [25, 41], iconAnchor: [12, 41]
                            })
                        }).addTo(map);
                    } else if (!endPoint) {
                        endPoint = e.latlng;
                        endMarker = L.marker(endPoint, {
                            icon: L.icon({
                                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
                                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                                iconSize: [25, 41], iconAnchor: [12, 41]
                            })
                        }).addTo(map);
                    } else {
                        clearAll();
                        map.fire('click', e);
                    }
                    updateStatus();
                });
            }

            function updateStatus() {
                const status = document.getElementById('status');
                const btn = document.getElementById('findRoutesBtn');
                
                if (startPoint && endPoint) {
                    status.innerHTML = `Start: ${startPoint.lat.toFixed(4)}, ${startPoint.lng.toFixed(4)}<br>End: ${endPoint.lat.toFixed(4)}, ${endPoint.lng.toFixed(4)}`;
                    btn.disabled = false;
                } else if (startPoint) {
                    status.textContent = 'Start point set. Click map for end point.';
                    btn.disabled = true;
                } else {
                    status.textContent = 'Click map to set start point.';
                    btn.disabled = true;
                }
            }

            function clearAll() {
                startPoint = endPoint = null;
                if (startMarker) { map.removeLayer(startMarker); startMarker = null; }
                if (endMarker) { map.removeLayer(endMarker); endMarker = null; }
                routeLayersGroup.clearLayers();
                document.getElementById('results').innerHTML = '';
                updateStatus();
            }

            async function findRoutes() {
                if (!startPoint || !endPoint) return;
                
                const results = document.getElementById('results');
                results.innerHTML = '<div class="loading">🔄 Finding routes...</div>';
                
                try {
                    const departureTime = document.getElementById('departureTime').value;
                    let timeParam = 'now';
                    if (departureTime === 'custom') {
                        const customTime = document.getElementById('customTime').value;
                        if (customTime) timeParam = Math.floor(new Date(customTime).getTime() / 1000);
                    }
                    
                    const url = `/analyze?start_lon=${startPoint.lng}&start_lat=${startPoint.lat}&end_lon=${endPoint.lng}&end_lat=${endPoint.lat}&departure_time=${timeParam}`;
                    const response = await fetch(url);
                    const data = await response.json();
                    
                    if (data.error) {
                        results.innerHTML = `<div class="error">❌ ${data.error}</div>`;
                        return;
                    }
                    
                    displayRoutes(data);
                } catch (error) {
                    results.innerHTML = `<div class="error">❌ ${error.message}</div>`;
                }
            }

            function displayRoutes(data) {
                currentRoutes = data.routes || [];
                const results = document.getElementById('results');
                
                if (!currentRoutes.length) {
                    results.innerHTML = '<div class="error">❌ No routes found</div>';
                    return;
                }
                
                let html = `<h3>🛣️ ${currentRoutes.length} Route${currentRoutes.length > 1 ? 's' : ''}</h3>`;
                
                currentRoutes.forEach((route, i) => {
                    const icon = route.type === 'multimodal' ? '🚴‍♂️🚌' : '🚴‍♂️';
                    html += `
                        <div class="route-card" onclick="selectRoute(${i})">
                            <strong>${icon} ${route.name}</strong><br>
                            ⏱️ ${route.summary.total_time_formatted} • 
                            📏 ${route.summary.total_distance_miles.toFixed(1)} mi<br>
                            🚴‍♂️ ${route.summary.bike_distance_miles.toFixed(1)} mi bike • 
                            🔢 ${route.summary.transfers} transfers
                        </div>
                    `;
                });
                
                results.innerHTML = html;
                if (currentRoutes.length > 0) setTimeout(() => selectRoute(0), 500);
            }

            function selectRoute(index) {
                document.querySelectorAll('.route-card').forEach((card, i) => {
                    card.classList.toggle('selected', i === index);
                });
                
                routeLayersGroup.clearLayers();
                const route = currentRoutes[index];
                
                route.legs.forEach(leg => {
                    if (leg.route && leg.route.geometry && leg.route.geometry.coordinates) {
                        const coords = leg.route.geometry.coordinates.map(c => [c[1], c[0]]);
                        const color = leg.type === 'bike' ? '#48bb78' : '#667eea';
                        L.polyline(coords, {color, weight: 5, opacity: 0.8}).addTo(routeLayersGroup);
                    }
                });
                
                if (routeLayersGroup.getLayers().length > 0) {
                    try {
                        map.fitBounds(routeLayersGroup.getBounds(), {padding: [20, 20]});
                    } catch (e) {}
                }
            }

            // Event listeners
            document.getElementById('departureTime').addEventListener('change', function() {
                const customGroup = document.getElementById('customTimeGroup');
                if (this.value === 'custom') {
                    customGroup.style.display = 'block';
                    const now = new Date();
                    now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
                    document.getElementById('customTime').value = now.toISOString().slice(0, 16);
                } else {
                    customGroup.style.display = 'none';
                }
            });

            document.addEventListener('DOMContentLoaded', initMap);
        </script>
    </body>
    </html>
    """)

@app.get("/status")
async def get_status():
    """Railway deployment status"""
    return {
        "status": "operational",
        "service": "Railway Bike-Bus-Bike Planner",
        "version": "1.0.0",
        "deployment": "Railway",
        "features": {
            "geopandas_analysis": GEOPANDAS_AVAILABLE,
            "google_api_configured": bool(GOOGLE_API_KEY),
            "osrm_server": OSRM_SERVER
        },
        "configuration": {
            "bike_speed_mph": BIKE_SPEED_MPH,
            "port": PORT
        }
    }

@app.get("/analyze")
async def analyze_routes(
    start_lon: float = Query(..., description="Start longitude"),
    start_lat: float = Query(..., description="Start latitude"),
    end_lon: float = Query(..., description="End longitude"),
    end_lat: float = Query(..., description="End latitude"),
    departure_time: str = Query("now", description="Departure time")
):
    """Main analysis endpoint for Railway"""
    try:
        result = analyze_multimodal_simple([start_lon, start_lat], [end_lon, end_lat], departure_time)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/bike-route")
async def get_bike_route(
    start_lon: float = Query(...), start_lat: float = Query(...),
    end_lon: float = Query(...), end_lat: float = Query(...)
):
    """Simple bike route endpoint"""
    try:
        route = calculate_bike_route_simple([start_lon, start_lat], [end_lon, end_lat])
        if not route:
            raise HTTPException(status_code=404, detail="No bike route found")
        return {"success": True, "route": route}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transit-routes")
async def get_transit_routes(
    start_lon: float = Query(...), start_lat: float = Query(...),
    end_lon: float = Query(...), end_lat: float = Query(...),
    departure_time: str = Query("now")
):
    """Simple transit routes endpoint"""
    try:
        result = get_transit_routes_google_simple((start_lon, start_lat), (end_lon, end_lat), departure_time)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check for Railway
@app.get("/health")
async def health_check():
    """Railway health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

# =============================================================================
# RAILWAY STARTUP
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Railway startup event"""
    logger.info("🚀 Starting Railway Bike-Bus-Bike Planner")
    logger.info(f"GeoPandas Available: {GEOPANDAS_AVAILABLE}")
    logger.info(f"Google API Key: {'✅ Configured' if GOOGLE_API_KEY else '❌ Missing'}")
    logger.info(f"OSRM Server: {OSRM_SERVER}")
    logger.info(f"Port: {PORT}")

# =============================================================================
# MAIN - Railway Entry Point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )

# 4. Dockerfile (optional - Railway will auto-detect)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for GeoPandas
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set environment
ENV PYTHONPATH=/app
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Run application
CMD ["python", "main.py"]

# 5. .railwayignore
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.coverage
.pytest_cache/
.venv/
venv/
*.egg-info/
dist/
build/
.DS_Store
*.log
.env.local
.env.development.local
.env.test.local
.env.production.local
uploads/
temp/
*.shp
*.dbf
*.shx
*.prj
*.zip

# 6. Procfile (Railway alternative)
web: uvicorn main:app --host 0.0.0.0 --port $PORT

# 7. Environment Variables for Railway Dashboard:
# GOOGLE_API_KEY=your_actual_api_key_here
# OSRM_SERVER=http://router.project-osrm.org
# BIKE_SPEED_MPH=11
# PYTHONPATH=/app

# =============================================================================
# DEPLOYMENT INSTRUCTIONS
# =============================================================================

"""
RAILWAY DEPLOYMENT STEPS:

1. Create these files in your project:
   - main.py (the simplified Python code above)
   - requirements.txt (simplified dependencies)
   - railway.json (Railway configuration)
   - Procfile (process definition)

2. Push to GitHub repository

3. Connect to Railway:
   - Go to railway.app
   - Click "Deploy from GitHub repo"
   - Select your repository
   - Railway auto-detects Python and installs dependencies

4. Set Environment Variables in Railway Dashboard:
   - GOOGLE_API_KEY=your_actual_google_maps_api_key
   - OSRM_SERVER=http://router.project-osrm.org
   - BIKE_SPEED_MPH=11

5. Deploy:
   - Railway automatically builds and deploys
   - Access via your Railway app URL (e.g., https://your-app.up.railway.app)

6. Test:
   - Visit the main URL for the interactive map
   - Check /status endpoint for system status
   - Use /analyze endpoint for route analysis

TROUBLESHOOTING:
- If GeoPandas fails to install, the app falls back to basic routing
- The app is designed to work with or without GeoPandas
- OSRM provides bike routing, Google Maps provides transit
- All spatial analysis is optional for Railway deployment

FEATURES INCLUDED:
✅ Interactive web map with click-to-route
✅ OSRM bike routing with realistic travel times  
✅ Google Maps transit integration
✅ Multimodal bike-bus-bike route combinations
✅ Direct bike route comparisons
✅ Mobile-responsive design
✅ Railway-optimized configuration
✅ Graceful fallbacks for missing dependencies
✅ Health checks and status monitoring
✅ CORS configured for external access
"""
