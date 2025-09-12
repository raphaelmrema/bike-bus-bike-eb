# osrm_google_bbb_planner.py
# OSRM (bike) + Google Directions (transit) – Bike–Bus–Bike planner
# FastAPI app, Experience Builder–friendly, using env var GOOGLE_API_KEY

import os
import json
import logging
import requests
import datetime
import math
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Ensure polyline is available (for decoding OSRM/Google polylines)
try:
    import polyline
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "polyline"])
    import polyline

# =============================================================================
# CONFIG
# =============================================================================

# Read your Google API key from environment (DON'T hardcode in code/repos)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# OSRM bicycle routing server (public demo)
OSRM_SERVER = os.getenv("OSRM_SERVER", "http://router.project-osrm.org")
USE_OSRM_DURATION = True
BIKE_SPEED_MPH = float(os.getenv("BIKE_SPEED_MPH", "11"))

# Google endpoints
GMAPS_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
GMAPS_PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

# CORS (keep open for dev; lock down in prod)
CORS_ALLOW_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "*"
).split(",")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("osrm-google-bbb")

# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="OSRM + Google Transit Bike–Bus–Bike Planner",
    description="Multimodal routing with OSRM (bike) and Google Directions (transit), ready for ArcGIS Experience Builder.",
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS if CORS_ALLOW_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# UTILITIES
# =============================================================================

def require_google_key():
    if not GOOGLE_API_KEY or len(GOOGLE_API_KEY.strip()) < 20:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not set or invalid.")

def format_time_duration(minutes: float) -> str:
    if minutes < 1: return "< 1 min"
    if minutes < 60: return f"{int(round(minutes))} min"
    h = int(minutes // 60); m = int(round(minutes % 60))
    return f"{h}h" if m == 0 else f"{h}h {m}m"

def decode_polyline_to_lonlat(encoded: str) -> List[List[float]]:
    pts = polyline.decode(encoded)
    # polyline returns [(lat, lon)], we want [[lon, lat], ...] for GeoJSON/Leaflet handling consistency
    return [[lon, lat] for (lat, lon) in pts]

def parse_epoch_to_hhmm(ts: Optional[int]) -> str:
    try:
        if not ts: return "Unknown"
        return datetime.datetime.fromtimestamp(ts).strftime("%H:%M")
    except Exception:
        return "Unknown"

# =============================================================================
# OSRM BICYCLE ROUTING
# =============================================================================

def calculate_bike_route_osrm(start_coords: List[float], end_coords: List[float], route_name="Bike Route"):
    """start/end: [lon, lat]"""
    try:
        coords = f"{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
        url = f"{OSRM_SERVER}/route/v1/cycling/{coords}"
        params = {"overview": "full", "geometries": "polyline", "steps": "false", "alternatives": "false"}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "Ok" or not data.get("routes"): return None

        route = data["routes"][0]
        distance_m = float(route.get("distance", 0.0))
        distance_mi = distance_m * 0.000621371

        if USE_OSRM_DURATION and route.get("duration") is not None:
            duration_min = float(route["duration"]) / 60.0
        else:
            duration_min = (distance_mi / BIKE_SPEED_MPH) * 60.0

        coords_lonlat = decode_polyline_to_lonlat(route["geometry"])

        # Simple placeholder “score”; replace with your facility-based scoring if desired
        score = max(0, min(100, 70 + (10 if distance_mi < 2 else -5)))

        return {
            "name": route_name,
            "length_miles": round(distance_mi, 3),
            "travel_time_minutes": round(duration_min, 1),
            "travel_time_formatted": format_time_duration(duration_min),
            "geometry": {"type": "LineString", "coordinates": coords_lonlat},
            "overall_score": score,
        }
    except Exception as e:
        logger.error(f"OSRM error: {e}")
        return None

# =============================================================================
# GOOGLE TRANSIT (DIRECTIONS) + STOPS (PLACES)
# =============================================================================

def _gmaps_departure_timestamp(departure_time: str) -> int:
    if departure_time == "now":
        return int(datetime.datetime.now().timestamp())
    try:
        return int(float(departure_time))  # epoch seconds
    except Exception:
        return int(datetime.datetime.now().timestamp())

def get_transit_routes_google(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    departure_time: str = "now",
    max_alternatives: int = 3
) -> Dict:
    """origin/destination: (lon, lat)"""
    require_google_key()
    try:
        ts = _gmaps_departure_timestamp(departure_time)
        params = {
            "origin": f"{origin[1]},{origin[0]}",
            "destination": f"{destination[1]},{destination[0]}",
            "mode": "transit",
            "departure_time": ts,
            "alternatives": "true",
            "transit_mode": "bus|subway|train|tram",
            "transit_routing_preference": "fewer_transfers",
            "key": GOOGLE_API_KEY,
        }
        r = requests.get(GMAPS_DIRECTIONS_URL, params=params, timeout=30)
        data = r.json()
        if data.get("status") != "OK":
            return {"error": data.get("error_message", f"Google Directions status: {data.get('status')}")}
        routes = []
        for idx, rd in enumerate(data.get("routes", [])[:max_alternatives]):
            parsed = parse_google_transit_route(rd, idx)
            if parsed: routes.append(parsed)
        if not routes: return {"error": "No transit routes found"}
        return {"routes": routes, "service": "Google Maps Transit", "total_routes": len(routes)}
    except Exception as e:
        logger.error(f"Google Directions error: {e}")
        return {"error": str(e)}

def parse_google_transit_route(route_data: Dict, route_index: int) -> Optional[Dict]:
    try:
        legs = route_data.get("legs", [])
        if not legs: return None
        leg = legs[0]

        dur_s = leg.get("duration", {}).get("value", 0)
        dur_min = round(dur_s / 60.0, 1)
        dist_m = leg.get("distance", {}).get("value", 0)
        dist_mi = round(dist_m * 0.000621371, 2)
        dep_ts = leg.get("departure_time", {}).get("value")
        arr_ts = leg.get("arrival_time", {}).get("value")

        steps_raw = leg.get("steps", [])
        steps = []
        route_geometry: List[List[float]] = []
        transit_boardings = 0
        walk_m = 0.0

        for i, s in enumerate(steps_raw):
            ps = parse_transit_step(s, i)
            if ps:
                steps.append(ps)
                if ps["travel_mode"] == "TRANSIT":
                    transit_boardings += 1
                if ps["travel_mode"] == "WALKING":
                    # ps has distance_miles; recompute meters from raw
                    walk_m += s.get("distance", {}).get("value", 0)
            if s.get("polyline", {}).get("points"):
                route_geometry.extend(decode_polyline_to_lonlat(s["polyline"]["points"]))

        transfers = max(0, transit_boardings - 1)
        return {
            "route_number": route_index + 1,
            "name": f"Transit Route {route_index + 1}" + (f" ({transfers} transfers)" if transfers else ""),
            "duration_minutes": dur_min,
            "duration_text": format_time_duration(dur_min),
            "distance_miles": dist_mi,
            "departure_time": parse_epoch_to_hhmm(dep_ts),
            "arrival_time": parse_epoch_to_hhmm(arr_ts),
            "transfers": transfers,
            "walking_distance_miles": round(walk_m * 0.000621371, 2),
            "route_geometry": route_geometry,
            "steps": steps,
            "service": "Google Maps Transit",
        }
    except Exception as e:
        logger.error(f"Parse Google route error: {e}")
        return None

def parse_transit_step(step: Dict, index: int) -> Optional[Dict]:
    try:
        mode = step.get("travel_mode", "UNKNOWN").upper()
        dur_s = step.get("duration", {}).get("value", 0)
        dist_m = step.get("distance", {}).get("value", 0)
        base = {
            "step_number": index + 1,
            "travel_mode": mode,
            "duration_minutes": round(dur_s / 60.0, 1),
            "duration_text": format_time_duration(dur_s / 60.0),
            "distance_miles": round(dist_m * 0.000621371, 2),
        }
        if mode == "TRANSIT":
            td = step.get("transit_details", {}) or {}
            dep = td.get("departure_stop", {}) or {}
            arr = td.get("arrival_stop", {}) or {}
            line = td.get("line", {}) or {}
            base.update({
                "transit_line": line.get("short_name") or line.get("name") or "Transit",
                "departure_stop_name": dep.get("name", "Stop"),
                "departure_stop_location": dep.get("location", {}),
                "arrival_stop_name": arr.get("name", "Stop"),
                "arrival_stop_location": arr.get("location", {}),
                "headsign": td.get("headsign", ""),
                "num_stops": td.get("num_stops", 0),
            })
        return base
    except Exception as e:
        logger.error(f"Parse step error: {e}")
        return None

def find_nearby_transit_stations_google(point_coords: List[float], radius_meters: int = 800, max_results: int = 6):
    """point_coords: [lon, lat]"""
    require_google_key()
    try:
        params = {
            "location": f"{point_coords[1]},{point_coords[0]}",
            "radius": radius_meters,
            "type": "transit_station",
            "key": GOOGLE_API_KEY,
        }
        r = requests.get(GMAPS_PLACES_NEARBY_URL, params=params, timeout=15)
        data = r.json()
        status = data.get("status", "UNKNOWN")
        if status not in ("OK", "ZERO_RESULTS"):
            logger.warning(f"Places nearby status: {status} – {data.get('error_message')}")
            return []
        results = data.get("results", [])[:max_results]
        out = []
        for item in results:
            loc = (item.get("geometry", {}) or {}).get("location", {})
            out.append({
                "id": item.get("place_id", ""),
                "name": item.get("name", "Transit Station"),
                "x": loc.get("lng"),
                "y": loc.get("lat"),
                "display_x": loc.get("lng"),
                "display_y": loc.get("lat"),
            })
        return out
    except Exception as e:
        logger.error(f"Places Nearby error: {e}")
        return []

# =============================================================================
# ANALYSIS: Build Bike–Bus–Bike & fallbacks
# =============================================================================

def _euclid_meters(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    # (lon,lat) → rough meters
    dx = (a[0] - b[0]) * 111000 * math.cos(math.radians((a[1] + b[1]) / 2.0))
    dy = (a[1] - b[1]) * 111000
    return math.hypot(dx, dy)

def should_use_transit_fallback(start_point, end_point, start_stops, end_stops, threshold_m=400):
    try:
        if not start_stops or not end_stops: return False
        s = start_stops[0]; e = end_stops[0]
        d1 = _euclid_meters(tuple(start_point), (s["display_x"], s["display_y"]))
        d2 = _euclid_meters(tuple(end_point), (e["display_x"], e["display_y"]))
        return (d1 < threshold_m and d2 < threshold_m)
    except Exception:
        return False

def analyze_complete_bike_bus_bike_routes(start_point: List[float], end_point: List[float], departure_time="now"):
    """
    start_point / end_point: [lon, lat]
    Returns route options: bike–bus–bike, transit-only fallback, and direct bike comparator.
    """
    routes = []

    # Nearby stations
    start_stops = find_nearby_transit_stations_google(start_point, radius_meters=800, max_results=4)
    end_stops   = find_nearby_transit_stations_google(end_point,   radius_meters=800, max_results=4)

    # Transit fallback if both bike legs would be tiny
    fallback_used = should_use_transit_fallback(start_point, end_point, start_stops, end_stops)
    if fallback_used:
        tr = get_transit_routes_google(tuple(start_point), tuple(end_point), departure_time)
        if tr.get("routes"):
            for i, r in enumerate(tr["routes"]):
                routes.append({
                    "id": len(routes)+1, "name": f"Transit Option {i+1}",
                    "type": "transit_fallback",
                    "summary": {
                        "total_time_minutes": r["duration_minutes"],
                        "total_time_formatted": r["duration_text"],
                        "total_distance_miles": r["distance_miles"],
                        "bike_distance_miles": 0.0, "transit_distance_miles": r["distance_miles"],
                        "bike_percentage": 0.0, "average_bike_score": 0.0,
                        "transfers": r.get("transfers", 0), "total_fare": 0,
                        "departure_time": r.get("departure_time", "Unknown"),
                        "arrival_time": r.get("arrival_time", "Unknown"),
                    },
                    "legs": [{
                        "type": "transit", "name": f"Google Transit Route {i+1}",
                        "description": "Direct transit (smart fallback)",
                        "route": r, "color": "#2196f3", "order": 1
                    }],
                    "fallback_reason": "Both bike segments < 400m"
                })

    # Bike → Transit → Bike
    if start_stops and end_stops:
        s = start_stops[0]; e = end_stops[0]
        # ensure distinct
        if s["id"] == e["id"]:
            if len(end_stops) > 1: e = end_stops[1]
            elif len(start_stops) > 1: e = start_stops[1]
        if s["id"] != e["id"]:
            bike1 = calculate_bike_route_osrm(start_point, [s["display_x"], s["display_y"]], "Start to Station")
            bike2 = calculate_bike_route_osrm([e["display_x"], e["display_y"]], end_point, "Station to Destination")
            if bike1 and bike2:
                tr = get_transit_routes_google((s["display_x"], s["display_y"]), (e["display_x"], e["display_y"]), departure_time)
                if tr.get("routes"):
                    for i, r in enumerate(tr["routes"]):
                        total_bike_mi = bike1["length_miles"] + bike2["length_miles"]
                        total_transit_mi = r["distance_miles"]
                        total_mi = total_bike_mi + total_transit_mi
                        # +5 min buffer for transfers
                        total_time = bike1["travel_time_minutes"] + r["duration_minutes"] + bike2["travel_time_minutes"] + 5.0
                        # length-weighted bike score
                        bike_score = 0.0
                        if total_bike_mi > 0:
                            bike_score = (bike1["overall_score"]*bike1["length_miles"] + bike2["overall_score"]*bike2["length_miles"]) / total_bike_mi
                        r_enh = dict(r); r_enh["start_stop"] = s; r_enh["end_stop"] = e
                        routes.append({
                            "id": len(routes)+1, "name": f"Bike–Bus–Bike Option {i+1}", "type": "bike_bus_bike",
                            "summary": {
                                "total_time_minutes": round(total_time, 1),
                                "total_time_formatted": format_time_duration(total_time),
                                "total_distance_miles": round(total_mi, 2),
                                "bike_distance_miles": round(total_bike_mi, 2),
                                "transit_distance_miles": round(total_transit_mi, 2),
                                "bike_percentage": round((total_bike_mi/total_mi)*100, 1) if total_mi>0 else 0.0,
                                "average_bike_score": round(bike_score, 1),
                                "transfers": r.get("transfers", 0), "total_fare": 0,
                                "departure_time": r.get("departure_time", "Unknown"),
                                "arrival_time": r.get("arrival_time", "Unknown"),
                            },
                            "legs": [
                                {"type":"bike","name":"OSRM Bike to Station","description":f"Bike to {s['name']}","route":bike1,"color":"#27ae60","order":1},
                                {"type":"transit","name":"Google Transit","description":f"Transit {s['name']} → {e['name']}","route":r_enh,"color":"#3498db","order":2},
                                {"type":"bike","name":"OSRM Station to Destination","description":f"Bike from {e['name']}","route":bike2,"color":"#27ae60","order":3},
                            ]
                        })

    # Direct bike comparator
    direct_bike = calculate_bike_route_osrm(start_point, end_point, "Direct OSRM Bike Route")
    if direct_bike:
        routes.append({
            "id": len(routes)+1, "name": "Direct OSRM Bike Route", "type": "direct_bike",
            "summary": {
                "total_time_minutes": direct_bike["travel_time_minutes"],
                "total_time_formatted": direct_bike["travel_time_formatted"],
                "total_distance_miles": direct_bike["length_miles"],
                "bike_distance_miles": direct_bike["length_miles"],
                "transit_distance_miles": 0.0,
                "bike_percentage": 100.0,
                "average_bike_score": direct_bike["overall_score"],
                "transfers": 0, "total_fare": 0,
                "departure_time": "Immediate", "arrival_time": "Flexible",
            },
            "legs": [{"type":"bike","name":"Direct OSRM Bike Route","description":"Complete OSRM bike route","route":direct_bike,"color":"#e74c3c","order":1}]
        })

    if not routes:
        raise HTTPException(status_code=400, detail="No routes found")

    routes.sort(key=lambda x: x["summary"]["total_time_minutes"])
    return {
        "success": True,
        "analysis_type": "osrm_google_transit",
        "fallback_used": fallback_used,
        "routing_engine": "OSRM + Google Transit",
        "routes": routes,
        "bus_stops": {"start_stops": start_stops, "end_stops": end_stops},
        "statistics": {
            "total_options": len(routes),
            "bike_bus_bike_options": len([r for r in routes if r["type"] == "bike_bus_bike"]),
            "direct_bike_options": len([r for r in routes if r["type"] == "direct_bike"]),
            "transit_fallback_options": len([r for r in routes if r["type"] == "transit_fallback"]),
            "fastest_option": routes[0]["name"],
            "fastest_time": routes[0]["summary"]["total_time_formatted"],
        },
        "bike_speed_mph": BIKE_SPEED_MPH,
        "osrm_server": OSRM_SERVER,
        "google_service": "Directions + Places",
        "analysis_timestamp": datetime.datetime.now().isoformat(),
    }

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def ui():
    # Lightweight, self-contained Leaflet UI
    return """
<!DOCTYPE html>
<html>
<head>
  <title>OSRM + Google Transit Bike–Bus–Bike Planner</title>
  <meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; }
    .header { background:#2c3e50; color:#fff; padding:14px 16px; }
    .container { display:flex; height: calc(100vh - 60px); }
    #map { flex:2; }
    .sidebar { flex:1; max-width:420px; padding:16px; background:#f8f9fa; overflow:auto; }
    .controls { margin-bottom:12px; }
    label { display:block; margin: 8px 0 4px; font-weight:600; }
    select, input { width:100%; padding:8px; }
    button { width:100%; padding:12px; margin-top:10px; background:#3498db; color:#fff; border:none; border-radius:4px; }
    button:disabled { background:#bbb; }
    .btn-clear{ background:#e74c3c; }
    .route-card{ background:#fff; border:1px solid #ddd; border-radius:8px; padding:12px; margin:10px 0; cursor:pointer; }
    .route-card.selected{ border-color:#3498db; background:#f5faff; }
    .route-summary{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px; }
    .summary-item{ background:#f1f3f4; border-radius:4px; text-align:center; padding:8px; }
    .summary-value{ font-weight:700; color:#3498db; }
    .coords{ font-size:12px; color:#555; background:#eef1f4; padding:6px; border-radius:4px; margin-top:8px; }
    .spinner{ border:3px solid #f3f3f3; border-top:3px solid #3498db; border-radius:50%; width:28px;height:28px; animation: spin 1s linear infinite; display:none; margin:16px auto;}
    @keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}
  </style>
</head>
<body>
  <div class="header"><h2>OSRM + Google Transit Bike–Bus–Bike Planner</h2></div>
  <div class="container">
    <div id="map"></div>
    <div class="sidebar">
      <div class="controls">
        <label>Departure Time</label>
        <select id="departureTime">
          <option value="now">Leave Now</option>
          <option value="custom">Custom</option>
        </select>
        <div id="customTimeGroup" style="display:none;">
          <label>Select Time</label>
          <input type="datetime-local" id="customTime">
        </div>
        <button id="findRoutesBtn" disabled>Find Routes</button>
        <button class="btn-clear" id="clearBtn">Clear Map</button>
        <div class="coords">
          <div><b>Start:</b> <span id="startCoords">Click map</span></div>
          <div><b>End:</b> <span id="endCoords">Click map</span></div>
        </div>
      </div>
      <div class="spinner" id="spinner"></div>
      <div id="results"></div>
    </div>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    let map = L.map('map').setView([30.3322,-81.6557], 12);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'© OpenStreetMap'}).addTo(map);
    let start = null, endp = null, startMarker=null, endMarker=null, routeLayers=L.layerGroup().addTo(map);
    let currentRoutes = [];

    const depSel = document.getElementById('departureTime');
    const customGroup = document.getElementById('customTimeGroup');
    const customInput = document.getElementById('customTime');
    depSel.addEventListener('change', ()=> {
      if (depSel.value==='custom'){ customGroup.style.display='block'; const n=new Date(); n.setMinutes(n.getMinutes()-n.getTimezoneOffset()); customInput.value=n.toISOString().slice(0,16);}
      else customGroup.style.display='none';
    });

    map.on('click', (e)=>{
      const lat=e.latlng.lat, lon=e.latlng.lng;
      if (!start){ if(startMarker) map.removeLayer(startMarker); start=[lon,lat]; startMarker=L.marker([lat,lon]).addTo(map).bindPopup('Start').openPopup(); document.getElementById('startCoords').textContent=lat.toFixed(5)+', '+lon.toFixed(5); }
      else if(!endp){ if(endMarker) map.removeLayer(endMarker); endp=[lon,lat]; endMarker=L.marker([lat,lon]).addTo(map).bindPopup('End').openPopup(); document.getElementById('endCoords').textContent=lat.toFixed(5)+', '+lon.toFixed(5); document.getElementById('findRoutesBtn').disabled=false; }
      else { clearAll(); map.fire('click', e); }
    });

    document.getElementById('clearBtn').onclick = clearAll;
    function clearAll(){
      if(startMarker){map.removeLayer(startMarker);} if(endMarker){map.removeLayer(endMarker);}
      startMarker=null; endMarker=null; start=null; endp=null; routeLayers.clearLayers();
      document.getElementById('startCoords').textContent='Click map'; document.getElementById('endCoords').textContent='Click map';
      document.getElementById('findRoutesBtn').disabled=true; document.getElementById('results').innerHTML='';
    }

    function spin(v){ document.getElementById('spinner').style.display = v?'block':'none'; }

    document.getElementById('findRoutesBtn').onclick = async ()=>{
      if(!start || !endp) return;
      spin(true); routeLayers.clearLayers();
      let dep = (depSel.value==='custom' && customInput.value) ? Math.floor(new Date(customInput.value).getTime()/1000) : 'now';
      const qs = new URLSearchParams({ start_lon:start[0], start_lat:start[1], end_lon:endp[0], end_lat:endp[1], departure_time: dep });
      const res = await fetch('/api/analyze?'+qs.toString()); const data = await res.json(); spin(false);
      if(!res.ok){ document.getElementById('results').innerHTML = '<div style="color:#a00;background:#fee;padding:8px;border-radius:6px;">'+(data.detail||'Analysis failed')+'</div>'; return; }
      currentRoutes = data.routes; renderResults(data); if(currentRoutes.length>0) selectRoute(0);
    };

    function renderResults(data){
      let html = '<h3>Found '+data.routes.length+' options</h3>';
      if(data.fallback_used){ html += '<div style="background:#e3f2fd;padding:8px;border-radius:6px;margin:8px 0;">Smart fallback: transit-only used.</div>'; }
      data.routes.forEach((r, idx)=>{
        html += '<div class="route-card" id="rc'+idx+'" onclick="selectRoute('+idx+')">'+
                '<div><b>'+r.name+'</b></div>'+
                '<div class="route-summary">'+
                '<div class="summary-item"><div class="summary-value">'+r.summary.total_time_formatted+'</div><div>Total Time</div></div>'+
                '<div class="summary-item"><div class="summary-value">'+(r.summary.total_distance_miles||0).toFixed(1)+' mi</div><div>Distance</div></div>'+
                '</div></div>';
      });
      document.getElementById('results').innerHTML = html;
    }

    window.selectRoute = function(index){
      document.querySelectorAll('.route-card').forEach(x=>x.classList.remove('selected'));
      const card = document.getElementById('rc'+index); if(card) card.classList.add('selected');
      routeLayers.clearLayers();
      const r = currentRoutes[index];
      const colors = {'bike':'#27ae60','transit':'#3498db'};
      (r.legs||[]).forEach(leg=>{
        if(leg.type==='bike' && leg.route.geometry){
          const coords = leg.route.geometry.coordinates.map(c=>[c[1],c[0]]);
          L.polyline(coords,{color:colors.bike,weight:5,opacity:0.9}).addTo(routeLayers);
        } else if(leg.type==='transit'){
          const coords = (leg.route.route_geometry||[]).map(c=>[c[1],c[0]]);
          if(coords.length) L.polyline(coords,{color:colors.transit,weight:5,opacity:0.9,dashArray:'10,5'}).addTo(routeLayers);
          (leg.route.steps||[]).forEach(s=>{
            if(s.travel_mode==='TRANSIT' && s.departure_stop_location && s.departure_stop_location.lat!==undefined){
              L.marker([s.departure_stop_location.lat, s.departure_stop_location.lng]).addTo(routeLayers)
               .bindPopup('Departure: '+(s.departure_stop_name||'Stop'));
              if(s.arrival_stop_location && s.arrival_stop_location.lat!==undefined){
                L.marker([s.arrival_stop_location.lat, s.arrival_stop_location.lng]).addTo(routeLayers)
                 .bindPopup('Arrival: '+(s.arrival_stop_name||'Stop'));
              }
            }
          });
        }
      });
      try{ if(routeLayers.getLayers().length>0) map.fitBounds(routeLayers.getBounds(),{padding:[18,18]}); }catch(e){}
    };
  </script>
</body>
</html>
    """

@app.get("/api/health")
async def health():
    # light connectivity flags
    status = {
        "status": "healthy",
        "google_key_present": bool(GOOGLE_API_KEY),
        "osrm_server": OSRM_SERVER,
        "bike_speed_mph": BIKE_SPEED_MPH,
        "time": datetime.datetime.now().isoformat()
    }
    try:
        requests.get(f"{OSRM_SERVER}/health", timeout=5)
        status["osrm_reachable"] = True
    except Exception:
        status["osrm_reachable"] = False
    return status

@app.get("/api/stops")
async def api_stops(lat: float, lon: float, radius_meters: int = 600):
    try:
        return {
            "stops": find_nearby_transit_stations_google([lon, lat], radius_meters, max_results=10),
            "center": {"lat": lat, "lon": lon},
            "radius_meters": radius_meters,
            "source": "Google Places"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/api/stops error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analyze")
async def api_analyze(
    start_lon: float = Query(...),
    start_lat: float = Query(...),
    end_lon: float = Query(...),
    end_lat: float = Query(...),
    departure_time: str = Query("now")
):
    # basic validation
    if not (-180 <= start_lon <= 180 and -90 <= start_lat <= 90): raise HTTPException(status_code=400, detail="Invalid start coordinates")
    if not (-180 <= end_lon <= 180 and -90 <= end_lat <= 90): raise HTTPException(status_code=400, detail="Invalid end coordinates")
    try:
        return analyze_complete_bike_bus_bike_routes([start_lon, start_lat], [end_lon, end_lat], departure_time)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/api/analyze error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def on_startup():
    logger.info("Starting OSRM + Google Transit BBB API…")
    logger.info(f"OSRM: {OSRM_SERVER}")
    logger.info(f"Google key present: {bool(GOOGLE_API_KEY)}")

if __name__ == "__main__":
    # Run directly: uses this process' env var GOOGLE_API_KEY
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False, log_level="info")
