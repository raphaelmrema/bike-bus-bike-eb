# bbb_osrm_google_enhanced_ui.py
# Bike–Bus–Bike planner: OSRM (bike) + Google Directions (transit)
# Frontend styled after "Enhanced Transit with GTFS" (cards + live panel),
# with distinct map colors for bike vs transit. Ready for Railway.

import os
import json
import math
import datetime
import logging
from typing import List, Dict, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Ensure polyline is available
try:
    import polyline
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "polyline"])
    import polyline

# =============================================================================
# CONFIG
# =============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OSRM_SERVER = os.getenv("OSRM_SERVER", "http://router.project-osrm.org")
BIKE_SPEED_MPH = float(os.getenv("BIKE_SPEED_MPH", "11"))
USE_OSRM_DURATION = True

GMAPS_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
GMAPS_PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

# Colors (frontend expects these)
COLOR_BIKE = "#27ae60"
COLOR_TRANSIT = "#1e88e5"
COLOR_DIRECT = "#8e44ad"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bbb-osrm-google-ui")

app = FastAPI(
    title="Bike–Bus–Bike (OSRM + Google Transit)",
    description="Multimodal planner with Enhanced-Transit-style frontend.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down in prod if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# UTILITIES
# =============================================================================
def require_google_key():
    if not GOOGLE_API_KEY or len(GOOGLE_API_KEY.strip()) < 20:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not set or invalid on the server.")

def fmt_time(mins: float) -> str:
    if mins < 1: return "< 1 min"
    if mins < 60: return f"{int(round(mins))} min"
    h = int(mins // 60)
    m = int(round(mins % 60))
    return f"{h}h" if m == 0 else f"{h}h {m}m"

def decode_to_lonlat(encoded: str) -> List[List[float]]:
    pts = polyline.decode(encoded)  # [(lat,lon), ...]
    return [[lon, lat] for (lat, lon) in pts]

def epoch_to_hhmm(ts: Optional[int]) -> str:
    try:
        if not ts:
            return "—"
        return datetime.datetime.fromtimestamp(ts).strftime("%H:%M")
    except Exception:
        return "—"

def _meters_distance(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    # a,b: (lon, lat)
    dx = (a[0]-b[0]) * 111000 * math.cos(math.radians((a[1]+b[1])/2))
    dy = (a[1]-b[1]) * 111000
    return math.hypot(dx, dy)

# =============================================================================
# OSRM (BIKE)
# =============================================================================
def osrm_bike(start: List[float], end: List[float], name="Bike Route"):
    try:
        coords = f"{start[0]},{start[1]};{end[0]},{end[1]}"
        url = f"{OSRM_SERVER}/route/v1/cycling/{coords}"
        params = {"overview": "full", "geometries":"polyline", "steps":"false", "alternatives":"false"}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return None
        route = data["routes"][0]
        dist_m = float(route.get("distance", 0))
        dist_mi = dist_m * 0.000621371
        if USE_OSRM_DURATION and route.get("duration") is not None:
            dur_min = float(route["duration"]) / 60.0
        else:
            dur_min = (dist_mi / BIKE_SPEED_MPH) * 60.0
        geom = decode_to_lonlat(route["geometry"])
        # placeholder score (you can plug your facility score later)
        score = max(0, min(100, 70 + (10 if dist_mi < 2 else -5)))
        return {
            "name": name,
            "length_miles": round(dist_mi, 3),
            "travel_time_minutes": round(dur_min, 1),
            "travel_time_formatted": fmt_time(dur_min),
            "geometry": {"type":"LineString", "coordinates": geom},
            "overall_score": score
        }
    except Exception as e:
        log.error(f"OSRM error: {e}")
        return None

# =============================================================================
# GOOGLE (TRANSIT + PLACES)
# =============================================================================
def _dep_ts(departure_time: str) -> int:
    if departure_time == "now":
        return int(datetime.datetime.now().timestamp())
    try:
        return int(float(departure_time))
    except Exception:
        return int(datetime.datetime.now().timestamp())

def google_transit(origin: Tuple[float,float], dest: Tuple[float,float], departure_time="now", max_alts=3) -> Dict:
    require_google_key()
    try:
        ts = _dep_ts(departure_time)
        params = {
            "origin": f"{origin[1]},{origin[0]}",
            "destination": f"{dest[1]},{dest[0]}",
            "mode": "transit",
            "departure_time": ts,
            "alternatives": "true",
            "transit_mode": "bus|subway|train|tram",
            "transit_routing_preference": "fewer_transfers",
            "key": GOOGLE_API_KEY
        }
        r = requests.get(GMAPS_DIRECTIONS_URL, params=params, timeout=30)
        data = r.json()
        if data.get("status") != "OK":
            return {"error": data.get("error_message", f"Google status: {data.get('status')}")}
        routes = []
        for idx, rd in enumerate(data.get("routes", [])[:max_alts]):
            parsed = _parse_google_route(rd, idx)
            if parsed: routes.append(parsed)
        if not routes:
            return {"error":"No transit routes found"}
        return {"routes": routes, "service":"Google Maps Transit", "total_routes": len(routes)}
    except Exception as e:
        log.error(f"Google Directions error: {e}")
        return {"error": str(e)}

def _parse_google_route(route_data: Dict, idx: int) -> Optional[Dict]:
    try:
        legs = route_data.get("legs", [])
        if not legs: return None
        leg = legs[0]
        dur_s = leg.get("duration", {}).get("value", 0)
        dur_min = round(dur_s/60.0, 1)
        dist_m = leg.get("distance", {}).get("value", 0)
        dist_mi = round(dist_m*0.000621371, 2)
        dep_ts = leg.get("departure_time", {}).get("value")
        arr_ts = leg.get("arrival_time", {}).get("value")
        steps_raw = leg.get("steps", [])
        steps = []
        geom = []
        boardings = 0
        walk_m = 0
        lines = []
        for i, s in enumerate(steps_raw):
            ps = _parse_step(s, i)
            if ps:
                steps.append(ps)
                if ps["travel_mode"] == "TRANSIT":
                    boardings += 1
                    if ps.get("transit_line"): lines.append(ps["transit_line"])
                if ps["travel_mode"] == "WALKING":
                    walk_m += s.get("distance", {}).get("value", 0)
            if s.get("polyline", {}).get("points"):
                geom.extend(decode_to_lonlat(s["polyline"]["points"]))
        transfers = max(0, boardings-1)
        return {
            "route_number": idx+1,
            "name": f"Transit Route {idx+1}" + (f" ({transfers} transfers)" if transfers else ""),
            "duration_minutes": dur_min,
            "duration_text": fmt_time(dur_min),
            "distance_miles": dist_mi,
            "departure_time": epoch_to_hhmm(dep_ts),
            "arrival_time": epoch_to_hhmm(arr_ts),
            "transfers": transfers,
            "walking_distance_miles": round(walk_m*0.000621371,2),
            "transit_lines": list(dict.fromkeys(lines)),
            "route_geometry": geom,
            "steps": steps,
            "service": "Google Maps Transit"
        }
    except Exception as e:
        log.error(f"Parse route error: {e}")
        return None

def _parse_step(step: Dict, idx: int) -> Optional[Dict]:
    try:
        mode = (step.get("travel_mode") or "UNKNOWN").upper()
        dur_s = step.get("duration", {}).get("value", 0)
        dist_m = step.get("distance", {}).get("value", 0)
        base = {
            "step_number": idx+1,
            "travel_mode": mode,
            "duration_minutes": round(dur_s/60.0,1),
            "duration_text": fmt_time(dur_s/60.0),
            "distance_miles": round(dist_m*0.000621371,2),
        }
        if mode == "TRANSIT":
            td = step.get("transit_details", {}) or {}
            dep = td.get("departure_stop", {}) or {}
            arr = td.get("arrival_stop", {}) or {}
            line = td.get("line", {}) or {}
            dep_t = (td.get("departure_time", {}) or {}).get("value")
            arr_t = (td.get("arrival_time", {}) or {}).get("value")
            base.update({
                "transit_line": line.get("short_name") or line.get("name") or "Transit",
                "departure_stop_name": dep.get("name", "Stop"),
                "departure_stop_location": dep.get("location", {}),
                "arrival_stop_name": arr.get("name", "Stop"),
                "arrival_stop_location": arr.get("location", {}),
                "scheduled_departure": epoch_to_hhmm(dep_t),
                "scheduled_arrival": epoch_to_hhmm(arr_t),
                "headsign": td.get("headsign",""),
                "num_stops": td.get("num_stops", 0),
            })
        return base
    except Exception as e:
        log.error(f"Parse step error: {e}")
        return None

def google_nearby_transit(point: List[float], radius_m=800, max_results=6):
    require_google_key()
    try:
        params = {
            "location": f"{point[1]},{point[0]}",
            "radius": radius_m,
            "type": "transit_station",
            "key": GOOGLE_API_KEY
        }
        r = requests.get(GMAPS_PLACES_NEARBY_URL, params=params, timeout=15)
        data = r.json()
        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            log.warning(f"Places status: {data.get('status')} {data.get('error_message')}")
            return []
        out = []
        for item in data.get("results", [])[:max_results]:
            loc = (item.get("geometry", {}) or {}).get("location", {})
            out.append({
                "id": item.get("place_id",""),
                "name": item.get("name","Transit Station"),
                "x": loc.get("lng"),
                "y": loc.get("lat"),
                "display_x": loc.get("lng"),
                "display_y": loc.get("lat"),
            })
        return out
    except Exception as e:
        log.error(f"Places error: {e}")
        return []

# =============================================================================
# ANALYSIS (Bike–Bus–Bike + fallbacks)
# =============================================================================
def _should_transit_only(start, end, s_stops, e_stops, thres_m=400):
    try:
        if not s_stops or not e_stops: return False
        s = s_stops[0]; e = e_stops[0]
        d1 = _meters_distance(tuple(start), (s["display_x"], s["display_y"]))
        d2 = _meters_distance(tuple(end), (e["display_x"], e["display_y"]))
        return (d1 < thres_m and d2 < thres_m)
    except: return False

def analyze_bbb(start_pt: List[float], end_pt: List[float], departure_time="now"):
    routes = []
    s_stops = google_nearby_transit(start_pt, 800, 4)
    e_stops = google_nearby_transit(end_pt, 800, 4)

    fallback_used = _should_transit_only(start_pt, end_pt, s_stops, e_stops)
    if fallback_used:
        tr = google_transit(tuple(start_pt), tuple(end_pt), departure_time)
        if tr.get("routes"):
            for i, r in enumerate(tr["routes"]):
                routes.append({
                    "id": len(routes)+1,
                    "name": f"Transit Option {i+1}",
                    "type": "transit_fallback",
                    "summary": {
                        "total_time_minutes": r["duration_minutes"],
                        "total_time_formatted": r["duration_text"],
                        "total_distance_miles": r["distance_miles"],
                        "bike_distance_miles": 0.0,
                        "transit_distance_miles": r["distance_miles"],
                        "bike_percentage": 0.0,
                        "average_bike_score": 0.0,
                        "transfers": r.get("transfers", 0),
                        "departure_time": r.get("departure_time","—"),
                        "arrival_time": r.get("arrival_time","—"),
                    },
                    "legs": [{
                        "type":"transit","name":f"Google Transit Route {i+1}",
                        "description":"Direct transit (smart fallback)", "route": r,
                        "color": COLOR_TRANSIT, "style":"dashed", "order": 1
                    }],
                    "fallback_reason":"Both bike segments < 400m"
                })

    if s_stops and e_stops:
        s = s_stops[0]; e = e_stops[0]
        if s["id"] == e["id"]:
            if len(e_stops) > 1: e = e_stops[1]
            elif len(s_stops) > 1: e = s_stops[1]
        if s["id"] != e["id"]:
            b1 = osrm_bike(start_pt, [s["display_x"], s["display_y"]], "Bike to Station")
            b2 = osrm_bike([e["display_x"], e["display_y"]], end_pt, "Bike from Station")
            if b1 and b2:
                tr = google_transit((s["display_x"], s["display_y"]), (e["display_x"], e["display_y"]), departure_time)
                if tr.get("routes"):
                    for i, r in enumerate(tr["routes"]):
                        bike_mi = b1["length_miles"] + b2["length_miles"]
                        tr_mi = r["distance_miles"]
                        total_mi = bike_mi + tr_mi
                        total_time = b1["travel_time_minutes"] + r["duration_minutes"] + b2["travel_time_minutes"] + 5.0
                        bike_score = 0.0
                        if bike_mi > 0:
                            bike_score = (b1["overall_score"]*b1["length_miles"] + b2["overall_score"]*b2["length_miles"]) / bike_mi
                        r_enh = dict(r); r_enh["start_stop"] = s; r_enh["end_stop"] = e
                        routes.append({
                            "id": len(routes)+1,
                            "name": f"Bike–Bus–Bike Option {i+1}",
                            "type": "bike_bus_bike",
                            "summary": {
                                "total_time_minutes": round(total_time,1),
                                "total_time_formatted": fmt_time(total_time),
                                "total_distance_miles": round(total_mi,2),
                                "bike_distance_miles": round(bike_mi,2),
                                "transit_distance_miles": round(tr_mi,2),
                                "bike_percentage": round((bike_mi/total_mi)*100,1) if total_mi>0 else 0.0,
                                "average_bike_score": round(bike_score,1),
                                "transfers": r.get("transfers", 0),
                                "departure_time": r.get("departure_time","—"),
                                "arrival_time": r.get("arrival_time","—"),
                            },
                            "legs": [
                                {"type":"bike","name":"Bike → Station","description":f"Bike to {s['name']}",
                                 "route": b1, "color": COLOR_BIKE, "style":"solid","order":1},
                                {"type":"transit","name":"Transit","description":f"{s['name']} → {e['name']}",
                                 "route": r_enh, "color": COLOR_TRANSIT, "style":"dashed","order":2},
                                {"type":"bike","name":"Station → Destination","description":f"Bike from {e['name']}",
                                 "route": b2, "color": COLOR_BIKE, "style":"solid","order":3},
                            ]
                        })

    direct = osrm_bike(start_pt, end_pt, "Direct Bike Route")
    if direct:
        routes.append({
            "id": len(routes)+1,
            "name": "Direct Bike Route",
            "type": "direct_bike",
            "summary": {
                "total_time_minutes": direct["travel_time_minutes"],
                "total_time_formatted": direct["travel_time_formatted"],
                "total_distance_miles": direct["length_miles"],
                "bike_distance_miles": direct["length_miles"],
                "transit_distance_miles": 0.0,
                "bike_percentage": 100.0,
                "average_bike_score": direct["overall_score"],
                "transfers": 0,
                "departure_time": "—",
                "arrival_time": "—",
            },
            "legs": [{
                "type":"bike","name":"Direct Bike Route",
                "description":"Complete OSRM bike route",
                "route": direct, "color": COLOR_DIRECT, "style":"solid", "order":1
            }]
        })

    if not routes:
        raise HTTPException(status_code=400, detail="No routes found.")

    routes.sort(key=lambda x: x["summary"]["total_time_minutes"])
    return {
        "success": True,
        "analysis_type": "osrm+google",
        "routes": routes,
        "bus_stops": {"start_stops": s_stops, "end_stops": e_stops},
        "statistics": {
            "total_options": len(routes),
            "fastest_option": routes[0]["name"],
            "fastest_time": routes[0]["summary"]["total_time_formatted"],
        },
        "colors": {"bike": COLOR_BIKE, "transit": COLOR_TRANSIT, "direct": COLOR_DIRECT},
        "ts": datetime.datetime.now().isoformat()
    }

# =============================================================================
# API ENDPOINTS
# =============================================================================
@app.get("/", response_class=HTMLResponse)
async def ui():
    # Enhanced-Transit-style UI: route cards + per-leg info + live panel
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Bike–Bus–Bike Planner</title>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <style>
    :root {{
      --bike: {COLOR_BIKE};
      --transit: {COLOR_TRANSIT};
      --direct: {COLOR_DIRECT};
    }}
    body {{ margin:0; font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial; background:#f6f8fb; }}
    .topbar {{ background:#1f2937; color:#fff; padding:12px 16px; }}
    .topbar h1 {{ font-size:18px; margin:0; }}
    .layout {{ display:flex; height:calc(100vh - 56px); }}
    #map {{ flex:2; }}
    .panel {{ flex:1; max-width:480px; background:#fff; border-left:1px solid #e5e7eb; display:flex; flex-direction:column; }}
    .controls {{ padding:12px 14px; border-bottom:1px solid #e5e7eb; background:#fafafa; }}
    .row {{ display:flex; gap:8px; align-items:center; }}
    label {{ font-size:13px; color:#374151; }}
    select, input {{ width:100%; padding:8px; border:1px solid #d1d5db; border-radius:6px; }}
    button {{ padding:10px; border:none; border-radius:6px; background:#1e88e5; color:#fff; cursor:pointer; }}
    button:disabled {{ background:#93c5fd; cursor:not-allowed; }}
    .split {{ display:grid; grid-template-columns: 1fr 1fr; gap:8px; }}
    .cards {{ padding:12px; overflow:auto; }}
    .card {{ background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:12px; margin-bottom:10px; cursor:pointer; }}
    .card.selected {{ border-color:#1e88e5; box-shadow:0 0 0 3px rgba(30,136,229,0.12); }}
    .hdr {{ display:flex; justify-content:space-between; align-items:center; }}
    .name {{ font-weight:700; color:#111827; }}
    .pill {{ font-size:12px; background:#eef2ff; color:#4338ca; padding:4px 8px; border-radius:999px; }}
    .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px; }}
    .stat {{ background:#f3f4f6; padding:8px; border-radius:8px; text-align:center; }}
    .stat .v {{ font-weight:700; color:#1e88e5; }}
    .legs {{ margin-top:8px; font-size:13px; color:#374151; }}
    .mini {{ font-size:12px; color:#6b7280; }}
    .live {{ border-top:1px dashed #e5e7eb; margin-top:10px; padding-top:8px; }}
    .good {{ color:#16a34a; }} .warn {{ color:#d97706; }} .bad {{ color:#dc2626; }}
    .err {{ color:#b91c1c; background:#fee2e2; padding:8px; border-radius:6px; margin:10px 12px; }}
    .spinner {{ display:none; margin:10px auto; border:3px solid #eee; border-top:3px solid #1e88e5; border-radius:50%; width:28px;height:28px; animation: spin 1s linear infinite; }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    .legend {{ padding:8px 12px; border-bottom:1px solid #e5e7eb; background:#fafafa; font-size:12px; color:#374151; }}
    .chip {{ display:inline-block; padding:4px 8px; border-radius:999px; margin-right:6px; }}
  </style>
</head>
<body>
  <div class="topbar"><h1>Bike–Bus–Bike (OSRM + Google Transit)</h1></div>
  <div class="layout">
    <div id="map"></div>
    <div class="panel">
      <div class="legend">
        <span class="chip" style="background: var(--bike); color:#fff;">Bike (solid)</span>
        <span class="chip" style="background: var(--transit); color:#fff;">Transit (dashed)</span>
        <span class="chip" style="background: var(--direct); color:#fff;">Direct Bike</span>
      </div>
      <div class="controls">
        <div class="split">
          <div>
            <label>Departure</label>
            <select id="depSel">
              <option value="now">Leave Now</option>
              <option value="custom">Custom</option>
            </select>
          </div>
          <div id="customWrap" style="display:none;">
            <label>Time</label>
            <input type="datetime-local" id="customTime"/>
          </div>
        </div>
        <div class="row" style="margin-top:8px; gap:8px;">
          <button id="findBtn" disabled>Find Routes</button>
          <button id="clearBtn" style="background:#ef4444;">Clear</button>
        </div>
        <div class="mini" id="coords">Start/End: click on the map</div>
      </div>
      <div class="spinner" id="spin"></div>
      <div id="cards" class="cards"></div>
      <div id="error" class="err" style="display:none;"></div>
    </div>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    let map = L.map('map').setView([30.3322,-81.6557], 12);
    L.tileLayer('https://{{"{"}}s{{"}"}}.tile.openstreetmap.org/{{"{"}}z{{"}"}}/{{"{"}}x{{"}"}}/{{"{"}}y{{"}"}}.png', {{ attribution:'© OpenStreetMap' }}).addTo(map);
    let start=null, endp=null, startM=null, endM=null, routes=[], routeLayer=L.layerGroup().addTo(map);

    const depSel=document.getElementById('depSel');
    const customWrap=document.getElementById('customWrap');
    const customTime=document.getElementById('customTime');
    const findBtn=document.getElementById('findBtn');
    const clearBtn=document.getElementById('clearBtn');
    const coordsEl=document.getElementById('coords');
    const cardsEl=document.getElementById('cards');
    const spinEl=document.getElementById('spin');
    const errEl=document.getElementById('error');

    depSel.addEventListener('change', ()=>{
      if(depSel.value==='custom'){
        customWrap.style.display='block';
        const n=new Date(); n.setMinutes(n.getMinutes()-n.getTimezoneOffset());
        customTime.value=n.toISOString().slice(0,16);
      } else customWrap.style.display='none';
    });

    map.on('click', (e)=>{
      const lat=e.latlng.lat, lon=e.latlng.lng;
      if(!start){
        if(startM) map.removeLayer(startM); start=[lon,lat];
        startM=L.marker([lat,lon]).addTo(map).bindPopup('Start').openPopup();
        coordsEl.textContent='Start set. Click End on map.';
      } else if(!endp){
        if(endM) map.removeLayer(endM); endp=[lon,lat];
        endM=L.marker([lat,lon]).addTo(map).bindPopup('End').openPopup();
        coordsEl.textContent='Ready. Click "Find Routes".';
        findBtn.disabled=false;
      } else {
        clearAll(); map.fire('click', e);
      }
    });

    clearBtn.onclick = clearAll;
    function clearAll(){
      if(startM){ map.removeLayer(startM); } if(endM){ map.removeLayer(endM); }
      start=null; endp=null; startM=null; endM=null;
      routeLayer.clearLayers(); findBtn.disabled=true; cardsEl.innerHTML='';
      errEl.style.display='none'; coordsEl.textContent='Start/End: click on the map';
    }

    function spin(v){ spinEl.style.display = v ? 'block' : 'none'; }

    findBtn.onclick = async ()=>{
      if(!start || !endp) return;
      spin(true); routeLayer.clearLayers(); cardsEl.innerHTML=''; errEl.style.display='none';
      let dep = 'now';
      if(depSel.value==='custom' && customTime.value){ dep = Math.floor(new Date(customTime.value).getTime()/1000); }
      const qs = new URLSearchParams({ start_lon:start[0], start_lat:start[1], end_lon:endp[0], end_lat:endp[1], departure_time: dep });
      const res = await fetch('/api/analyze?'+qs.toString());
      const data = await res.json(); spin(false);
      if(!res.ok){ errEl.style.display='block'; errEl.textContent = data.detail || 'Analysis failed'; return; }
      routes = data.routes; renderCards(routes);
      if(routes.length>0) selectRoute(0);
    };

    function renderCards(rs){
      cardsEl.innerHTML='';
      rs.forEach((r, idx)=>{
        const div = document.createElement('div');
        div.className='card'; div.id='card'+idx;
        div.onclick=()=>selectRoute(idx);
        let html = `
          <div class="hdr">
            <div class="name">${{r.name}}</div>
            <div class="pill">${{r.type.replace('_',' ')}}</div>
          </div>
          <div class="grid2">
            <div class="stat"><div class="v">${{r.summary.total_time_formatted}}</div><div>Total Time</div></div>
            <div class="stat"><div class="v">${{(r.summary.total_distance_miles||0).toFixed(1)}} mi</div><div>Distance</div></div>
          </div>
          <div class="legs">
        `;
        (r.legs||[]).forEach((leg)=>{
          const miles = (leg.route.length_miles || leg.route.distance_miles || 0).toFixed(1);
          const ttext = leg.route.travel_time_formatted || leg.route.duration_text || '';
          const color = leg.type==='transit' ? 'var(--transit)' : (leg.type==='bike' ? 'var(--bike)' : 'var(--direct)');
          html += `<div style="margin:6px 0;">
            <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${{color}};margin-right:6px;"></span>
            <b>${{leg.name}}</b> — ${{miles}} mi ${ttext?(' • '+ttext):''}
          </div>`;
        });
        html += `</div>`;

        // Live panel (like Enhanced Transit): for each transit leg, show a small "Live Departures" block
        const tlegs = (r.legs||[]).filter(l=>l.type==='transit');
        if(tlegs.length){
          html += `<div class="live"><b>Live Departures</b><div class="mini" style="margin-top:6px;">(sample panel; auto-updates possible by polling /api/realtime)</div>`;
          tlegs.forEach((leg, i)=>{
            const s = (leg.route.steps||[]).find(x=>x.travel_mode==='TRANSIT');
            if(s && s.departure_stop_location && s.departure_stop_location.lat!==undefined){
              const lat = s.departure_stop_location.lat, lng = s.departure_stop_location.lng;
              const stopName = s.departure_stop_name || 'Stop';
              const domId = `live_${{idx}}_${{i}}`;
              html += `
                <div style="margin-top:6px;">
                  <div><b>${{stopName}}</b> — line ${ {s.transit_line || ''} }</div>
                  <div id="${{domId}}" class="mini">Fetching live status…</div>
                </div>
                <script>
                  (async ()=>{
                    try {{
                      const prms = new URLSearchParams({{ stop_lat: ${lat}, stop_lng: ${lng} }});
                      const r = await fetch('/api/realtime?'+prms.toString());
                      const d = await r.json();
                      const el = document.getElementById('${{domId}}');
                      if(!r.ok) throw new Error(d.detail||'err');
                      // Render a simple "next 3" style list
                      const rows = (d.departures||[]).slice(0,3).map(x=> {{
                        const cls = x.status==='On time' ? 'good' : (x.status.includes('delay')?'warn':'bad');
                        return `<span class="${{cls}}">${{x.time}} (${ {x.status} })</span>`;
                      }});
                      el.innerHTML = rows.length ? rows.join(' • ') : 'No upcoming vehicles';
                    }} catch(e) {{
                      const el = document.getElementById('${{domId}}');
                      if(el) el.textContent = 'Live not available';
                    }}
                  }})();
                </script>
              `;
            }
          });
          html += `</div>`;
        }

        div.innerHTML = html;
        cardsEl.appendChild(div);
      });
    }

    function selectRoute(idx){
      document.querySelectorAll('.card').forEach(c=>c.classList.remove('selected'));
      const card=document.getElementById('card'+idx); if(card) card.classList.add('selected');
      routeLayer.clearLayers();
      const r = routes[idx];

      (r.legs||[]).forEach(leg=>{
        let coords=[];
        if(leg.type==='bike' && leg.route.geometry){
          coords = leg.route.geometry.coordinates.map(c=>[c[1],c[0]]);
          L.polyline(coords, {{ color: getColor(leg), weight:5, opacity:0.95 }}).addTo(routeLayer);
        } else if(leg.type==='transit'){
          coords = (leg.route.route_geometry||[]).map(c=>[c[1],c[0]]);
          if(coords.length) L.polyline(coords, {{ color: getColor(leg), weight:5, opacity:0.95, dashArray:'10,6' }}).addTo(routeLayer);
          // Stops
          (leg.route.steps||[]).forEach(s=>{
            if(s.travel_mode==='TRANSIT' && s.departure_stop_location && s.departure_stop_location.lat!==undefined){
              L.marker([s.departure_stop_location.lat, s.departure_stop_location.lng])
               .addTo(routeLayer).bindPopup('Departure: '+(s.departure_stop_name||'Stop'));
              if(s.arrival_stop_location && s.arrival_stop_location.lat!==undefined){
                L.marker([s.arrival_stop_location.lat, s.arrival_stop_location.lng])
                 .addTo(routeLayer).bindPopup('Arrival: '+(s.arrival_stop_name||'Stop'));
              }
            }
          });
        } else if(leg.type==='direct'){
          if(leg.route.geometry){ coords = leg.route.geometry.coordinates.map(c=>[c[1],c[0]]);
            L.polyline(coords, {{ color: getColor(leg), weight:5, opacity:0.95 }}).addTo(routeLayer);
          }
        }
      });

      try {{ if(routeLayer.getLayers().length) map.fitBounds(routeLayer.getBounds(), {{ padding:[18,18] }}); }} catch(e){{}}
    }

    function getColor(leg){
      if(leg.type==='transit') return 'var(--transit)';
      if(leg.type==='bike') return 'var(--bike)';
      return 'var(--direct)';
    }
  </script>
</body>
</html>
    """

@app.get("/api/health")
async def health():
    st = {
        "status": "healthy",
        "google_key_present": bool(GOOGLE_API_KEY),
        "osrm": OSRM_SERVER,
        "bike_speed_mph": BIKE_SPEED_MPH,
        "time": datetime.datetime.now().isoformat()
    }
    try:
        requests.get(f"{OSRM_SERVER}/health", timeout=5)
        st["osrm_reachable"] = True
    except Exception:
        st["osrm_reachable"] = False
    return st

@app.get("/api/stops")
async def api_stops(lat: float, lon: float, radius_meters: int = 600):
    try:
        stops = google_nearby_transit([lon,lat], radius_meters, 10)
        return {"stops": stops, "center":{"lat":lat,"lon":lon}, "radius_meters": radius_meters}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/realtime")
async def api_realtime(stop_lat: float, stop_lng: float):
    """
    Live panel endpoint used by the frontend. Currently simulates "On time / minor delay".
    You can replace this logic with real GTFS-RT lookups and mapping.
    """
    try:
        now = datetime.datetime.now()
        # simple rolling "next 3" every 7/12/18 minutes
        mins = [7, 12, 18]
        deps = []
        for m in mins:
            t = (now + datetime.timedelta(minutes=m)).strftime("%H:%M")
            # naive status simulate
            status = "On time" if m % 2 == 0 else "minor delay ~2m"
            deps.append({"time": t, "status": status})
        return {"stop": {"lat": stop_lat, "lng": stop_lng}, "departures": deps}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analyze")
async def api_analyze(
    start_lon: float = Query(...), start_lat: float = Query(...),
    end_lon: float = Query(...), end_lat: float = Query(...),
    departure_time: str = Query("now")
):
    # basic coordinate validation
    if not (-180 <= start_lon <= 180 and -90 <= start_lat <= 90): raise HTTPException(status_code=400, detail="Invalid start coordinates")
    if not (-180 <= end_lon <= 180 and -90 <= end_lat <= 90): raise HTTPException(status_code=400, detail="Invalid end coordinates")
    try:
        return analyze_bbb([start_lon,start_lat],[end_lon,end_lat], departure_time)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def on_startup():
    log.info("Starting Bike–Bus–Bike (OSRM + Google) service…")
    log.info(f"OSRM: {OSRM_SERVER}")
    log.info(f"Google key present: {bool(GOOGLE_API_KEY)}")

if __name__ == "__main__":
    uvicorn.run("bbb_osrm_google_enhanced_ui:app", host="0.0.0.0", port=int(os.getenv("PORT","8000")), reload=False)
