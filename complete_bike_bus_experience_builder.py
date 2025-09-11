# complete_bike_bus_experience_builder_otp.py
# Complete Bike-Bus-Bike Route Planner with OSRM + OpenTripPlanner
# Designed for ArcGIS Experience Builder

import os
import json
import logging
import requests
import datetime
import zipfile
import pandas as pd
import io
import math
import re
from typing import List, Dict
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# =============================================================================
# CONFIGURATION
# =============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://experience.arcgis.com,https://*.maps.arcgis.com,http://localhost:*,https://*.railway.app"
).split(",")

# OSRM Server
OSRM_SERVER = os.getenv("OSRM_SERVER", "http://router.project-osrm.org")
USE_OSRM_DURATION = True

# OTP Server
TRANSIT_ENGINE = os.getenv("TRANSIT_ENGINE", "otp")  # "otp" or "google"
OTP_SERVER = os.getenv("OTP_SERVER", "http://localhost:8080/otp/routers/default/plan")

# Bicycle speed (fallback)
BIKE_SPEED_MPH = 11
BIKE_SPEED_FEET_PER_SECOND = BIKE_SPEED_MPH * 5280 / 3600

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# FASTAPI SETUP
# =============================================================================

app = FastAPI(
    title="OSRM + OTP Bike-Bus-Bike Route Planner API",
    description="Multimodal transportation planner using OSRM for bikes and OTP for transit",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# ENHANCED GTFS MANAGER
# =============================================================================

class EnhancedGTFSManager:
    def __init__(self):
        self.gtfs_data = {}
        self.stops_df = None
        self.routes_df = None
        self.stop_times_df = None
        self.trips_df = None
        self.calendar_df = None
        self.is_loaded = False

    def load_gtfs_data(self, gtfs_urls=None):
        if gtfs_urls is None:
            gtfs_urls = [
                "https://ride.jtafla.com/gtfs-archive/gtfs.zip",
                "https://schedules.jtafla.com/schedulesgtfs/download"
            ]
        for url in gtfs_urls:
            try:
                resp = requests.get(url, timeout=30, verify=False)
                if resp.ok and resp.content.startswith(b'PK'):
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                        if "stops.txt" in z.namelist():
                            self.stops_df = pd.read_csv(z.open("stops.txt"))
                        if "routes.txt" in z.namelist():
                            self.routes_df = pd.read_csv(z.open("routes.txt"))
                        if "stop_times.txt" in z.namelist():
                            self.stop_times_df = pd.read_csv(z.open("stop_times.txt"))
                        if "trips.txt" in z.namelist():
                            self.trips_df = pd.read_csv(z.open("trips.txt"))
                        self.is_loaded = True
                        logger.info("✅ GTFS loaded successfully")
                        return True
            except Exception as e:
                logger.warning(f"GTFS load failed from {url}: {e}")
        return False

    def find_nearby_stops(self, lat, lon, radius_km=0.5):
        if not self.is_loaded or self.stops_df is None:
            return []
        try:
            stops = self.stops_df.copy()
            stops['distance_km'] = ((stops['stop_lat'] - lat)**2 + (stops['stop_lon'] - lon)**2)**0.5 * 111
            nearby = stops[stops['distance_km'] <= radius_km].sort_values('distance_km')
            return nearby[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'distance_km']].head(5).to_dict('records')
        except Exception as e:
            logger.error(f"Error finding nearby stops: {e}")
            return []

gtfs_manager = EnhancedGTFSManager()

# =============================================================================
# UTILITIES
# =============================================================================

def format_time_duration(minutes):
    if minutes < 1:
        return "< 1 min"
    elif minutes < 60:
        return f"{int(minutes)} min"
    else:
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours}h {mins}m" if mins else f"{hours}h"

# =============================================================================
# OSRM BIKE ROUTING
# =============================================================================

def calculate_bike_route_osrm(start_coords, end_coords, route_name="Bike Route"):
    try:
        coords = f"{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
        url = f"{OSRM_SERVER}/route/v1/cycling/{coords}"
        params = {"overview": "full", "geometries": "geojson"}
        r = requests.get(url, params=params, timeout=30).json()
        if r.get("code") != "Ok":
            return None
        route = r["routes"][0]
        distance_m = route["distance"]
        duration_min = route["duration"] / 60 if USE_OSRM_DURATION else (distance_m * 0.000621371 / BIKE_SPEED_MPH) * 60
        return {
            "name": route_name,
            "distance_miles": distance_m * 0.000621371,
            "travel_time_minutes": duration_min,
            "travel_time_formatted": format_time_duration(duration_min),
            "geometry": route["geometry"]
        }
    except Exception as e:
        logger.error(f"OSRM error: {e}")
        return None

# =============================================================================
# OTP TRANSIT ROUTING
# =============================================================================

def get_transit_routes_otp(origin, destination, departure_time="now"):
    try:
        if departure_time == "now":
            dt = datetime.datetime.now()
        else:
            try:
                dt = datetime.datetime.fromtimestamp(int(departure_time))
            except:
                dt = datetime.datetime.now()

        params = {
            "fromPlace": f"{origin[1]},{origin[0]}",
            "toPlace": f"{destination[1]},{destination[0]}",
            "mode": "TRANSIT,WALK",
            "date": dt.strftime("%Y-%m-%d"),
            "time": dt.strftime("%H:%M:%S"),
            "numItineraries": 3
        }

        resp = requests.get(OTP_SERVER, params=params, timeout=30)
        resp.raise_for_status()
        itineraries = resp.json().get("plan", {}).get("itineraries", [])
        routes = []
        for i, itin in enumerate(itineraries):
            routes.append({
                "route_number": i+1,
                "name": f"OTP Transit {i+1}",
                "duration_minutes": round(itin["duration"] / 60, 1),
                "transfers": itin.get("transfers", 0),
                "legs": itin.get("legs", [])
            })
        return {"routes": routes, "service": "OTP"}
    except Exception as e:
        logger.error(f"OTP error: {e}")
        return {"error": str(e)}

def get_transit_routes(origin, destination, departure_time="now"):
    if TRANSIT_ENGINE.lower() == "otp":
        return get_transit_routes_otp(origin, destination, departure_time)
    else:
        raise HTTPException(status_code=500, detail="Google API disabled in this build")

# =============================================================================
# MAIN ANALYSIS ENGINE
# =============================================================================

def analyze_complete_bike_bus_bike_routes(start_point, end_point, departure_time="now"):
    routes = []

    # Nearest stops
    start_stops = gtfs_manager.find_nearby_stops(start_point[1], start_point[0])
    end_stops = gtfs_manager.find_nearby_stops(end_point[1], end_point[0])

    if start_stops and end_stops:
        bike1 = calculate_bike_route_osrm(start_point, [start_stops[0]["stop_lon"], start_stops[0]["stop_lat"]], "Start to Stop")
        bike2 = calculate_bike_route_osrm([end_stops[0]["stop_lon"], end_stops[0]["stop_lat"]], end_point, "Stop to End")

        if bike1 and bike2:
            transit_result = get_transit_routes(
                [start_stops[0]["stop_lon"], start_stops[0]["stop_lat"]],
                [end_stops[0]["stop_lon"], end_stops[0]["stop_lat"]],
                departure_time
            )
            for i, t in enumerate(transit_result.get("routes", [])):
                total_time = bike1["travel_time_minutes"] + bike2["travel_time_minutes"] + t["duration_minutes"]
                routes.append({
                    "id": i+1,
                    "name": f"Bike-Bus-Bike {i+1}",
                    "summary": {
                        "total_time_minutes": total_time,
                        "total_time_formatted": format_time_duration(total_time),
                        "bike_distance_miles": bike1["distance_miles"] + bike2["distance_miles"],
                        "transfers": t.get("transfers", 0)
                    },
                    "legs": [bike1, t, bike2]
                })

    # Direct bike
    direct_bike = calculate_bike_route_osrm(start_point, end_point, "Direct Bike")
    if direct_bike:
        routes.append({
            "id": len(routes)+1,
            "name": "Direct Bike",
            "summary": {
                "total_time_minutes": direct_bike["travel_time_minutes"],
                "total_time_formatted": direct_bike["travel_time_formatted"],
                "bike_distance_miles": direct_bike["distance_miles"],
                "transfers": 0
            },
            "legs": [direct_bike]
        })

    if not routes:
        raise HTTPException(status_code=400, detail="No routes found")

    return {"success": True, "routes": routes, "service": f"OSRM + {TRANSIT_ENGINE.upper()}"}

# =============================================================================
# FASTAPI ENDPOINTS
# =============================================================================

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "osrm_server": OSRM_SERVER,
        "otp_server": OTP_SERVER,
        "transit_engine": TRANSIT_ENGINE,
        "gtfs_loaded": gtfs_manager.is_loaded,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/api/analyze")
async def analyze_routes(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    departure_time: str = "now"
):
    return analyze_complete_bike_bus_bike_routes([start_lon, start_lat], [end_lon, end_lat], departure_time)

@app.on_event("startup")
async def startup_event():
    gtfs_manager.load_gtfs_data()
    logger.info("🚀 Server started with OSRM + OTP")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("complete_bike_bus_experience_builder_otp:app", host="0.0.0.0", port=port, reload=False)
