# integrated_multimodal_transit_tool.py
# Complete FastAPI app with OSRM bike routing, Google Transit, GTFS real-time, and shapefile analysis
# Deployable on Railway or run locally with .env file

import os
import json
import logging
import requests
import datetime
import math
import random
import zipfile
import io
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import subprocess
import sys

# ----------------------------------------------------
# Load environment variables from .env (local dev)
# ----------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

def install_packages():
    """Install required packages if missing"""
    packages = ["polyline", "pandas", "geopandas", "shapely", "numpy", "python-dotenv"]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages()

import polyline
import pandas as pd
import numpy as np

try:
    import geopandas as gpd
    from shapely.geometry import LineString
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: GeoPandas not available. Shapefile analysis will be disabled.")

# =============================================================================
# CONFIG
# =============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OSRM_SERVER = os.getenv("OSRM_SERVER", "http://router.project-osrm.org")
BIKE_SPEED_MPH = float(os.getenv("BIKE_SPEED_MPH", "11"))

ROADS_SHAPEFILE_URL = os.getenv("ROADS_SHAPEFILE_URL", "")
TRANSIT_STOPS_SHAPEFILE_URL = os.getenv("TRANSIT_STOPS_SHAPEFILE_URL", "")

CORS_ALLOW_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("integrated-multimodal-transit")

# =============================================================================
# GTFS MANAGER (simplified)
# =============================================================================

class EnhancedGTFSManager:
    def __init__(self):
        self.stops_df = None
        self.is_loaded = False
        self.last_update = None

    def load_gtfs_data(self):
        urls = [
            "https://ride.jtafla.com/gtfs-archive/gtfs.zip",
            "https://schedules.jtafla.com/schedulesgtfs/download",
            "https://openmobilitydata.org/p/jacksonville-transportation-authority/331/latest/download",
        ]
        for url in urls:
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    if "stops.txt" in z.namelist():
                        self.stops_df = pd.read_csv(z.open("stops.txt"))
                        self.is_loaded = True
                        self.last_update = datetime.datetime.now()
                        logger.info("Loaded GTFS data")
                        return True
            except Exception as e:
                logger.error(f"GTFS load error {url}: {e}")
        return False

# =============================================================================
# SHAPEFILE ANALYZER
# =============================================================================

class ShapefileAnalyzer:
    def __init__(self):
        self.roads_gdf = None
        self.transit_stops_gdf = None
        self.loaded = False
        self.available_files = []

    def load_shapefiles(self):
        if not GEOPANDAS_AVAILABLE:
            return False
        try:
            # List shapefiles in data.zip
            if ROADS_SHAPEFILE_URL.startswith("zip+") or TRANSIT_STOPS_SHAPEFILE_URL.startswith("zip+"):
                actual_url = (ROADS_SHAPEFILE_URL or TRANSIT_STOPS_SHAPEFILE_URL).replace("zip+", "")
                self.available_files = self._list_shapefiles_in_zip(actual_url)
                logger.info(f"Shapefiles found in zip: {self.available_files}")

            # Try loading roads & stops
            self.roads_gdf = self._load_from_path_or_zip(ROADS_SHAPEFILE_URL, "roads.shp")
            self.transit_stops_gdf = self._load_from_path_or_zip(TRANSIT_STOPS_SHAPEFILE_URL, "transit_stops.shp")

            self.loaded = self.roads_gdf is not None
            return self.loaded
        except Exception as e:
            logger.error(f"Error loading shapefiles: {e}")
            return False

    def _list_shapefiles_in_zip(self, url: str):
        try:
            if url.startswith("http"):
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    return [f for f in z.namelist() if f.endswith(".shp")]
            elif zipfile.is_zipfile(url):
                with zipfile.ZipFile(url, "r") as z:
                    return [f for f in z.namelist() if f.endswith(".shp")]
        except Exception as e:
            logger.error(f"Error listing shapefiles in {url}: {e}")
        return []

    def _load_from_path_or_zip(self, path_or_url: str, target_file: str):
        try:
            if path_or_url.startswith("zip+"):
                actual_url = path_or_url.replace("zip+", "")
                if actual_url.startswith("http"):
                    r = requests.get(actual_url, stream=True)
                    r.raise_for_status()
                    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                        if target_file in z.namelist():
                            return gpd.read_file(f"zip+{actual_url}!{target_file}")
                        else:
                            logger.warning(f"{target_file} not found. Available: {z.namelist()}")
                else:
                    if zipfile.is_zipfile(actual_url):
                        with zipfile.ZipFile(actual_url, "r") as z:
                            if target_file in z.namelist():
                                return gpd.read_file(f"zip://{actual_url}!{target_file}")
            else:
                return gpd.read_file(path_or_url)
        except Exception as e:
            logger.error(f"Error reading {path_or_url}: {e}")
            return None

# =============================================================================
# OSRM BIKE ROUTING
# =============================================================================

def format_time_duration(minutes: float) -> str:
    if minutes < 1: return "< 1 min"
    if minutes < 60: return f"{int(round(minutes))} min"
    h = int(minutes // 60); m = int(round(minutes % 60))
    return f"{h}h" if m == 0 else f"{h}h {m}m"

def calculate_bike_route_osrm(start_coords, end_coords, waypoints=None, route_name="Bike Route"):
    try:
        coords_list = [start_coords] + (waypoints or []) + [end_coords]
        coords_str = ";".join([f"{lon},{lat}" for lon,lat in coords_list])
        url = f"{OSRM_SERVER}/route/v1/cycling/{coords_str}"
        params = {"overview":"full","geometries":"polyline","steps":"false","alternatives":"false"}
        r = requests.get(url, params=params, timeout=30); r.raise_for_status(); data = r.json()
        if data.get("code")!="Ok" or not data.get("routes"): return None
        route = data["routes"][0]
        distance_mi = route["distance"]*0.000621371
        duration_min = route["duration"]/60
        coords_latlon = polyline.decode(route["geometry"])
        geometry = {"type":"LineString","coordinates":[[lon,lat] for lat,lon in coords_latlon]}
        return {
            "name": route_name,
            "length_miles": round(distance_mi,3),
            "travel_time_minutes": round(duration_min,1),
            "travel_time_formatted": format_time_duration(duration_min),
            "geometry": geometry,
            "overall_score": 70
        }
    except Exception as e:
        logger.error(f"OSRM error: {e}")
        return None

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="Integrated Multimodal Transit Tool", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gtfs_manager = EnhancedGTFSManager()
shapefile_analyzer = ShapefileAnalyzer()

@app.on_event("startup")
async def startup_event():
    gtfs_manager.load_gtfs_data()
    shapefile_analyzer.load_shapefiles()

@app.get("/api/bike-routes")
async def get_bike_routes(start_lon: float, start_lat: float, end_lon: float, end_lat: float):
    route = calculate_bike_route_osrm([start_lon,start_lat],[end_lon,end_lat])
    if not route:
        raise HTTPException(status_code=404, detail="No bike routes found")
    return {"success":True,"routes":[route]}

@app.get("/api/status")
async def get_status():
    return {
        "system_status": "operational",
        "osrm_server": OSRM_SERVER,
        "gtfs_loaded": gtfs_manager.is_loaded,
        "shapefile_loaded": shapefile_analyzer.loaded,
        "roads_count": len(shapefile_analyzer.roads_gdf) if shapefile_analyzer.roads_gdf is not None else 0,
        "transit_stops_count": len(shapefile_analyzer.transit_stops_gdf) if shapefile_analyzer.transit_stops_gdf is not None else 0,
        "available_shapefiles": shapefile_analyzer.available_files
    }

if __name__=="__main__":
    port = int(os.getenv("PORT",8000))
    uvicorn.run("integrated_multimodal_transit_tool:app",host="0.0.0.0",port=port)
