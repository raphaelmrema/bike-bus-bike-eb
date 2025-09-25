# enhanced_osrm_google_transit_with_geopandas.py
# Complete FastAPI app with GeoPandas-based enhanced bike routing analysis
# Error-corrected version for Railway deployment

import os
import json
import logging
import requests
import datetime
import math
import sys
import zipfile
import numpy as np
import re
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, Query, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Auto-install required packages
def _ensure_packages():
    """Auto-install required packages if missing"""
    reqs = ["geopandas", "shapely", "polyline", "rtree", "pyproj", "pandas", "numpy"]
    missing = []
    for r in reqs:
        try:
            if r == "rtree":
                import rtree.index
            elif r == "polyline":
                import polyline
            else:
                __import__(r)
        except ImportError:
            missing.append(r)
    
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        import subprocess
        for pkg in missing:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                print(f"Successfully installed {pkg}")
            except Exception as e:
                print(f"Failed to install {pkg}: {e}")

_ensure_packages()

try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point, LineString
    from shapely.ops import nearest_points
    import polyline
    GEOPANDAS_AVAILABLE = True
    print("GeoPandas loaded successfully")
except ImportError as e:
    print(f"GeoPandas not available: {e}")
    GEOPANDAS_AVAILABLE = False

# =============================================================================
# CONFIG
# =============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OSRM_SERVER = os.getenv("OSRM_SERVER", "http://router.project-osrm.org")
USE_OSRM_DURATION = True
BIKE_SPEED_MPH = float(os.getenv("BIKE_SPEED_MPH", "11"))

# Shapefile paths - support both direct files and compressed archives
ROADS_SHAPEFILE = os.getenv("ROADS_SHAPEFILE", "./data/roads.shp")
ROADS_ARCHIVE = os.getenv("ROADS_ARCHIVE", "./data/roads.zip")
TRANSIT_STOPS_SHAPEFILE = os.getenv("TRANSIT_STOPS_SHAPEFILE", "./data/transit_stops.shp")
TRANSIT_STOPS_ARCHIVE = os.getenv("TRANSIT_STOPS_ARCHIVE", "./data/transit_stops.zip")

GMAPS_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
GMAPS_PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

CORS_ALLOW_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced-osrm-transit")

# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="Enhanced OSRM + Google Transit + GeoPandas Planner",
    description="Bike-Bus-Bike routing with advanced GeoPandas-based bicycle analysis",
    version="3.1.0",
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

def extract_if_archived(archive_path: str, extract_dir: str = "./data") -> bool:
    """Extract compressed shapefile if archive exists"""
    if os.path.exists(archive_path):
        try:
            logger.info(f"Extracting {archive_path}")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            logger.info(f"Successfully extracted {archive_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to extract {archive_path}: {e}")
            return False
    return False

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
    return [[lon, lat] for (lat, lon) in pts]

def parse_epoch_to_hhmm(ts: Optional[int]) -> str:
    try:
        if not ts: return "Unknown"
        return datetime.datetime.fromtimestamp(ts).strftime("%H:%M")
    except Exception:
        return "Unknown"

def _map_field(row_or_columns, candidates):
    """Find field name from candidates (case-insensitive)"""
    if hasattr(row_or_columns, 'keys'):
        keys = list(row_or_columns.keys())
    else:
        keys = list(row_or_columns)
    
    lower_map = {str(k).lower(): k for k in keys}
    
    for cand in candidates:
        lc = str(cand).lower()
        if lc in lower_map:
            return lower_map[lc]
    return None

# =============================================================================
# LEVEL OF TRAFFIC STRESS (LTS) CLASSIFICATION
# =============================================================================

def classify_lts(facility_type, speed_limit, lanes):
    """Classify Level of Traffic Stress (LTS) from 1 (best) to 4 (worst)"""
    ft = str(facility_type or "").upper().strip()
    
    try:
        sp = float(speed_limit) if speed_limit is not None else 30
    except (ValueError, TypeError):
        sp = 30
    
    try:
        ln = int(lanes) if lanes is not None else 2
    except (ValueError, TypeError):
        ln = 2

    protected_facilities = {
        "PROTECTED BIKELANE", "PROTECTED BIKE LANE", 
        "SHARED USE PATH", "MIXED USE PATH", "BIKE PATH"
    }
    
    buffered_facilities = {
        "BUFFERED BIKELANE", "BUFFERED BIKE LANE", 
        "UNBUFFERED BIKELANE", "UNBUFFERED BIKE LANE"
    }
    
    shared_facilities = {
        "SHARED LANE", "BIKE ROUTE"
    }

    if ft in protected_facilities:
        return 1
    elif ft in buffered_facilities and sp <= 30:
        return 2
    elif ft in shared_facilities and sp <= 35 and ln <= 2:
        return 3
    else:
        return 4

# =============================================================================
# ENHANCED BIKE ROUTING CLASS
# =============================================================================

class EnhancedBikeRouter:
    def __init__(self):
        self.roads_gdf = None
        self.roads_gdf_m = None
        self.transit_stops_gdf = None
        self.geopandas_enabled = GEOPANDAS_AVAILABLE

    def load_shapefiles(self):
        """Load shapefiles with support for compressed archives"""
        if not self.geopandas_enabled:
            logger.warning("GeoPandas not available - using basic OSRM routing only")
            return

        try:
            # Handle roads shapefile/archive
            roads_loaded = False
            
            # Try extracting archive first
            if os.path.exists(ROADS_ARCHIVE):
                extract_if_archived(ROADS_ARCHIVE)
            
            # Load roads shapefile
            if os.path.exists(ROADS_SHAPEFILE):
                logger.info(f"Loading roads from: {ROADS_SHAPEFILE}")
                self.roads_gdf = gpd.read_file(ROADS_SHAPEFILE)
                logger.info(f"Loaded roads: {len(self.roads_gdf)} features")
                self._standardize_roads_columns()
                self.roads_gdf_m = self._to_meters(self.roads_gdf)
                roads_loaded = True
            else:
                logger.warning(f"Roads shapefile not found: {ROADS_SHAPEFILE}")

            # Handle transit stops shapefile/archive  
            if os.path.exists(TRANSIT_STOPS_ARCHIVE):
                extract_if_archived(TRANSIT_STOPS_ARCHIVE)
            
            if os.path.exists(TRANSIT_STOPS_SHAPEFILE):
                logger.info(f"Loading transit stops from: {TRANSIT_STOPS_SHAPEFILE}")
                self.transit_stops_gdf = gpd.read_file(TRANSIT_STOPS_SHAPEFILE)
                logger.info(f"Loaded transit stops: {len(self.transit_stops_gdf)} features")
            else:
                logger.warning(f"Transit stops shapefile not found: {TRANSIT_STOPS_SHAPEFILE}")

            if roads_loaded:
                logger.info("Enhanced bike routing with GeoPandas analysis enabled")
            else:
                logger.info("Using basic OSRM routing only")
                
        except Exception as e:
            logger.error(f"Error loading shapefiles: {e}")
            logger.info("Falling back to basic OSRM routing")

    def _standardize_roads_columns(self):
        """Standardize column names for consistent access"""
        if self.roads_gdf is None:
            return
            
        cols = self.roads_gdf.columns

        total_score_f = _map_field(cols, [
            "TOTAL_SCOR", "Total_Scor", "TOTAL_SCORE", "Total_Score", 
            "Bike_Score", "bike_score", "BIKE_SCORE"
        ])

        fac_type_f = _map_field(cols, [
            "FACILITY_T", "Facility_Type", "FACILITY_TYPE", "facility_type"
        ])

        speed_f = _map_field(cols, [
            "SPEED", "Speed_Limit", "SPEED_LIMIT", "SPD_LIM"
        ])

        lanes_f = _map_field(cols, [
            "Lane_Count", "Lanes", "LANES", "Num_Lanes", "NUM_LANES"
        ])

        try:
            if total_score_f:
                score_values = pd.to_numeric(self.roads_gdf[total_score_f], errors='coerce').fillna(0)
                self.roads_gdf["Bike_Score__canon"] = score_values
                self.roads_gdf["Directional_Score__canon"] = 100
            else:
                self.roads_gdf["Bike_Score__canon"] = 0
                self.roads_gdf["Directional_Score__canon"] = 100
        except:
            self.roads_gdf["Bike_Score__canon"] = 0
            self.roads_gdf["Directional_Score__canon"] = 100

        self.roads_gdf["Facility_Type__canon"] = (
            self.roads_gdf[fac_type_f].fillna("NO BIKELANE") 
            if fac_type_f else "NO BIKELANE"
        )

        try:
            self.roads_gdf["Speed_Limit__canon"] = (
                pd.to_numeric(self.roads_gdf[speed_f], errors='coerce').fillna(30)
                if speed_f else 30
            )
        except:
            self.roads_gdf["Speed_Limit__canon"] = 30

        try:
            self.roads_gdf["Lanes__canon"] = (
                pd.to_numeric(self.roads_gdf[lanes_f], errors='coerce').fillna(2)
                if lanes_f else 2
            )
        except:
            self.roads_gdf["Lanes__canon"] = 2

        logger.info("Standardized road attribute columns")

    def _to_meters(self, gdf):
        """Return a projected (meters) copy for accurate calculations"""
        if gdf is None or gdf.empty:
            return gdf
            
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        
        try:
            target_crs = gdf.estimate_utm_crs()
        except Exception:
            target_crs = "EPSG:3857"
            
        try:
            return gdf.to_crs(target_crs)
        except Exception:
            return gdf

    def _sample_line_points(self, line, num_points=5):
        """Sample points along a line at regular intervals."""
        try:
            if line.is_empty or line.length == 0:
                return []
                
            points = []
            for i in range(num_points):
                distance = (i / (num_points - 1)) * line.length if num_points > 1 else 0
                point = line.interpolate(distance)
                points.append((point.x, point.y))
            return points
        except Exception:
            return []

    def _calculate_average_bearing(self, points):
        """Calculate average bearing (direction) from a series of points."""
        try:
            if len(points) < 2:
                return 0
                
            bearings = []
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                
                dx = x2 - x1
                dy = y2 - y1
                
                if dx == 0 and dy == 0:
                    continue
                    
                bearing = math.atan2(dy, dx) * 180 / math.pi
                bearing = (bearing + 360) % 360
                bearings.append(bearing)
            
            if not bearings:
                return 0
                
            sin_sum = sum(math.sin(math.radians(b)) for b in bearings)
            cos_sum = sum(math.cos(math.radians(b)) for b in bearings)
            
            mean_bearing = math.atan2(sin_sum, cos_sum) * 180 / math.pi
            return (mean_bearing + 360) % 360
            
        except Exception:
            return 0

    def _calculate_alignment_score(self, route_line, road_line):
        """Calculate how well the road segment aligns with the route direction."""
        try:
            route_points = self._sample_line_points(route_line, num_points=5)
            road_points = self._sample_line_points(road_line, num_points=5)
            
            if len(route_points) < 2 or len(road_points) < 2:
                return 0.5
            
            route_bearing = self._calculate_average_bearing(route_points)
            road_bearing = self._calculate_average_bearing(road_points)
            
            angle_diff = abs(route_bearing - road_bearing)
            angle_diff = min(angle_diff, 360 - angle_diff)
            angle_diff = min(angle_diff, 180 - angle_diff)
            
            alignment_score = 1 - (angle_diff / 90.0)
            return max(0, min(1, alignment_score))
            
        except Exception:
            return 0.5

    def _calculate_proximity_score(self, route_line, road_line):
        """Calculate proximity score based on distance."""
        try:
            distance = route_line.distance(road_line)
            
            if distance <= 1:
                return 1.0
            elif distance <= 10:
                return 0.8 + 0.2 * (10 - distance) / 9
            elif distance <= 25:
                return 0.3 + 0.5 * (25 - distance) / 15
            else:
                return max(0, 0.3 * math.exp(-(distance - 25) / 25))
                
        except Exception:
            return 0.1

    def _calculate_coverage_score(self, route_line, road_line):
        """Calculate how much of the route length is covered."""
        try:
            road_buffer = road_line.buffer(15)
            intersection = route_line.intersection(road_buffer)
            
            if intersection.is_empty:
                return 0
                
            coverage_length = intersection.length if hasattr(intersection, 'length') else 0
            route_length = route_line.length
            
            coverage_ratio = coverage_length / route_length if route_length > 0 else 0
            return min(1.0, coverage_ratio)
            
        except Exception:
            return 0.1

    def analyze_route_segments(self, route_geometry):
        """Enhanced route segment analysis using alignment-based selection"""
        if not self.geopandas_enabled or self.roads_gdf is None:
            # Return basic default analysis
            return [], 50.0, {"BASIC OSRM": {"length_miles": 0.0, "percentage": 100.0, "avg_score": 50.0}}

        try:
            if not route_geometry or not route_geometry.get("coordinates"):
                return [], 50.0, {"NO DATA": {"length_miles": 0.0, "percentage": 100.0, "avg_score": 0.0}}

            coords = route_geometry["coordinates"]
            if len(coords) < 2:
                return [], 50.0, {"NO DATA": {"length_miles": 0.0, "percentage": 100.0, "avg_score": 0.0}}
                
            route_ll = LineString([(float(lon), float(lat)) for lon, lat in coords])
            route_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[route_ll], crs="EPSG:4326")

            if self.roads_gdf is None or self.roads_gdf.empty:
                route_length_miles = route_ll.length * 69.0
                return [], 50.0, {"NO SHAPEFILE DATA": {"length_miles": route_length_miles, "percentage": 100.0, "avg_score": 50.0}}

            route_m = self._to_meters(route_gdf)
            roads_m = self.roads_gdf_m

            if route_m is None or roads_m is None:
                route_length_miles = route_ll.length * 69.0
                return [], 50.0, {"PROJECTION ERROR": {"length_miles": route_length_miles, "percentage": 100.0, "avg_score": 50.0}}

            route_line_m = route_m.geometry.iloc[0]
            
            initial_buffer = 50
            route_buffered = route_line_m.buffer(initial_buffer)
            candidate_mask = roads_m.geometry.intersects(route_buffered)
            candidates = roads_m[candidate_mask].copy()

            if candidates.empty:
                route_length_miles = route_ll.length * 69.0
                return [], 50.0, {"NO CANDIDATES": {"length_miles": route_length_miles, "percentage": 100.0, "avg_score": 50.0}}

            selected_segments = []
            
            for idx, road_row in candidates.iterrows():
                road_geom = road_row.geometry
                if road_geom is None or road_geom.is_empty:
                    continue
                    
                alignment_score = self._calculate_alignment_score(route_line_m, road_geom)
                proximity_score = self._calculate_proximity_score(route_line_m, road_geom)
                coverage_score = self._calculate_coverage_score(route_line_m, road_geom)
                
                overall_match_score = (
                    alignment_score * 0.4 +
                    proximity_score * 0.4 +
                    coverage_score * 0.2
                )
                
                if overall_match_score >= 0.3:
                    bike_score = float(road_row.get("Bike_Score__canon", 0))
                    dir_score = float(road_row.get("Directional_Score__canon", 100))
                    facility_type = str(road_row.get("Facility_Type__canon", "NO BIKELANE"))
                    speed_limit = float(road_row.get("Speed_Limit__canon", 30))
                    lanes = int(road_row.get("Lanes__canon", 2))

                    composite_score = (bike_score * dir_score) / 100.0
                    lts = classify_lts(facility_type, speed_limit, lanes)
                    
                    try:
                        seg_length_m = float(road_geom.length) * proximity_score
                    except:
                        seg_length_m = 100.0

                    if seg_length_m > 0:
                        selected_segments.append({
                            "facility_type": facility_type,
                            "bike_score": round(bike_score, 1),
                            "directional_score": round(dir_score, 1),
                            "composite_score": round(composite_score, 2),
                            "LTS": lts,
                            "length_ft": round(seg_length_m * 3.28084, 1),
                            "length_m": seg_length_m,
                            "speed_limit": speed_limit,
                            "lanes": lanes,
                            "match_quality": round(overall_match_score, 3)
                        })

            if not selected_segments:
                route_length_miles = route_ll.length * 69.0
                return [], 50.0, {"NO MATCHES": {"length_miles": route_length_miles, "percentage": 100.0, "avg_score": 50.0}}

            total_length_m = sum(seg["length_m"] for seg in selected_segments)
            weighted_score_sum = sum(seg["composite_score"] * seg["length_m"] for seg in selected_segments)
            overall_score = weighted_score_sum / total_length_m if total_length_m > 0 else 50.0

            facility_buckets = {}
            for seg in selected_segments:
                facility_type = seg["facility_type"]
                length_m = seg["length_m"]
                
                if facility_type not in facility_buckets:
                    facility_buckets[facility_type] = {"length_m": 0.0, "count": 0, "score_sum": 0.0}
                
                facility_buckets[facility_type]["length_m"] += length_m
                facility_buckets[facility_type]["count"] += 1
                facility_buckets[facility_type]["score_sum"] += seg["composite_score"]

            facility_stats = {}
            for facility_type, stats in facility_buckets.items():
                length_miles = stats["length_m"] * 0.000621371
                percentage = (stats["length_m"] / total_length_m) * 100.0 if total_length_m > 0 else 0.0
                avg_score = stats["score_sum"] / stats["count"] if stats["count"] > 0 else 0.0
                
                facility_stats[facility_type] = {
                    "length_miles": round(length_miles, 3),
                    "percentage": round(percentage, 1),
                    "count": stats["count"],
                    "avg_score": round(avg_score, 1)
                }
            
            return selected_segments, round(overall_score, 1), facility_stats

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return [], 50.0, {"ANALYSIS ERROR": {"length_miles": 0.0, "percentage": 100.0, "avg_score": 0.0}}

    def find_nearby_transit_stops(self, lat: float, lon: float, radius_km: float = 0.5) -> List[Dict]:
        """Find nearby transit stops using GeoPandas"""
        if not self.geopandas_enabled or self.transit_stops_gdf is None:
            return []
        
        try:
            point = Point(lon, lat)
            point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs="EPSG:4326")
            
            # Convert to meters for accurate distance calculation
            point_m = self._to_meters(point_gdf)
            stops_m = self._to_meters(self.transit_stops_gdf)
            
            if point_m is None or stops_m is None:
                return []
            
            # Calculate distances
            distances = stops_m.geometry.distance(point_m.geometry.iloc[0])
            radius_m = radius_km * 1000
            
            # Filter by radius
            nearby_mask = distances <= radius_m
            nearby_stops = self.transit_stops_gdf[nearby_mask].copy()
            nearby_stops['distance_m'] = distances[nearby_mask]
            
            # Sort by distance
            nearby_stops = nearby_stops.sort_values('distance_m')
            
            # Extract stop information
            stops = []
            for idx, stop in nearby_stops.head(5).iterrows():
                geom = stop.geometry
                if geom is None:
                    continue
                    
                # Try to find name field
                name_field = _map_field(stop.index, ["stop_name", "name", "NAME", "Stop_Name"])
                stop_name = stop[name_field] if name_field else f"Transit Stop {idx}"
                
                stops.append({
                    "id": str(idx),
                    "name": str(stop_name),
                    "lat": geom.y,
                    "lon": geom.x,
                    "distance_m": round(stop['distance_m'], 1),
                    "distance_km": round(stop['distance_m'] / 1000, 3)
                })
            
            return stops
            
        except Exception as e:
            logger.error(f"Error finding nearby transit stops: {e}")
            return []

# Create global router instance
enhanced_router = EnhancedBikeRouter()

# =============================================================================
# STARTUP EVENT - Extract archives and load shapefiles
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Extract archives and load shapefiles on startup"""
    logger.info("Starting up enhanced bike-bus-bike routing service...")
    
    # Extract any compressed archives
    if os.path.exists(ROADS_ARCHIVE):
        extract_if_archived(ROADS_ARCHIVE)
    if os.path.exists(TRANSIT_STOPS_ARCHIVE):
        extract_if_archived(TRANSIT_STOPS_ARCHIVE)
    
    # Load shapefiles
    enhanced_router.load_shapefiles()
    
    logger.info("Startup complete")

# =============================================================================
# ENHANCED BIKE ROUTING FUNCTIONS
# =============================================================================

def calculate_bike_route_enhanced(start_coords: List[float], end_coords: List[float], route_name="Enhanced Bike Route"):
    """Create enhanced bike route using OSRM + GeoPandas analysis"""
    try:
        coords = f"{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
        url = f"{OSRM_SERVER}/route/v1/cycling/{coords}"
        params = {"overview": "full", "geometries": "polyline", "steps": "false", "alternatives": "false"}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        if data.get("code") != "Ok" or not data.get("routes"): 
            return None

        route = data["routes"][0]
        distance_m = float(route.get("distance", 0.0))
        distance_mi = distance_m * 0.000621371

        if USE_OSRM_DURATION and route.get("duration") is not None:
            duration_min = float(route["duration"]) / 60.0
        else:
            duration_min = (distance_mi / BIKE_SPEED_MPH) * 60.0

        coords_lonlat = decode_polyline_to_lonlat(route["geometry"])
        
        route_data = {
            "name": route_name,
            "length_miles": round(distance_mi, 3),
            "travel_time_minutes": round(duration_min, 1),
            "travel_time_formatted": format_time_duration(duration_min),
            "geometry": {"type": "LineString", "coordinates": coords_lonlat},
            "route_type": "bike"
        }
        
        # Enhanced analysis using GeoPandas (if available)
        segments, overall_score, facility_stats = enhanced_router.analyze_route_segments(route_data["geometry"])
        
        route_data.update({
            "segments": segments,
            "overall_score": overall_score,
            "facility_stats": facility_stats
        })

        return route_data
    except Exception as e:
        logger.error(f"Enhanced bike route error: {e}")
        return None

# =============================================================================
# GOOGLE TRANSIT FUNCTIONS
# =============================================================================

def get_transit_routes_google(origin: Tuple[float, float], destination: Tuple[float, float], departure_time: str = "now", max_alternatives: int = 3) -> Dict:
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
            "transit_routing_preference": "fewer_transfers",
            "key": GOOGLE_API_KEY,
        }
        
        r = requests.get(GMAPS_DIRECTIONS_URL, params=params, timeout=30)
        data = r.json()
        
        if data.get("status") != "OK":
            error_msg = data.get("error_message", f"Google Directions status: {data.get('status')}")
            return {"error": error_msg}
        
        routes = []
        for idx, rd in enumerate(data.get("routes", [])[:max_alternatives]):
            parsed = parse_google_transit_route_enhanced(rd, idx)
            if parsed: 
                routes.append(parsed)
        
        if not routes: 
            return {"error": "No transit routes found"}
        
        return {"routes": routes, "service": "Google Maps Transit + Real-time", "total_routes": len(routes)}
    except Exception as e:
        logger.error(f"Google Directions error: {e}")
        return {"error": str(e)}

def parse_google_transit_route_enhanced(route_data: Dict, route_index: int) -> Optional[Dict]:
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

        for i, s in enumerate(steps_raw):
            ps = parse_transit_step_enhanced(s, i)
            if ps:
                steps.append(ps)
                if ps["travel_mode"] == "TRANSIT":
                    transit_boardings += 1
            if s.get("polyline", {}).get("points"):
                route_geometry.extend(decode_polyline_to_lonlat(s["polyline"]["points"]))

        transfers = max(0, transit_boardings - 1)
        
        enhanced_route = {
            "route_number": route_index + 1,
            "name": f"Transit Route {route_index + 1}" + (f" ({transfers} transfers)" if transfers else ""),
            "duration_minutes": dur_min,
            "duration_text": format_time_duration(dur_min),
            "distance_miles": dist_mi,
            "departure_time": parse_epoch_to_hhmm(dep_ts),
            "arrival_time": parse_epoch_to_hhmm(arr_ts),
            "transfers": transfers,
            "route_geometry": route_geometry,
            "steps": steps,
            "service": "Google Maps Transit",
            "route_type": "transit",
            "realtime_enhanced": True,
            "enhancement_timestamp": datetime.datetime.now().isoformat()
        }
        
        return enhanced_route
    except Exception as e:
        logger.error(f"Transit route parsing error: {e}")
        return None

def parse_transit_step_enhanced(step: Dict, step_index: int) -> Optional[Dict]:
    try:
        mode = step.get("travel_mode", "UNKNOWN")
        
        # Clean HTML from instructions
        instruction = step.get("html_instructions", "")
        instruction = re.sub(r'<[^>]*>', '', instruction)
        
        dur_s = step.get("duration", {}).get("value", 0)
        dur_min = round(dur_s / 60.0, 1)
        dist_m = step.get("distance", {}).get("value", 0)
        dist_mi = round(dist_m * 0.000621371, 2)

        step_data = {
            "step_number": step_index + 1,
            "travel_mode": mode,
            "instruction": instruction,
            "duration_seconds": dur_s,
            "duration_minutes": dur_min,
            "duration_text": format_time_duration(dur_min),
            "distance_meters": dist_m,
            "distance_miles": dist_mi
        }

        if mode == "TRANSIT" and "transit_details" in step:
            transit = step["transit_details"]
            line = transit.get("line", {})
            vehicle = line.get("vehicle", {})
            agencies = line.get("agencies", [])

            step_data.update({
                "transit_line": line.get("short_name", line.get("name", "Unknown Line")),
                "transit_line_color": line.get("color", "#1f8dd6"),
                "transit_vehicle_type": vehicle.get("type", "BUS"),
                "transit_vehicle_name": vehicle.get("name", "Bus"),
                "transit_agency": agencies[0].get("name", "Transit Agency") if agencies else "Transit Agency"
            })

            departure_stop = transit.get("departure_stop", {})
            arrival_stop = transit.get("arrival_stop", {})

            step_data.update({
                "departure_stop_name": departure_stop.get("name", "Unknown Stop"),
                "departure_stop_location": departure_stop.get("location", {}),
                "arrival_stop_name": arrival_stop.get("name", "Unknown Stop"),
                "arrival_stop_location": arrival_stop.get("location", {}),
                "num_stops": transit.get("num_stops", 0)
            })

            departure_time = transit.get("departure_time", {})
            arrival_time = transit.get("arrival_time", {})

            step_data.update({
                "scheduled_departure": parse_epoch_to_hhmm(departure_time.get("value")),
                "scheduled_arrival": parse_epoch_to_hhmm(arrival_time.get("value")),
                "departure_timestamp": departure_time.get("value", 0),
                "arrival_timestamp": arrival_time.get("value", 0)
            })

            step_data["headsign"] = transit.get("headsign", "")

            if "fare" in line:
                fare = line["fare"]
                step_data.update({
                    "fare_text": fare.get("text", ""),
                    "fare_value": fare.get("value", 0),
                    "fare_currency": fare.get("currency", "USD")
                })

        return step_data
    except Exception as e:
        logger.error(f"Transit step parsing error: {e}")
        return None

# =============================================================================
# BIKE-BUS-BIKE ROUTE ANALYSIS
# =============================================================================

def analyze_bike_bus_bike_routes(start_coords: List[float], end_coords: List[float], departure_time: str = "now"):
    """Main function to analyze bike-bus-bike routes"""
    try:
        logger.info(f"Analyzing bike-bus-bike routes from {start_coords} to {end_coords}")
        
        # Find nearby transit stops for start and end points
        start_stops = enhanced_router.find_nearby_transit_stops(start_coords[1], start_coords[0], radius_km=0.8)
        end_stops = enhanced_router.find_nearby_transit_stops(end_coords[1], end_coords[0], radius_km=0.8)
        
        if not start_stops or not end_stops:
            # Fall back to Google Places API if no local transit data
            start_stops = find_nearby_transit_google(start_coords[1], start_coords[0])
            end_stops = find_nearby_transit_google(end_coords[1], end_coords[0])
        
        if not start_stops:
            return {"error": "No transit stops found near start location"}
        if not end_stops:
            return {"error": "No transit stops found near end location"}

        # Use the closest stops
        start_stop = start_stops[0]
        end_stop = end_stops[0]
        
        # Ensure we have different stops
        if start_stop["id"] == end_stop["id"]:
            if len(start_stops) > 1:
                end_stop = start_stops[1]
            elif len(end_stops) > 1:
                end_stop = end_stops[1]
            else:
                return {"error": "Only one transit stop found - not suitable for multimodal routing"}

        # Create bike legs
        bike_leg_1 = calculate_bike_route_enhanced(start_coords, [start_stop["lon"], start_stop["lat"]], "Bike to Transit")
        bike_leg_2 = calculate_bike_route_enhanced([end_stop["lon"], end_stop["lat"]], end_coords, "Transit to Destination")
        
        if not bike_leg_1:
            return {"error": "Could not create bike route to transit stop"}
        if not bike_leg_2:
            return {"error": "Could not create bike route from transit stop"}

        # Get transit options
        transit_result = get_transit_routes_google(
            (start_stop["lon"], start_stop["lat"]), 
            (end_stop["lon"], end_stop["lat"]), 
            departure_time
        )
        
        if "error" in transit_result:
            transit_routes = []
        else:
            transit_routes = transit_result.get("routes", [])

        # Create complete multimodal routes
        complete_routes = []
        for i, transit_route in enumerate(transit_routes):
            total_time = (bike_leg_1["travel_time_minutes"] + 
                         transit_route["duration_minutes"] + 
                         bike_leg_2["travel_time_minutes"] + 
                         5)  # Add 5 minutes for transfers/boarding
            
            total_distance = (bike_leg_1["length_miles"] + 
                            transit_route["distance_miles"] + 
                            bike_leg_2["length_miles"])
            
            bike_distance = bike_leg_1["length_miles"] + bike_leg_2["length_miles"]
            
            # Calculate weighted bike safety score
            total_bike_length = bike_leg_1["length_miles"] + bike_leg_2["length_miles"]
            if total_bike_length > 0:
                weighted_score = ((bike_leg_1["overall_score"] * bike_leg_1["length_miles"]) +
                                (bike_leg_2["overall_score"] * bike_leg_2["length_miles"])) / total_bike_length
            else:
                weighted_score = 0

            complete_route = {
                "id": i + 1,
                "name": f"Bike-Bus-Bike Option {i + 1}",
                "type": "multimodal",
                "summary": {
                    "total_time_minutes": round(total_time, 1),
                    "total_time_formatted": format_time_duration(total_time),
                    "total_distance_miles": round(total_distance, 2),
                    "bike_distance_miles": round(bike_distance, 2),
                    "transit_distance_miles": round(transit_route["distance_miles"], 2),
                    "bike_percentage": round((bike_distance / total_distance) * 100, 1) if total_distance > 0 else 0,
                    "average_bike_score": round(weighted_score, 1),
                    "transfers": transit_route.get("transfers", 0),
                    "departure_time": transit_route.get("departure_time", "Unknown"),
                    "arrival_time": transit_route.get("arrival_time", "Unknown")
                },
                "legs": [
                    {
                        "type": "bike",
                        "name": "Bike to Transit",
                        "route": bike_leg_1,
                        "order": 1
                    },
                    {
                        "type": "transit",
                        "name": f"Transit Route {i + 1}",
                        "route": transit_route,
                        "start_stop": start_stop,
                        "end_stop": end_stop,
                        "order": 2
                    },
                    {
                        "type": "bike",
                        "name": "Transit to Destination",
                        "route": bike_leg_2,
                        "order": 3
                    }
                ]
            }
            complete_routes.append(complete_route)

        # Add direct bike route for comparison
        direct_bike_route = calculate_bike_route_enhanced(start_coords, end_coords, "Direct Bike Route")
        if direct_bike_route:
            direct_route = {
                "id": len(complete_routes) + 1,
                "name": "Direct Bike Route",
                "type": "direct_bike",
                "summary": {
                    "total_time_minutes": direct_bike_route["travel_time_minutes"],
                    "total_time_formatted": direct_bike_route["travel_time_formatted"],
                    "total_distance_miles": direct_bike_route["length_miles"],
                    "bike_distance_miles": direct_bike_route["length_miles"],
                    "transit_distance_miles": 0,
                    "bike_percentage": 100,
                    "average_bike_score": direct_bike_route["overall_score"],
                    "transfers": 0,
                    "departure_time": "Immediate",
                    "arrival_time": "Flexible"
                },
                "legs": [
                    {
                        "type": "bike",
                        "name": "Direct Bike Route",
                        "route": direct_bike_route,
                        "order": 1
                    }
                ]
            }
            complete_routes.append(direct_route)

        # Sort by total time
        complete_routes.sort(key=lambda x: x["summary"]["total_time_minutes"])

        result = {
            "success": True,
            "analysis_type": "bike_bus_bike",
            "routes": complete_routes,
            "transit_stops": {
                "start_stop": start_stop,
                "end_stop": end_stop,
                "all_start_stops": start_stops[:3],
                "all_end_stops": end_stops[:3]
            },
            "statistics": {
                "total_options": len(complete_routes),
                "multimodal_options": len([r for r in complete_routes if r["type"] == "multimodal"]),
                "direct_bike_options": len([r for r in complete_routes if r["type"] == "direct_bike"]),
                "fastest_option": complete_routes[0]["name"] if complete_routes else None,
                "fastest_time": complete_routes[0]["summary"]["total_time_formatted"] if complete_routes else None
            },
            "bike_speed_mph": BIKE_SPEED_MPH,
            "analysis_timestamp": datetime.datetime.now().isoformat()
        }

        return result

    except Exception as e:
        logger.error(f"Bike-bus-bike analysis error: {e}")
        return {"error": str(e)}

def find_nearby_transit_google(lat: float, lon: float, radius_m: int = 500) -> List[Dict]:
    """Find nearby transit stops using Google Places API as fallback"""
    require_google_key()
    try:
        params = {
            "location": f"{lat},{lon}",
            "radius": radius_m,
            "type": "transit_station",
            "key": GOOGLE_API_KEY
        }
        
        r = requests.get(GMAPS_PLACES_NEARBY_URL, params=params, timeout=30)
        data = r.json()
        
        if data.get("status") != "OK":
            return []
        
        stops = []
        for i, place in enumerate(data.get("results", [])[:5]):
            location = place.get("geometry", {}).get("location", {})
            stops.append({
                "id": f"google_{place.get('place_id', i)}",
                "name": place.get("name", f"Transit Stop {i+1}"),
                "lat": location.get("lat", lat),
                "lon": location.get("lng", lon),
                "distance_m": 0,  # Google doesn't provide exact distance
                "distance_km": 0,
                "source": "google_places"
            })
        
        return stops
    except Exception as e:
        logger.error(f"Google Places transit search error: {e}")
        return []

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Main page with interactive map"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced Bike-Bus-Bike Route Planner</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
        <style>
            body { margin: 0; padding: 0; font-family: 'Segoe UI', sans-serif; }
            .header {
                background: linear-gradient(135deg, #2c3e50, #3498db);
                color: white; padding: 20px; text-align: center;
            }
            .container { display: flex; height: calc(100vh - 80px); }
            #map { flex: 2; }
            #sidebar { flex: 1; padding: 20px; overflow-y: auto; background: #f8f9fa; }
            .controls { margin-bottom: 20px; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select, button {
                width: 100%; padding: 8px; border: 1px solid #ddd;
                border-radius: 4px; font-size: 14px;
            }
            button { background: #3498db; color: white; border: none; cursor: pointer; }
            button:hover { background: #2980b9; }
            button:disabled { background: #bdc3c7; cursor: not-allowed; }
            .route-card {
                background: white; border-radius: 8px; padding: 15px;
                margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                cursor: pointer; border: 2px solid transparent;
            }
            .route-card:hover { border-color: #3498db; }
            .route-card.selected { border-color: #2ecc71; background: #f8fff8; }
            .route-name { font-weight: bold; color: #2c3e50; margin-bottom: 10px; }
            .route-summary { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
            .summary-item { text-align: center; }
            .summary-value { font-weight: bold; color: #3498db; }
            .summary-label { font-size: 12px; color: #7f8c8d; }
            .legs { margin-top: 15px; }
            .leg { margin: 8px 0; padding: 8px; border-left: 4px solid #bdc3c7; }
            .leg.bike { border-left-color: #27ae60; background: #f8fff8; }
            .leg.transit { border-left-color: #3498db; background: #f8fbff; }
            .error { color: #e74c3c; padding: 10px; background: #fdf2f2; border-radius: 4px; }
            .loading { text-align: center; color: #7f8c8d; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚴‍♂️🚌🚴‍♀️ Enhanced Bike-Bus-Bike Route Planner</h1>
            <p>Advanced multimodal routing with GeoPandas-powered bicycle infrastructure analysis</p>
        </div>
        
        <div class="container">
            <div id="map"></div>
            <div id="sidebar">
                <div class="controls">
                    <h3>📍 Route Planning</h3>
                    <p><strong>Instructions:</strong> Click the map to set start (green) and end (red) points, then find routes!</p>
                    
                    <div class="form-group">
                        <label>🕐 Departure Time:</label>
                        <select id="departureTime">
                            <option value="now">Leave Now</option>
                            <option value="custom">Custom Time</option>
                        </select>
                    </div>
                    
                    <div class="form-group" id="customTimeGroup" style="display: none;">
                        <input type="datetime-local" id="customTime">
                    </div>
                    
                    <button id="findRoutesBtn" disabled>🔍 Find Bike-Bus-Bike Routes</button>
                    <button onclick="clearAll()">🗑️ Clear All</button>
                    
                    <div id="coordinates" style="font-size: 12px; margin-top: 10px; color: #666;"></div>
                </div>
                
                <div id="results"></div>
            </div>
        </div>

        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
            let map, startPoint = null, endPoint = null;
            let startMarker = null, endMarker = null, routeLayersGroup;
            let currentRoutes = [];

            function initMap() {
                map = L.map('map').setView([30.3322, -81.6557], 12);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors'
                }).addTo(map);
                
                routeLayersGroup = L.layerGroup().addTo(map);
                
                map.on('click', function(e) {
                    if (!startPoint) {
                        startPoint = e.latlng;
                        if (startMarker) map.removeLayer(startMarker);
                        startMarker = L.marker(startPoint, {
                            icon: L.icon({
                                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
                                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                                iconSize: [25, 41], iconAnchor: [12, 41]
                            })
                        }).addTo(map);
                        updateCoordinates();
                    } else if (!endPoint) {
                        endPoint = e.latlng;
                        if (endMarker) map.removeLayer(endMarker);
                        endMarker = L.marker(endPoint, {
                            icon: L.icon({
                                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
                                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                                iconSize: [25, 41], iconAnchor: [12, 41]
                            })
                        }).addTo(map);
                        updateCoordinates();
                    } else {
                        clearAll();
                        map.fire('click', e);
                    }
                });
            }

            function updateCoordinates() {
                const coords = document.getElementById('coordinates');
                let text = '';
                if (startPoint) text += `Start: ${startPoint.lat.toFixed(5)}, ${startPoint.lng.toFixed(5)}\\n`;
                if (endPoint) text += `End: ${endPoint.lat.toFixed(5)}, ${endPoint.lng.toFixed(5)}`;
                coords.textContent = text;
                
                document.getElementById('findRoutesBtn').disabled = !(startPoint && endPoint);
            }

            function clearAll() {
                startPoint = null; endPoint = null;
                if (startMarker) { map.removeLayer(startMarker); startMarker = null; }
                if (endMarker) { map.removeLayer(endMarker); endMarker = null; }
                routeLayersGroup.clearLayers();
                document.getElementById('results').innerHTML = '';
                updateCoordinates();
                currentRoutes = [];
            }

            async function findRoutes() {
                if (!startPoint || !endPoint) return;
                
                const results = document.getElementById('results');
                results.innerHTML = '<div class="loading">🔄 Analyzing bike-bus-bike routes...</div>';
                
                try {
                    const departureTime = document.getElementById('departureTime').value;
                    let timeParam = 'now';
                    
                    if (departureTime === 'custom') {
                        const customTime = document.getElementById('customTime').value;
                        if (customTime) {
                            timeParam = Math.floor(new Date(customTime).getTime() / 1000);
                        }
                    }
                    
                    const response = await fetch(`/analyze-bike-bus-bike?start_lon=${startPoint.lng}&start_lat=${startPoint.lat}&end_lon=${endPoint.lng}&end_lat=${endPoint.lat}&departure_time=${timeParam}`);
                    const data = await response.json();
                    
                    if (data.error) {
                        results.innerHTML = `<div class="error">❌ ${data.error}</div>`;
                        return;
                    }
                    
                    displayResults(data);
                    
                } catch (error) {
                    results.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
                }
            }

            function displayResults(data) {
                currentRoutes = data.routes || [];
                const results = document.getElementById('results');
                
                if (currentRoutes.length === 0) {
                    results.innerHTML = '<div class="error">❌ No routes found</div>';
                    return;
                }
                
                let html = `<h3>🛣️ Found ${currentRoutes.length} Route Options</h3>`;
                
                currentRoutes.forEach((route, index) => {
                    const typeIcon = route.type === 'multimodal' ? '🚴‍♂️🚌🚴‍♀️' : '🚴‍♂️';
                    const typeLabel = route.type === 'multimodal' ? 'Multimodal' : 'Direct Bike';
                    
                    html += `
                        <div class="route-card" onclick="selectRoute(${index})">
                            <div class="route-name">${typeIcon} ${route.name}</div>
                            <div class="route-summary">
                                <div class="summary-item">
                                    <div class="summary-value">${route.summary.total_time_formatted}</div>
                                    <div class="summary-label">Total Time</div>
                                </div>
                                <div class="summary-item">
                                    <div class="summary-value">${route.summary.total_distance_miles.toFixed(1)} mi</div>
                                    <div class="summary-label">Distance</div>
                                </div>
                                <div class="summary-item">
                                    <div class="summary-value">${route.summary.bike_distance_miles.toFixed(1)} mi</div>
                                    <div class="summary-label">Bike Distance</div>
                                </div>
                                <div class="summary-item">
                                    <div class="summary-value">${route.summary.average_bike_score}</div>
                                    <div class="summary-label">Bike Safety</div>
                                </div>
                            </div>
                            <div class="legs">
                                ${route.legs.map(leg => `
                                    <div class="leg ${leg.type}">
                                        <strong>${leg.type === 'bike' ? '🚴‍♂️' : '🚌'} ${leg.name}</strong><br>
                                        ${leg.route.length_miles?.toFixed(2) || leg.route.distance_miles?.toFixed(2) || '0'} mi • 
                                        ${leg.route.travel_time_formatted || leg.route.duration_text || 'N/A'}
                                        ${leg.type === 'bike' && leg.route.overall_score ? ` • Safety: ${leg.route.overall_score}` : ''}
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    `;
                });
                
                results.innerHTML = html;
                
                // Auto-select first route
                if (currentRoutes.length > 0) {
                    setTimeout(() => selectRoute(0), 500);
                }
            }

            function selectRoute(index) {
                // Update selection styling
                document.querySelectorAll('.route-card').forEach((card, i) => {
                    card.classList.toggle('selected', i === index);
                });
                
                // Clear existing route layers
                routeLayersGroup.clearLayers();
                
                const route = currentRoutes[index];
                if (!route) return;
                
                // Display each leg
                route.legs.forEach((leg, legIndex) => {
                    displayLegOnMap(leg, legIndex);
                });
                
                // Add transit stops for multimodal routes
                if (route.type === 'multimodal') {
                    const transitLeg = route.legs.find(leg => leg.type === 'transit');
                    if (transitLeg && transitLeg.start_stop && transitLeg.end_stop) {
                        // Start stop
                        L.marker([transitLeg.start_stop.lat, transitLeg.start_stop.lon], {
                            icon: L.divIcon({
                                html: '🚌',
                                iconSize: [20, 20],
                                className: 'emoji-icon'
                            })
                        }).bindPopup(`<strong>🚌 ${transitLeg.start_stop.name}</strong>`).addTo(routeLayersGroup);
                        
                        // End stop
                        L.marker([transitLeg.end_stop.lat, transitLeg.end_stop.lon], {
                            icon: L.divIcon({
                                html: '🚌',
                                iconSize: [20, 20],
                                className: 'emoji-icon'
                            })
                        }).bindPopup(`<strong>🚌 ${transitLeg.end_stop.name}</strong>`).addTo(routeLayersGroup);
                    }
                }
                
                // Fit map to show all layers
                if (routeLayersGroup.getLayers().length > 0) {
                    try {
                        map.fitBounds(routeLayersGroup.getBounds(), {padding: [20, 20]});
                    } catch (e) {
                        console.warn('Could not fit bounds');
                    }
                }
            }

            function displayLegOnMap(leg, legIndex) {
                const route = leg.route;
                if (!route.geometry || !route.geometry.coordinates) return;
                
                const coords = route.geometry.coordinates.map(coord => [coord[1], coord[0]]);
                const color = leg.type === 'bike' ? '#27ae60' : '#3498db';
                
                const polyline = L.polyline(coords, {
                    color: color,
                    weight: 6,
                    opacity: 0.8
                }).addTo(routeLayersGroup);
                
                polyline.bindPopup(`
                    <strong>${leg.type === 'bike' ? '🚴‍♂️' : '🚌'} ${leg.name}</strong><br>
                    Distance: ${route.length_miles?.toFixed(2) || route.distance_miles?.toFixed(2)} mi<br>
                    Time: ${route.travel_time_formatted || route.duration_text}
                    ${leg.type === 'bike' && route.overall_score ? `<br>Safety Score: ${route.overall_score}` : ''}
                `);
            }

            // Event listeners
            document.getElementById('departureTime').addEventListener('change', function() {
                const customTimeGroup = document.getElementById('customTimeGroup');
                if (this.value === 'custom') {
                    customTimeGroup.style.display = 'block';
                    const now = new Date();
                    now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
                    document.getElementById('customTime').value = now.toISOString().slice(0, 16);
                } else {
                    customTimeGroup.style.display = 'none';
                }
            });

            document.getElementById('findRoutesBtn').addEventListener('click', findRoutes);

            // Initialize map
            document.addEventListener('DOMContentLoaded', initMap);
        </script>
    </body>
    </html>
    """

@app.get("/status")
async def get_status():
    """API status and configuration info"""
    return {
        "status": "operational",
        "service": "Enhanced OSRM + Google Transit + GeoPandas Planner",
        "version": "3.1.0",
        "features": {
            "geopandas_analysis": enhanced_router.geopandas_enabled,
            "roads_data": enhanced_router.roads_gdf is not None,
            "transit_stops_data": enhanced_router.transit_stops_gdf is not None,
            "google_api": bool(GOOGLE_API_KEY),
            "osrm_server": OSRM_SERVER
        },
        "configuration": {
            "bike_speed_mph": BIKE_SPEED_MPH,
            "use_osrm_duration": USE_OSRM_DURATION
        },
        "data_status": {
            "roads_features": len(enhanced_router.roads_gdf) if enhanced_router.roads_gdf is not None else 0,
            "transit_stops": len(enhanced_router.transit_stops_gdf) if enhanced_router.transit_stops_gdf is not None else 0
        }
    }

@app.get("/bike-route")
async def get_bike_route(
    start_lon: float = Query(..., description="Start longitude"),
    start_lat: float = Query(..., description="Start latitude"), 
    end_lon: float = Query(..., description="End longitude"),
    end_lat: float = Query(..., description="End latitude")
):
    """Get enhanced bicycle route with GeoPandas analysis"""
    try:
        route = calculate_bike_route_enhanced([start_lon, start_lat], [end_lon, end_lat])
        if not route:
            raise HTTPException(status_code=404, detail="No bike route found")
        
        return {
            "success": True,
            "route": route,
            "analysis_timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Bike route endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transit-routes")  
async def get_transit_routes_endpoint(
    start_lon: float = Query(..., description="Start longitude"),
    start_lat: float = Query(..., description="Start latitude"),
    end_lon: float = Query(..., description="End longitude"), 
    end_lat: float = Query(..., description="End latitude"),
    departure_time: str = Query("now", description="Departure time (epoch or 'now')")
):
    """Get transit routes using Google Maps API"""
    try:
        result = get_transit_routes_google(
            (start_lon, start_lat), 
            (end_lon, end_lat), 
            departure_time
        )
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transit routes endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze-bike-bus-bike")
async def analyze_bike_bus_bike_endpoint(
    start_lon: float = Query(..., description="Start longitude"),
    start_lat: float = Query(..., description="Start latitude"),
    end_lon: float = Query(..., description="End longitude"),
    end_lat: float = Query(..., description="End latitude"), 
    departure_time: str = Query("now", description="Departure time (epoch or 'now')")
):
    """Complete bike-bus-bike route analysis"""
    try:
        result = analyze_bike_bus_bike_routes(
            [start_lon, start_lat], 
            [end_lon, end_lat], 
            departure_time
        )
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bike-bus-bike analysis endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/nearby-transit")
async def get_nearby_transit_stops(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    radius_km: float = Query(0.5, description="Search radius in kilometers")
):
    """Find nearby transit stops"""
    try:
        # Try GeoPandas first
        stops = enhanced_router.find_nearby_transit_stops(lat, lon, radius_km)
        
        # Fall back to Google Places if no local data
        if not stops:
            stops = find_nearby_transit_google(lat, lon, int(radius_km * 1000))
        
        return {
            "success": True,
            "stops": stops,
            "search_location": {"lat": lat, "lon": lon},
            "search_radius_km": radius_km,
            "found_count": len(stops)
        }
    except Exception as e:
        logger.error(f"Nearby transit endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-shapefile")
async def upload_shapefile(
    file: UploadFile = File(...),
    shapefile_type: str = Query(..., description="'roads' or 'transit_stops'")
):
    """Upload and process shapefile data"""
    try:
        if not file.filename.endswith(('.zip', '.shp')):
            raise HTTPException(status_code=400, detail="File must be .zip archive or .shp file")
        
        # Save uploaded file
        upload_dir = "./uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract if it's a zip file
        if file.filename.endswith('.zip'):
            extract_if_archived(file_path, upload_dir)
            
            # Find the .shp file in extracted content
            shp_files = []
            for root, dirs, files in os.walk(upload_dir):
                for f in files:
                    if f.endswith('.shp'):
                        shp_files.append(os.path.join(root, f))
            
            if not shp_files:
                raise HTTPException(status_code=400, detail="No .shp file found in archive")
            
            file_path = shp_files[0]
        
        # Load the shapefile based on type
        if shapefile_type == "roads":
            if enhanced_router.geopandas_enabled:
                enhanced_router.roads_gdf = gpd.read_file(file_path)
                enhanced_router._standardize_roads_columns()
                enhanced_router.roads_gdf_m = enhanced_router._to_meters(enhanced_router.roads_gdf)
                logger.info(f"Loaded roads shapefile: {len(enhanced_router.roads_gdf)} features")
            else:
                raise HTTPException(status_code=500, detail="GeoPandas not available")
                
        elif shapefile_type == "transit_stops":
            if enhanced_router.geopandas_enabled:
                enhanced_router.transit_stops_gdf = gpd.read_file(file_path)
                logger.info(f"Loaded transit stops shapefile: {len(enhanced_router.transit_stops_gdf)} features")
            else:
                raise HTTPException(status_code=500, detail="GeoPandas not available")
        else:
            raise HTTPException(status_code=400, detail="shapefile_type must be 'roads' or 'transit_stops'")
        
        return {
            "success": True,
            "message": f"Successfully loaded {shapefile_type} shapefile",
            "features_count": len(enhanced_router.roads_gdf if shapefile_type == "roads" else enhanced_router.transit_stops_gdf),
            "file_path": file_path
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Shapefile upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced OSRM + Google Transit + GeoPandas Planner")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to") 
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info("🚴‍♂️🚌🚴‍♀️ Starting Enhanced Bike-Bus-Bike Route Planner")
    logger.info(f"GeoPandas Available: {GEOPANDAS_AVAILABLE}")
    logger.info(f"OSRM Server: {OSRM_SERVER}")
    logger.info(f"Google API Key Configured: {bool(GOOGLE_API_KEY)}")
    logger.info(f"Bike Speed: {BIKE_SPEED_MPH} mph")
    
    uvicorn.run(
        "enhanced_osrm_google_transit_with_geopandas:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
