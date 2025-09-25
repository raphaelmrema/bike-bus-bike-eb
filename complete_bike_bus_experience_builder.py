# enhanced_osrm_google_transit_with_geopandas.py
# Complete FastAPI app with GeoPandas-based enhanced bike routing analysis
# Optimized for Railway deployment with Git LFS support for large shapefiles

import os
import json
import logging
import requests
import datetime
import math
import sys
import zipfile
import numpy as np
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
# GOOGLE TRANSIT FUNCTIONS (unchanged from your original)
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
            "
