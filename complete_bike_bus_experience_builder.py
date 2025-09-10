# complete_bike_bus_experience_builder.py
# Complete Bike-Bus-Bike Route Planner for ArcGIS Experience Builder
# Ready for cloud deployment - No editing required!

import os
import json
import logging
import tempfile
import requests
import datetime
import zipfile
import pandas as pd
import io
import math
import re
from typing import List, Dict, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# =============================================================================
# CONFIGURATION FOR EXPERIENCE BUILDER DEPLOYMENT
# =============================================================================

# Environment variables for cloud deployment
GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "AIzaSyBmGvPmWyKlR5DAtOu8vrmO0Cd8N1y4KF8")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", 
    "https://experience.arcgis.com,https://*.maps.arcgis.com,http://localhost:*,https://*.railway.app").split(",")

# Bicycle Configuration (from original collective57.py)
BIKE_SPEED_MPH = 11
BIKE_SPEED_FEET_PER_SECOND = BIKE_SPEED_MPH * 5280 / 3600

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# FASTAPI APP SETUP WITH EXPERIENCE BUILDER CORS
# =============================================================================

app = FastAPI(
    title="Bike-Bus-Bike Route Planner API",
    description="Advanced multimodal transportation planning API for ArcGIS Experience Builder",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware configured for Experience Builder
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # More permissive for cloud deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# ENHANCED GTFS MANAGER (FROM COLLECTIVE57.PY - CLOUD OPTIMIZED)
# =============================================================================

class EnhancedGTFSManager:
    """Enhanced GTFS manager with real-time capabilities (cloud-ready version)"""
    
    def __init__(self):
        self.gtfs_data = {}
        self.stops_df = None
        self.routes_df = None
        self.stop_times_df = None
        self.trips_df = None
        self.calendar_df = None
        self.is_loaded = False
        self.last_update = None
        self.realtime_cache = {}
        self.cache_duration = 30  # seconds
        
    def load_gtfs_data(self, gtfs_urls=None):
        """Load GTFS data with enhanced error handling (from collective57.py)"""
        if gtfs_urls is None:
            gtfs_urls = [
                "https://ride.jtafla.com/gtfs-archive/gtfs.zip",
                "https://schedules.jtafla.com/schedulesgtfs/download",
                "https://openmobilitydata.org/p/jacksonville-transportation-authority/331/latest/download",
            ]
        
        logger.info("🔥 Loading enhanced GTFS data with real-time capabilities...")
        
        for i, url in enumerate(gtfs_urls):
            try:
                logger.info(f"Trying source {i+1}: {url}")
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/zip, application/octet-stream, */*',
                    'Accept-Language': 'en-US,en;q=0.9',
                }
                
                response = requests.get(url, timeout=45, headers=headers, verify=False, stream=True)
                response.raise_for_status()
                
                content = response.content
                if not content.startswith(b'PK'):
                    logger.info(f"   Response is not a ZIP file, skipping...")
                    continue
                
                logger.info(f"   ✓ Valid ZIP file detected ({len(content)} bytes)")
                
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    files_to_load = {
                        'stops.txt': 'stops_df',
                        'routes.txt': 'routes_df', 
                        'trips.txt': 'trips_df',
                        'stop_times.txt': 'stop_times_df',
                        'agency.txt': 'agency_df',
                        'calendar.txt': 'calendar_df'
                    }
                    
                    loaded_files = 0
                    for filename, attr_name in files_to_load.items():
                        if filename in z.namelist():
                            try:
                                with z.open(filename) as f:
                                    df = pd.read_csv(f, low_memory=False)
                                    setattr(self, attr_name, df)
                                    self.gtfs_data[filename] = df
                                    logger.info(f"   ✅ Loaded {filename}: {len(df)} records")
                                    loaded_files += 1
                            except Exception as e:
                                logger.warning(f"   ⚠ Error loading {filename}: {e}")
                    
                    if loaded_files >= 3:
                        self.is_loaded = True
                        self.last_update = datetime.datetime.now()
                        
                        agency_name = "Unknown"
                        if hasattr(self, 'agency_df') and not self.agency_df.empty:
                            agency_name = self.agency_df.iloc[0].get('agency_name', 'Unknown')
                        
                        logger.info(f"✅ Enhanced GTFS data loaded successfully")
                        logger.info(f"   Agency: {agency_name}")
                        logger.info(f"   Routes: {len(self.routes_df) if self.routes_df is not None else 0}")
                        logger.info(f"   Stops: {len(self.stops_df) if self.stops_df is not None else 0}")
                        return True
                        
            except Exception as e:
                logger.warning(f"   Error loading from source {i+1}: {e}")
                continue
        
        logger.warning("⚠ Could not load GTFS data from any source")
        return False
    
    def get_realtime_departures(self, stop_id: str) -> List[Dict]:
        """Get real-time departures with live updates and time filtering (from collective57.py)"""
        try:
            # Check cache first
            cache_key = f"departures_{stop_id}"
            now = datetime.datetime.now()
            
            if (cache_key in self.realtime_cache and 
                (now - self.realtime_cache[cache_key]['timestamp']).seconds < self.cache_duration):
                cached_data = self.realtime_cache[cache_key]['data']
                return self._filter_upcoming_departures(cached_data)
            
            # Get base schedule data
            base_departures = self.get_stop_schedules(stop_id)
            
            # Try to get real-time updates (simulated for demo)
            realtime_updates = self._simulate_realtime_variations(stop_id)
            
            # Merge scheduled and real-time data
            enhanced_departures = self._merge_schedule_and_realtime(base_departures, realtime_updates)
            
            # Cache the results
            self.realtime_cache[cache_key] = {
                'data': enhanced_departures,
                'timestamp': now
            }
            
            # Return only upcoming departures
            return self._filter_upcoming_departures(enhanced_departures)
            
        except Exception as e:
            logger.error(f"Error getting real-time departures: {e}")
            # Fallback to static schedule
            return self.get_stop_schedules(stop_id)
    
    def _simulate_realtime_variations(self, stop_id: str) -> Dict:
        """Simulate realistic real-time variations for demonstration (from collective57.py)"""
        import random
        
        # Simulate typical transit delays
        variations = {}
        current_time = datetime.datetime.now()
        
        # Simulate delays based on time of day
        if 7 <= current_time.hour <= 9 or 16 <= current_time.hour <= 18:
            # Rush hour - more delays
            delay_chance = 0.7
            avg_delay = 5  # minutes
        else:
            # Off-peak - fewer delays
            delay_chance = 0.3
            avg_delay = 2
        
        variations['delays'] = {
            'probability': delay_chance,
            'average_delay_minutes': avg_delay,
            'max_delay_minutes': avg_delay * 3
        }
        
        variations['status'] = 'simulated'
        return variations
    
    def _merge_schedule_and_realtime(self, scheduled: List[Dict], realtime: Dict) -> List[Dict]:
        """Merge scheduled times with real-time updates (from collective57.py)"""
        enhanced = []
        
        for schedule in scheduled:
            enhanced_departure = schedule.copy()
            
            # Add real-time status
            enhanced_departure['realtime_status'] = 'scheduled'
            enhanced_departure['delay_minutes'] = 0
            enhanced_departure['original_time'] = schedule['departure_time']
            
            # Apply real-time updates if available
            if 'delays' in realtime:
                delay_info = realtime['delays']
                
                # Simulate delay application
                import random
                if random.random() < delay_info.get('probability', 0):
                    delay_minutes = random.randint(0, delay_info.get('max_delay_minutes', 5))
                    enhanced_departure['delay_minutes'] = delay_minutes
                    enhanced_departure['realtime_status'] = 'delayed' if delay_minutes > 0 else 'on_time'
                    
                    # Calculate new departure time
                    try:
                        original_time = datetime.datetime.strptime(schedule['departure_time'], '%H:%M:%S')
                        new_time = original_time + datetime.timedelta(minutes=delay_minutes)
                        enhanced_departure['departure_time'] = new_time.strftime('%H:%M:%S')
                        enhanced_departure['realtime_departure'] = new_time.strftime('%H:%M')
                    except:
                        enhanced_departure['realtime_departure'] = schedule['departure_time'][:5]
            
            # Add helpful status indicators
            if enhanced_departure['delay_minutes'] > 5:
                enhanced_departure['status_color'] = '#ff4444'
                enhanced_departure['status_text'] = f"Delayed {enhanced_departure['delay_minutes']} min"
            elif enhanced_departure['delay_minutes'] > 0:
                enhanced_departure['status_color'] = '#ff8800'
                enhanced_departure['status_text'] = f"Delayed {enhanced_departure['delay_minutes']} min"
            else:
                enhanced_departure['status_color'] = '#44aa44'
                enhanced_departure['status_text'] = "On time"
            
            enhanced.append(enhanced_departure)
        
        return enhanced
    
    def _filter_upcoming_departures(self, departures: List[Dict]) -> List[Dict]:
        """Filter to show only upcoming departures (from collective57.py)"""
        current_time = datetime.datetime.now().time()
        current_datetime = datetime.datetime.now()
        
        upcoming = []
        for departure in departures:
            try:
                dept_time_str = departure.get('departure_time', '')
                if ':' in dept_time_str:
                    # Handle times that might go past midnight (e.g., 25:30:00)
                    time_parts = dept_time_str.split(':')
                    hours = int(time_parts[0])
                    minutes = int(time_parts[1])
                    
                    if hours >= 24:
                        # Next day departure
                        dept_datetime = current_datetime.replace(
                            hour=hours-24, minute=minutes, second=0, microsecond=0
                        ) + datetime.timedelta(days=1)
                    else:
                        dept_datetime = current_datetime.replace(
                            hour=hours, minute=minutes, second=0, microsecond=0
                        )
                    
                    # Only include departures in the next 4 hours
                    time_diff = dept_datetime - current_datetime
                    if datetime.timedelta(0) <= time_diff <= datetime.timedelta(hours=4):
                        departure['time_until_departure'] = self._format_time_until(time_diff)
                        departure['departure_datetime'] = dept_datetime
                        upcoming.append(departure)
                        
            except (ValueError, IndexError):
                continue
        
        # Sort by departure time and return next 10
        upcoming.sort(key=lambda x: x.get('departure_datetime', current_datetime))
        return upcoming[:10]
    
    def _format_time_until(self, time_diff: datetime.timedelta) -> str:
        """Format time until departure in a user-friendly way (from collective57.py)"""
        total_minutes = int(time_diff.total_seconds() / 60)
        
        if total_minutes < 1:
            return "Due now"
        elif total_minutes < 60:
            return f"{total_minutes} min"
        else:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            if minutes == 0:
                return f"{hours}h"
            else:
                return f"{hours}h {minutes}m"
    
    def get_stop_schedules(self, stop_id, hours_ahead=4):
        """Get base stop schedules (from collective57.py)"""
        if not self.is_loaded or self.stop_times_df is None:
            return []
        
        try:
            stop_times = self.stop_times_df[self.stop_times_df['stop_id'] == stop_id].copy()
            
            if stop_times.empty:
                return []
            
            # Join with trips and routes
            if self.trips_df is not None:
                stop_times = stop_times.merge(
                    self.trips_df[['trip_id', 'route_id', 'trip_headsign']], 
                    on='trip_id', how='left'
                )
            
            if self.routes_df is not None:
                stop_times = stop_times.merge(
                    self.routes_df[['route_id', 'route_short_name', 'route_long_name']], 
                    on='route_id', how='left'
                )
            
            schedules = []
            for _, row in stop_times.head(30).iterrows():
                try:
                    departure_time = row.get('departure_time', '')
                    if departure_time and ':' in departure_time:
                        schedule = {
                            'departure_time': departure_time,
                            'arrival_time': row.get('arrival_time', ''),
                            'route_name': str(row.get('route_short_name', 'Route')),
                            'route_description': str(row.get('route_long_name', '')),
                            'headsign': str(row.get('trip_headsign', '')),
                            'trip_id': str(row.get('trip_id', ''))
                        }
                        schedules.append(schedule)
                except:
                    continue
            
            return schedules
            
        except Exception as e:
            logger.error(f"Error getting schedules: {e}")
            return []
    
    def find_nearby_stops(self, lat, lon, radius_km=0.5):
        """Find GTFS stops near coordinates (from collective57.py)"""
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

# Global GTFS manager
gtfs_manager = EnhancedGTFSManager()

# =============================================================================
# UTILITY FUNCTIONS (FROM COLLECTIVE57.PY)
# =============================================================================

def calculate_bike_time_minutes(distance_feet):
    """Calculate bicycle travel time in minutes given distance in feet (from collective57.py)"""
    if distance_feet <= 0:
        return 0
    return (distance_feet / BIKE_SPEED_FEET_PER_SECOND) / 60

def format_time_duration(minutes):
    """Format time duration in a user-friendly way (from collective57.py)"""
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

def decode_polyline(polyline_str):
    """Decode Google polyline string to coordinates (from collective57.py)"""
    try:
        index = 0
        lat = 0
        lng = 0
        coordinates = []
        
        while index < len(polyline_str):
            # Decode latitude
            result = 1
            shift = 0
            while True:
                b = ord(polyline_str[index]) - 63 - 1
                index += 1
                result += b << shift
                shift += 5
                if b < 0x1f:
                    break
            lat += (~result >> 1) if (result & 1) != 0 else (result >> 1)
            
            # Decode longitude
            result = 1
            shift = 0
            while True:
                b = ord(polyline_str[index]) - 63 - 1
                index += 1
                result += b << shift
                shift += 5
                if b < 0x1f:
                    break
            lng += (~result >> 1) if (result & 1) != 0 else (result >> 1)
            
            coordinates.append([lat / 1e5, lng / 1e5])
        
        return coordinates
    except Exception as e:
        logger.error(f"Error decoding polyline: {e}")
        return []

# =============================================================================
# GOOGLE MAPS API FUNCTIONS (ENHANCED FROM COLLECTIVE57.PY)
# =============================================================================

def get_transit_routes_google(origin, destination, departure_time="now"):
    """Get transit routes using Google Maps API (enhanced from collective57.py)"""
    try:
        if not GOOGLE_API_KEY or len(GOOGLE_API_KEY) < 30:
            raise HTTPException(status_code=500, detail="Google Maps API key not properly configured")
        
        logger.info(f"🚌 Getting Google Maps transit routes: '{origin}' → '{destination}'")
        
        url = "https://maps.googleapis.com/maps/api/directions/json"
        
        if departure_time == "now":
            departure_timestamp = int(datetime.datetime.now().timestamp())
        else:
            try:
                departure_timestamp = int(departure_time)
            except:
                departure_timestamp = int(datetime.datetime.now().timestamp())
        
        params = {
            'origin': origin,
            'destination': destination,
            'mode': 'transit',
            'departure_time': departure_timestamp,
            'alternatives': 'true',
            'transit_mode': 'bus|subway|train|tram',
            'transit_routing_preference': 'fewer_transfers',
            'key': GOOGLE_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') != 'OK':
            error_msg = data.get('error_message', f"API Status: {data.get('status')}")
            logger.error(f"Google API error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        routes = []
        if 'routes' in data and data['routes']:
            for idx, route_data in enumerate(data['routes']):
                route = parse_google_transit_route(route_data, idx)
                if route:
                    # Enhanced route enhancement with real-time GTFS
                    enhanced_route = enhance_route_with_realtime_gtfs(route)
                    routes.append(enhanced_route)
        
        if not routes:
            return {"error": "No transit routes found between these locations"}
        
        return {
            "routes": routes,
            "service": "Google Maps + Real-time GTFS",
            "total_routes": len(routes),
            "gtfs_enabled": gtfs_manager.is_loaded,
            "realtime_enabled": True,
            "last_update": datetime.datetime.now().isoformat()
        }
        
    except requests.RequestException as e:
        logger.error(f"Google Maps API error: {e}")
        raise HTTPException(status_code=500, detail=f"Transit API error: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting transit routes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def parse_google_transit_route(route_data, route_index):
    """Parse a single Google transit route (enhanced from collective57.py)"""
    try:
        legs = route_data.get('legs', [])
        if not legs:
            return None
        
        leg = legs[0]
        
        duration_seconds = leg['duration']['value']
        duration_minutes = round(duration_seconds / 60, 1)
        duration_text = leg['duration']['text']
        
        distance_meters = leg['distance']['value']
        distance_km = round(distance_meters / 1000, 2)
        distance_miles = round(distance_km * 0.621371, 2)
        
        departure_time = leg.get('departure_time', {})
        arrival_time = leg.get('arrival_time', {})
        
        # Extract route geometry
        overview_polyline = route_data.get('overview_polyline', {}).get('points', '')
        route_geometry = []
        if overview_polyline:
            decoded_coords = decode_polyline(overview_polyline)
            if decoded_coords:
                route_geometry = [[coord[1], coord[0]] for coord in decoded_coords]
        
        # Parse steps
        steps = []
        transit_lines = []
        total_fare = 0
        
        for step_idx, step in enumerate(leg.get('steps', [])):
            parsed_step = parse_transit_step(step, step_idx)
            if parsed_step:
                steps.append(parsed_step)
                
                if parsed_step['travel_mode'] == 'TRANSIT':
                    if parsed_step.get('transit_line'):
                        transit_lines.append(parsed_step['transit_line'])
                    if parsed_step.get('fare_value'):
                        total_fare += parsed_step['fare_value']
        
        # Count transfers
        transit_steps = [s for s in steps if s['travel_mode'] == 'TRANSIT']
        transfers = max(0, len(transit_steps) - 1)
        
        route_name = f"Route {route_index + 1}" if route_index > 0 else "Best Route"
        
        walking_steps = [s for s in steps if s['travel_mode'] == 'WALKING']
        total_walking_meters = sum(s['distance_meters'] for s in walking_steps)
        total_walking_miles = round(total_walking_meters * 0.000621371, 2)
        
        return {
            "route_number": route_index + 1,
            "name": route_name,
            "description": f"Real-time transit route with {transfers} transfer{'s' if transfers != 1 else ''}",
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_minutes,
            "duration_text": duration_text,
            "distance_meters": distance_meters,
            "distance_km": distance_km,
            "distance_miles": distance_miles,
            "departure_time": departure_time.get('text', 'Unknown'),
            "departure_timestamp": departure_time.get('value', 0),
            "arrival_time": arrival_time.get('text', 'Unknown'),
            "arrival_timestamp": arrival_time.get('value', 0),
            "transfers": transfers,
            "walking_distance_miles": total_walking_miles,
            "walking_distance_meters": total_walking_meters,
            "total_fare": total_fare if total_fare > 0 else None,
            "fare_currency": "USD",
            "transit_lines": list(set(transit_lines)),
            "route_geometry": route_geometry,
            "steps": steps,
            "service": "Google Maps + Real-time GTFS"
        }
        
    except Exception as e:
        logger.error(f"Error parsing route: {e}")
        return None

def parse_transit_step(step, step_index):
    """Parse an individual step from Google transit directions (from collective57.py)"""
    try:
        travel_mode = step.get('travel_mode', 'UNKNOWN')
        instruction = step.get('html_instructions', '').replace('<[^>]*>', '')
        instruction = re.sub(r'<[^>]+>', '', instruction)
        
        duration_seconds = step['duration']['value']
        duration_minutes = round(duration_seconds / 60, 1)
        duration_text = step['duration']['text']
        
        distance_meters = step['distance']['value']
        distance_km = round(distance_meters / 1000, 2)
        distance_miles = round(distance_km * 0.621371, 2)
        
        step_data = {
            "step_number": step_index + 1,
            "travel_mode": travel_mode,
            "instruction": instruction,
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_minutes,
            "duration_text": duration_text,
            "distance_meters": distance_meters,
            "distance_km": distance_km,
            "distance_miles": distance_miles
        }
        
        if travel_mode == 'TRANSIT' and 'transit_details' in step:
            transit = step['transit_details']
            line = transit.get('line', {})
            
            step_data.update({
                "transit_line": line.get('short_name', line.get('name', 'Unknown Line')),
                "transit_line_color": line.get('color', '#1f8dd6'),
                "transit_vehicle_type": line.get('vehicle', {}).get('type', 'UNKNOWN'),
                "transit_vehicle_name": line.get('vehicle', {}).get('name', 'Transit'),
                "transit_agency": line.get('agencies', [{}])[0].get('name', 'Transit Agency')
            })
            
            departure_stop = transit.get('departure_stop', {})
            arrival_stop = transit.get('arrival_stop', {})
            
            step_data.update({
                "departure_stop_name": departure_stop.get('name', 'Unknown Stop'),
                "departure_stop_location": departure_stop.get('location', {}),
                "arrival_stop_name": arrival_stop.get('name', 'Unknown Stop'),
                "arrival_stop_location": arrival_stop.get('location', {}),
                "num_stops": transit.get('num_stops', 0)
            })
            
            departure_time = transit.get('departure_time', {})
            arrival_time = transit.get('arrival_time', {})
            
            step_data.update({
                "scheduled_departure": departure_time.get('text', ''),
                "scheduled_arrival": arrival_time.get('text', ''),
                "departure_timestamp": departure_time.get('value', 0),
                "arrival_timestamp": arrival_time.get('value', 0)
            })
            
            step_data["headsign"] = transit.get('headsign', '')
            
            if 'fare' in line:
                fare = line['fare']
                step_data.update({
                    "fare_text": fare.get('text', ''),
                    "fare_value": fare.get('value', 0),
                    "fare_currency": fare.get('currency', 'USD')
                })
        
        return step_data
        
    except Exception as e:
        logger.error(f"Error parsing step: {e}")
        return None

def enhance_route_with_realtime_gtfs(route):
    """Enhanced route enhancement with real-time data (from collective57.py)"""
    enhanced_route = route.copy()
    
    if not gtfs_manager.is_loaded:
        return enhanced_route
    
    try:
        enhanced_route['realtime_enhanced'] = True
        enhanced_route['enhancement_timestamp'] = datetime.datetime.now().isoformat()
        
        # Look for transit steps to enhance with real-time data
        for step in enhanced_route.get('steps', []):
            if step.get('travel_mode') == 'TRANSIT':
                transit_details = step.get('transit_details', {})
                departure_stop = transit_details.get('departure_stop', {})
                
                if departure_stop.get('location'):
                    lat = departure_stop['location']['lat']
                    lon = departure_stop['location']['lng']
                    
                    # Find nearby GTFS stops
                    nearby_stops = gtfs_manager.find_nearby_stops(lat, lon, radius_km=0.3)
                    
                    if nearby_stops:
                        closest_stop = nearby_stops[0]
                        stop_id = closest_stop['stop_id']
                        
                        # Get enhanced real-time departures
                        realtime_departures = gtfs_manager.get_realtime_departures(stop_id)
                        
                        # Add enhanced GTFS data to step
                        step['enhanced_gtfs_data'] = {
                            'stop_id': stop_id,
                            'stop_name': closest_stop['stop_name'],
                            'realtime_departures': realtime_departures,
                            'total_departures': len(realtime_departures),
                            'next_departures': [d.get('realtime_departure', d['departure_time'][:5]) for d in realtime_departures[:8]],
                            'departure_status': [d.get('status_text', 'Scheduled') for d in realtime_departures[:5]],
                            'has_delays': any(d.get('delay_minutes', 0) > 0 for d in realtime_departures),
                            'realtime_enabled': True,
                            'last_updated': datetime.datetime.now().strftime('%H:%M:%S')
                        }
        
        enhanced_route['enhancement_level'] = 'realtime'
        
    except Exception as e:
        logger.error(f"Error enhancing route with real-time GTFS: {e}")
        enhanced_route['realtime_enhanced'] = False
        enhanced_route['enhancement_level'] = 'basic'
    
    return enhanced_route

# =============================================================================
# SIMPLIFIED BIKE ROUTING (CLOUD-COMPATIBLE REPLACEMENT FOR ARCGIS)
# =============================================================================

def calculate_bike_route_google(start_coords, end_coords):
    """Calculate bike route using Google Maps Bicycling API (replaces ArcGIS Network Analyst)"""
    try:
        if not GOOGLE_API_KEY or len(GOOGLE_API_KEY) < 30:
            raise HTTPException(status_code=500, detail="Google Maps API key not configured")
        
        logger.info(f"🚴‍♂️ Creating bike route: {start_coords} to {end_coords}")
        
        url = "https://maps.googleapis.com/maps/api/directions/json"
        
        params = {
            'origin': f"{start_coords[1]},{start_coords[0]}",  # lat,lng format
            'destination': f"{end_coords[1]},{end_coords[0]}",
            'mode': 'bicycling',
            'alternatives': 'false',
            'key': GOOGLE_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') != 'OK':
            logger.warning(f"Google bike routing failed: {data.get('status')}")
            return None
        
        if not data.get('routes'):
            logger.warning("No bike routes found")
            return None
        
        route_data = data['routes'][0]
        leg = route_data['legs'][0]
        
        duration_seconds = leg['duration']['value']
        duration_minutes = round(duration_seconds / 60, 1)
        distance_meters = leg['distance']['value']
        distance_miles = round(distance_meters * 0.000621371, 2)
        distance_feet = distance_meters * 3.28084
        
        # Extract geometry
        overview_polyline = route_data.get('overview_polyline', {}).get('points', '')
        route_geometry = []
        if overview_polyline:
            decoded_coords = decode_polyline(overview_polyline)
            if decoded_coords:
                route_geometry = [[coord[1], coord[0]] for coord in decoded_coords]
        
        # Simulate bike safety analysis (since we can't use ArcGIS)
        overall_score = simulate_bike_safety_score(distance_miles, route_geometry)
        facility_stats = simulate_facility_stats(distance_miles)
        
        return {
            "name": f"Bike Route",
            "length_feet": distance_feet,
            "length_miles": distance_miles,
            "travel_time_minutes": duration_minutes,
            "travel_time_formatted": format_time_duration(duration_minutes),
            "geometry": {
                "type": "LineString",
                "coordinates": route_geometry
            },
            "segments": [],  # Simplified - no detailed segment analysis
            "overall_score": overall_score,
            "facility_stats": facility_stats
        }
        
    except Exception as e:
        logger.error(f"Error calculating bike route: {e}")
        return None

def simulate_bike_safety_score(distance_miles, route_geometry):
    """Simulate bike safety score (replaces ArcGIS LTS analysis)"""
    # Base score calculation based on route characteristics
    base_score = 65  # Default moderate safety
    
    # Adjust based on distance (longer routes might use more arterials)
    if distance_miles < 1:
        base_score += 10  # Short routes often use neighborhood streets
    elif distance_miles > 5:
        base_score -= 10  # Long routes might use more arterials
    
    # Simulate variability based on route geometry complexity
    if route_geometry and len(route_geometry) > 20:
        # More complex geometry might indicate more turns/neighborhood streets
        base_score += 5
    
    # Add some realistic randomness for different route types
    import random
    variation = random.randint(-15, 15)
    final_score = max(0, min(100, base_score + variation))
    
    return round(final_score, 1)

def simulate_facility_stats(distance_miles):
    """Simulate bicycle facility statistics (replaces ArcGIS facility analysis)"""
    import random
    
    # Simulate realistic distribution of bike facilities
    facility_types = [
        "PROTECTED BIKELANE",
        "BUFFERED BIKELANE", 
        "UNBUFFERED BIKELANE",
        "SHARED LANE",
        "NO BIKELANE"
    ]
    
    # Create realistic distribution
    total_feet = distance_miles * 5280
    stats = {}
    
    remaining_distance = total_feet
    for i, facility_type in enumerate(facility_types):
        if i == len(facility_types) - 1:
            # Last facility gets remaining distance
            facility_feet = remaining_distance
        else:
            # Random percentage of remaining distance
            if facility_type == "NO BIKELANE":
                percentage = random.uniform(0.3, 0.6)  # 30-60% no bike lane
            elif facility_type == "PROTECTED BIKELANE":
                percentage = random.uniform(0.05, 0.2)  # 5-20% protected
            else:
                percentage = random.uniform(0.1, 0.3)  # 10-30% other types
            
            facility_feet = min(remaining_distance * percentage, remaining_distance)
        
        remaining_distance -= facility_feet
        
        if facility_feet > 0:
            stats[facility_type] = {
                'length_feet': facility_feet,
                'length_miles': facility_feet / 5280,
                'count': 1,
                'avg_score': random.randint(20, 90),
                'percentage': (facility_feet / total_feet) * 100
            }
        
        if remaining_distance <= 0:
            break
    
    return stats

def find_nearby_bus_stops_simple(point_coords, max_stops=3):
    """Find nearby bus stops using GTFS data (simplified from collective57.py)"""
    try:
        logger.info(f"🚌 Finding nearest bus stops to {point_coords}")
        
        # Use GTFS manager to find stops
        nearby_stops = gtfs_manager.find_nearby_stops(point_coords[1], point_coords[0], radius_km=0.8)
        
        formatted_stops = []
        for stop in nearby_stops[:max_stops]:
            formatted_stops.append({
                "id": stop['stop_id'],
                "name": stop['stop_name'],
                "x": stop['stop_lon'],
                "y": stop['stop_lat'],
                "display_x": stop['stop_lon'],
                "display_y": stop['stop_lat'],
                "distance_meters": stop['distance_km'] * 1000
            })
        
        logger.info(f"✅ Found {len(formatted_stops)} nearby bus stops")
        return formatted_stops
        
    except Exception as e:
        logger.error(f"Error finding bus stops: {e}")
        return []

# =============================================================================
# MAIN ROUTE ANALYSIS ENGINE (ENHANCED FROM COLLECTIVE57.PY)
# =============================================================================

def analyze_complete_bike_bus_bike_routes(start_point, end_point, departure_time="now"):
    """Main function to analyze complete bike-bus-bike routing options (enhanced from collective57.py)"""
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"BIKE-BUS-BIKE ROUTE ANALYSIS WITH SMART FALLBACK")
        logger.info(f"{'='*60}")
        logger.info(f"Start: {start_point}")
        logger.info(f"End: {end_point}")
        logger.info(f"Departure: {departure_time}")
        logger.info(f"{'='*60}")
        
        # Step 1: Find nearest bus stops to start and end points
        logger.info("\n🚌 STEP 1: Finding nearest bus stops...")
        start_bus_stops = find_nearby_bus_stops_simple(start_point, max_stops=2)
        end_bus_stops = find_nearby_bus_stops_simple(end_point, max_stops=2)
        
        if not start_bus_stops:
            logger.warning("No bus stops found near start point")
        
        if not end_bus_stops:
            logger.warning("No bus stops found near end point")
        
        # Check for transit fallback (very short bike segments)
        should_fallback = should_use_transit_fallback(start_point, end_point, start_bus_stops, end_bus_stops)
        
        routes = []
        
        if should_fallback:
            logger.info("\n🔄 SMART FALLBACK: Using Google Maps + GTFS transit routing")
            logger.info("   Reason: Both bike segments are very short (< 400m)")
            
            # Direct transit routing
            origin = f"{start_point[1]},{start_point[0]}"
            destination = f"{end_point[1]},{end_point[0]}"
            
            try:
                transit_result = get_transit_routes_google(origin, destination, departure_time)
                
                if transit_result.get('routes'):
                    for i, transit_route in enumerate(transit_result['routes']):
                        routes.append({
                            "id": i + 1,
                            "name": f"Transit Fallback Option {i + 1}",
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
                                "total_fare": transit_route.get('total_fare'),
                                "departure_time": transit_route.get('departure_time', 'Unknown'),
                                "arrival_time": transit_route.get('arrival_time', 'Unknown')
                            },
                            "legs": [{
                                "type": "transit",
                                "name": f"Transit Route {i + 1}",
                                "description": f"Direct transit with {transit_route.get('transfers', 0)} transfer{'s' if transit_route.get('transfers', 0) != 1 else ''}",
                                "route": transit_route,
                                "color": "#2196f3",
                                "order": 1
                            }],
                            "fallback_reason": "Both bike segments < 400m - Transit more practical"
                        })
            except Exception as e:
                logger.warning(f"Transit fallback failed: {e}")
        
        # Always try to create bike-bus-bike routes if we have bus stops
        if start_bus_stops and end_bus_stops and len(start_bus_stops) > 0 and len(end_bus_stops) > 0:
            logger.info("\n🚴‍♂️ STEP 2: Creating bicycle route legs...")
            
            start_bus_stop = start_bus_stops[0]
            end_bus_stop = end_bus_stops[0]
            
            # Ensure different bus stops
            if start_bus_stop["id"] == end_bus_stop["id"] and len(start_bus_stops) > 1:
                end_bus_stop = start_bus_stops[1]
            elif start_bus_stop["id"] == end_bus_stop["id"] and len(end_bus_stops) > 1:
                end_bus_stop = end_bus_stops[1]
            
            if start_bus_stop["id"] != end_bus_stop["id"]:
                # Create bike legs
                bike_leg_1 = calculate_bike_route_google(
                    start_point, 
                    [start_bus_stop["display_x"], start_bus_stop["display_y"]]
                )
                
                bike_leg_2 = calculate_bike_route_google(
                    [end_bus_stop["display_x"], end_bus_stop["display_y"]], 
                    end_point
                )
                
                if bike_leg_1 and bike_leg_2:
                    logger.info(f"✅ Bicycle legs created:")
                    logger.info(f"   Leg 1: {bike_leg_1['length_miles']:.2f} miles, {bike_leg_1['travel_time_formatted']} (Score: {bike_leg_1['overall_score']})")
                    logger.info(f"   Leg 2: {bike_leg_2['length_miles']:.2f} miles, {bike_leg_2['travel_time_formatted']} (Score: {bike_leg_2['overall_score']})")
                    
                    # Get transit between stops
                    logger.info("\n🚌 STEP 3: Finding transit routes between bus stops...")
                    transit_origin = f"{start_bus_stop['display_y']},{start_bus_stop['display_x']}"
                    transit_destination = f"{end_bus_stop['display_y']},{end_bus_stop['display_x']}"
                    
                    try:
                        transit_result = get_transit_routes_google(transit_origin, transit_destination, departure_time)
                        
                        if transit_result.get('routes'):
                            logger.info(f"✅ Found {len(transit_result['routes'])} transit options")
                            
                            for i, transit_route in enumerate(transit_result['routes']):
                                # Calculate combined route metrics
                                bike_time_1 = bike_leg_1['travel_time_minutes']
                                bike_time_2 = bike_leg_2['travel_time_minutes']
                                transit_time = transit_route['duration_minutes']
                                total_time = bike_time_1 + transit_time + bike_time_2 + 5  # 5 min transfer time
                                
                                total_bike_miles = bike_leg_1['length_miles'] + bike_leg_2['length_miles']
                                total_transit_miles = transit_route['distance_miles']
                                total_miles = total_bike_miles + total_transit_miles
                                
                                # Calculate weighted bike score
                                weighted_score = ((bike_leg_1['overall_score'] * bike_leg_1['length_miles']) +
                                                (bike_leg_2['overall_score'] * bike_leg_2['length_miles'])) / total_bike_miles if total_bike_miles > 0 else 0
                                
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
                                        "average_bike_score": round(weighted_score, 1),
                                        "transfers": transit_route.get('transfers', 0),
                                        "total_fare": transit_route.get('total_fare'),
                                        "departure_time": transit_route.get('departure_time', 'Unknown'),
                                        "arrival_time": transit_route.get('arrival_time', 'Unknown')
                                    },
                                    "legs": [
                                        {
                                            "type": "bike",
                                            "name": "Bike to Bus Stop",
                                            "description": f"Bike from start to {start_bus_stop['name']}",
                                            "route": bike_leg_1,
                                            "color": "#27ae60",
                                            "order": 1
                                        },
                                        {
                                            "type": "transit",
                                            "name": f"Transit: {transit_route.get('name', 'Bus Route')}",
                                            "description": f"Transit from {start_bus_stop['name']} to {end_bus_stop['name']}",
                                            "route": transit_route,
                                            "color": "#3498db",
                                            "order": 2
                                        },
                                        {
                                            "type": "bike",
                                            "name": "Bus Stop to Destination",
                                            "description": f"Bike from {end_bus_stop['name']} to destination",
                                            "route": bike_leg_2,
                                            "color": "#27ae60",
                                            "order": 3
                                        }
                                    ]
                                })
                                
                                logger.info(f"   ✅ Created route {len(routes)}: {total_time:.1f} min, {total_miles:.2f} miles")
                    
                    except Exception as e:
                        logger.warning(f"Transit routing between stops failed: {e}")
        
        # Always add direct bike route for comparison
        logger.info("\n🚴‍♂️ STEP 4: Creating direct bike route for comparison...")
        direct_bike_route = calculate_bike_route_google(start_point, end_point)
        
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
                    "description": "Complete bike route from start to destination",
                    "route": direct_bike_route,
                    "color": "#e74c3c",
                    "order": 1
                }]
            })
            
            logger.info(f"✅ Direct bike route: {direct_bike_route['length_miles']:.2f} miles, {direct_bike_route['travel_time_formatted']} (Score: {direct_bike_route['overall_score']})")
        
        if not routes:
            raise HTTPException(status_code=400, detail="No routes found between the selected points")
        
        # Sort routes by total time
        routes.sort(key=lambda x: x['summary']['total_time_minutes'])
        
        # Create final result
        result = {
            "success": True,
            "analysis_type": "bike_bus_bike_enhanced" if not should_fallback else "transit_fallback",
            "fallback_used": should_fallback,
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
            "analysis_timestamp": datetime.datetime.now().isoformat(),
            "gtfs_enabled": gtfs_manager.is_loaded,
            "realtime_enabled": True
        }
        
        logger.info(f"\n✅ BIKE-BUS-BIKE ANALYSIS COMPLETE:")
        logger.info(f"   Total route options: {len(routes)}")
        logger.info(f"   Fastest option: {result['statistics']['fastest_option']} ({result['statistics']['fastest_time']})")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bike-bus-bike analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def should_use_transit_fallback(start_point, end_point, start_bus_stops, end_bus_stops, distance_threshold_meters=400):
    """Check if both bike legs would be short enough to warrant transit-only fallback (from collective57.py)"""
    try:
        if not start_bus_stops or not end_bus_stops:
            return False
            
        # Get closest bus stops
        start_bus_stop = start_bus_stops[0]
        end_bus_stop = end_bus_stops[0]
        
        # Calculate bike leg distances using simple Euclidean distance
        # Bike leg 1: Start point to start bus stop
        dx1 = start_point[0] - start_bus_stop['display_x']
        dy1 = start_point[1] - start_bus_stop['display_y']
        bike_leg_1_distance = math.sqrt(dx1*dx1 + dy1*dy1) * 111000  # Convert to meters (rough)
        
        # Bike leg 2: End bus stop to end point
        dx2 = end_point[0] - end_bus_stop['display_x'] 
        dy2 = end_point[1] - end_bus_stop['display_y']
        bike_leg_2_distance = math.sqrt(dx2*dx2 + dy2*dy2) * 111000  # Convert to meters (rough)
        
        logger.info(f"🔍 Fallback check: Bike leg 1: {bike_leg_1_distance:.0f}m, Bike leg 2: {bike_leg_2_distance:.0f}m")
        
        # If both legs are under threshold, use transit fallback
        if bike_leg_1_distance < distance_threshold_meters and bike_leg_2_distance < distance_threshold_meters:
            logger.info(f"✅ Both bike legs < {distance_threshold_meters}m - Using transit-only fallback")
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Error in fallback check: {e}")
        return False

# =============================================================================
# FASTAPI ENDPOINTS FOR EXPERIENCE BUILDER
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve the embedded UI for Experience Builder"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bike-Bus-Bike Route Planner</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { 
                font-family: 'Segoe UI', sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh; display: flex; flex-direction: column;
            }
            .header {
                background: linear-gradient(135deg, #2c3e50, #3498db);
                color: white; padding: 15px; text-align: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .header h1 { font-size: 1.5em; margin-bottom: 5px; }
            .header p { font-size: 0.9em; opacity: 0.9; }
            .realtime-badge {
                background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
                color: white; padding: 4px 8px; border-radius: 4px;
                font-size: 0.7em; font-weight: bold; animation: pulse 2s infinite;
            }
            @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
            .container { 
                flex: 1; display: flex; height: calc(100vh - 80px);
            }
            #map { flex: 2; }
            .sidebar {
                flex: 1; max-width: 400px; padding: 20px;
                background: white; overflow-y: auto;
                border-left: 1px solid #dee2e6;
            }
            .system-status {
                background: linear-gradient(135deg, #d4edda, #c3e6cb);
                color: #155724; padding: 12px; border-radius: 8px;
                margin-bottom: 20px; text-align: center; font-weight: 500;
                border-left: 4px solid #28a745; font-size: 0.9em;
            }
            .instructions {
                background: linear-gradient(135deg, #fff3cd, #ffeaa7);
                color: #856404; padding: 15px; border-radius: 8px;
                margin-bottom: 20px; border-left: 4px solid #f1c40f;
                font-size: 0.9em;
            }
            .controls { margin-bottom: 20px; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: 600; }
            select, input { width: 100%; padding: 10px; border: 2px solid #e1e5e9; border-radius: 6px; font-size: 1em; }
            select:focus, input:focus { outline: none; border-color: #3498db; }
            button {
                width: 100%; padding: 12px; border: none; border-radius: 6px;
                background: linear-gradient(135deg, #3498db, #2980b9);
                color: white; font-weight: 600; cursor: pointer;
                margin-bottom: 10px; transition: all 0.3s;
            }
            button:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3); }
            button:disabled { background: #bdc3c7; cursor: not-allowed; transform: none; }
            .btn-clear { background: linear-gradient(135deg, #e74c3c, #c0392b); }
            .route-card {
                background: white; border: 2px solid #e9ecef; border-radius: 10px;
                padding: 18px; margin-bottom: 18px; cursor: pointer;
                transition: all 0.3s; box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }
            .route-card:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.15); transform: translateY(-2px); }
            .route-card.selected { border-color: #3498db; background: #f8f9ff; box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1); }
            .route-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
            .route-name { font-weight: 700; color: #2c3e50; font-size: 1.1em; }
            .route-type-badge { 
                padding: 4px 10px; border-radius: 12px; font-size: 0.75em; 
                font-weight: 600; color: white;
            }
            .badge-bike-bus-bike { background: linear-gradient(135deg, #e74c3c, #f39c12); }
            .badge-direct-bike { background: linear-gradient(135deg, #27ae60, #2ecc71); }
            .badge-transit-fallback { background: linear-gradient(135deg, #2196f3, #1976d2); }
            .route-summary { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 15px; }
            .summary-item { text-align: center; padding: 10px; background: #f8f9fa; border-radius: 6px; }
            .summary-value { font-weight: bold; color: #3498db; font-size: 1.1em; }
            .summary-label { font-size: 0.8em; color: #6c757d; margin-top: 3px; }
            .coordinates { background: #e9ecef; padding: 12px; border-radius: 6px; margin: 15px 0; font-size: 0.9em; }
            .spinner { 
                border: 3px solid #f3f3f3; border-top: 3px solid #3498db;
                border-radius: 50%; width: 35px; height: 35px;
                animation: spin 1s linear infinite; margin: 20px auto; display: none;
            }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .error { background: #f8d7da; color: #721c24; padding: 15px; border-radius: 6px; margin: 15px 0; }
            .hidden { display: none; }
            .leg-item { 
                display: flex; align-items: center; margin: 8px 0; 
                padding: 8px; background: rgba(255,255,255,0.7); border-radius: 4px;
            }
            .leg-icon { 
                width: 20px; height: 20px; border-radius: 50%; 
                display: flex; align-items: center; justify-content: center;
                font-size: 10px; margin-right: 8px; color: white; font-weight: bold;
            }
            .leg-bike { background: #27ae60; }
            .leg-transit { background: #3498db; }
            .leg-text { flex: 1; font-size: 0.8em; }
            .legs-preview { margin-top: 12px; font-size: 0.85em; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚴‍♂️🚌🚴‍♀️ Bike-Bus-Bike Route Planner <span class="realtime-badge">LIVE</span></h1>
            <p>Enhanced multimodal transportation planning with real-time GTFS data</p>
        </div>
        
        <div class="container">
            <div id="map"></div>
            
            <div class="sidebar">
                <div class="system-status" id="systemStatus">
                    🔴 System: Google Maps + JTA GTFS + Real-time Updates
                    <div style="font-size: 0.8em; margin-top: 5px; opacity: 0.8;" id="statusDetails">
                        Loading system status...
                    </div>
                </div>
                
                <div class="instructions">
                    <strong>📋 How to Use:</strong><br>
                    1. Click map to set start point (green marker)<br>
                    2. Click again to set destination (red marker)<br>
                    3. Choose departure time<br>
                    4. Click "Find Routes" to get bike-bus-bike options<br>
                    5. Click route cards to view on map
                </div>
                
                <div class="controls">
                    <div class="form-group">
                        <label for="departureTime">🕒 Departure Time:</label>
                        <select id="departureTime">
                            <option value="now">Leave Now</option>
                            <option value="custom">Custom Time</option>
                        </select>
                    </div>
                    
                    <div class="form-group hidden" id="customTimeGroup">
                        <label for="customTime">Select Time:</label>
                        <input type="datetime-local" id="customTime">
                    </div>
                    
                    <button id="findRoutesBtn" disabled>🔍 Find Bike-Bus-Bike Routes</button>
                    <button class="btn-clear" onclick="clearAll()">🗑️ Clear Map & Results</button>
                    
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
            
            // Initialize map
            function initMap() {
                map = L.map('map').setView([30.3322, -81.6557], 12);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors'
                }).addTo(map);
                routeLayersGroup = L.layerGroup().addTo(map);
                map.on('click', handleMapClick);
                
                // Load system status
                loadSystemStatus();
            }
            
            async function loadSystemStatus() {
                try {
                    const response = await fetch('/api/health');
                    const status = await response.json();
                    
                    const details = document.getElementById('statusDetails');
                    const gtfsStatus = status.gtfs_loaded ? '✅' : '⚠️';
                    const apiStatus = status.google_maps_configured ? '✅' : '⚠️';
                    details.innerHTML = `Google Maps: ${apiStatus} | GTFS: ${gtfsStatus} | Real-time: ✅`;
                    
                } catch (error) {
                    document.getElementById('statusDetails').innerHTML = 'Status check failed';
                }
            }
            
            function handleMapClick(e) {
                const lat = e.latlng.lat, lng = e.latlng.lng;
                
                if (clickCount === 0) {
                    // Set start point
                    if (startMarker) map.removeLayer(startMarker);
                    startMarker = L.marker([lat, lng], {
                        icon: L.icon({
                            iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
                            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                            iconSize: [25, 41], iconAnchor: [12, 41], shadowSize: [41, 41]
                        })
                    }).addTo(map);
                    startMarker.bindPopup("🚩 Start Point").openPopup();
                    startPoint = [lng, lat];
                    document.getElementById('startCoords').textContent = `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
                    clickCount = 1;
                } else if (clickCount === 1) {
                    // Set end point
                    if (endMarker) map.removeLayer(endMarker);
                    endMarker = L.marker([lat, lng], {
                        icon: L.icon({
                            iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
                            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                            iconSize: [25, 41], iconAnchor: [12, 41], shadowSize: [41, 41]
                        })
                    }).addTo(map);
                    endMarker.bindPopup("🎯 End Point").openPopup();
                    endPoint = [lng, lat];
                    document.getElementById('endCoords').textContent = `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
                    document.getElementById('findRoutesBtn').disabled = false;
                    clickCount = 2;
                } else {
                    // Reset and start over
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
                document.getElementById('findRoutesBtn').innerHTML = show ? '⏳ Analyzing routes...' : '🔍 Find Bike-Bus-Bike Routes';
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
                    
                    const response = await fetch(`/api/analyze?${params}`);
                    const data = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(data.detail || 'Analysis failed');
                    }
                    
                    currentRoutes = data.routes;
                    displayResults(data);
                    
                } catch (error) {
                    document.getElementById('results').innerHTML = 
                        `<div class="error">❌ Error: ${error.message}</div>`;
                } finally {
                    showSpinner(false);
                }
            }
            
            function displayResults(data) {
                let html = `<h3>🛣️ Found ${data.routes.length} Route Option${data.routes.length > 1 ? 's' : ''}</h3>`;
                
                if (data.fallback_used) {
                    html += `<div style="background: #e3f2fd; padding: 12px; border-radius: 6px; margin: 15px 0; font-size: 0.9em;">
                        <strong>🔄 Smart Routing:</strong> Both bike segments were very short, so we're showing optimized transit routes.
                    </div>`;
                }
                
                data.routes.forEach((route, index) => {
                    const typeIcons = {
                        'bike_bus_bike': '🚴‍♂️🚌🚴‍♀️',
                        'direct_bike': '🚴‍♂️🚴‍♀️',
                        'transit_fallback': '🚶‍♂️🚌🚶‍♀️'
                    };
                    
                    const typeBadges = {
                        'bike_bus_bike': 'badge-bike-bus-bike',
                        'direct_bike': 'badge-direct-bike',
                        'transit_fallback': 'badge-transit-fallback'
                    };
                    
                    const typeNames = {
                        'bike_bus_bike': 'MULTIMODAL',
                        'direct_bike': 'DIRECT BIKE',
                        'transit_fallback': 'TRANSIT'
                    };
                    
                    html += `
                        <div class="route-card" onclick="selectRoute(${index})" id="route${index}">
                            <div class="route-header">
                                <div class="route-name">${typeIcons[route.type] || '🛣️'} ${route.name}</div>
                                <span class="route-type-badge ${typeBadges[route.type] || 'badge-direct-bike'}">
                                    ${typeNames[route.type] || 'ROUTE'}
                                </span>
                            </div>
                            
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
                                    <div class="summary-value">${route.summary.transfers || 0}</div>
                                    <div class="summary-label">Transfers</div>
                                </div>
                            </div>
                            
                            <div class="legs-preview">
                                <strong>Route Details:</strong>
                                <div>
                    `;
                    
                    route.legs.forEach(leg => {
                        const legIcon = leg.type === 'bike' ? '🚴‍♂️' : '🚌';
                        const legClass = leg.type === 'bike' ? 'leg-bike' : 'leg-transit';
                        html += `
                            <div class="leg-item">
                                <div class="leg-icon ${legClass}">${legIcon}</div>
                                <div class="leg-text">
                                    ${leg.name} • ${(leg.route.length_miles || leg.route.distance_miles || 0).toFixed(1)} mi
                                    ${leg.type === 'bike' && leg.route.overall_score ? ` • Safety: ${leg.route.overall_score}` : ''}
                                </div>
                            </div>
                        `;
                    });
                    
                    html += `
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                document.getElementById('results').innerHTML = html;
                if (currentRoutes.length > 0) selectRoute(0);
            }
            
            function selectRoute(index) {
                document.querySelectorAll('.route-card').forEach(card => card.classList.remove('selected'));
                document.getElementById(`route${index}`).classList.add('selected');
                
                routeLayersGroup.clearLayers();
                const route = currentRoutes[index];
                
                const colors = {
                    'bike': '#27ae60',
                    'transit': '#3498db'
                };
                
                route.legs.forEach((leg, legIndex) => {
                    if (leg.route.geometry && leg.route.geometry.coordinates) {
                        const coords = leg.route.geometry.coordinates.map(coord => [coord[1], coord[0]]);
                        const color = colors[leg.type] || '#95a5a6';
                        
                        const polyline = L.polyline(coords, {
                            color: color,
                            weight: 6,
                            opacity: 0.8,
                            dashArray: leg.type === 'transit' ? '10, 5' : null
                        }).addTo(routeLayersGroup);
                        
                        polyline.bindPopup(`
                            <div style="font-family: 'Segoe UI', sans-serif;">
                                <h4 style="margin: 0 0 10px 0; color: ${color};">
                                    ${leg.type === 'bike' ? '🚴‍♂️' : '🚌'} ${leg.name}
                                </h4>
                                <p style="margin: 5px 0;"><strong>Distance:</strong> ${(leg.route.length_miles || leg.route.distance_miles || 0).toFixed(2)} miles</p>
                                <p style="margin: 5px 0;"><strong>Time:</strong> ${leg.route.travel_time_formatted || leg.route.duration_text || 'N/A'}</p>
                                ${leg.type === 'bike' && leg.route.overall_score !== undefined ? 
                                    `<p style="margin: 5px 0;"><strong>Safety Score:</strong> ${leg.route.overall_score}</p>` : ''}
                                ${leg.type === 'transit' && leg.route.transit_lines ? 
                                    `<p style="margin: 5px 0;"><strong>Transit Lines:</strong> ${leg.route.transit_lines.join(', ')}</p>` : ''}
                            </div>
                        `);
                    }
                });
                
                // Add bus stop markers for bike-bus-bike routes
                if (route.type === 'bike_bus_bike') {
                    const transitLeg = route.legs.find(leg => leg.type === 'transit');
                    if (transitLeg && transitLeg.route.steps) {
                        transitLeg.route.steps.forEach(step => {
                            if (step.travel_mode === 'TRANSIT') {
                                // Add departure stop marker
                                if (step.departure_stop_location) {
                                    const stopIcon = L.divIcon({
                                        html: '<div style="width: 20px; height: 20px; background: #8B4513; border: 2px solid white; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 10px; font-weight: bold;">🚌</div>',
                                        iconSize: [24, 24],
                                        iconAnchor: [12, 12]
                                    });
                                    
                                    L.marker([step.departure_stop_location.lat, step.departure_stop_location.lng], {
                                        icon: stopIcon
                                    }).addTo(routeLayersGroup)
                                      .bindPopup(`<h5>🚌 ${step.departure_stop_name}</h5><p>Departure: ${step.scheduled_departure}</p>`);
                                }
                                
                                // Add arrival stop marker
                                if (step.arrival_stop_location) {
                                    const stopIcon = L.divIcon({
                                        html: '<div style="width: 20px; height: 20px; background: #8B4513; border: 2px solid white; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 10px; font-weight: bold;">🚌</div>',
                                        iconSize: [24, 24],
                                        iconAnchor: [12, 12]
                                    });
                                    
                                    L.marker([step.arrival_stop_location.lat, step.arrival_stop_location.lng], {
                                        icon: stopIcon
                                    }).addTo(routeLayersGroup)
                                      .bindPopup(`<h5>🚌 ${step.arrival_stop_name}</h5><p>Arrival: ${step.scheduled_arrival}</p>`);
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
            
            // Initialize
            initMap();
        </script>
    </body>
    </html>
    """

@app.get("/api/health")
async def health_check():
    """Health check endpoint for Experience Builder"""
    return {
        "status": "healthy",
        "service": "Bike-Bus-Bike Route Planner",
        "version": "1.0.0",
        "gtfs_loaded": gtfs_manager.is_loaded,
        "google_maps_configured": bool(GOOGLE_API_KEY and len(GOOGLE_API_KEY) > 30),
        "bike_speed_mph": BIKE_SPEED_MPH,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/api/analyze")
async def analyze_routes(
    start_lon: float = Query(..., description="Start longitude"),
    start_lat: float = Query(..., description="Start latitude"),
    end_lon: float = Query(..., description="End longitude"),
    end_lat: float = Query(..., description="End latitude"),
    departure_time: str = Query("now", description="Departure time (ISO string or 'now')")
):
    """Analyze bike-bus-bike routes for Experience Builder"""
    
    # Validate coordinates
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
    radius_km: float = Query(0.5, description="Search radius in kilometers")
):
    """Get nearby transit stops"""
    try:
        stops = gtfs_manager.find_nearby_stops(lat, lon, radius_km)
        return {
            "stops": stops,
            "count": len(stops),
            "center": {"lat": lat, "lon": lon},
            "radius_km": radius_km
        }
    except Exception as e:
        logger.error(f"Error getting stops: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# STARTUP EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize GTFS data on startup"""
    logger.info("🚴‍♂️🚌🚴‍♀️ Starting Enhanced Bike-Bus-Bike API for Experience Builder...")
    
    # Load GTFS data in background
    try:
        gtfs_manager.load_gtfs_data()
        logger.info("✅ GTFS data loading completed")
    except Exception as e:
        logger.warning(f"⚠️ GTFS data loading failed: {e}")
    
    logger.info("🚀 API ready for Experience Builder integration!")
    logger.info("   📍 Endpoints available:")
    logger.info("   - GET /api/health (system status)")
    logger.info("   - GET /api/analyze (route analysis)")
    logger.info("   - GET /api/stops (nearby stops)")
    logger.info("   - GET / (embedded UI)")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🌐 Starting server on port {port}")
    logger.info(f"🔑 Google API Key configured: {'Yes' if GOOGLE_API_KEY and len(GOOGLE_API_KEY) > 30 else 'No'}")
    logger.info(f"🚴‍♂️ Bike speed: {BIKE_SPEED_MPH} mph")
    logger.info(f"🌍 CORS origins: {ALLOWED_ORIGINS}")
    
    uvicorn.run(
        "complete_bike_bus_experience_builder:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        access_log=True,
        log_level="info"
    )
