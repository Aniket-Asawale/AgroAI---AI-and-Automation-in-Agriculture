"""
AgroSensor — Open-Meteo Weather Integration
Fetches real weather data and caches it to a JSON file.
Both the simulator (sensor_conn.py) and the API (main.py) read from this cache.
No external dependencies — uses urllib from stdlib.
"""

import json
import logging
import time
import urllib.request
import urllib.error
from pathlib import Path
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)

# Paths relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = _PROJECT_ROOT / "weather_config.json"
CACHE_PATH = _PROJECT_ROOT / "weather_cache.json"

CACHE_TTL_SECONDS = 3600  # 1 hour

# ─── Soil Type Profiles ───
# Each soil type defines physical properties that affect simulated sensor values.
SOIL_PROFILES = {
    "Alluvial": {
        "description": "Fertile river-deposited soil, good drainage & nutrient retention",
        "water_retention": 0.65,   # 0-1, how well soil holds moisture
        "drainage_rate": 0.6,      # 0-1, how fast water drains (higher = faster)
        "ph_base": 7.0,            # natural pH tendency
        "ph_range": (6.5, 8.0),
        "ec_base": 1100,           # baseline EC (μS/cm)
        "ec_range": (400, 2200),
        "n_base": 140, "p_base": 90, "k_base": 170,  # baseline NPK (mg/kg)
    },
    "Black (Regur)": {
        "description": "Clay-rich, high moisture retention, cracks when dry",
        "water_retention": 0.85,
        "drainage_rate": 0.3,
        "ph_base": 7.8,
        "ph_range": (7.0, 8.5),
        "ec_base": 1400,
        "ec_range": (600, 2800),
        "n_base": 110, "p_base": 70, "k_base": 200,
    },
    "Red": {
        "description": "Iron-rich, acidic, low fertility, porous",
        "water_retention": 0.40,
        "drainage_rate": 0.75,
        "ph_base": 5.8,
        "ph_range": (4.5, 6.5),
        "ec_base": 800,
        "ec_range": (200, 1500),
        "n_base": 80, "p_base": 50, "k_base": 100,
    },
    "Laterite": {
        "description": "Leached tropical soil, low nutrients, acidic",
        "water_retention": 0.35,
        "drainage_rate": 0.80,
        "ph_base": 5.5,
        "ph_range": (4.5, 6.0),
        "ec_base": 600,
        "ec_range": (150, 1200),
        "n_base": 60, "p_base": 40, "k_base": 80,
    },
    "Sandy": {
        "description": "Coarse-grained, very low retention, drains fast",
        "water_retention": 0.20,
        "drainage_rate": 0.90,
        "ph_base": 6.5,
        "ph_range": (5.5, 7.5),
        "ec_base": 500,
        "ec_range": (100, 1000),
        "n_base": 50, "p_base": 30, "k_base": 60,
    },
    "Clay": {
        "description": "Fine-grained, very high retention, poor drainage",
        "water_retention": 0.90,
        "drainage_rate": 0.20,
        "ph_base": 7.2,
        "ph_range": (6.5, 8.5),
        "ec_base": 1500,
        "ec_range": (700, 3000),
        "n_base": 130, "p_base": 80, "k_base": 180,
    },
}

# Pre-defined Indian cities with soil type assignments
CITIES = [
    {"name": "Kolhapur",   "state": "Maharashtra",  "lat": 16.695,  "lon": 74.2333, "soil_type": "Black (Regur)", "sensors": 1},
    {"name": "Satara",     "state": "Maharashtra",  "lat": 17.3333, "lon": 74.7167, "soil_type": "Laterite",      "sensors": 1},
    {"name": "Solapur",    "state": "Maharashtra",  "lat": 18.0167, "lon": 75.3667, "soil_type": "Black (Regur)", "sensors": 1},
    {"name": "Thane",      "state": "Maharashtra",  "lat": 19.2183, "lon": 72.9783, "soil_type": "Laterite",      "sensors": 1},
    {"name": "Mumbai",     "state": "Maharashtra",  "lat": 19.076,  "lon": 72.8777, "soil_type": "Laterite",      "sensors": 1},
    {"name": "Pune",       "state": "Maharashtra",  "lat": 18.5204, "lon": 73.8567, "soil_type": "Red",           "sensors": 1},
    {"name": "Nashik",     "state": "Maharashtra",  "lat": 19.9975, "lon": 73.7898, "soil_type": "Black (Regur)", "sensors": 1},
    {"name": "Nagpur",     "state": "Maharashtra",  "lat": 21.1458, "lon": 79.0882, "soil_type": "Black (Regur)", "sensors": 1},
    {"name": "Delhi",      "state": "Delhi",        "lat": 28.7041, "lon": 77.1025, "soil_type": "Alluvial",      "sensors": 1},
    {"name": "Bangalore",  "state": "Karnataka",    "lat": 12.9716, "lon": 77.5946, "soil_type": "Red",           "sensors": 1},
    {"name": "Hyderabad",  "state": "Telangana",    "lat": 17.385,  "lon": 78.4867, "soil_type": "Red",           "sensors": 1},
    {"name": "Chennai",    "state": "Tamil Nadu",    "lat": 13.0827, "lon": 80.2707, "soil_type": "Sandy",         "sensors": 1},
    {"name": "Jaipur",     "state": "Rajasthan",    "lat": 26.9124, "lon": 75.7873, "soil_type": "Sandy",         "sensors": 1},
    {"name": "Lucknow",    "state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462, "soil_type": "Alluvial",      "sensors": 1},
    {"name": "Ahmedabad",  "state": "Gujarat",      "lat": 23.0225, "lon": 72.5714, "soil_type": "Alluvial",      "sensors": 1},
]

# WMO weather code descriptions
WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

_lock = Lock()


def load_config() -> dict:
    """Load city config from weather_config.json. Creates default if missing."""
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    # Default: Kolhapur
    default = {"city": "Kolhapur", "lat": 16.695, "lon": 74.2333}
    save_config(default)
    return default


def save_config(cfg: dict) -> None:
    """Write city config to weather_config.json."""
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def load_cache() -> Optional[dict]:
    """Load cached weather data if it exists and is fresh."""
    if not CACHE_PATH.exists():
        return None
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if time.time() - data.get("_fetched_at", 0) < CACHE_TTL_SECONDS:
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _save_cache(data: dict) -> None:
    """Write weather data to cache file."""
    data["_fetched_at"] = time.time()
    CACHE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def fetch_weather(lat: float, lon: float) -> Optional[dict]:
    """Fetch current + hourly weather from Open-Meteo API."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation,rain,"
        f"weather_code,wind_speed_10m,apparent_temperature"
        f"&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,rain"
        f"&forecast_days=1&timezone=auto"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AgroSensor/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        # Extract what we need
        current = raw.get("current", {})
        result = {
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation", 0),
            "rain": current.get("rain", 0),
            "weather_code": current.get("weather_code", 0),
            "wind_speed": current.get("wind_speed_10m", 0),
            "apparent_temperature": current.get("apparent_temperature"),
            "description": WMO_CODES.get(current.get("weather_code", 0), "Unknown"),
            "is_raining": current.get("rain", 0) > 0 or current.get("precipitation", 0) > 0.1,
            "hourly_temp": raw.get("hourly", {}).get("temperature_2m", []),
            "hourly_humidity": raw.get("hourly", {}).get("relative_humidity_2m", []),
            "hourly_rain_prob": raw.get("hourly", {}).get("precipitation_probability", []),
            "lat": lat,
            "lon": lon,
        }
        return result
    except Exception as e:
        logger.warning("Failed to fetch weather from Open-Meteo: %s", e)
        return None


def get_weather(force_refresh: bool = False) -> Optional[dict]:
    """
    Get current weather data. Uses cache if fresh, otherwise fetches.
    Thread-safe. Returns None if both cache and API fail.
    """
    with _lock:
        if not force_refresh:
            cached = load_cache()
            if cached:
                return cached

        cfg = load_config()
        data = fetch_weather(cfg["lat"], cfg["lon"])
        if data:
            data["city"] = cfg.get("city", "Unknown")
            _save_cache(data)
            return data

        # If fetch failed, return stale cache if available
        if CACHE_PATH.exists():
            try:
                return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return None


def set_city(city_name: str, lat: float | None = None, lon: float | None = None) -> Optional[dict]:
    """
    Set the active city by name.
    If lat/lon are provided (from DB), use them directly.
    Otherwise fall back to the hardcoded CITIES list (legacy/seed).
    Returns updated weather or None.
    """
    if lat is not None and lon is not None:
        save_config({"city": city_name, "lat": lat, "lon": lon})
    else:
        city = next((c for c in CITIES if c["name"].lower() == city_name.lower()), None)
        if not city:
            return None
        save_config({"city": city["name"], "lat": city["lat"], "lon": city["lon"]})
    # Clear cache to force refresh
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()
    return get_weather(force_refresh=True)

