"""
Pydantic request/response models for the AgroSensor API.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# --- Sensor Reading Schemas ---

class SensorReadingOut(BaseModel):
    """Single sensor reading response."""
    id: int
    sensor_id: str
    timestamp: datetime
    city: Optional[str] = None
    soil_type: Optional[str] = None
    temperature: Optional[float] = None
    moisture: Optional[float] = None
    ec: Optional[float] = None
    ph: Optional[float] = None
    nitrogen: Optional[float] = None
    phosphorus: Optional[float] = None
    potassium: Optional[float] = None
    is_valid: bool = True

    model_config = {"from_attributes": True}


class SensorReadingLive(BaseModel):
    """Live reading response (includes raw data for debugging)."""
    sensor_id: str
    timestamp: datetime
    city: Optional[str] = None
    soil_type: Optional[str] = None
    temperature: float
    moisture: float
    ec: float
    ph: float
    nitrogen: float
    phosphorus: float
    potassium: float
    raw_registers: list[int]
    is_valid: bool


# --- History ---

class HistoryQuery(BaseModel):
    """Query parameters for historical data."""
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    param: Optional[str] = None
    page: int = Field(default=1, ge=1)
    limit: int = Field(default=50, ge=1, le=5000)


class HistoryResponse(BaseModel):
    """Paginated history response."""
    total: int
    page: int
    limit: int
    readings: list[SensorReadingOut]


# --- Health ---

class HealthStatus(BaseModel):
    """System health check response."""
    status: str  # "healthy" / "degraded" / "unhealthy"
    sensor_connected: bool
    database_connected: bool
    mqtt_connected: bool = False
    mqtt_messages: int = 0
    data_source: str = "serial"    # "mqtt_cloud" or "serial"
    last_reading_at: Optional[datetime] = None
    uptime_seconds: float
    version: str
    polling_interval_seconds: float = 300.0


# --- Sensor Metadata ---

class SensorMetadataOut(BaseModel):
    """Sensor configuration and identity."""
    sensor_id: str
    sensor_type: str
    manufacturer: str
    modbus_address: int
    com_port: str
    baud_rate: int
    location_description: Optional[str] = None
    city: Optional[str] = None
    soil_type: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# --- On-demand read ---

class TriggerReadResponse(BaseModel):
    """Response from POST /api/sensor/read."""
    success: bool
    message: str
    reading: Optional[SensorReadingLive] = None


# --- Locations (dynamic city management) ---

class LocationOut(BaseModel):
    """Location response from DB."""
    id: int
    name: str
    state: str
    lat: float
    lon: float
    soil_type: str = "Alluvial"
    num_sensors: int = 1
    is_active: bool = False
    created_at: datetime

    model_config = {"from_attributes": True}


class LocationCreate(BaseModel):
    """Request to add a new city/location."""
    name: str = Field(..., min_length=2, max_length=50)
    state: str = Field(..., min_length=2, max_length=50)
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    soil_type: str = Field(default="Alluvial", max_length=50)
    num_sensors: int = Field(default=1, ge=1, le=10)


class LocationUpdate(BaseModel):
    """Request to update a location."""
    name: Optional[str] = None
    state: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    soil_type: Optional[str] = None
    num_sensors: Optional[int] = None
    is_active: Optional[bool] = None


# --- Weather ---

class CityOut(BaseModel):
    """Single city option (for weather dropdown)."""
    id: int
    name: str
    state: str
    lat: float
    lon: float
    soil_type: str = ""
    sensors: int = 1
    is_active: bool = False

    model_config = {"from_attributes": True}


class WeatherResponse(BaseModel):
    """Current weather data from Open-Meteo."""
    city: str = ""
    soil_type: str = ""
    soil_description: str = ""
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    precipitation: float = 0
    rain: float = 0
    weather_code: int = 0
    wind_speed: float = 0
    apparent_temperature: Optional[float] = None
    description: str = "Unknown"
    is_raining: bool = False


class CityListResponse(BaseModel):
    """List of available cities with current selection."""
    cities: list[CityOut]
    current: str


class SetCityRequest(BaseModel):
    """Request to change the active city."""
    city: str

