"""
FastAPI route definitions for the AgroSensor API.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select, desc
from sqlalchemy.ext.asyncio import AsyncSession

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCATIONS_CACHE = _PROJECT_ROOT / "locations_cache.json"

from api.schemas import (
    CityListResponse,
    CityOut,
    HealthStatus,
    HistoryResponse,
    LocationCreate,
    LocationOut,
    LocationUpdate,
    SensorMetadataOut,
    SensorReadingLive,
    SensorReadingOut,
    SetCityRequest,
    TriggerReadResponse,
    WeatherResponse,
)
from config import settings
from database.connection import get_async_session
from database.models import Location, SensorData, SensorMetadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Will be set by main.py at startup
_sensor_reader = None
_mqtt_subscriber = None   # Set by main.py when MQTT mode is active
_start_time = time.time()

# Persistent FieldEnvironment cache for MQTT "Read Now"
# Keyed by city name → FieldEnvironment instance so values evolve naturally
# instead of resetting to soil-profile baselines on every click.
_env_cache: dict = {}


def set_sensor_reader(reader):
    """Called by main.py to inject the sensor reader instance."""
    global _sensor_reader
    _sensor_reader = reader


# ─── GET /api/health ───

@router.get("/health", response_model=HealthStatus)
async def health_check(session: AsyncSession = Depends(get_async_session)):
    """System, sensor, and database health status."""
    # DB check
    db_ok = True
    try:
        from sqlalchemy import text
        await session.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    # Sensor / MQTT check
    mqtt_mode = settings.MQTT_ENABLED
    mqtt_ok = False
    mqtt_msgs = 0
    if mqtt_mode and _mqtt_subscriber is not None:
        mqtt_ok = _mqtt_subscriber.is_connected
        mqtt_msgs = _mqtt_subscriber.message_count

    sensor_ok = mqtt_ok if mqtt_mode else (_sensor_reader is not None and _sensor_reader.is_connected)

    # Last reading
    last_reading_at = None
    try:
        result = await session.execute(
            select(SensorData.timestamp)
            .order_by(desc(SensorData.timestamp))
            .limit(1)
        )
        row = result.scalar_one_or_none()
        if row:
            last_reading_at = row
    except Exception:
        pass

    status = "healthy" if (db_ok and sensor_ok) else ("degraded" if db_ok else "unhealthy")

    return HealthStatus(
        status=status,
        sensor_connected=sensor_ok,
        database_connected=db_ok,
        mqtt_connected=mqtt_ok,
        mqtt_messages=mqtt_msgs,
        data_source="mqtt_cloud" if mqtt_mode else "serial",
        last_reading_at=last_reading_at,
        uptime_seconds=time.time() - _start_time,
        version=settings.APP_VERSION,
        polling_interval_seconds=settings.POLLING_INTERVAL_SECONDS,
    )


# ─── GET /api/sensor/live ───

@router.get("/sensor/live", response_model=SensorReadingOut)
async def get_live_reading(
    city: Optional[str] = Query(None, description="Filter by city name"),
    session: AsyncSession = Depends(get_async_session),
):
    """Returns the most recent sensor reading from the database, optionally filtered by city."""
    query = select(SensorData)
    if city:
        query = query.where(SensorData.city == city)
    query = query.order_by(desc(SensorData.timestamp)).limit(1)
    result = await session.execute(query)
    reading = result.scalar_one_or_none()
    if not reading:
        raise HTTPException(status_code=404, detail="No sensor readings available yet")
    return reading


# ─── GET /api/sensor/history ───

@router.get("/sensor/history", response_model=HistoryResponse)
async def get_history(
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    city: Optional[str] = Query(None, description="Filter by city name"),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=5000),
    session: AsyncSession = Depends(get_async_session),
):
    """Paginated historical readings with optional date range filter and city filter."""
    # Ensure timezone-aware datetimes (asyncpg + timestamptz needs aware datetimes)
    if start and start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end and end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    query = select(SensorData)
    count_query = select(func.count(SensorData.id))

    if city:
        query = query.where(SensorData.city == city)
        count_query = count_query.where(SensorData.city == city)
    if start:
        query = query.where(SensorData.timestamp >= start)
        count_query = count_query.where(SensorData.timestamp >= start)
    if end:
        query = query.where(SensorData.timestamp <= end)
        count_query = count_query.where(SensorData.timestamp <= end)

    total = (await session.execute(count_query)).scalar() or 0

    query = query.order_by(desc(SensorData.timestamp)).offset((page - 1) * limit).limit(limit)
    result = await session.execute(query)
    readings = result.scalars().all()

    return HistoryResponse(
        total=total,
        page=page,
        limit=limit,
        readings=[SensorReadingOut.model_validate(r) for r in readings],
    )


# ─── POST /api/sensor/read ───

@router.post("/sensor/read", response_model=TriggerReadResponse)
async def trigger_read(session: AsyncSession = Depends(get_async_session)):
    """Trigger an immediate sensor read, store it, and return the result."""

    # ─── MQTT Cloud Mode: generate a fresh simulated reading ───
    if settings.MQTT_ENABLED:
        from tools.sensor_conn import FieldEnvironment

        # Determine active city & soil type
        from tools.weather import load_config as _load_cfg
        cfg = _load_cfg()
        city_name = cfg.get("city", "Kolhapur")
        soil_type = ""
        loc_result = await session.execute(
            select(Location).where(func.lower(Location.name) == city_name.lower())
        )
        loc = loc_result.scalar_one_or_none()
        if loc:
            soil_type = loc.soil_type

        # Reuse a persistent FieldEnvironment so values evolve naturally
        # instead of resetting to baseline on every click (which caused spikes).
        cache_key = city_name.lower()
        if cache_key not in _env_cache:
            _env_cache[cache_key] = FieldEnvironment(soil_type=soil_type, sensor_index=0)
            logger.info("Created persistent FieldEnvironment for '%s' (%s)", city_name, soil_type)
        env = _env_cache[cache_key]
        raw_regs = env.read()

        sensor_id = f"AGRO-{city_name.upper()}-001"
        now_ts = datetime.now(timezone.utc)

        # Ensure sensor_metadata row exists (FK constraint)
        from sqlalchemy import select as sa_select
        meta_result = await session.execute(
            sa_select(SensorMetadata).where(SensorMetadata.sensor_id == sensor_id)
        )
        if meta_result.scalar_one_or_none() is None:
            meta = SensorMetadata(
                sensor_id=sensor_id,
                city=city_name,
                soil_type=soil_type,
                status="active",
                location_id=loc.id if loc else None,
            )
            session.add(meta)
            await session.flush()

        # Store in DB
        db_record = SensorData(
            sensor_id=sensor_id,
            timestamp=now_ts,
            city=city_name,
            soil_type=soil_type,
            temperature=raw_regs[0] / 10.0,
            moisture=raw_regs[1] / 10.0,
            ec=float(raw_regs[2]),
            ph=raw_regs[3] / 100.0,
            nitrogen=float(raw_regs[4]),
            phosphorus=float(raw_regs[5]),
            potassium=float(raw_regs[6]),
            raw_frame=str(raw_regs),
            is_valid=True,
        )
        session.add(db_record)
        await session.flush()

        live = SensorReadingLive(
            sensor_id=sensor_id,
            timestamp=now_ts,
            city=city_name,
            soil_type=soil_type,
            temperature=raw_regs[0] / 10.0,
            moisture=raw_regs[1] / 10.0,
            ec=float(raw_regs[2]),
            ph=raw_regs[3] / 100.0,
            nitrogen=float(raw_regs[4]),
            phosphorus=float(raw_regs[5]),
            potassium=float(raw_regs[6]),
            raw_registers=raw_regs,
            is_valid=True,
        )
        return TriggerReadResponse(
            success=True,
            message="Fresh simulated reading generated",
            reading=live,
        )

    # ─── Legacy COM Port Mode ───
    if _sensor_reader is None or not _sensor_reader.is_connected:
        return TriggerReadResponse(success=False, message="Sensor not connected")

    reading = _sensor_reader.read()
    if reading is None:
        return TriggerReadResponse(success=False, message="Failed to read sensor")

    # Determine active city & soil type from DB
    from tools.weather import load_config as _load_cfg
    cfg = _load_cfg()
    city_name = cfg.get("city", "Kolhapur")
    soil_type = ""
    loc_result = await session.execute(
        select(Location).where(func.lower(Location.name) == city_name.lower())
    )
    loc = loc_result.scalar_one_or_none()
    if loc:
        soil_type = loc.soil_type

    # Build a sensor_id matching the new naming convention
    sensor_id = f"AGRO-{city_name.upper()}-001"

    # Store in DB
    db_record = SensorData(
        sensor_id=sensor_id,
        timestamp=reading.timestamp,
        city=city_name,
        soil_type=soil_type,
        temperature=reading.temperature,
        moisture=reading.moisture,
        ec=reading.ec,
        ph=reading.ph,
        nitrogen=reading.nitrogen,
        phosphorus=reading.phosphorus,
        potassium=reading.potassium,
        raw_frame=str(reading.raw_registers),
        is_valid=reading.is_valid,
    )
    session.add(db_record)
    await session.flush()

    live = SensorReadingLive(
        sensor_id=sensor_id,
        timestamp=reading.timestamp,
        city=city_name,
        soil_type=soil_type,
        temperature=reading.temperature,
        moisture=reading.moisture,
        ec=reading.ec,
        ph=reading.ph,
        nitrogen=reading.nitrogen,
        phosphorus=reading.phosphorus,
        potassium=reading.potassium,
        raw_registers=reading.raw_registers,
        is_valid=reading.is_valid,
    )

    return TriggerReadResponse(success=True, message="Read successful", reading=live)


# ─── GET /api/sensor/metadata ───

@router.get("/sensor/metadata", response_model=SensorMetadataOut)
async def get_sensor_metadata(
    city: Optional[str] = Query(None, description="Filter by city name"),
    session: AsyncSession = Depends(get_async_session),
):
    """Returns sensor configuration and identity for the active city."""
    if not city:
        from tools.weather import load_config as _load_cfg
        city = _load_cfg().get("city", "Kolhapur")
    sensor_id = f"AGRO-{city.upper()}-001"
    result = await session.execute(
        select(SensorMetadata).where(SensorMetadata.sensor_id == sensor_id)
    )
    meta = result.scalar_one_or_none()
    if not meta:
        raise HTTPException(status_code=404, detail="Sensor metadata not found")
    return meta



# ─── Location CRUD Endpoints ───

async def _sync_locations_cache(session: AsyncSession):
    """Write ALL locations to JSON file for the MQTT publisher / sensor simulator."""
    result = await session.execute(
        select(Location).order_by(Location.name)
    )
    locs = result.scalars().all()
    cache = [
        {"name": l.name, "state": l.state, "lat": l.lat, "lon": l.lon,
         "soil_type": l.soil_type, "num_sensors": l.num_sensors,
         "is_active": l.is_active}
        for l in locs
    ]
    try:
        _LOCATIONS_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("Failed to write locations cache")


@router.get("/locations", response_model=list[LocationOut])
async def list_locations(session: AsyncSession = Depends(get_async_session)):
    """List all locations from the database."""
    result = await session.execute(
        select(Location).where(Location.is_active == True).order_by(Location.name)
    )
    return result.scalars().all()


@router.post("/locations", response_model=LocationOut, status_code=201)
async def create_location(loc: LocationCreate, session: AsyncSession = Depends(get_async_session)):
    """Add a new city/location. Also creates sensor metadata rows."""
    # Check for duplicate name
    existing = await session.execute(
        select(Location).where(func.lower(Location.name) == loc.name.lower())
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Location '{loc.name}' already exists")

    new_loc = Location(
        name=loc.name, state=loc.state,
        lat=loc.lat, lon=loc.lon,
        soil_type=loc.soil_type, num_sensors=1,
        is_active=False,
    )
    session.add(new_loc)
    await session.flush()  # get new_loc.id

    # Auto-create single sensor metadata
    sensor_id = f"AGRO-{loc.name.upper()}-001"
    meta = SensorMetadata(
        sensor_id=sensor_id,
        sensor_type=settings.SENSOR_TYPE,
        manufacturer=settings.SENSOR_MANUFACTURER,
        modbus_address=settings.SENSOR_MODBUS_ADDRESS,
        com_port=settings.SENSOR_COM_PORT,
        baud_rate=settings.SENSOR_BAUD_RATE,
        location_description=f"{loc.name}, {loc.state}",
        city=loc.name,
        soil_type=loc.soil_type,
        location_id=new_loc.id,
        status="active",
    )
    session.add(meta)

    logger.info("Created location '%s' with sensor %s", loc.name, sensor_id)
    await _sync_locations_cache(session)
    return new_loc


@router.delete("/locations/{location_id}", status_code=204)
async def delete_location(location_id: int, session: AsyncSession = Depends(get_async_session)):
    """Soft-delete a location (marks inactive). Sensor data is preserved."""
    result = await session.execute(select(Location).where(Location.id == location_id))
    loc = result.scalar_one_or_none()
    if not loc:
        raise HTTPException(status_code=404, detail="Location not found")
    loc.is_active = False
    await session.flush()
    await _sync_locations_cache(session)
    logger.info("Deactivated location '%s'", loc.name)


@router.patch("/locations/{location_id}/toggle", response_model=LocationOut)
async def toggle_location(location_id: int, session: AsyncSession = Depends(get_async_session)):
    """Toggle a location's sensor on/off (is_active flag)."""
    result = await session.execute(select(Location).where(Location.id == location_id))
    loc = result.scalar_one_or_none()
    if not loc:
        raise HTTPException(status_code=404, detail="Location not found")
    loc.is_active = not loc.is_active
    await session.flush()
    await _sync_locations_cache(session)
    logger.info("Toggled location '%s' -> %s", loc.name, "ON" if loc.is_active else "OFF")
    return loc


# ─── Weather Endpoints ───

async def _get_city_soil_info(city_name: str, session: AsyncSession) -> tuple[str, str]:
    """Return (soil_type, soil_description) for a city from the DB."""
    from tools.weather import SOIL_PROFILES
    result = await session.execute(
        select(Location).where(func.lower(Location.name) == city_name.lower())
    )
    loc = result.scalar_one_or_none()
    if loc:
        desc = SOIL_PROFILES.get(loc.soil_type, {}).get("description", "")
        return loc.soil_type, desc
    return "", ""


@router.get("/weather", response_model=WeatherResponse)
async def get_weather(session: AsyncSession = Depends(get_async_session)):
    """Get current weather data from Open-Meteo (cached hourly)."""
    from tools.weather import get_weather as _get_weather
    data = _get_weather()
    if not data:
        raise HTTPException(status_code=503, detail="Weather data unavailable")
    city_name = data.get("city", "")
    soil_type, soil_desc = await _get_city_soil_info(city_name, session)
    return WeatherResponse(
        city=city_name,
        soil_type=soil_type,
        soil_description=soil_desc,
        temperature=data.get("temperature"),
        humidity=data.get("humidity"),
        precipitation=data.get("precipitation", 0),
        rain=data.get("rain", 0),
        weather_code=data.get("weather_code", 0),
        wind_speed=data.get("wind_speed", 0),
        apparent_temperature=data.get("apparent_temperature"),
        description=data.get("description", "Unknown"),
        is_raining=data.get("is_raining", False),
    )


@router.get("/weather/cities", response_model=CityListResponse)
async def list_cities(session: AsyncSession = Depends(get_async_session)):
    """List ALL cities from the DB (active + inactive) so dashboard can show toggle state."""
    from tools.weather import load_config
    result = await session.execute(
        select(Location).order_by(Location.name)
    )
    locations = result.scalars().all()
    cfg = load_config()
    cities = [
        CityOut(
            id=loc.id, name=loc.name, state=loc.state, lat=loc.lat, lon=loc.lon,
            soil_type=loc.soil_type, sensors=loc.num_sensors,
            is_active=loc.is_active,
        )
        for loc in locations
    ]
    return CityListResponse(cities=cities, current=cfg.get("city", "Kolhapur"))


@router.put("/weather/city", response_model=WeatherResponse)
async def change_city(req: SetCityRequest, session: AsyncSession = Depends(get_async_session)):
    """Change the active city. Looks up from DB, triggers a fresh weather fetch."""
    from tools.weather import set_city as _set_city
    # Look up city in DB first (dynamic)
    result = await session.execute(
        select(Location).where(func.lower(Location.name) == req.city.lower())
    )
    loc = result.scalar_one_or_none()
    if loc:
        data = _set_city(req.city, lat=loc.lat, lon=loc.lon)
    else:
        # Fallback to hardcoded list (legacy)
        data = _set_city(req.city)
    if not data:
        raise HTTPException(status_code=400, detail=f"Unknown city: {req.city}")
    city_name = data.get("city", "")
    soil_type, soil_desc = await _get_city_soil_info(city_name, session)
    return WeatherResponse(
        city=city_name,
        soil_type=soil_type,
        soil_description=soil_desc,
        temperature=data.get("temperature"),
        humidity=data.get("humidity"),
        precipitation=data.get("precipitation", 0),
        rain=data.get("rain", 0),
        weather_code=data.get("weather_code", 0),
        wind_speed=data.get("wind_speed", 0),
        apparent_temperature=data.get("apparent_temperature"),
        description=data.get("description", "Unknown"),
        is_raining=data.get("is_raining", False),
    )
