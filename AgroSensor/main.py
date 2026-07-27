"""
AgroSensor — FastAPI Application Entry Point
Handles startup/shutdown, MQTT cloud subscription, and serves the dashboard.

Data flow:
    Sensor Simulator (mqtt_publisher.py)
        → HiveMQ Cloud (MQTT broker)
        → This server (MQTT subscriber)
        → PostgreSQL
"""

import asyncio
import logging
import queue
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router, set_sensor_reader
from config import settings
from database.connection import async_engine, get_session_ctx
from database.models import Base, Location, SensorData, SensorMetadata
from test_sensor.modbus_client import ModbusClient
from test_sensor.sensor_reader import SensorReader

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Silence noisy third-party loggers
for _noisy in ("uvicorn.access", "uvicorn.error", "sqlalchemy.engine", "asyncio"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Global state
_reader: SensorReader | None = None
_polling_task: asyncio.Task | None = None
_weather_task: asyncio.Task | None = None
_mqtt_subscriber = None           # MQTTSubscriber instance
_mqtt_pending: queue.Queue = queue.Queue()  # thread-safe queue for MQTT thread → asyncio


async def _init_db():
    """Create tables if they don't exist."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ensured")


async def _seed_locations():
    """Seed the locations table from CITIES if empty, then ensure sensor metadata."""
    from sqlalchemy import select, func as sa_func
    from tools.weather import CITIES

    async with get_session_ctx() as session:
        # Check if locations table is already populated
        count = (await session.execute(select(sa_func.count(Location.id)))).scalar() or 0
        if count == 0:
            logger.info("Seeding %d default locations...", len(CITIES))
            for city_info in CITIES:
                # Only Kolhapur is active by default
                is_active = city_info["name"].lower() == "kolhapur"
                loc = Location(
                    name=city_info["name"],
                    state=city_info.get("state", ""),
                    lat=city_info["lat"],
                    lon=city_info["lon"],
                    soil_type=city_info.get("soil_type", "Alluvial"),
                    num_sensors=1,
                    is_active=is_active,
                )
                session.add(loc)
            await session.flush()
            logger.info("Seeded %d locations (Kolhapur active)", len(CITIES))

        # Now ensure sensor metadata exists for all active locations
        locations = (await session.execute(
            select(Location).where(Location.is_active == True)
        )).scalars().all()

        for loc in locations:
            sensor_id = f"AGRO-{loc.name.upper()}-001"
            result = await session.execute(
                select(SensorMetadata).where(SensorMetadata.sensor_id == sensor_id)
            )
            if result.scalar_one_or_none() is None:
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
                    location_id=loc.id,
                    status="active",
                )
                session.add(meta)
        logger.info("Sensor metadata ensured for %d locations", len(locations))


async def _refresh_weather():
    """Background task: refresh weather data every hour."""
    from tools.weather import get_weather
    logger.info("Weather refresh task started (interval=3600s)")
    while True:
        try:
            data = get_weather(force_refresh=False)
            if data:
                logger.debug("Weather: %s — %s, %.1f°C",
                             data.get("city", "?"), data.get("description", "?"),
                             data.get("temperature", 0))
            else:
                logger.warning("Weather refresh: no data available")
            await asyncio.sleep(3600)  # 1 hour
        except asyncio.CancelledError:
            logger.info("Weather refresh task cancelled")
            break
        except Exception:
            logger.exception("Weather refresh error")
            await asyncio.sleep(300)  # Retry in 5 min on error


async def _poll_sensor():
    """Background task: poll sensors via COM port (legacy fallback when MQTT disabled)."""
    from sqlalchemy import select as sa_select
    from tools.sensor_conn import get_active_readings, _get_active_city
    from test_sensor.sensor_reader import scale_registers, validate_reading

    logger.info("Sensor polling started (interval=%ss)", settings.POLLING_INTERVAL_SECONDS)
    while True:
        try:
            if _reader is None or not _reader.is_connected:
                logger.warning("Sensor not connected, skipping poll")
            else:
                city = _get_active_city()
                soil_type = ""
                async with get_session_ctx() as session:
                    loc_result = await session.execute(
                        sa_select(Location).where(Location.name == city)
                    )
                    loc = loc_result.scalar_one_or_none()
                    if loc:
                        soil_type = loc.soil_type

                readings = get_active_readings()
                now = datetime.now(timezone.utc)

                async with get_session_ctx() as session:
                    for suffix, _idx, raw_regs in readings:
                        sensor_id = f"AGRO-{suffix}"
                        try:
                            scaled = scale_registers(raw_regs)
                            is_valid = validate_reading(scaled)
                        except ValueError:
                            logger.warning("Invalid registers from %s", sensor_id)
                            continue
                        record = SensorData(
                            sensor_id=sensor_id, timestamp=now,
                            city=city, soil_type=soil_type,
                            temperature=scaled["temperature"], moisture=scaled["moisture"],
                            ec=scaled["ec"], ph=scaled["ph"],
                            nitrogen=scaled["nitrogen"], phosphorus=scaled["phosphorus"],
                            potassium=scaled["potassium"],
                            raw_frame=str(raw_regs), is_valid=is_valid,
                        )
                        session.add(record)
                    logger.debug("Poll: stored %d readings for %s", len(readings), city)

            await asyncio.sleep(settings.POLLING_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            logger.info("Polling task cancelled")
            break
        except Exception:
            logger.exception("Polling error")
            await asyncio.sleep(settings.POLLING_INTERVAL_SECONDS)


# ─── MQTT Cloud Integration ───

def _on_mqtt_reading(payload: dict):
    """
    Callback from MQTT subscriber thread — enqueues reading for async DB storage.
    This runs in the paho-mqtt network thread, so we push to an asyncio Queue.
    """
    try:
        _mqtt_pending.put_nowait(payload)
    except Exception as e:
        logger.warning("Failed to enqueue MQTT reading: %s", e)


def _queue_get_with_timeout():
    """Blocking get with timeout — ensures the executor thread always returns."""
    try:
        return _mqtt_pending.get(timeout=5)
    except queue.Empty:
        return None


async def _mqtt_db_writer():
    """
    Background asyncio task: drains the MQTT queue and stores readings in PostgreSQL.
    Uses queue.Queue (thread-safe) with run_in_executor for blocking get().
    """
    from sqlalchemy import select as sa_select
    loop = asyncio.get_running_loop()
    logger.info("☁️  MQTT → DB writer started")
    while True:
        try:
            payload = await loop.run_in_executor(None, _queue_get_with_timeout)
        except asyncio.CancelledError:
            logger.info("MQTT DB writer cancelled")
            break

        if payload is None:
            continue  # queue was empty, loop again

        try:
            city = payload.get("city", "")
            sensor_id = payload.get("sensor_id", "")
            ts_str = payload.get("timestamp", "")

            # Parse timestamp
            try:
                ts = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                ts = datetime.now(timezone.utc)

            async with get_session_ctx() as session:
                # Look up soil_type from DB
                soil_type = ""
                loc_result = await session.execute(
                    sa_select(Location).where(Location.name == city)
                )
                loc = loc_result.scalar_one_or_none()
                if loc:
                    soil_type = loc.soil_type

                # Ensure sensor_metadata row exists (FK constraint)
                meta_result = await session.execute(
                    sa_select(SensorMetadata).where(SensorMetadata.sensor_id == sensor_id)
                )
                if meta_result.scalar_one_or_none() is None:
                    meta = SensorMetadata(
                        sensor_id=sensor_id,
                        city=city,
                        soil_type=soil_type,
                        status="active",
                        location_id=loc.id if loc else None,
                    )
                    session.add(meta)
                    await session.flush()
                    logger.info("Auto-created sensor metadata: %s", sensor_id)

                record = SensorData(
                    sensor_id=sensor_id,
                    timestamp=ts,
                    city=city,
                    soil_type=soil_type,
                    temperature=payload.get("temperature", 0),
                    moisture=payload.get("moisture", 0),
                    ec=payload.get("ec", 0),
                    ph=payload.get("ph", 0),
                    nitrogen=payload.get("nitrogen", 0),
                    phosphorus=payload.get("phosphorus", 0),
                    potassium=payload.get("potassium", 0),
                    raw_frame=payload.get("raw_frame", ""),
                    is_valid=True,
                )
                session.add(record)
            logger.debug("☁️  Stored MQTT reading: %s / %s", sensor_id, city)

        except Exception:
            logger.exception("Error storing MQTT reading")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global _reader, _polling_task, _weather_task, _mqtt_subscriber

    # 1. Init database & seed locations
    await _init_db()
    await _seed_locations()

    _mqtt_writer_task = None

    if settings.MQTT_ENABLED:
        # ─── MQTT Cloud Mode ───
        from tools.mqtt_subscriber import MQTTSubscriber

        _mqtt_subscriber = MQTTSubscriber(on_reading_callback=_on_mqtt_reading)
        _mqtt_subscriber.start()
        logger.info("☁️  MQTT cloud mode enabled — listening for sensor data from HiveMQ")

        # Start the async DB writer that drains the MQTT queue
        _mqtt_writer_task = asyncio.create_task(_mqtt_db_writer())

        # Export subscriber reference so health endpoint can check it
        from api import routes as _routes
        _routes._mqtt_subscriber = _mqtt_subscriber
    else:
        # ─── Legacy COM Port Mode ───
        try:
            client = ModbusClient()
            _reader = SensorReader(client=client)
            if _reader.connect():
                logger.info("Sensor connected on %s", settings.SENSOR_COM_PORT)
                set_sensor_reader(_reader)
            else:
                logger.warning("Sensor not available — running in DB-only mode")
                _reader = None
        except Exception:
            logger.exception("Failed to initialize sensor")
            _reader = None

        if settings.POLLING_ENABLED and _reader is not None:
            _polling_task = asyncio.create_task(_poll_sensor())

    # Weather refresh (always — provides data for dashboard)
    _weather_task = asyncio.create_task(_refresh_weather())

    yield  # App is running

    # ─── Shutdown ───
    if _mqtt_subscriber:
        _mqtt_subscriber.stop()
        logger.info("MQTT subscriber stopped")

    for task in (_polling_task, _weather_task, _mqtt_writer_task):
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    if _reader:
        _reader.disconnect()
        logger.info("Sensor disconnected")

    await async_engine.dispose()
    logger.info("Database connections closed")


# --- FastAPI App ---
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    print("Application is running on localhost : http://127.0.0.1:8000")
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
    
