"""
Unit tests for API endpoints.
Uses SQLite in-memory DB and mocked sensor reader.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

from httpx import AsyncClient, ASGITransport
from sqlalchemy import StaticPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from database.models import Base, SensorMetadata, SensorData
from api.schemas import SensorReadingLive
from test_sensor.sensor_reader import SensorReading


# ═══════════════════════════════════════════════════════════
# Test app factory — bypasses PostgreSQL and real sensor
# ═══════════════════════════════════════════════════════════

@pytest_asyncio.fixture(scope="function")
async def test_app():
    """Create a FastAPI test app with SQLite in-memory and no real sensor."""
    from fastapi import FastAPI
    import config as agro_config

    # Default config uses MQTT mode; tests inject a mock Modbus reader instead.
    agro_config.settings.MQTT_ENABLED = False

    from api.routes import router, set_sensor_reader
    from database.connection import get_async_session

    # 1. In-memory SQLite
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )

    # 2. Override DB dependency
    async def override_get_session():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # 3. Seed data
    async with session_factory() as session:
        meta = SensorMetadata(
            sensor_id="AGRO-7IN1-001",
            sensor_type="7-in-1 Soil Sensor",
            manufacturer="Test",
            status="active",
        )
        session.add(meta)
        await session.flush()
        for i in range(3):
            session.add(SensorData(
                sensor_id="AGRO-7IN1-001",
                timestamp=datetime(2026, 3, 11, 10, i, 0, tzinfo=timezone.utc),
                temperature=25.0 + i, moisture=45.0, ec=1200.0,
                ph=6.5, nitrogen=100.0, phosphorus=80.0, potassium=150.0,
                is_valid=True,
            ))
        await session.commit()

    # 4. Build app
    app = FastAPI()
    app.dependency_overrides[get_async_session] = override_get_session
    app.include_router(router)

    # 5. Mock sensor reader (connected)
    mock_reader = MagicMock()
    mock_reader.is_connected = True
    mock_reader.read.return_value = SensorReading(
        temperature=26.0, moisture=50.0, ec=1300.0, ph=6.8,
        nitrogen=110.0, phosphorus=85.0, potassium=160.0,
        timestamp=datetime(2026, 3, 11, 12, 0, 0, tzinfo=timezone.utc),
        raw_registers=[260, 500, 1300, 680, 110, 85, 160],
        is_valid=True,
    )
    set_sensor_reader(mock_reader)

    yield app

    # Cleanup
    set_sensor_reader(None)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def client(test_app):
    """Async HTTP test client."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ═══════════════════════════════════════════════════════════
# Endpoint tests
# ═══════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["database_connected"] is True
    assert data["sensor_connected"] is True
    assert data["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_live_reading(client):
    resp = await client.get("/api/sensor/live")
    assert resp.status_code == 200
    data = resp.json()
    assert data["sensor_id"] == "AGRO-7IN1-001"
    assert data["temperature"] == pytest.approx(27.0)  # latest seeded (i=2)


@pytest.mark.asyncio
async def test_history_endpoint(client):
    resp = await client.get("/api/sensor/history?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert len(data["readings"]) == 3


@pytest.mark.asyncio
async def test_history_pagination(client):
    resp = await client.get("/api/sensor/history?limit=2&page=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert len(data["readings"]) == 2


@pytest.mark.asyncio
async def test_trigger_read(client):
    resp = await client.post("/api/sensor/read")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["reading"]["temperature"] == pytest.approx(26.0)


@pytest.mark.asyncio
async def test_sensor_metadata(client):
    # Metadata route resolves sensor_id as AGRO-{city.upper()}-001; seed uses 7IN1.
    resp = await client.get("/api/sensor/metadata?city=7IN1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["sensor_id"] == "AGRO-7IN1-001"
    assert data["sensor_type"] == "7-in-1 Soil Sensor"

