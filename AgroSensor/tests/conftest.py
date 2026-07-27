"""
Shared test fixtures for AgroSensor tests.
Uses SQLite in-memory database so tests run without PostgreSQL.
"""

import asyncio
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy import StaticPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from database.models import Base, SensorMetadata, SensorData


# --- Async SQLite engine for tests ---
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for all async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def async_engine():
    """Create a fresh in-memory SQLite engine per test."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def async_session(async_engine):
    """Provide an async session bound to the test engine."""
    session_factory = async_sessionmaker(
        bind=async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture(scope="function")
async def seeded_session(async_session):
    """Session pre-loaded with one SensorMetadata + 3 SensorData rows."""
    meta = SensorMetadata(
        sensor_id="TEST-SENSOR-001",
        sensor_type="7-in-1 Soil Sensor",
        manufacturer="Test Manufacturer",
        modbus_address=1,
        com_port="COM99",
        baud_rate=9600,
        location_description="Unit Test",
        status="active",
    )
    async_session.add(meta)
    await async_session.flush()

    for i in range(3):
        reading = SensorData(
            sensor_id="TEST-SENSOR-001",
            timestamp=datetime(2026, 3, 11, 10, i, 0, tzinfo=timezone.utc),
            temperature=25.0 + i,
            moisture=45.0 + i,
            ec=1200.0 + i * 10,
            ph=6.5 + i * 0.1,
            nitrogen=100.0 + i,
            phosphorus=80.0 + i,
            potassium=150.0 + i,
            raw_frame=f"[250, 450, 1200, 650, 100, 80, 150]",
            is_valid=True,
        )
        async_session.add(reading)

    await async_session.commit()
    yield async_session

