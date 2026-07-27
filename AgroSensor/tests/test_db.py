"""
Unit tests for database models and queries.
Uses SQLite in-memory via conftest.py fixtures — no PostgreSQL needed.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from sqlalchemy import select, func, desc

from database.models import Base, SensorMetadata, SensorData


# ═══════════════════════════════════════════════════════════
# Model creation tests
# ═══════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_create_sensor_metadata(async_session):
    """Insert a SensorMetadata row and verify all fields persist."""
    meta = SensorMetadata(
        sensor_id="DB-TEST-001",
        sensor_type="7-in-1 Soil Sensor",
        manufacturer="Test",
        modbus_address=1,
        com_port="COM5",
        baud_rate=9600,
        location_description="Lab",
        status="active",
    )
    async_session.add(meta)
    await async_session.commit()

    result = await async_session.execute(
        select(SensorMetadata).where(SensorMetadata.sensor_id == "DB-TEST-001")
    )
    row = result.scalar_one()
    assert row.sensor_id == "DB-TEST-001"
    assert row.baud_rate == 9600
    assert row.status == "active"
    assert row.created_at is not None


@pytest.mark.asyncio
async def test_create_sensor_data(async_session):
    """Insert metadata + data row, verify FK relationship works."""
    # Must insert metadata first (FK constraint)
    meta = SensorMetadata(sensor_id="DB-TEST-002", status="active")
    async_session.add(meta)
    await async_session.flush()

    reading = SensorData(
        sensor_id="DB-TEST-002",
        timestamp=datetime(2026, 3, 11, 12, 0, 0, tzinfo=timezone.utc),
        temperature=25.5,
        moisture=42.0,
        ec=1100.0,
        ph=6.8,
        nitrogen=120.0,
        phosphorus=90.0,
        potassium=180.0,
        raw_frame="[255, 420, 1100, 680, 120, 90, 180]",
        is_valid=True,
    )
    async_session.add(reading)
    await async_session.commit()

    result = await async_session.execute(
        select(SensorData).where(SensorData.sensor_id == "DB-TEST-002")
    )
    row = result.scalar_one()
    assert row.temperature == pytest.approx(25.5)
    assert row.ph == pytest.approx(6.8)
    assert row.is_valid is True


# ═══════════════════════════════════════════════════════════
# Query tests using seeded data
# ═══════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_query_latest_reading(seeded_session):
    """Should return the most recent reading (timestamp-ordered)."""
    result = await seeded_session.execute(
        select(SensorData)
        .where(SensorData.sensor_id == "TEST-SENSOR-001")
        .order_by(desc(SensorData.timestamp))
        .limit(1)
    )
    latest = result.scalar_one()
    # The 3rd reading (i=2) has the latest timestamp: 10:02:00
    assert latest.temperature == pytest.approx(27.0)
    assert latest.moisture == pytest.approx(47.0)


@pytest.mark.asyncio
async def test_query_count(seeded_session):
    """Should have exactly 3 seeded readings."""
    result = await seeded_session.execute(
        select(func.count(SensorData.id))
        .where(SensorData.sensor_id == "TEST-SENSOR-001")
    )
    assert result.scalar() == 3


@pytest.mark.asyncio
async def test_query_by_time_range(seeded_session):
    """Filter readings by timestamp range."""
    start = datetime(2026, 3, 11, 10, 1, 0, tzinfo=timezone.utc)
    result = await seeded_session.execute(
        select(SensorData)
        .where(SensorData.sensor_id == "TEST-SENSOR-001")
        .where(SensorData.timestamp >= start)
        .order_by(SensorData.timestamp)
    )
    rows = result.scalars().all()
    assert len(rows) == 2  # 10:01:00 and 10:02:00


@pytest.mark.asyncio
async def test_metadata_relationship(seeded_session):
    """Verify the metadata ↔ readings relationship."""
    result = await seeded_session.execute(
        select(SensorMetadata).where(SensorMetadata.sensor_id == "TEST-SENSOR-001")
    )
    meta = result.scalar_one()
    assert meta.sensor_type == "7-in-1 Soil Sensor"


@pytest.mark.asyncio
async def test_sensor_id_unique_constraint(async_session):
    """Inserting duplicate sensor_id should raise an error."""
    meta1 = SensorMetadata(sensor_id="UNIQUE-001", status="active")
    meta2 = SensorMetadata(sensor_id="UNIQUE-001", status="active")
    async_session.add(meta1)
    await async_session.flush()
    async_session.add(meta2)
    with pytest.raises(Exception):  # IntegrityError
        await async_session.flush()

