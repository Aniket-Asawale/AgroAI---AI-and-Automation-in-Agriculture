"""
SQLAlchemy ORM models for the AgroSensor database.
Tables:
  - sensor_metadata: sensor identity, configuration, status
  - sensor_data: timestamped 7-parameter readings
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class Location(Base):
    """Dynamic city/location registry — replaces hardcoded CITIES list."""
    __tablename__ = "locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    state: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    soil_type: Mapped[str] = mapped_column(String(50), nullable=False, default="Alluvial")
    num_sensors: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    sensors: Mapped[list["SensorMetadata"]] = relationship(back_populates="location", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Location(name='{self.name}', soil_type='{self.soil_type}')>"


class SensorMetadata(Base):
    __tablename__ = "sensor_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sensor_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    sensor_type: Mapped[str] = mapped_column(String(100), default="7-in-1 Soil Sensor")
    manufacturer: Mapped[str] = mapped_column(String(100), default="Bombay Electronics / White-label")
    modbus_address: Mapped[int] = mapped_column(Integer, default=1)
    com_port: Mapped[str] = mapped_column(String(20), default="COM3")
    baud_rate: Mapped[int] = mapped_column(Integer, default=9600)
    location_description: Mapped[str] = mapped_column(Text, nullable=True)
    city: Mapped[str] = mapped_column(String(50), nullable=True, default=None, index=True)
    soil_type: Mapped[str] = mapped_column(String(50), nullable=True, default=None)
    location_id: Mapped[int] = mapped_column(Integer, ForeignKey("locations.id"), nullable=True)
    calibration_offset_json: Mapped[str] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    location: Mapped["Location"] = relationship(back_populates="sensors")
    readings: Mapped[list["SensorData"]] = relationship(back_populates="sensor", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<SensorMetadata(sensor_id='{self.sensor_id}', city='{self.city}', status='{self.status}')>"


class SensorData(Base):
    __tablename__ = "sensor_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sensor_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("sensor_metadata.sensor_id"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Location context
    city: Mapped[str] = mapped_column(String(50), nullable=True, default=None)
    soil_type: Mapped[str] = mapped_column(String(50), nullable=True, default=None)

    # 7 measured parameters
    temperature: Mapped[float] = mapped_column(Float, nullable=True)   # °C
    moisture: Mapped[float] = mapped_column(Float, nullable=True)      # % RH
    ec: Mapped[float] = mapped_column(Float, nullable=True)            # μS/cm
    ph: Mapped[float] = mapped_column(Float, nullable=True)            # pH
    nitrogen: Mapped[float] = mapped_column(Float, nullable=True)      # mg/kg
    phosphorus: Mapped[float] = mapped_column(Float, nullable=True)    # mg/kg
    potassium: Mapped[float] = mapped_column(Float, nullable=True)     # mg/kg

    # Debug / audit
    raw_frame: Mapped[str] = mapped_column(Text, nullable=True)
    is_valid: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationship
    sensor: Mapped["SensorMetadata"] = relationship(back_populates="readings")

    # Indexes for fast queries
    __table_args__ = (
        Index("idx_sensor_data_sensor_time", "sensor_id", timestamp.desc()),
        Index("idx_sensor_data_timestamp", timestamp.desc()),
        Index("idx_sensor_data_city", "city"),
    )

    def __repr__(self) -> str:
        return (
            f"<SensorData(sensor_id='{self.sensor_id}', "
            f"timestamp='{self.timestamp}', valid={self.is_valid})>"
        )

