"""
AgroSensor - Central Configuration
All settings are configurable via environment variables or .env file.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # --- Application ---
    APP_NAME: str = "AgroSensor"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # --- Database (PostgreSQL) ---
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "agro"
    DB_NAME: str = "agrosensor"

    @property
    def DATABASE_URL_SYNC(self) -> str:
        """Synchronous database URL for Alembic migrations."""
        return f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def DATABASE_URL_ASYNC(self) -> str:
        """Async database URL for FastAPI/SQLAlchemy async sessions."""
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # --- Sensor / Modbus ---
    SENSOR_COM_PORT: str = "COM3"
    SENSOR_BAUD_RATE: int = 9600
    SENSOR_MODBUS_ADDRESS: int = 1
    SENSOR_TIMEOUT: float = 1.0
    SENSOR_REGISTER_START: int = 0x0000
    SENSOR_REGISTER_COUNT: int = 7
    SENSOR_WARMUP_SECONDS: float = 15.0

    # --- Polling ---
    POLLING_INTERVAL_SECONDS: float = 100.0
    POLLING_ENABLED: bool = True

    # --- Sensor Identity ---
    SENSOR_ID: str = "AGRO-7IN1-001"
    SENSOR_TYPE: str = "7-in-1 Soil Sensor"
    SENSOR_MANUFACTURER: str = "Bombay Electronics / White-label"
    SENSOR_LOCATION: str = "Development Lab"

    # --- MQTT Cloud (HiveMQ) ---
    MQTT_BROKER: str = "62b38161a1354807bf5b943683a20f06.s1.eu.hivemq.cloud"
    MQTT_PORT: int = 8883
    MQTT_USERNAME: str = "agrosensor"
    MQTT_PASSWORD: str = "AgroMQTT2026!"
    MQTT_TOPIC_PREFIX: str = "agro"          # topics: agro/<city>/sensor-001
    MQTT_USE_TLS: bool = True
    MQTT_ENABLED: bool = True

    # --- API Server ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # --- Validation Ranges (from sensor specs) ---
    MOISTURE_MIN: float = 0.0
    MOISTURE_MAX: float = 100.0
    TEMPERATURE_MIN: float = -40.0
    TEMPERATURE_MAX: float = 80.0
    EC_MIN: float = 0.0
    EC_MAX: float = 20000.0
    PH_MIN: float = 3.0
    PH_MAX: float = 9.0
    NPK_MIN: float = 0.0
    NPK_MAX: float = 2999.0

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


settings = Settings()

