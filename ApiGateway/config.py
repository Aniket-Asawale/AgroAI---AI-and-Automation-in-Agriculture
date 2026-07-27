"""
API Gateway Configuration — Backend service URLs and settings.
"""

from pydantic_settings import BaseSettings


class GatewaySettings(BaseSettings):
    """All settings can be overridden via environment variables."""

    # Gateway
    GATEWAY_HOST: str = "127.0.0.1"
    GATEWAY_PORT: int = 8080
    DEBUG: bool = True

    # Backend services
    AGROSENSOR_URL: str = "http://127.0.0.1:8000"
    CROP_RECOMMENDATION_URL: str = "http://127.0.0.1:8001"
    AUTH_URL: str = "http://127.0.0.1:8002"
    PLANT_DISEASE_URL: str = "http://127.0.0.1:8003"

    # Timeouts (seconds)
    REQUEST_TIMEOUT: float = 30.0
    # API Key Authentication (Exporting to other teams)
    # Set to False for development (no X-API-Key required)
    # Set to True with valid keys for production
    REQUIRE_API_KEY: bool = False
    VALID_API_KEYS: str = "agro_dev_key_123,agro_team_b_456,agro_ai_dev"

    model_config = {"env_prefix": "GATEWAY_"}


settings = GatewaySettings()

