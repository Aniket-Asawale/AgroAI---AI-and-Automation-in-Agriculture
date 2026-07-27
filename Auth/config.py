"""
Auth Service Configuration — PostgreSQL connection and JWT settings.
"""

from pydantic_settings import BaseSettings


class AuthSettings(BaseSettings):
    """All settings can be overridden via environment variables prefixed with AUTH_."""

    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8002
    DEBUG: bool = True

    # PostgreSQL
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "agrodb"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "proShadow"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # JWT
    JWT_SECRET: str = "change-me-in-production-use-a-strong-random-key"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    model_config = {"env_prefix": "AUTH_"}


settings = AuthSettings()

