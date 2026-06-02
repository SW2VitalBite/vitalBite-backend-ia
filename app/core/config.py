from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    APP_NAME: str = "VitalBite Backend IA"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8081"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


settings = Settings()
