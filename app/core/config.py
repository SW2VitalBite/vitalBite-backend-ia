from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración central del microservicio de IA.

    Carga las variables desde `.env` (ver `.env.example`). Las propiedades
    derivadas exponen valores ya parseados para uso directo en la aplicación.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # === App ===
    APP_NAME: str = "VitalBite Backend IA"
    APP_VERSION: str = "0.2.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    API_V1_PREFIX: str = "/api/v1"

    # === CORS ===
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8081,http://localhost:4200"

    # === Seguridad ===
    API_KEY: str = "vitalbite_ia_secret_key_dev_change_in_prod"

    # === Comunicación con Core NestJS ===
    CORE_SERVICE_URL: str = "http://localhost:3000"
    CORE_GRAPHQL_URL: str = "http://localhost:3000/graphql"
    INTERNAL_TOKEN: str = "internal_service_token"

    # === Machine Learning ===
    MODELS_DIR: str = "app/models/artifacts"
    MIN_PATIENTS_FOR_KMEANS: int = 10
    RF_RISK_THRESHOLD_HIGH: float = 0.70
    RF_RISK_THRESHOLD_MEDIUM: float = 0.40

    # === OCR / Deep Learning ===
    OCR_MIN_CONFIDENCE: float = 0.60
    IMAGE_MAX_SIZE_MB: int = 5
    DL_MODEL_PATH: str = "app/models/artifacts/food_classifier.h5"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def models_dir_path(self) -> Path:
        return Path(self.MODELS_DIR)

    @property
    def IMAGE_MAX_SIZE_BYTES(self) -> int:
        return self.IMAGE_MAX_SIZE_MB * 1024 * 1024


settings = Settings()
