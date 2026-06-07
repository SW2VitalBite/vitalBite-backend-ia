"""Health check extendido — GET /api/v1/health.

Reporta el estado del servicio y la disponibilidad de cada artefacto de modelo
(Random Forest, scaler, CNN) sin cargar las librerías pesadas. Útil para los
health checks de orquestadores en la nube.
"""

from fastapi import APIRouter

from app.core.config import settings
from app.services.model_loader import models_status

router = APIRouter()


@router.get("")
def health_check() -> dict:
    modelos = models_status()
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "modelos": modelos,
        "modelos_cargados": all(
            v for k, v in modelos.items() if k != "food_classifier_cnn"
        ),
    }
