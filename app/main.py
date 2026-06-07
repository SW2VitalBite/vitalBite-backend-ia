from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exceptions import (
    CoreServiceError,
    InsufficientDataError,
    InsufficientPatientsError,
    ModelNotLoadedError,
    core_service_handler,
    insufficient_data_handler,
    insufficient_patients_handler,
    model_not_loaded_handler,
)
from app.core.logging import configure_logging, get_logger

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    configure_logging("DEBUG" if settings.DEBUG else "INFO")
    logger.info(
        "iniciando microservicio de IA",
        extra={"version": settings.APP_VERSION, "prefix": settings.API_V1_PREFIX},
    )
    # Precarga de la CNN (no bloquea el arranque si TF/artefacto no están)
    try:
        from app.services.food_classification_service import preload_cnn_model

        preload_cnn_model()
    except Exception as exc:  # defensivo: nunca abortar el arranque por la CNN
        logger.warning("preload CNN omitido", extra={"motivo": str(exc)})

    yield
    # --- shutdown ---
    logger.info("deteniendo microservicio de IA")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Microservicio de IA para VitalBite. Inferencia de ML/DL: "
        "Random Forest (CU10), K-means (CU11) y OCR + CNN (CU9)."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Handlers de excepciones de dominio → códigos HTTP semánticos
app.add_exception_handler(InsufficientDataError, insufficient_data_handler)
app.add_exception_handler(InsufficientPatientsError, insufficient_patients_handler)
app.add_exception_handler(ModelNotLoadedError, model_not_loaded_handler)
app.add_exception_handler(CoreServiceError, core_service_handler)

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/", tags=["root"])
def root() -> dict[str, str]:
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }
