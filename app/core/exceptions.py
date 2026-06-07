"""Jerarquía de excepciones de dominio y sus manejadores HTTP.

Estas excepciones separan los errores de negocio (datos insuficientes, pocos
pacientes, modelo no disponible) de los errores técnicos genéricos, y se mapean
a códigos HTTP semánticos mediante los handlers registrados en ``main.py``.
"""

from fastapi import Request
from fastapi.responses import JSONResponse


class InsufficientDataError(Exception):
    """El expediente del paciente no tiene suficientes features para predecir (CU10)."""

    def __init__(self, missing_fields: list[str]):
        self.missing_fields = missing_fields
        super().__init__(f"Datos insuficientes. Campos faltantes: {missing_fields}")


class InsufficientPatientsError(Exception):
    """No hay suficientes pacientes en el tenant para ejecutar K-means (CU11)."""

    def __init__(self, count: int, minimum: int):
        self.count = count
        self.minimum = minimum
        super().__init__(
            f"Se requieren al menos {minimum} pacientes. Se recibieron {count}."
        )


class ModelNotLoadedError(Exception):
    """Un artefacto ML (.pkl / .h5) no se encontró o no pudo cargarse."""

    def __init__(self, model_path: str, reason: str | None = None):
        self.model_path = model_path
        self.reason = reason
        msg = f"Modelo no disponible: {model_path}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


class CoreServiceError(Exception):
    """Falla al comunicarse con el Core NestJS (ej. obtener alergias del paciente)."""

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def insufficient_data_handler(
    request: Request, exc: InsufficientDataError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "datos_insuficientes",
            "message": "El expediente del paciente no tiene suficientes datos para la predicción.",
            "campos_faltantes": exc.missing_fields,
        },
    )


async def insufficient_patients_handler(
    request: Request, exc: InsufficientPatientsError
) -> JSONResponse:
    faltan = max(exc.minimum - exc.count, 0)
    return JSONResponse(
        status_code=422,
        content={
            "error": "pacientes_insuficientes",
            "message": (
                f"Se requieren al menos {exc.minimum} pacientes con datos registrados. "
                f"El tenant actual tiene {exc.count}. "
                f"Registre {faltan} expedientes adicionales para activar la segmentación."
            ),
            "pacientes_actuales": exc.count,
            "minimo_requerido": exc.minimum,
            "faltan": faltan,
        },
    )


async def model_not_loaded_handler(
    request: Request, exc: ModelNotLoadedError
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": "modelo_no_disponible",
            "message": str(exc),
            "model_path": exc.model_path,
        },
    )


async def core_service_handler(
    request: Request, exc: CoreServiceError
) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={"error": "core_service_error", "message": exc.detail},
    )
