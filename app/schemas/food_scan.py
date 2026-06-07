"""Schemas de entrada/salida para CU9 — Scanner Nutricional (OCR + Deep Learning).

La imagen viaja como ``multipart/form-data`` (campo ``image``); ``patient_id`` y
``mode`` viajan como campos del formulario. Por eso no hay un modelo de request
del cuerpo JSON, solo el de respuesta.
"""

from pydantic import BaseModel, Field

from app.schemas.common import ScanMode


class NutrientInfo(BaseModel):
    calorias: float | None = None
    carbohidratos_g: float | None = None
    proteinas_g: float | None = None
    grasas_totales_g: float | None = None
    grasas_saturadas_g: float | None = None
    sodio_mg: float | None = None
    azucares_g: float | None = None
    fibra_g: float | None = None
    ingredientes: list[str] = Field(default_factory=list)


class FoodPrediction(BaseModel):
    clase: str
    probabilidad: float


class FoodScanResponse(BaseModel):
    modo: ScanMode
    semaforo: str = Field(..., description="SEGURO | PRECAUCION | RIESGO")
    advertencias: list[str] = Field(default_factory=list)
    nutrientes: NutrientInfo = Field(default_factory=NutrientInfo)
    predicciones_alimento: list[FoodPrediction] = Field(default_factory=list)
    confianza: float = Field(..., ge=0, le=1)
    requiere_retoma: bool = False
    mensaje_retoma: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "modo": "label",
                "semaforo": "RIESGO",
                "advertencias": [
                    "⚠️ Contiene alérgeno: maní",
                    "🧂 Alto contenido de sodio (>600 mg por porción)",
                ],
                "nutrientes": {
                    "calorias": 250,
                    "carbohidratos_g": 30,
                    "proteinas_g": 5,
                    "grasas_totales_g": 12,
                    "sodio_mg": 720,
                    "ingredientes": ["harina", "maní", "azúcar"],
                },
                "predicciones_alimento": [],
                "confianza": 0.87,
                "requiere_retoma": False,
                "mensaje_retoma": None,
            }
        }
    }
