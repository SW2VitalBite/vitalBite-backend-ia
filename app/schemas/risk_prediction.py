"""Schemas de entrada/salida para CU10 — Predicción de Riesgo (Random Forest)."""

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    edad: float | None = Field(None, ge=5, le=120, description="Edad en años")
    sexo: int | None = Field(None, ge=0, le=1, description="0=Femenino, 1=Masculino")
    peso_kg: float | None = Field(None, ge=20, le=300)
    talla_m: float | None = Field(None, ge=0.5, le=2.5)
    imc: float | None = Field(
        None, ge=10, le=70, description="Se calcula automáticamente si no se provee"
    )
    variacion_peso_3m_kg: float | None = Field(None, ge=-50, le=50)
    porcentaje_grasa: float | None = Field(None, ge=3, le=70)
    nivel_actividad: int | None = Field(
        None, ge=0, le=4, description="0=Sedentario ... 4=Muy activo"
    )
    calidad_dieta_score: float | None = Field(None, ge=0, le=10)
    num_comorbilidades: int | None = Field(None, ge=0, le=20)


class RiskPredictionRequest(BaseModel):
    patient_id: str = Field(..., description="UUID del paciente")
    tenant_id: str = Field(..., description="ID del tenant del nutricionista")
    features: PatientFeatures

    model_config = {
        "json_schema_extra": {
            "example": {
                "patient_id": "11111111-1111-1111-1111-111111111111",
                "tenant_id": "tenant-demo",
                "features": {
                    "edad": 58,
                    "sexo": 1,
                    "peso_kg": 98,
                    "talla_m": 1.70,
                    "imc": 33.9,
                    "variacion_peso_3m_kg": 6.5,
                    "porcentaje_grasa": 38,
                    "nivel_actividad": 0,
                    "calidad_dieta_score": 2.5,
                    "num_comorbilidades": 3,
                },
            }
        }
    }


class RiskPredictionResponse(BaseModel):
    nivel_riesgo: str = Field(..., description="Bajo | Medio | Alto")
    probabilidad: float = Field(
        ..., ge=0, le=1, description="Probabilidad de la clase predicha"
    )
    probabilidades: dict[str, float] = Field(
        ..., description="Probabilidades de cada clase"
    )
    factores_criticos: list[str] = Field(
        ..., description="Top factores que determinan el riesgo"
    )
    recomendacion: str = Field(
        ..., description="Recomendación clínica generada automáticamente"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "nivel_riesgo": "Alto",
                "probabilidad": 0.83,
                "probabilidades": {"Bajo": 0.05, "Medio": 0.12, "Alto": 0.83},
                "factores_criticos": [
                    "Índice de Masa Corporal elevado",
                    "Presencia de comorbilidades registradas",
                    "Calidad de la dieta por debajo del promedio",
                ],
                "recomendacion": (
                    "Atención prioritaria requerida. Revisar plan de alimentación "
                    "y considerar interconsulta médica."
                ),
            }
        }
    }
