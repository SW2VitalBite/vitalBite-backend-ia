"""Schemas de entrada/salida para CU11 — Segmentación de Pacientes (K-means)."""

from pydantic import BaseModel, Field


class PatientFeatureInput(BaseModel):
    patient_id: str
    imc: float | None = Field(None, ge=10, le=70)
    porcentaje_grasa: float | None = Field(None, ge=3, le=70)
    masa_muscular_kg: float | None = Field(None, ge=5, le=100)
    variacion_peso_3m: float | None = Field(None, ge=-50, le=50)
    nivel_actividad: float | None = Field(None, ge=0, le=4)
    adherencia_dieta_pct: float | None = Field(None, ge=0, le=100)
    asistencia_citas_pct: float | None = Field(None, ge=0, le=100)


class SegmentationRequest(BaseModel):
    tenant_id: str = Field(..., description="ID del tenant del nutricionista")
    patients: list[PatientFeatureInput] = Field(..., min_length=1)


class PatientClusterPoint(BaseModel):
    patient_id: str
    cluster_id: int
    x: float  # Componente principal 1 (PCA)
    y: float  # Componente principal 2 (PCA)
    z: float  # Componente principal 3 (PCA)


class ClusterInfo(BaseModel):
    cluster_id: int
    label: str = Field(..., description="Etiqueta descriptiva del grupo")
    total_pacientes: int
    porcentaje: float = Field(..., description="Porcentaje del total de pacientes")
    feature_dominante: str = Field(..., description="Variable más representativa del cluster")
    caracteristicas: dict[str, float] = Field(
        ..., description="Media de cada feature en el cluster"
    )


class SegmentationResponse(BaseModel):
    k_clusters: int = Field(..., description="Número de clusters encontrados")
    clusters: list[ClusterInfo]
    pca_points: list[PatientClusterPoint]
    variance_explained: list[float] = Field(
        ..., description="Varianza explicada por cada componente PCA"
    )
    silhouette_score: float | None = Field(
        None, description="Calidad del clustering (-1 a 1); >0.5 = clusters bien diferenciados"
    )
    total_pacientes: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "k_clusters": 3,
                "clusters": [
                    {
                        "cluster_id": 0,
                        "label": "Riesgo Alto — Sedentario con Sobrepeso",
                        "total_pacientes": 15,
                        "porcentaje": 37.5,
                        "feature_dominante": "IMC",
                        "caracteristicas": {
                            "IMC": 32.4,
                            "% Grasa": 38.2,
                            "Actividad Física": 0.8,
                        },
                    }
                ],
                "pca_points": [
                    {
                        "patient_id": "uuid-1",
                        "cluster_id": 0,
                        "x": 0.34,
                        "y": -0.12,
                        "z": 0.87,
                    }
                ],
                "variance_explained": [0.42, 0.28, 0.15],
                "silhouette_score": 0.41,
                "total_pacientes": 40,
            }
        }
    }
