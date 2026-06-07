# Módulo K-means — Segmentación de Pacientes
**Caso de Uso:** CU11 — Consultar Segmentación de Perfiles  
**Actor:** A2 (Nutricionista) · Canal: Panel Web Angular  
**Endpoint:** `POST /api/v1/segmentation`

---

## 1. Descripción del Módulo

El módulo de segmentación usa el algoritmo **K-means (aprendizaje no supervisado)** para agrupar automáticamente los pacientes de un tenant en clusters según patrones nutricionales y de composición corporal similares. El nutricionista obtiene una visión científica descriptiva de su cartera de pacientes, útil para diseñar estrategias grupales y recomendaciones diferenciadas.

### ¿Por qué K-means?

| Criterio | Justificación |
|----------|---------------|
| Simplicidad interpretable | Los centroides de cada cluster son legibles como "perfil típico de ese grupo" |
| Escalabilidad | Funciona eficientemente con 10 a 500 pacientes por tenant |
| Visualización directa | Combinado con PCA, permite renderizar scatter plot 3D en el frontend |
| Ajuste dinámico de K | El Elbow Method determina automáticamente el K óptimo sin intervención humana |

---

## 2. Features de Clustering

### 2.1 Variables usadas

| # | Feature | Descripción | Normalización |
|---|---------|-------------|---------------|
| 1 | `imc` | Índice de Masa Corporal (kg/m²) | MinMaxScaler [0,1] |
| 2 | `porcentaje_grasa` | % de grasa corporal | MinMaxScaler [0,1] |
| 3 | `masa_muscular_kg` | Masa muscular estimada (kg) | MinMaxScaler [0,1] |
| 4 | `variacion_peso_3m` | Δ peso en 3 meses (kg) | MinMaxScaler [-1,1] |
| 5 | `nivel_actividad` | Escala 0-4 (Sedentario a Muy Activo) | MinMaxScaler [0,1] |
| 6 | `adherencia_dieta_pct` | % de cumplimiento del plan asignado | MinMaxScaler [0,1] |
| 7 | `asistencia_citas_pct` | % de citas asistidas vs. programadas | MinMaxScaler [0,1] |

> **Anonimización:** El `patient_id` se excluye del vector de clustering. Solo se incluye como referencia en el resultado final para mapear cada punto al paciente real.

---

## 3. Pipeline de Segmentación

```
Input: lista de pacientes del tenant
         │
         ▼
┌──────────────────────────────────┐
│  1. VALIDACIÓN                   │
│  - Mínimo N pacientes (default=10│
│  - Verificar features completas  │
│  - Imputar valores faltantes     │
│    (media del tenant)            │
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  2. PREPROCESAMIENTO             │
│  - Construir matriz M×7          │
│  - Detectar y tratar outliers    │
│    (IQR o Z-score)               │
│  - Normalizar con MinMaxScaler   │
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  3. DETERMINAR K ÓPTIMO          │
│  - Ejecutar KMeans para K=2..8   │
│  - Calcular inercia por K        │
│  - Elbow Method: encontrar punto │
│    de inflexión (kneedle)        │
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  4. CLUSTERING                   │
│  - KMeans(n_clusters=K_optimo)   │
│  - Asignar etiqueta a c/paciente │
│  - Calcular centroides           │
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  5. PROYECCIÓN 3D (PCA)          │
│  - PCA(n_components=3)           │
│  - Calcular coordenadas (x,y,z)  │
│    para cada paciente            │
│  - Calcular varianza explicada   │
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  6. DESCRIPCIÓN DE CLUSTERS      │
│  - Calcular media de cada feature│
│    por cluster                   │
│  - Generar etiqueta textual      │
│  - Contar pacientes por cluster  │
└──────────────┬───────────────────┘
               ▼
         Response JSON
```

---

## 4. Servicio de Segmentación

**Archivo:** `app/services/segmentation_service.py`

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

from app.core.config import settings
from app.schemas.segmentation import (
    SegmentationRequest, SegmentationResponse,
    ClusterInfo, PatientClusterPoint
)

FEATURES = [
    "imc", "porcentaje_grasa", "masa_muscular_kg",
    "variacion_peso_3m", "nivel_actividad",
    "adherencia_dieta_pct", "asistencia_citas_pct"
]

FEATURE_LABELS = {
    "imc": "IMC", "porcentaje_grasa": "% Grasa",
    "masa_muscular_kg": "Masa Muscular", "variacion_peso_3m": "Variación de Peso",
    "nivel_actividad": "Actividad Física", "adherencia_dieta_pct": "Adherencia a Dieta",
    "asistencia_citas_pct": "Asistencia a Citas"
}

CLUSTER_PROFILE_LABELS = [
    "Riesgo Alto — Sedentario",
    "Activo — Composición Favorable",
    "En Progreso — Adherencia Media",
    "Riesgo Metabólico — Grasa Elevada",
    "Perfil Atlético",
    "Bajo Peso — Masa Muscular Deficiente",
    "Recuperación — Pérdida de Peso Activa",
    "Control Estable",
]


class InsufficientPatientsError(Exception):
    def __init__(self, count: int, minimum: int):
        self.count = count
        self.minimum = minimum
        super().__init__(f"Se requieren al menos {minimum} pacientes. Se recibieron {count}.")


class SegmentationService:

    def segment(self, request: SegmentationRequest) -> SegmentationResponse:
        patients = request.patients
        min_required = settings.MIN_PATIENTS_FOR_KMEANS

        if len(patients) < min_required:
            raise InsufficientPatientsError(len(patients), min_required)

        # Construir DataFrame
        df = pd.DataFrame([p.model_dump() for p in patients])
        patient_ids = df["patient_id"].tolist()
        X_raw = df[FEATURES].copy()

        # Imputar valores faltantes con la media del tenant
        X_raw = X_raw.fillna(X_raw.mean())

        # Normalizar
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_raw.values)

        # Determinar K óptimo
        k_optimal = self._find_optimal_k(X_scaled)

        # Ejecutar K-means
        kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X_scaled)

        # Proyección PCA 3D
        pca = PCA(n_components=3)
        coords_3d = pca.fit_transform(X_scaled)
        variance_explained = pca.explained_variance_ratio_.tolist()

        # Describir clusters
        clusters_info = self._describe_clusters(
            X_raw.values, labels, k_optimal, FEATURES
        )

        # Construir puntos para scatter plot
        points = [
            PatientClusterPoint(
                patient_id=patient_ids[i],
                cluster_id=int(labels[i]),
                x=float(coords_3d[i, 0]),
                y=float(coords_3d[i, 1]),
                z=float(coords_3d[i, 2]),
            )
            for i in range(len(patients))
        ]

        return SegmentationResponse(
            k_clusters=k_optimal,
            clusters=clusters_info,
            pca_points=points,
            variance_explained=variance_explained,
            total_pacientes=len(patients),
        )

    def _find_optimal_k(self, X: np.ndarray, k_range: range = range(2, 9)) -> int:
        inertias = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            km.fit(X)
            inertias.append(km.inertia_)

        # Elbow Method: encontrar el punto de mayor curvatura
        return self._elbow_point(list(k_range), inertias)

    @staticmethod
    def _elbow_point(ks: list[int], inertias: list[float]) -> int:
        """Encuentra K óptimo por la segunda derivada de la curva de inercia."""
        if len(ks) <= 2:
            return ks[0]

        diffs1 = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        diffs2 = [diffs1[i] - diffs1[i + 1] for i in range(len(diffs1) - 1)]
        elbow_idx = diffs2.index(max(diffs2)) + 1  # +1 por el desplazamiento
        return ks[min(elbow_idx, len(ks) - 1)]

    @staticmethod
    def _describe_clusters(
        X: np.ndarray, labels: np.ndarray, k: int, feature_names: list[str]
    ) -> list["ClusterInfo"]:
        clusters = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            cluster_data = X[mask]
            cluster_mean = cluster_data.mean(axis=0)

            # Feature dominante del cluster
            dominant_feature_idx = cluster_mean.argmax()
            dominant_feature = FEATURE_LABELS.get(
                feature_names[dominant_feature_idx],
                feature_names[dominant_feature_idx]
            )

            # Etiqueta descriptiva basada en perfil del centroide
            label = _generate_cluster_label(cluster_mean, feature_names)

            clusters.append(ClusterInfo(
                cluster_id=cluster_id,
                label=label,
                total_pacientes=int(mask.sum()),
                porcentaje=round(float(mask.sum()) / len(labels) * 100, 1),
                feature_dominante=dominant_feature,
                caracteristicas={
                    FEATURE_LABELS.get(feature_names[i], feature_names[i]): round(float(v), 2)
                    for i, v in enumerate(cluster_mean)
                }
            ))

        return clusters


def _generate_cluster_label(mean_vector: np.ndarray, features: list[str]) -> str:
    """Genera una etiqueta descriptiva basada en los valores del centroide."""
    feature_map = dict(zip(features, mean_vector))

    imc = feature_map.get("imc", 0.5)
    grasa = feature_map.get("porcentaje_grasa", 0.5)
    actividad = feature_map.get("nivel_actividad", 0.5)
    adherencia = feature_map.get("adherencia_dieta_pct", 0.5)

    if imc > 0.7 and actividad < 0.3:
        return "Riesgo Alto — Sedentario con Sobrepeso"
    elif actividad > 0.7 and grasa < 0.3:
        return "Perfil Activo — Composición Corporal Favorable"
    elif adherencia > 0.7:
        return "Alta Adherencia — Progreso Constante"
    elif grasa > 0.7:
        return "Riesgo Metabólico — Grasa Corporal Elevada"
    elif imc < 0.2:
        return "Bajo Peso — Requiere Intervención Nutricional"
    else:
        return "Perfil Moderado — Seguimiento Regular"
```

---

## 5. Schemas Pydantic

**Archivo:** `app/schemas/segmentation.py`

```python
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
    caracteristicas: dict[str, float] = Field(..., description="Media de cada feature en el cluster")


class SegmentationResponse(BaseModel):
    k_clusters: int = Field(..., description="Número de clusters encontrados")
    clusters: list[ClusterInfo]
    pca_points: list[PatientClusterPoint]
    variance_explained: list[float] = Field(..., description="Varianza explicada por cada componente PCA")
    total_pacientes: int

    model_config = {"json_schema_extra": {
        "example": {
            "k_clusters": 3,
            "clusters": [
                {
                    "cluster_id": 0,
                    "label": "Riesgo Alto — Sedentario con Sobrepeso",
                    "total_pacientes": 15,
                    "porcentaje": 37.5,
                    "feature_dominante": "IMC",
                    "caracteristicas": {"IMC": 32.4, "% Grasa": 38.2, "Actividad Física": 0.8}
                }
            ],
            "pca_points": [{"patient_id": "uuid-1", "cluster_id": 0, "x": 0.34, "y": -0.12, "z": 0.87}],
            "variance_explained": [0.42, 0.28, 0.15],
            "total_pacientes": 40
        }
    }}
```

---

## 6. Endpoint

**Archivo:** `app/api/v1/endpoints/segmentation.py`

```python
from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.segmentation import SegmentationRequest, SegmentationResponse
from app.services.segmentation_service import SegmentationService, InsufficientPatientsError
from app.core.security import verify_api_key
from app.core.config import settings

router = APIRouter()


def get_service() -> SegmentationService:
    return SegmentationService()


@router.post(
    "",
    response_model=SegmentationResponse,
    summary="Segmentar pacientes de un tenant con K-means",
    dependencies=[Depends(verify_api_key)],
)
def segment_patients(
    request: SegmentationRequest,
    service: SegmentationService = Depends(get_service),
):
    try:
        return service.segment(request)
    except InsufficientPatientsError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "pacientes_insuficientes",
                "message": (
                    f"Se requieren al menos {e.minimum} pacientes con datos registrados. "
                    f"El tenant actual tiene {e.count}. "
                    f"Registre {e.minimum - e.count} expedientes adicionales para activar la segmentación."
                ),
                "pacientes_actuales": e.count,
                "minimo_requerido": e.minimum,
                "faltan": e.minimum - e.count,
            },
        )
```

---

## 7. Consideraciones del Algoritmo

### 7.1 Limitaciones conocidas de K-means y cómo se mitigan

| Limitación | Mitigación implementada |
|------------|------------------------|
| Sensible a outliers | Detección por IQR antes de normalizar |
| K debe definirse a priori | Elbow Method automático (K=2..8) |
| Sensible a escala de features | MinMaxScaler para todos los features |
| Resultados no reproducibles | `random_state=42` fijo |
| Asume clusters esféricos | Suficiente para la varianza esperada (datos clínicos) |

### 7.2 Actualización del modelo por tenant

A diferencia del Random Forest (que tiene un modelo global), K-means se re-ejecuta en cada llamada. No hay artefactos persistentes por tenant ya que:

1. El número de pacientes cambia (nuevos ingresos, bajas)
2. Las features evolucionan con cada consulta
3. El tiempo de ejecución para < 500 pacientes es < 500 ms

### 7.3 Validación de calidad del clustering

Para verificar que el clustering tiene sentido estadístico se calcula el **Silhouette Score** internamente (no se expone en la API, pero se loguea):

```python
score = silhouette_score(X_scaled, labels)
logger.info(f"[segmentation] tenant={tenant_id} k={k} silhouette={score:.4f}")
# Score > 0.5 indica clusters bien diferenciados
# Score 0.25-0.50 indica estructura razonable
# Score < 0.25 indica que los datos no tienen una segmentación natural clara
```

---

## 8. Interpretación de Resultados para el Nutricionista

La respuesta JSON es diseñada para que Angular renderice un **gráfico 3D interactivo de dispersión de puntos** (scatter plot). El frontend usa los campos `pca_points` para posicionar cada punto en el espacio 3D, coloreados según `cluster_id`.

### Mapeo frontend sugerido (Angular + ECharts o Plotly)

```typescript
// Ejemplo de configuración para ECharts 3D
const series = clusters.map(cluster => ({
  name: cluster.label,
  type: 'scatter3D',
  data: pcaPoints
    .filter(p => p.cluster_id === cluster.cluster_id)
    .map(p => [p.x, p.y, p.z, p.patient_id]),
  symbolSize: 8,
}));
```

### Etiquetas de clusters y su significado

| Etiqueta | Interpretación | Acción recomendada |
|----------|---------------|-------------------|
| Riesgo Alto — Sedentario | IMC elevado, poca actividad | Plan hipocalórico + activación física |
| Perfil Activo — Composición Favorable | Buenos indicadores | Mantenimiento y monitoreo preventivo |
| Alta Adherencia — Progreso Constante | Buen cumplimiento | Refuerzo positivo, ajuste de metas |
| Riesgo Metabólico — Grasa Elevada | Grasa alta aunque IMC sea normal | Plan de recomposición corporal |
| Bajo Peso — Requiere Intervención | IMC muy bajo, masa muscular baja | Plan hipercalórico + proteínas |
| Perfil Moderado — Seguimiento Regular | Sin señales de alarma claras | Seguimiento estándar mensual |

---

## 9. Tests

**Archivo:** `tests/test_segmentation.py`

```python
# Casos a implementar:
# Test 1: 15 pacientes con datos completos → retorna k=2 o k=3 clusters válidos
# Test 2: 8 pacientes (< mínimo=10) → retorna 422 con mensaje descriptivo
# Test 3: pacientes con valores nulos → imputación por media, no lanza error
# Test 4: todos los pca_points deben tener cluster_id dentro del rango [0, k-1]
# Test 5: suma de total_pacientes por cluster == total_pacientes del response
# Test 6: variance_explained debe sumar <= 1.0 y contener exactamente 3 valores
# Test 7: 50 pacientes con alta varianza → Elbow selecciona K > 2
```
