# Módulo Random Forest — Predicción de Riesgo Nutricional
**Caso de Uso:** CU10 — Evaluar Predicción de Riesgo de Salud  
**Actor:** A2 (Nutricionista) · Canal: Panel Web Angular  
**Endpoint:** `POST /api/v1/risk-prediction`

---

## 1. Descripción del Módulo

El módulo de predicción usa un clasificador **Random Forest (Bosque Aleatorio)** para estimar la probabilidad de que un paciente desarrolle complicaciones o descompensaciones nutricionales a futuro. El resultado orienta al nutricionista con un nivel de riesgo (`Bajo`, `Medio`, `Alto`) y los factores clínicos que más influyen en dicho nivel.

### ¿Por qué Random Forest?

| Criterio | Ventaja en este contexto |
|----------|--------------------------|
| Robustez a outliers | Los datos clínicos tienen valores atípicos (ej. pesos extremos) |
| Manejo de features mixtas | Combina variables numéricas y categóricas sin normalización obligatoria |
| `feature_importances_` | Permite explicar qué factores determinan el riesgo (exigencia clínica) |
| No requiere grandes datasets | Funciona bien con 500-5000 registros de pacientes |
| Resistencia a overfitting | El bagging y la aleatoriedad de árboles reduce la varianza |

---

## 2. Features del Modelo

### 2.1 Vector de Entrada (10 features)

| # | Feature | Tipo | Rango/Valores | Fuente en CU3 |
|---|---------|------|---------------|---------------|
| 1 | `edad` | Numérica continua | 5 – 100 años | Expediente paciente |
| 2 | `sexo` | Categórica binaria | 0 = Femenino, 1 = Masculino | Expediente paciente |
| 3 | `peso_kg` | Numérica continua | 20 – 300 kg | Medida corporal |
| 4 | `talla_m` | Numérica continua | 0.50 – 2.50 m | Medida corporal |
| 5 | `imc` | Numérica continua | 10 – 60 | Calculado: peso/talla² |
| 6 | `variacion_peso_3m_kg` | Numérica continua | -30 – +30 | Δ entre última y 3ª medida |
| 7 | `porcentaje_grasa` | Numérica continua | 3 – 60 % | Medida corporal |
| 8 | `nivel_actividad` | Ordinal | 0=Sedentario, 1=Leve, 2=Moderado, 3=Activo, 4=Muy activo | Registro hábitos |
| 9 | `calidad_dieta_score` | Numérica continua | 0 – 10 | Auto-reporte paciente |
| 10 | `num_comorbilidades` | Numérica discreta | 0 – 10 | Expediente clínico |

### 2.2 Variable Objetivo (Target)

| Clase | Valor | Criterio clínico |
|-------|-------|------------------|
| `Bajo` | 0 | IMC normal + sin comorbilidades + dieta equilibrada |
| `Medio` | 1 | IMC limítrofe O alguna comorbilidad O dieta deficiente |
| `Alto` | 2 | IMC fuera de rango + comorbilidades + patrones de riesgo |

---

## 3. Generación de Datos Sintéticos para Entrenamiento

Para el entorno académico sin datos clínicos reales, se generan datos sintéticos con reglas epidemiológicas conocidas.

**Archivo:** `app/models/training/generate_synthetic_data.py`

```python
import numpy as np
import pandas as pd

def generate_nutritional_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    edad = rng.integers(15, 80, n_samples).astype(float)
    sexo = rng.integers(0, 2, n_samples).astype(float)
    peso_kg = rng.normal(75, 18, n_samples).clip(40, 200)
    talla_m = rng.normal(1.68, 0.10, n_samples).clip(1.45, 2.05)
    imc = peso_kg / (talla_m ** 2)
    variacion_peso = rng.normal(0, 3, n_samples)
    porcentaje_grasa = rng.normal(28, 10, n_samples).clip(5, 55)
    nivel_actividad = rng.integers(0, 5, n_samples).astype(float)
    calidad_dieta = rng.uniform(0, 10, n_samples)
    num_comorbilidades = rng.integers(0, 6, n_samples).astype(float)

    # Regla de etiquetado clínico
    risk_score = (
        (imc > 30).astype(float) * 2 +
        (imc < 18.5).astype(float) * 1.5 +
        (variacion_peso > 5).astype(float) * 1.2 +
        (porcentaje_grasa > 35).astype(float) * 1.0 +
        (nivel_actividad < 1).astype(float) * 0.8 +
        (calidad_dieta < 4).astype(float) * 0.7 +
        num_comorbilidades * 0.5
    )

    riesgo = np.where(risk_score >= 4, 2, np.where(risk_score >= 2, 1, 0))

    return pd.DataFrame({
        "edad": edad, "sexo": sexo, "peso_kg": peso_kg,
        "talla_m": talla_m, "imc": imc, "variacion_peso_3m_kg": variacion_peso,
        "porcentaje_grasa": porcentaje_grasa, "nivel_actividad": nivel_actividad,
        "calidad_dieta_score": calidad_dieta, "num_comorbilidades": num_comorbilidades,
        "riesgo": riesgo
    })
```

---

## 4. Script de Entrenamiento

**Archivo:** `app/models/training/train_random_forest.py`

```python
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from generate_synthetic_data import generate_nutritional_dataset

ARTIFACTS_DIR = Path("app/models/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "edad", "sexo", "peso_kg", "talla_m", "imc",
    "variacion_peso_3m_kg", "porcentaje_grasa", "nivel_actividad",
    "calidad_dieta_score", "num_comorbilidades"
]
TARGET = "riesgo"
RISK_LABELS = {0: "Bajo", 1: "Medio", 2: "Alto"}


def train():
    print("Generando datos sintéticos...")
    df = generate_nutritional_dataset(n_samples=3000)

    X = df[FEATURES].values
    y = df[TARGET].values

    # Normalización (guardamos el scaler para usarlo en inferencia)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Entrenando Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,        # 200 árboles
        max_depth=12,            # Profundidad máxima controlada
        min_samples_split=10,    # Mínimo de muestras para dividir un nodo
        min_samples_leaf=5,      # Mínimo de muestras en hoja
        class_weight="balanced", # Compensa desbalance entre clases
        random_state=42,
        n_jobs=-1                # Usar todos los núcleos disponibles
    )
    model.fit(X_train, y_train)

    # Evaluación
    y_pred = model.predict(X_test)
    print("\n=== REPORTE DE CLASIFICACIÓN ===")
    print(classification_report(y_test, y_pred, target_names=list(RISK_LABELS.values())))

    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="f1_macro")
    print(f"F1 Macro (CV 5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Importancia de features
    importances = dict(zip(FEATURES, model.feature_importances_))
    print("\n=== IMPORTANCIA DE FEATURES ===")
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {imp:.4f}")

    # Guardar artefactos
    joblib.dump(model, ARTIFACTS_DIR / "risk_rf_model.pkl")
    joblib.dump(scaler, ARTIFACTS_DIR / "risk_scaler.pkl")
    joblib.dump(importances, ARTIFACTS_DIR / "risk_feature_importances.pkl")
    print(f"\nModelo guardado en {ARTIFACTS_DIR}")


if __name__ == "__main__":
    train()
```

---

## 5. Servicio de Inferencia

**Archivo:** `app/services/risk_prediction_service.py`

```python
import joblib
import numpy as np
from pathlib import Path
from functools import lru_cache

from app.core.config import settings
from app.schemas.risk_prediction import RiskPredictionRequest, RiskPredictionResponse

FEATURES_ORDER = [
    "edad", "sexo", "peso_kg", "talla_m", "imc",
    "variacion_peso_3m_kg", "porcentaje_grasa", "nivel_actividad",
    "calidad_dieta_score", "num_comorbilidades"
]

RISK_LABELS = {0: "Bajo", 1: "Medio", 2: "Alto"}

FEATURE_DESCRIPTIONS = {
    "imc": "Índice de Masa Corporal elevado",
    "porcentaje_grasa": "Porcentaje de grasa corporal fuera de rango",
    "variacion_peso_3m_kg": "Variación de peso significativa en los últimos 3 meses",
    "num_comorbilidades": "Presencia de comorbilidades registradas",
    "calidad_dieta_score": "Calidad de la dieta por debajo del promedio",
    "nivel_actividad": "Nivel de actividad física insuficiente",
}


@lru_cache(maxsize=1)
def _load_model():
    path = Path(settings.MODELS_DIR) / "risk_rf_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_scaler():
    path = Path(settings.MODELS_DIR) / "risk_scaler.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Scaler no encontrado: {path}")
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_importances() -> dict:
    path = Path(settings.MODELS_DIR) / "risk_feature_importances.pkl"
    return joblib.load(path) if path.exists() else {}


def _identify_critical_factors(
    features: dict, importances: dict, predicted_class: int
) -> list[str]:
    """Retorna las 3 features más importantes con descripción legible."""
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    factors = []

    for feat_name, _ in top_features:
        if feat_name in FEATURE_DESCRIPTIONS and feat_name in features:
            factors.append(FEATURE_DESCRIPTIONS[feat_name])
        if len(factors) == 3:
            break

    return factors


class RiskPredictionService:
    MIN_REQUIRED_FEATURES = 7  # Mínimo de features no nulas para habilitar predicción

    def predict(self, request: RiskPredictionRequest) -> RiskPredictionResponse:
        features_dict = request.features.model_dump()

        # Validar datos mínimos
        non_null = sum(1 for v in features_dict.values() if v is not None)
        if non_null < self.MIN_REQUIRED_FEATURES:
            missing = [k for k, v in features_dict.items() if v is None]
            raise InsufficientDataError(missing_fields=missing)

        # Calcular IMC si no viene calculado
        if features_dict.get("imc") is None and features_dict.get("peso_kg") and features_dict.get("talla_m"):
            features_dict["imc"] = features_dict["peso_kg"] / (features_dict["talla_m"] ** 2)

        # Construir vector respetando orden del entrenamiento
        vector = np.array(
            [features_dict.get(f, 0.0) or 0.0 for f in FEATURES_ORDER],
            dtype=np.float64
        ).reshape(1, -1)

        model = _load_model()
        scaler = _load_scaler()
        importances = _load_importances()

        vector_scaled = scaler.transform(vector)
        probabilities = model.predict_proba(vector_scaled)[0]
        predicted_class = int(np.argmax(probabilities))

        factores = _identify_critical_factors(features_dict, importances, predicted_class)

        return RiskPredictionResponse(
            nivel_riesgo=RISK_LABELS[predicted_class],
            probabilidad=float(probabilities[predicted_class]),
            probabilidades={
                "Bajo": float(probabilities[0]),
                "Medio": float(probabilities[1]),
                "Alto": float(probabilities[2]),
            },
            factores_criticos=factores,
            recomendacion=_generate_recommendation(predicted_class, factores),
        )


def _generate_recommendation(level: int, factors: list[str]) -> str:
    base = {
        0: "Mantener hábitos actuales y continuar monitoreo preventivo trimestral.",
        1: "Se recomienda ajuste de plan dietético y seguimiento mensual de métricas.",
        2: "Atención prioritaria requerida. Revisar plan de alimentación y considerar interconsulta médica.",
    }
    return base[level]


class InsufficientDataError(Exception):
    def __init__(self, missing_fields: list[str]):
        self.missing_fields = missing_fields
        super().__init__(f"Datos insuficientes. Campos faltantes: {missing_fields}")
```

---

## 6. Schemas Pydantic

**Archivo:** `app/schemas/risk_prediction.py`

```python
from pydantic import BaseModel, Field, model_validator

class PatientFeatures(BaseModel):
    edad: float | None = Field(None, ge=5, le=120, description="Edad en años")
    sexo: int | None = Field(None, ge=0, le=1, description="0=Femenino, 1=Masculino")
    peso_kg: float | None = Field(None, ge=20, le=300)
    talla_m: float | None = Field(None, ge=0.5, le=2.5)
    imc: float | None = Field(None, ge=10, le=70, description="Se calcula automáticamente si no se provee")
    variacion_peso_3m_kg: float | None = Field(None, ge=-50, le=50)
    porcentaje_grasa: float | None = Field(None, ge=3, le=70)
    nivel_actividad: int | None = Field(None, ge=0, le=4)
    calidad_dieta_score: float | None = Field(None, ge=0, le=10)
    num_comorbilidades: int | None = Field(None, ge=0, le=20)


class RiskPredictionRequest(BaseModel):
    patient_id: str = Field(..., description="UUID del paciente")
    tenant_id: str = Field(..., description="ID del tenant del nutricionista")
    features: PatientFeatures


class RiskPredictionResponse(BaseModel):
    nivel_riesgo: str = Field(..., description="Bajo | Medio | Alto")
    probabilidad: float = Field(..., ge=0, le=1, description="Probabilidad de la clase predicha")
    probabilidades: dict[str, float] = Field(..., description="Probabilidades de cada clase")
    factores_criticos: list[str] = Field(..., description="Top 3 factores que determinan el riesgo")
    recomendacion: str = Field(..., description="Recomendación clínica generada automáticamente")

    model_config = {"json_schema_extra": {
        "example": {
            "nivel_riesgo": "Alto",
            "probabilidad": 0.83,
            "probabilidades": {"Bajo": 0.05, "Medio": 0.12, "Alto": 0.83},
            "factores_criticos": [
                "Índice de Masa Corporal elevado",
                "Presencia de comorbilidades registradas",
                "Calidad de la dieta por debajo del promedio"
            ],
            "recomendacion": "Atención prioritaria requerida..."
        }
    }}
```

---

## 7. Endpoint

**Archivo:** `app/api/v1/endpoints/risk_prediction.py`

```python
from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.risk_prediction import RiskPredictionRequest, RiskPredictionResponse
from app.services.risk_prediction_service import RiskPredictionService, InsufficientDataError
from app.core.security import verify_api_key

router = APIRouter()


def get_service() -> RiskPredictionService:
    return RiskPredictionService()


@router.post(
    "",
    response_model=RiskPredictionResponse,
    summary="Predecir riesgo nutricional con Random Forest",
    dependencies=[Depends(verify_api_key)],
)
def predict_risk(
    request: RiskPredictionRequest,
    service: RiskPredictionService = Depends(get_service),
):
    try:
        return service.predict(request)
    except InsufficientDataError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "datos_insuficientes",
                "message": "El expediente del paciente no tiene suficientes datos para la predicción.",
                "campos_faltantes": e.missing_fields,
            },
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "modelo_no_disponible", "message": str(e)},
        )
```

---

## 8. Hiperparámetros del Modelo

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `n_estimators` | 200 | Balance entre rendimiento y tiempo de inferencia (<100ms) |
| `max_depth` | 12 | Evita sobreajuste en dataset pequeño |
| `min_samples_split` | 10 | Regularización; evita nodos con pocos ejemplos |
| `min_samples_leaf` | 5 | Suaviza las predicciones en bordes de decisión |
| `class_weight` | `"balanced"` | Compensa si una clase de riesgo tiene pocos ejemplos |
| `random_state` | 42 | Reproducibilidad académica |
| `n_jobs` | -1 | Paralelismo total (todos los núcleos CPU) |

---

## 9. Métricas de Evaluación Esperadas

Con datos sintéticos de 3000 muestras y 80/20 split:

| Métrica | Valor Esperado |
|---------|---------------|
| Accuracy | ~78 – 85% |
| F1 Macro | ~0.76 – 0.83 |
| F1 Clase "Alto" | ~0.80 – 0.88 |
| F1 Clase "Medio" | ~0.70 – 0.78 |
| Tiempo inferencia (1 paciente) | < 20 ms |

---

## 10. Tests

**Archivo:** `tests/test_risk_prediction.py`

```python
# Casos a implementar:
# Test 1: paciente con IMC=35, dieta=2, comorbilidades=3 → debe retornar "Alto"
# Test 2: paciente joven, IMC normal, activo → debe retornar "Bajo"
# Test 3: <7 features en request → debe retornar 422 con lista de campos faltantes
# Test 4: modelo .pkl no existe → debe retornar 503
# Test 5: verificar que factores_criticos siempre retorna lista con al menos 1 elemento
# Test 6: probabilidades siempre suman 1.0 (con tolerancia 0.001)
```
