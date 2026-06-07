# Arquitectura Técnica — Microservicio de IA (VitalBite)

## 1. Visión General

El microservicio de IA es un componente **stateless** especializado en inferencia. No posee base de datos propia; recibe payloads del Core NestJS o de la app móvil, ejecuta modelos de ML/DL en memoria y retorna resultados estructurados en JSON. Sigue una arquitectura en capas (Layered Architecture) inspirada en Clean Architecture.

```
┌──────────────────────────────────────────────────────┐
│              vitalBite-backend-ia                     │
│                                                      │
│  ┌─────────────┐   ┌───────────────┐                 │
│  │  API Layer  │   │  Schema Layer │                 │
│  │  (routes)   │──▶│  (Pydantic)   │                 │
│  └──────┬──────┘   └───────────────┘                 │
│         │                                            │
│  ┌──────▼──────────────────────────────────────┐     │
│  │              Service Layer                  │     │
│  │  ┌──────────────┐  ┌──────────────────────┐ │     │
│  │  │ OCR Service  │  │ Food Classifier Svc  │ │     │
│  │  │ (EasyOCR)    │  │ (CNN MobileNetV2)    │ │     │
│  │  └──────────────┘  └──────────────────────┘ │     │
│  │  ┌──────────────┐  ┌──────────────────────┐ │     │
│  │  │  RF Service  │  │  K-means Service     │ │     │
│  │  │ (Random      │  │  (Segmentation)      │ │     │
│  │  │  Forest)     │  │                      │ │     │
│  │  └──────────────┘  └──────────────────────┘ │     │
│  └──────────────────────┬──────────────────────┘     │
│                         │                            │
│  ┌──────────────────────▼──────────────────────┐     │
│  │              Model Layer                    │     │
│  │  artifacts/ (.pkl, .h5, .pt)                │     │
│  └─────────────────────────────────────────────┘     │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 2. Capas de la Arquitectura

### 2.1 API Layer — `app/api/`

Contiene únicamente los **route handlers** de FastAPI. Cada endpoint:
- Valida el request con Pydantic (Schema Layer)
- Delega toda la lógica al Service Layer correspondiente
- Formatea y retorna la respuesta Pydantic

**Regla:** Cero lógica de negocio en esta capa. Si el handler tiene más de 10 líneas, la lógica debe moverse al servicio.

```python
# Ejemplo de handler correcto (thin controller)
@router.post("/risk-prediction", response_model=RiskPredictionResponse)
async def predict_risk(
    payload: RiskPredictionRequest,
    service: RiskPredictionService = Depends(get_risk_service),
):
    return await service.predict(payload)
```

### 2.2 Schema Layer — `app/schemas/`

Modelos Pydantic v2 que definen contratos de entrada/salida. Toda validación de formato y tipos ocurre aquí.

| Archivo | Propósito |
|---------|-----------|
| `risk_prediction.py` | Request/Response para CU10 (Random Forest) |
| `segmentation.py` | Request/Response para CU11 (K-means) |
| `food_scan.py` | Request/Response para CU9 (OCR + DL) |
| `common.py` | Tipos compartidos: `SemaforoRiesgo`, `NivelRiesgo`, `ConfidenceScore` |

### 2.3 Service Layer — `app/services/`

Contiene toda la lógica de inferencia, preprocesamiento y post-procesamiento. Esta capa **no depende** de FastAPI.

| Archivo | Responsabilidad |
|---------|----------------|
| `risk_prediction_service.py` | Cargar modelo RF, normalizar features, ejecutar `predict_proba`, mapear a niveles de riesgo |
| `segmentation_service.py` | Preprocesar pacientes, ejecutar K-means, proyectar PCA 3D, describir clusters |
| `ocr_service.py` | Inicializar EasyOCR, preprocesar imagen, extraer texto, parsear nutrientes |
| `food_classification_service.py` | Cargar CNN, preprocesar imagen, ejecutar inferencia, mapear a categorías |
| `nutrition_analyzer_service.py` | Cruzar resultado OCR/CNN con alergias del paciente y construir semáforo |
| `model_loader.py` | Singleton para carga perezosa de artefactos ML en memoria |

### 2.4 Model Layer — `app/models/`

Contiene los artefactos serializados de modelos entrenados y los scripts de entrenamiento.

```
app/models/
├── artifacts/                   # Modelos serializados (excluidos de git con .gitignore)
│   ├── risk_rf_model.pkl        # Random Forest entrenado
│   ├── risk_scaler.pkl          # StandardScaler para features de RF
│   ├── kmeans_{tenant_id}.pkl   # K-means por tenant (generado en runtime)
│   ├── pca_transformer.pkl      # PCA para proyección 3D
│   └── food_classifier.h5       # CNN MobileNetV2 fine-tuned (>50 MB)
└── training/                    # Scripts de entrenamiento (solo desarrollo)
    ├── train_random_forest.py
    ├── train_food_classifier.py
    └── generate_synthetic_data.py
```

---

## 3. Flujo de Datos por Módulo

### 3.1 Flujo CU9 — Scanner Nutricional (OCR + DL)

```
App Móvil (React Native)
    │
    │  POST /api/v1/food-scan
    │  Content-Type: multipart/form-data
    │  { image: <archivo>, mode: "label"|"plate", patient_id: UUID }
    ▼
FastAPI — food_scan.py (router)
    │
    │  1. Validar tamaño imagen (< 5 MB)
    │  2. Detectar modo de operación
    ▼
    ├──[modo = "label"]──→ ocr_service.py
    │                          │
    │                          │  1. Preprocesar: escala de grises, umbralización
    │                          │  2. EasyOCR: extraer texto crudo
    │                          │  3. Parsear: regex para nutrientes (cal, carb, prot, grasas)
    │                          │  4. Calcular confianza promedio
    │                          ▼
    │                      { nutrientes_raw, confianza }
    │
    └──[modo = "plate"]──→ food_classification_service.py
                               │
                               │  1. Resize imagen a 224x224
                               │  2. Normalizar pixeles [0,1]
                               │  3. Inferencia CNN → vector de probabilidades
                               │  4. Top-3 categorías de alimento
                               ▼
                           { categorias, probabilidades }
    │
    ▼
nutrition_analyzer_service.py
    │
    │  1. Obtener alergias del paciente (llamada interna al Core NestJS)
    │  2. Cruzar ingredientes/categorías con lista de alergias
    │  3. Evaluar semáforo: SEGURO / PRECAUCIÓN / RIESGO
    │  4. Generar lista de advertencias personalizadas
    ▼
Response JSON → App Móvil
```

### 3.2 Flujo CU10 — Predicción de Riesgo (Random Forest)

```
Core NestJS (Angular request)
    │
    │  POST /api/v1/risk-prediction
    │  { patient_features: { edad, peso, talla, imc, ... } }
    ▼
FastAPI — risk_prediction.py (router)
    │
    ▼
risk_prediction_service.py
    │
    │  1. Validar features completas (mín. 7 de 10)
    │  2. Construir vector numpy [10 features]
    │  3. Normalizar con StandardScaler cargado
    │  4. Ejecutar rf_model.predict_proba([vector])
    │  5. Obtener probabilidades: [P_bajo, P_medio, P_alto]
    │  6. Identificar features con mayor importancia (feature_importances_)
    │  7. Mapear a nivel textual: "Bajo" / "Medio" / "Alto"
    ▼
Response JSON → Core NestJS → PostgreSQL (almacenado) → Angular
```

### 3.3 Flujo CU11 — Segmentación K-means

```
Core NestJS
    │
    │  POST /api/v1/segmentation
    │  { tenant_id: UUID, patients: [{ patient_id, features... }] }
    ▼
FastAPI — segmentation.py (router)
    │
    ▼
segmentation_service.py
    │
    │  1. Validar mínimo N pacientes (configurable, default=10)
    │  2. Construir matriz de features (m pacientes × n features)
    │  3. Normalizar con MinMaxScaler
    │  4. Determinar K óptimo con Elbow Method (K=2..8)
    │  5. Ejecutar KMeans(n_clusters=K_optimo)
    │  6. Proyectar a 3D con PCA(n_components=3)
    │  7. Describir cada cluster: feature dominante, tamaño, etiqueta
    ▼
Response JSON → Core NestJS → Angular (gráfico 3D scatter)
```

---

## 4. Diseño de Endpoints REST

> **Nota de arquitectura:** El protocolo REST en este microservicio es válido per las reglas de proyecto (AGENTS.md), ya que FastAPI es un servicio interno de inferencia. GraphQL solo es obligatorio para la comunicación cliente ↔ Core NestJS.

### Resumen de Endpoints

| Método | Ruta | Descripción | Autenticación |
|--------|------|-------------|---------------|
| GET | `/` | Info del servicio | Pública |
| GET | `/api/v1/health` | Estado de salud + modelos cargados | Pública |
| POST | `/api/v1/food-scan` | Analizar imagen (OCR o clasificación) | API Key |
| POST | `/api/v1/risk-prediction` | Predecir riesgo nutricional | API Key |
| POST | `/api/v1/segmentation` | Segmentar pacientes por tenant | API Key |

### Esquema de autenticación

Todos los endpoints protegidos requieren la cabecera:
```
X-API-Key: <valor configurado en .env>
```

La validación se implementa como una dependencia FastAPI reutilizable:

```python
# app/core/security.py
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
```

---

## 5. Gestión de Modelos en Memoria

Para evitar cargar los modelos en cada request (costo alto de I/O), se usa un **singleton de carga perezosa** (`lazy loading`):

```python
# app/services/model_loader.py
import joblib
from functools import lru_cache

@lru_cache(maxsize=None)
def get_rf_model():
    return joblib.load(settings.MODELS_DIR / "risk_rf_model.pkl")

@lru_cache(maxsize=None)
def get_rf_scaler():
    return joblib.load(settings.MODELS_DIR / "risk_scaler.pkl")
```

El modelo CNN de TensorFlow/Keras se carga al inicio del servidor en el evento `startup` para evitar el cold start en el primer request:

```python
# app/main.py
@app.on_event("startup")
async def load_models():
    from app.services.food_classification_service import preload_cnn_model
    await preload_cnn_model()
```

---

## 6. Manejo de Errores

Se define una jerarquía de excepciones de dominio para diferenciar errores de negocio de errores técnicos:

| Excepción | HTTP Status | Caso de uso |
|-----------|-------------|-------------|
| `InsufficientDataError` | 422 | Paciente sin suficientes medidas para RF |
| `InsufficientPatientsError` | 422 | Menos pacientes que umbral para K-means |
| `LowConfidenceError` | 200 con `confianza < umbral` | Imagen borrosa en OCR/DL |
| `ModelNotLoadedError` | 503 | Artefacto `.pkl`/`.h5` no encontrado |
| `CoreServiceError` | 502 | Falla al obtener alergias del Core NestJS |

---

## 7. Decisiones de Arquitectura

| Decisión | Alternativa Considerada | Justificación |
|----------|------------------------|---------------|
| EasyOCR para etiquetas | Tesseract | EasyOCR tiene mejor soporte multilenguaje, API Python más simple y mejor rendimiento en imágenes con texto curvo o inclinado |
| MobileNetV2 (TF/Keras) como base CNN | ResNet50, EfficientNet | MobileNetV2 está optimizado para inferencia en tiempo real, modelo base < 15 MB, ideal para despliegue en contenedor |
| Joblib para serialización de sklearn | Pickle nativo | Joblib es el estándar recomendado por scikit-learn para objetos con arrays numpy grandes |
| PCA 3D para visualización K-means | t-SNE, UMAP | PCA es determinista (resultado reproducible), computacionalmente más ligero y más fácil de interpretar para el frontend |
| K óptimo con Elbow Method | Silhouette Score | El Elbow Method es más rápido y suficiente para la varianza esperada (10-200 pacientes por tenant) |
