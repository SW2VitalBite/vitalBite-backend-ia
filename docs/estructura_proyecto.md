# Estructura del Proyecto — vitalBite-backend-ia

Árbol completo de directorios y archivos que debe existir al finalizar la implementación. Los archivos marcados con ✅ ya existen. Los marcados con 🔲 deben crearse.

---

## Árbol de Directorios

```
vitalBite-backend-ia/
│
├── 📁 app/
│   │
│   ├── 📁 api/
│   │   └── 📁 v1/
│   │       ├── 📁 endpoints/
│   │       │   ├── ✅ health.py              → GET /api/v1/health
│   │       │   ├── 🔲 food_scan.py           → POST /api/v1/food-scan       (CU9)
│   │       │   ├── 🔲 risk_prediction.py     → POST /api/v1/risk-prediction  (CU10)
│   │       │   └── 🔲 segmentation.py        → POST /api/v1/segmentation     (CU11)
│   │       ├── ✅ router.py                  → Registra todos los sub-routers
│   │       └── 🔲 __init__.py
│   │
│   ├── 📁 core/
│   │   ├── ✅ config.py                      → Settings (pydantic-settings)
│   │   ├── 🔲 security.py                    → Dependencia verify_api_key
│   │   ├── 🔲 logging.py                     → Logger estructurado JSON
│   │   ├── 🔲 exceptions.py                  → Excepciones de dominio personalizadas
│   │   └── 🔲 __init__.py
│   │
│   ├── 📁 schemas/
│   │   ├── 🔲 common.py                      → Tipos compartidos (Semaforo, NivelRiesgo)
│   │   ├── 🔲 food_scan.py                   → FoodScanRequest / FoodScanResponse
│   │   ├── 🔲 risk_prediction.py             → RiskPredictionRequest / RiskPredictionResponse
│   │   ├── 🔲 segmentation.py                → SegmentationRequest / SegmentationResponse
│   │   └── 🔲 __init__.py
│   │
│   ├── 📁 services/
│   │   ├── 🔲 model_loader.py                → Singleton lazy loading de artefactos
│   │   ├── 🔲 ocr_service.py                 → EasyOCR: extracción y parseo de nutrientes
│   │   ├── 🔲 food_classification_service.py → CNN MobileNetV2: clasificación de alimentos
│   │   ├── 🔲 nutrition_analyzer_service.py  → Cruce alergias + semáforo de riesgo
│   │   ├── 🔲 risk_prediction_service.py     → Random Forest: inferencia y factores críticos
│   │   ├── 🔲 segmentation_service.py        → K-means: clustering + PCA 3D
│   │   └── 🔲 __init__.py
│   │
│   ├── 📁 models/
│   │   ├── 📁 artifacts/                     → Modelos serializados (en .gitignore)
│   │   │   ├── 🔲 risk_rf_model.pkl          → Clasificador Random Forest entrenado
│   │   │   ├── 🔲 risk_scaler.pkl            → StandardScaler para features de RF
│   │   │   ├── 🔲 risk_feature_importances.pkl → Importancia de features RF
│   │   │   └── 🔲 food_classifier.h5         → CNN MobileNetV2 fine-tuned (>50 MB)
│   │   ├── 📁 training/                      → Scripts de entrenamiento (solo desarrollo)
│   │   │   ├── 🔲 generate_synthetic_data.py → Genera dataset sintético para RF
│   │   │   ├── 🔲 train_random_forest.py     → Entrena y serializa el modelo RF
│   │   │   └── 🔲 train_food_classifier.py   → Entrena y serializa CNN (MobileNetV2)
│   │   └── 🔲 __init__.py
│   │
│   └── ✅ main.py                             → FastAPI app, CORS, startup events
│
├── 📁 docs/
│   ├── ✅ plan_implementacion.md              → Plan maestro por fases
│   ├── ✅ arquitectura_ia.md                  → Arquitectura técnica y capas
│   ├── ✅ modulo_ocr_deep_learning.md         → Diseño del scanner (CU9)
│   ├── ✅ modulo_random_forest.md             → Diseño del predictor de riesgo (CU10)
│   ├── ✅ modulo_kmeans.md                    → Diseño de la segmentación (CU11)
│   └── ✅ estructura_proyecto.md              → Este archivo
│
├── 📁 tests/
│   ├── 🔲 conftest.py                         → Fixtures compartidas de pytest
│   ├── 🔲 test_health.py                      → Test del endpoint /health
│   ├── 🔲 test_food_scan.py                   → Tests del scanner OCR/DL (CU9)
│   ├── 🔲 test_risk_prediction.py             → Tests del predictor RF (CU10)
│   ├── 🔲 test_segmentation.py                → Tests del K-means (CU11)
│   └── 📁 fixtures/
│       ├── 🔲 sample_label.jpg                → Imagen de etiqueta nutricional de prueba
│       └── 🔲 sample_plate.jpg                → Imagen de plato de comida de prueba
│
├── 📁 scripts/
│   └── 🔲 seed_models.sh                      → Script para entrenar todos los modelos (CI/CD)
│
├── ✅ requirements.txt                         → Dependencias de producción (actualizar)
├── 🔲 requirements-dev.txt                    → pytest, ruff, httpx (solo desarrollo)
├── 🔲 .env.example                            → Variables de entorno de ejemplo
├── 🔲 .gitignore                              → Excluir .venv/, artifacts/*.pkl, *.h5
├── 🔲 Dockerfile                              → Imagen de producción multi-stage
├── 🔲 docker-compose.yml                      → Desarrollo local con hot-reload
├── ✅ README.md                               → Documentación de inicio (actualizar)
└── 🔲 pytest.ini                             → Configuración de pytest
```

---

## Detalle de los Archivos Críticos a Crear

### `app/core/security.py`

```python
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from app.core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API Key inválida o no autorizada."
        )
    return api_key
```

---

### `app/core/exceptions.py`

```python
from fastapi import Request
from fastapi.responses import JSONResponse

class InsufficientDataError(Exception):
    def __init__(self, missing_fields: list[str]):
        self.missing_fields = missing_fields

class InsufficientPatientsError(Exception):
    def __init__(self, count: int, minimum: int):
        self.count = count
        self.minimum = minimum

class ModelNotLoadedError(Exception):
    def __init__(self, model_path: str):
        self.model_path = model_path

async def insufficient_data_handler(request: Request, exc: InsufficientDataError):
    return JSONResponse(
        status_code=422,
        content={"error": "datos_insuficientes", "campos_faltantes": exc.missing_fields}
    )
```

---

### `.env.example`

```env
# === App ===
APP_NAME=VitalBite Backend IA
APP_VERSION=0.2.0
DEBUG=true
HOST=0.0.0.0
PORT=8001
API_V1_PREFIX=/api/v1

# === CORS ===
CORS_ORIGINS=http://localhost:3000,http://localhost:8081,http://localhost:4200

# === Seguridad ===
API_KEY=vitalbite_ia_secret_key_dev_change_in_prod

# === Comunicación con Core NestJS ===
CORE_SERVICE_URL=http://localhost:3000
CORE_GRAPHQL_URL=http://localhost:3000/graphql
INTERNAL_TOKEN=internal_service_token

# === Machine Learning ===
MODELS_DIR=app/models/artifacts
MIN_PATIENTS_FOR_KMEANS=10
RF_RISK_THRESHOLD_HIGH=0.70
RF_RISK_THRESHOLD_MEDIUM=0.40

# === OCR / Deep Learning ===
OCR_MIN_CONFIDENCE=0.60
IMAGE_MAX_SIZE_MB=5
DL_MODEL_PATH=app/models/artifacts/food_classifier.h5
```

---

### `requirements.txt` (versión actualizada)

```txt
# === Framework ===
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.9.0
pydantic-settings>=2.6.0
python-multipart>=0.0.12

# === Machine Learning ===
scikit-learn>=1.5.0
joblib>=1.4.0
numpy>=1.26.0
pandas>=2.2.0

# === Deep Learning ===
tensorflow>=2.17.0
Pillow>=10.4.0
opencv-python-headless>=4.10.0

# === OCR ===
easyocr>=1.7.0

# === HTTP Client (comunicación interna) ===
httpx>=0.27.0
```

---

### `requirements-dev.txt`

```txt
pytest>=8.2.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
ruff>=0.4.0
```

---

### `Dockerfile`

```dockerfile
# Etapa 1: Construcción de dependencias
FROM python:3.11-slim AS builder

WORKDIR /install
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/packages -r requirements.txt

# Etapa 2: Imagen de producción
FROM python:3.11-slim AS production

WORKDIR /app

# Copiar dependencias instaladas
COPY --from=builder /install/packages /usr/local

# Copiar código fuente
COPY app/ ./app/
COPY .env.example .env

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8001/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
```

---

### `docker-compose.yml`

```yaml
version: "3.9"

services:
  vitalbite-ia:
    build:
      context: .
      target: production
    container_name: vitalbite-ia
    ports:
      - "8001:8001"
    volumes:
      - ./app:/app/app          # Hot-reload en desarrollo
      - ./app/models/artifacts:/app/app/models/artifacts
    environment:
      - DEBUG=true
      - CORE_SERVICE_URL=http://host.docker.internal:3000
    env_file:
      - .env
    restart: unless-stopped
```

---

### `pytest.ini`

```ini
[pytest]
testpaths = tests
asyncio_mode = auto
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

---

## `.gitignore` (adiciones recomendadas)

```gitignore
# Entorno virtual
.venv/
__pycache__/
*.pyc

# Variables de entorno
.env

# Artefactos de modelos (archivos grandes)
app/models/artifacts/*.pkl
app/models/artifacts/*.h5
app/models/artifacts/*.pt
app/models/artifacts/*.joblib

# Cobertura de tests
.coverage
htmlcov/
.pytest_cache/

# Logs
*.log
logs/
```

---

## Orden de Creación Recomendado

```
1. app/core/exceptions.py
2. app/core/security.py
3. app/core/logging.py
4. app/schemas/common.py
5. app/schemas/risk_prediction.py
6. app/schemas/segmentation.py
7. app/schemas/food_scan.py
8. app/models/training/generate_synthetic_data.py
9. app/models/training/train_random_forest.py
10. app/services/model_loader.py
11. app/services/risk_prediction_service.py
12. app/api/v1/endpoints/risk_prediction.py
13. [Ejecutar: python app/models/training/train_random_forest.py]
14. tests/test_risk_prediction.py
15. app/services/segmentation_service.py
16. app/api/v1/endpoints/segmentation.py
17. tests/test_segmentation.py
18. app/services/ocr_service.py
19. app/services/food_classification_service.py
20. app/services/nutrition_analyzer_service.py
21. app/api/v1/endpoints/food_scan.py
22. app/models/training/train_food_classifier.py
23. tests/test_food_scan.py
24. app/api/v1/router.py  [actualizar con los 3 nuevos routers]
25. app/main.py           [actualizar con startup events]
26. Dockerfile + docker-compose.yml
27. .env.example + requirements.txt actualizado
```
