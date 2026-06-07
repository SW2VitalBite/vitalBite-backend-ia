# VitalBite Backend IA

Microservicio de inteligencia artificial para VitalBite. Expone una API REST con
FastAPI (servicio interno de inferencia) para tres casos de uso:

| CU | Endpoint | Algoritmo |
|----|----------|-----------|
| **CU9** — Scanner Nutricional | `POST /api/v1/food-scan` | OCR (EasyOCR) + CNN (MobileNetV2) |
| **CU10** — Predicción de Riesgo | `POST /api/v1/risk-prediction` | Random Forest |
| **CU11** — Segmentación de Pacientes | `POST /api/v1/segmentation` | K-means + PCA 3D |

> El protocolo REST aquí es válido por las reglas del proyecto (AGENTS.md):
> FastAPI es un servicio interno de inferencia. GraphQL solo es obligatorio para
> la comunicación cliente ↔ Core NestJS.

## Requisitos

- Python 3.11+ (recomendado 3.10–3.12 para habilitar el Deep Learning de CU9)
- pip

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt   # producción  (dev: pip install -r requirements-dev.txt)
copy .env.example .env            # Windows  (Linux/Mac: cp .env.example .env)
```

### Entrenar los modelos (genera los artefactos)

```bash
python -m app.models.training.train_random_forest   # CU10 → artifacts/risk_rf_model.pkl
bash scripts/seed_models.sh                          # entrena RF (+ CNN si hay dataset)
```

## Ejecución

> **Importante:** ejecuta siempre dentro del venv. El modelo `food_classifier.h5`
> se entrenó con el Keras del venv; usar el Python global (otra versión de Keras)
> provoca un error al deserializar `BatchNormalization` (`renorm`). Para garantizar
> el intérprete correcto, invoca uvicorn como módulo del venv:

```bash
# Opción robusta (no depende de activar el venv):
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8001   # Windows
# o, tras activar el venv (.venv\Scripts\activate):
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

> Para logs más limpios en la demo, pon `DEBUG=false` en `.env` (evita el ruido
> DEBUG de multipart/PIL/h5py).

- Swagger UI: <http://localhost:8001/docs>
- Health check: <http://localhost:8001/api/v1/health>

Los endpoints de inferencia exigen la cabecera `X-API-Key` (valor en `.env`).

## Docker

```bash
docker compose up --build      # desarrollo con hot-reload
```

## Tests

```bash
pytest -q
```

## Estado del Deep Learning (CU9)

- **Clasificación de platos (modo `plate`):** activa. La CNN MobileNetV2 se
  entrena con **Food-101** y se sirve desde `food_classifier.h5`. TensorFlow
  (>=2.20, funciona en Python 3.13) está activo en `requirements.txt`. En
  Windows nativo corre en **CPU** (la GPU requiere WSL2).

  ```bash
  # Entrenar (subconjunto factible en CPU)
  python -m app.models.training.train_food_classifier \
      --max-per-class 400 --epochs-frozen 6 --epochs-finetune 3
  # Dataset por defecto: D:\FoodNetProject\FoodNet\Food Datasets\food-101\images
  ```

  Las clases se leen dinámicamente de `food_class_names.json` (generado al
  entrenar), por lo que el código no depende del número de clases.

- **OCR de etiquetas (modo `label`):** **opcional**. `opencv` y `easyocr` están
  comentados en `requirements.txt`; sin ellos ese modo responde `503`. La lógica
  del semáforo, el parseo de nutrientes y la validación de imágenes funcionan sin
  esas librerías.

Ver `docs/resumen_implementacion.md` para el detalle completo.

## Estructura del proyecto

```
app/
├── api/v1/          # Rutas y endpoints versionados
├── core/            # config, security (API Key), logging JSON, exceptions
├── models/          # artifacts/ (modelos serializados) + training/ (scripts)
├── schemas/         # Esquemas Pydantic (request/response)
└── services/        # Lógica de inferencia (RF, K-means, OCR, CNN, analizador)
```
