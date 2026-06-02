# VitalBite Backend IA

Microservicio de inteligencia artificial para VitalBite. Expone APIs REST con FastAPI para:

- Reconocimiento de etiquetas nutricionales (OCR / Deep Learning)
- Predicción de riesgo nutricional (Random Forest)
- Segmentación de pacientes (K-means)

## Requisitos

- Python 3.11+
- pip

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Ejecución

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

Documentación interactiva: [http://localhost:8001/docs](http://localhost:8001/docs)

## Estructura del proyecto

```
app/
├── api/v1/          # Rutas y endpoints versionados
├── core/            # Configuración y utilidades base
├── models/          # Modelos ML entrenados (artefactos)
├── schemas/         # Esquemas Pydantic (request/response)
└── services/        # Lógica de negocio e inferencia
```
