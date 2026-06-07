# Plan de Implementación — Microservicio de IA (VitalBite)
**Tecnología base:** FastAPI + Python 3.11 · Puerto 8001  
**Fecha de creación:** 05/06/2026  
**Estado:** En diseño

---

## 1. Propósito del Microservicio

El microservicio `vitalBite-backend-ia` es el componente de inteligencia artificial de la plataforma SaaS VitalBite. Su responsabilidad exclusiva es ejecutar inferencias de Machine Learning y Deep Learning de forma asíncrona, desacoplada del Core empresarial (NestJS). No persiste datos propios; actúa como motor analítico que recibe parámetros, ejecuta modelos entrenados y devuelve resultados estructurados al Core o directamente a la app móvil.

### Casos de Uso que implementa

| CU   | Nombre                                          | Algoritmo                    | Actor  |
|------|-------------------------------------------------|------------------------------|--------|
| CU9  | Analizar Etiquetas y Alimentos con Deep Learning | CNN + OCR (Tesseract/EasyOCR)| A1 (Paciente) |
| CU10 | Evaluar Predicción de Riesgo de Salud           | Random Forest (supervisado)  | A2 (Nutricionista) |
| CU11 | Consultar Segmentación de Perfiles              | K-means (no supervisado)     | A2 (Nutricionista) |

---

## 2. Documentación Relacionada

| Documento | Descripción |
|-----------|-------------|
| [`arquitectura_ia.md`](./arquitectura_ia.md) | Arquitectura técnica completa del microservicio |
| [`modulo_ocr_deep_learning.md`](./modulo_ocr_deep_learning.md) | Diseño e implementación del scanner nutricional (CU9) |
| [`modulo_random_forest.md`](./modulo_random_forest.md) | Diseño e implementación del predictor de riesgo (CU10) |
| [`modulo_kmeans.md`](./modulo_kmeans.md) | Diseño e implementación de la segmentación de pacientes (CU11) |
| [`estructura_proyecto.md`](./estructura_proyecto.md) | Árbol de carpetas y archivos a crear |

---

## 3. Fases de Implementación

### FASE 0 — Preparación del entorno (Semana 1)

**Objetivo:** Tener la base del proyecto funcional con dependencias, configuración y estructura de carpetas.

| Tarea | Descripción | Prioridad |
|-------|-------------|-----------|
| 0.1 | Actualizar `requirements.txt` con todas las librerías de ML/DL | Alta |
| 0.2 | Ampliar `app/core/config.py` con variables de entorno para DB y modelos | Alta |
| 0.3 | Crear la estructura de directorios completa (ver `estructura_proyecto.md`) | Alta |
| 0.4 | Crear `.env.example` con todas las variables requeridas | Alta |
| 0.5 | Configurar cliente HTTP para comunicación interna con el Core NestJS | Media |
| 0.6 | Añadir modelos `__init__.py` en cada paquete nuevo | Alta |

**Entregable:** Proyecto corre con `uvicorn app.main:app --reload`, con Swagger funcional en `/docs`.

---

### FASE 1 — Módulo Random Forest: Predicción de Riesgo (CU10) (Semana 2-3)

**Objetivo:** Implementar el endpoint que predice riesgo nutricional (Bajo / Medio / Alto) a partir de datos antropométricos del paciente.

**Justificación de prioridad:** Es el módulo más simple en cuanto a entrada/salida y permite validar la arquitectura de servicios antes de abordar DL.

| Tarea | Descripción |
|-------|-------------|
| 1.1 | Definir `schemas/risk_prediction.py` (request con 10 features, response con nivel + probabilidad + factores) |
| 1.2 | Crear `services/risk_prediction_service.py` con carga de modelo `.pkl` y función `predict()` |
| 1.3 | Crear `api/v1/endpoints/risk_prediction.py` con `POST /risk-prediction` |
| 1.4 | Generar datos sintéticos para entrenamiento inicial (script `scripts/train_random_forest.py`) |
| 1.5 | Entrenar modelo y serializar artefacto en `app/models/artifacts/risk_rf_model.pkl` |
| 1.6 | Implementar lógica de validación de datos insuficientes (retornar `422` con campos faltantes) |
| 1.7 | Escribir tests unitarios con `pytest` |

**Features de entrada del modelo:**
- Edad, sexo, peso actual, talla, IMC calculado
- Variación de peso en últimos 3 meses
- Porcentaje de grasa corporal
- Nivel de actividad física (0-4 escala Likert)
- Calidad de dieta auto-reportada (0-10)
- Número de comorbilidades registradas

**Entregable:** `POST /api/v1/risk-prediction` retorna JSON `{ "nivel_riesgo": "Alto", "probabilidad": 0.82, "factores_criticos": [...] }`.

---

### FASE 2 — Módulo K-means: Segmentación de Pacientes (CU11) (Semana 3-4)

**Objetivo:** Implementar el endpoint que agrupa a los pacientes de un tenant en clusters nutricionales, retornando coordenadas para visualización 3D en Angular.

| Tarea | Descripción |
|-------|-------------|
| 2.1 | Definir `schemas/segmentation.py` (request con array de pacientes anonimizados, response con clusters + centroides) |
| 2.2 | Crear `services/segmentation_service.py` con lógica de K-means (K óptimo con Elbow Method) |
| 2.3 | Crear `api/v1/endpoints/segmentation.py` con `POST /segmentation` |
| 2.4 | Implementar preprocesamiento: normalización MinMaxScaler, detección de outliers |
| 2.5 | Implementar validación de mínimo de pacientes (umbral configurable en `.env`) |
| 2.6 | Serializar el modelo entrenado por tenant en `app/models/artifacts/kmeans_{tenant_id}.pkl` |
| 2.7 | Retornar proyección PCA 3D para gráfico de dispersión en el frontend |
| 2.8 | Escribir tests unitarios con `pytest` |

**Features usadas en clustering:**
- IMC, porcentaje de grasa corporal, masa muscular
- Variación de peso, nivel de actividad física
- Adherencia al plan dietético (%)
- Número de citas asistidas / totales

**Entregable:** `POST /api/v1/segmentation` retorna `{ "clusters": [...], "centroides": [...], "pca_points": [...] }`.

---

### FASE 3 — Módulo OCR + Deep Learning: Scanner Nutricional (CU9) (Semana 4-6)

**Objetivo:** Implementar el endpoint que recibe una imagen desde la app móvil, ejecuta OCR sobre etiquetas nutricionales o clasifica el plato de comida con una CNN, y retorna un diagnóstico de riesgo personalizado.

**Nota:** Esta fase es la más compleja y requiere decisiones sobre modelo base de CNN (transfer learning).

| Tarea | Descripción |
|-------|-------------|
| 3.1 | Definir `schemas/food_scan.py` (request: imagen base64 o multipart, modo `label`/`plate`) |
| 3.2 | Crear `services/ocr_service.py` con EasyOCR para extracción de texto de etiquetas |
| 3.3 | Crear `services/food_classification_service.py` con modelo CNN (MobileNetV2 fine-tuned) |
| 3.4 | Crear `services/nutrition_analyzer_service.py` para cruce con alergias del paciente |
| 3.5 | Crear `api/v1/endpoints/food_scan.py` con `POST /food-scan` (acepta multipart/form-data) |
| 3.6 | Implementar pipeline de procesamiento: validar imagen → routing → OCR o CNN → análisis → respuesta |
| 3.7 | Parsear texto OCR: extraer calorías, carbohidratos, proteínas, grasas, sodio, ingredientes |
| 3.8 | Implementar semáforo de riesgo: clasificar como `SEGURO`, `PRECAUCIÓN`, `RIESGO` |
| 3.9 | Gestionar imágenes con baja confianza (umbral < 0.60) |
| 3.10 | Escribir tests con imágenes de prueba |

**Entregable:** `POST /api/v1/food-scan` retorna `{ "modo": "label", "semaforo": "RIESGO", "nutrientes": {...}, "advertencias": [...], "confianza": 0.87 }`.

---

### FASE 4 — Integración, Despliegue y Hardening (Semana 6-7)

**Objetivo:** Conectar el microservicio con el Core NestJS, containerizar y preparar para despliegue en nube.

| Tarea | Descripción |
|-------|-------------|
| 4.1 | Añadir autenticación por API Key (cabecera `X-API-Key`) en todos los endpoints |
| 4.2 | Implementar middleware de logging estructurado (JSON logs) |
| 4.3 | Configurar health check extendido: `/api/v1/health` con estado de modelos cargados |
| 4.4 | Crear `Dockerfile` multi-stage optimizado para producción |
| 4.5 | Crear `docker-compose.yml` para desarrollo local |
| 4.6 | Documentar variables de entorno en `.env.example` |
| 4.7 | Validar comunicación con Core NestJS (llamada interna REST desde NestJS → FastAPI) |
| 4.8 | Configurar GitHub Actions CI: lint (ruff) + tests (pytest) |

---

## 4. Dependencias de Datos entre Fases (Diagrama)

```
CU3 (Medidas Corporales — NestJS)
        │
        ├──→ CU10 (Random Forest): requiere historial de medidas del paciente
        │
        └──→ CU11 (K-means): requiere medidas de al menos N pacientes del tenant

CU9 (Scanner Nutricional) → opera de forma INDEPENDIENTE
        │
        └──→ requiere únicamente: imagen desde móvil + perfil de alergias del paciente
```

> **Regla de negocio (AGENTS.md):** Los modelos CU10 y CU11 no pueden ejecutarse si el expediente carece de datos del CU3.

---

## 5. Stack Tecnológico Detallado

| Categoría | Librería | Versión Mínima | Uso |
|-----------|----------|----------------|-----|
| Framework API | FastAPI | 0.115.0 | Router, endpoints, middleware |
| Servidor ASGI | Uvicorn | 0.32.0 | Servidor de producción |
| Validación | Pydantic v2 | 2.9.0 | Schemas de request/response |
| ML supervisado | scikit-learn | 1.5.0 | Random Forest, K-means, PCA, escalado |
| ML utils | joblib | 1.4.0 | Serialización de modelos `.pkl` |
| Deep Learning | TensorFlow / Keras | 2.17.0 | CNN MobileNetV2 para clasificación |
| OCR | EasyOCR | 1.7.0 | Extracción de texto de etiquetas |
| Procesamiento imagen | Pillow | 10.4.0 | Carga, redimensionado, normalización |
| Procesamiento imagen | OpenCV | 4.10.0 | Preprocesamiento, mejora de contraste |
| Data | NumPy | 1.26.0 | Operaciones matriciales |
| Data | Pandas | 2.2.0 | Manipulación de features |
| HTTP cliente | httpx | 0.27.0 | Comunicación interna con Core NestJS |
| Tests | pytest | 8.2.0 | Tests unitarios e integración |
| Linter | ruff | 0.4.0 | Linting y formateo |

---

## 6. Variables de Entorno Requeridas

```env
# App
APP_NAME=VitalBite Backend IA
APP_VERSION=0.2.0
DEBUG=true
HOST=0.0.0.0
PORT=8001
API_V1_PREFIX=/api/v1

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8081

# Seguridad
API_KEY=vitalbite_ia_secret_key_dev

# Core NestJS (comunicación interna)
CORE_SERVICE_URL=http://localhost:3000
CORE_GRAPHQL_URL=http://localhost:3000/graphql

# Machine Learning
MODELS_DIR=app/models/artifacts
MIN_PATIENTS_FOR_KMEANS=10
RF_RISK_THRESHOLD_HIGH=0.70
RF_RISK_THRESHOLD_MEDIUM=0.40

# OCR / Deep Learning
OCR_MIN_CONFIDENCE=0.60
IMAGE_MAX_SIZE_MB=5
DL_MODEL_PATH=app/models/artifacts/food_classifier.h5
```

---

## 7. Estructura de Comunicación con el Core NestJS

El Core NestJS actúa como orquestador. Las llamadas siguen este patrón:

```
App Móvil (React Native)
    │
    │ POST /api/v1/food-scan  (imagen multipart)
    ▼
FastAPI (puerto 8001)
    │
    │ [procesamiento local: OCR/CNN]
    │
    │ GET perfil alergias del paciente
    ▼
Core NestJS (GraphQL, puerto 3000) → PostgreSQL

App Web Angular (Nutricionista)
    │
    │ GraphQL mutation ejecutarPrediccion(input) → Core NestJS
    │
    ▼
Core NestJS
    │ POST /api/v1/risk-prediction  (REST interno)
    ▼
FastAPI (puerto 8001)
    │
    └─→ Retorna predicción → Core NestJS → almacena en PostgreSQL → responde a Angular
```

---

## 8. Criterios de Aceptación Académicos

| Criterio | Descripción | CU Asociado |
|----------|-------------|-------------|
| OCR funcional | Extraer al menos 5 campos nutricionales de una etiqueta fotografiada | CU9 |
| Clasificación de alimentos | Identificar tipo de alimento con > 60% de confianza | CU9 |
| Semáforo de riesgo | Retornar `SEGURO`, `PRECAUCIÓN` o `RIESGO` con justificación | CU9 |
| Random Forest entrenado | Modelo con al menos 3 niveles de riesgo y features documentadas | CU10 |
| Predicción con explicación | Retornar lista de factores críticos que justifican el nivel de riesgo | CU10 |
| K-means operativo | Clusterizar mínimo 10 pacientes en al menos 3 grupos | CU11 |
| Visualización 3D | Retornar coordenadas PCA para renderizar scatter plot en Angular | CU11 |
| Swagger documentado | Todos los endpoints con ejemplos en `/docs` | Todos |
| Containerizado | `Dockerfile` funcional, imagen < 2 GB | Todos |
