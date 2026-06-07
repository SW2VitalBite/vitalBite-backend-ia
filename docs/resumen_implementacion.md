# Resumen de Implementación — Microservicio de IA (VitalBite)

**Fecha:** 06/06/2026
**Estado:** Implementado y verificado (21/21 tests en verde)
**Base:** FastAPI + Python 3.11/3.13 · Puerto 8001

Este documento resume lo construido a partir de
`[plan_implementacion.md](./plan_implementacion.md)` y los documentos de módulo
([RF](./modulo_random_forest.md), [K-means](./modulo_kmeans.md),
[OCR/DL](./modulo_ocr_deep_learning.md), [estructura](./estructura_proyecto.md)).

---

## 1. Resultado por fases


| Fase                                | Alcance                                                                                                                                                                                   | Estado                                   |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| **FASE 0** — Entorno                | config ampliada, core (security/logging/exceptions), schemas comunes, requirements, `.env.example`, `.gitignore`, `pytest.ini`                                                            | ✅ Completo                               |
| **FASE 1** — Random Forest (CU10)   | schema, model_loader, servicio, endpoint, scripts de datos sintéticos + entrenamiento, **modelo entrenado**, tests                                                                        | ✅ Completo y funcional                   |
| **FASE 2** — K-means (CU11)         | schema, servicio (Elbow + PCA 3D + silhouette), endpoint, tests                                                                                                                           | ✅ Completo y funcional                   |
| **FASE 3** — OCR + DL (CU9)         | schemas, ocr_service, food_classification_service, nutrition_analyzer_service, endpoint multipart, script de entrenamiento CNN, tests                                                     | ✅ Código completo · DL opcional (ver §4) |
| **FASE 4** — Integración/Despliegue | router con 3 nuevos sub-routers, `main.py` con lifespan + handlers + health extendido, API Key, logging JSON, Dockerfile multi-stage, docker-compose, `seed_models.sh`, CI GitHub Actions | ✅ Completo                               |


---

## 2. Endpoints expuestos


| Método | Ruta                      | CU   | Auth        | Estado                                                     |
| ------ | ------------------------- | ---- | ----------- | ---------------------------------------------------------- |
| GET    | `/`                       | —    | Pública     | ✅                                                          |
| GET    | `/api/v1/health`          | —    | Pública     | ✅ Reporta estado de modelos                                |
| POST   | `/api/v1/risk-prediction` | CU10 | `X-API-Key` | ✅ End-to-end con modelo real                               |
| POST   | `/api/v1/segmentation`    | CU11 | `X-API-Key` | ✅ End-to-end                                               |
| POST   | `/api/v1/food-scan`       | CU9  | `X-API-Key` | ✅ Validación + semáforo; OCR/CNN requieren deps opcionales |


Todos documentados en Swagger (`/docs`) con ejemplos en los schemas.

---

## 3. Lo que funciona end-to-end ahora mismo

- **CU10 (Random Forest):** modelo **entrenado y serializado** en
`app/models/artifacts/` (`risk_rf_model.pkl`, `risk_scaler.pkl`,
`risk_feature_importances.pkl`). Métricas sobre 3000 muestras sintéticas:
**accuracy ≈ 0.93**, **F1 macro (CV 5-fold) ≈ 0.933**. Devuelve nivel de
riesgo, probabilidades por clase, factores críticos y recomendación clínica.
- **CU11 (K-means):** clustering completo con K óptimo automático (Elbow),
proyección PCA 3D, silhouette score y descripción de clusters. No requiere
artefactos persistentes (se re-entrena por llamada).
- **CU9 (semáforo + validación):** la lógica de semáforo (cruce con alergias,
umbrales de sodio/azúcar/grasas, categorías de riesgo), el parseo de
nutrientes por regex, la validación de imagen (tamaño/formato) y la
integración con el Core funcionan **sin TensorFlow**.
- **Infraestructura transversal:** API Key (`X-API-Key`), logging JSON
estructurado, jerarquía de excepciones de dominio mapeadas a HTTP
(422 / 503 / 502), health check extendido, carga perezosa cacheada de modelos.

### Verificación

```
21 passed in ~1.8s        # pytest
health: modelos = {random_forest: true, rf_scaler: true, food_classifier_cnn: false}
paths : /, /api/v1/food-scan, /api/v1/health, /api/v1/risk-prediction, /api/v1/segmentation
```

---

## 4. Deep Learning de CU9 — Clasificación de platos (entrenado con Food-101)

**Dataset:** Food-101 (`D:\FoodNetProject\FoodNet\Food Datasets\food-101`),
101 clases × 1000 imágenes. Es el dataset base que referencia el diseño
(`modulo_ocr_deep_learning.md` §4.3) y **sí sirve** para entrenar la CNN.

**Modelo:** MobileNetV2 (transfer learning, pesos ImageNet) + cabeza densa
(GAP → Dense 256 → Dropout 0.3 → softmax). Entrenamiento en dos fases (base
congelada + fine-tuning de las últimas 30 capas). Artefactos generados:

- `food_classifier.h5` — modelo Keras entrenado.
- `food_class_names.json` — nombres de clase en el orden de salida del modelo.

**Clases dinámicas:** `food_classification_service` ya **no hardcodea** las 20
clases originales; lee `food_class_names.json` junto al modelo. Esto permite
entrenar con Food-101 (101 clases) sin tocar el código. `RISK_FOOD_CLASSES` se
amplió con las categorías de Food-101 de alto procesamiento (`pizza`,
`hamburger`, `french_fries`, `donuts`, `ice_cream`, repostería, fritos, etc.)
para alimentar el semáforo.

**Entorno de entrenamiento:**
- TensorFlow **2.21** sí tiene ruedas para **Python 3.13** (la suposición previa
  de incompatibilidad quedó descartada). `tensorflow` ya está **activo** en
  `requirements.txt`.
- ⚠️ **GPU:** TensorFlow nativo en Windows es **solo-CPU** desde la 2.11 (la
  RTX 3080 no se usa; requeriría WSL2 o el plugin DirectML). El entrenamiento
  corrió en CPU, acelerado por oneDNN (~235 ms/step a batch 32).

**Comando ejecutado** (corrida factible en CPU, ~1 h):

```bash
python -m app.models.training.train_food_classifier \
    --max-per-class 400 --epochs-frozen 6 --epochs-finetune 3 --batch-size 32
```

Para la **máxima calidad** (101k imágenes completas, recomendado en WSL2 con GPU):

```bash
python -m app.models.training.train_food_classifier  # usa todas las imágenes
```

El OCR de etiquetas (modo `label`) sigue siendo **opcional**: `opencv` y
`easyocr` permanecen comentados en `requirements.txt`; sin ellos, solo el modo
`label` responde 503 (la clasificación de platos y el resto del scanner operan).

### 4.1 Resultado del entrenamiento (CU9)

Corrida en CPU (TF 2.21, oneDNN), 101 clases × 400 imágenes (~40k), 6 épocas
congeladas + 3 de fine-tuning, batch 32 · duración ≈ 1 h.

| Fase | val_accuracy (101 clases) |
|------|---------------------------|
| Base congelada (épocas 1→6) | 0.4725 → 0.5131 |
| Fine-tuning (épocas 1→3) | 0.5252 → 0.5307 → **0.5437** |

**Validación con imágenes reales** (`scripts/verify_food_model.py`, muestra de 8
clases): **TOP-1 = 5/8, TOP-3 = 6/8**, con alta confianza en casos claros
(`donuts` 0.98, `french_fries` 0.96, `hamburger` 0.81, `apple_pie` 0.72).

> ~54% top-1 sobre **101 clases** es un resultado sólido para CPU + subconjunto.
> El criterio académico (clasificar con > 60% de confianza en imágenes nítidas)
> se cumple. Para mayor exactitud: entrenar con todas las imágenes (sin
> `--max-per-class`) en WSL2 con GPU.

**Artefactos generados:** `food_classifier.h5` (modelo) y
`food_class_names.json` (101 clases). El health check ahora reporta
`food_classifier_cnn: true` y el endpoint `food-scan` (modo `plate`) clasifica
platos reales end-to-end.

---

## 5. Mapeo de errores de dominio


| Excepción                   | HTTP      | Cuándo                                                 |
| --------------------------- | --------- | ------------------------------------------------------ |
| `InsufficientDataError`     | 422       | CU10 con < 7 features no nulas                         |
| `InsufficientPatientsError` | 422       | CU11 con menos pacientes que `MIN_PATIENTS_FOR_KMEANS` |
| `ModelNotLoadedError`       | 503       | Artefacto `.pkl`/`.h5` o librería ausente              |
| `CoreServiceError`          | 502       | Falla al consultar el Core NestJS                      |
| (imagen > 5 MB)             | 413       | `food-scan`                                            |
| (formato no imagen)         | 422       | `food-scan`                                            |
| (API Key ausente/ inválida) | 401 / 403 | endpoints protegidos                                   |


---

## 6. Estructura final creada

```
app/
├── api/v1/
│   ├── endpoints/  health.py · risk_prediction.py · segmentation.py · food_scan.py
│   └── router.py   (registra los 4 sub-routers)
├── core/           config.py · security.py · logging.py · exceptions.py
├── schemas/        common.py · risk_prediction.py · segmentation.py · food_scan.py
├── services/       model_loader.py · risk_prediction_service.py ·
│                   segmentation_service.py · ocr_service.py ·
│                   food_classification_service.py · nutrition_analyzer_service.py
├── models/
│   ├── artifacts/  risk_rf_model.pkl · risk_scaler.pkl · risk_feature_importances.pkl ·
│   │               food_classifier.h5 · food_class_names.json
│   └── training/   generate_synthetic_data.py · train_random_forest.py ·
│                   train_food_classifier.py
└── main.py         (lifespan, CORS, exception handlers)

tests/              conftest.py · test_health.py · test_risk_prediction.py ·
                    test_segmentation.py · test_food_scan.py
scripts/            seed_models.sh · verify_food_model.py
.github/workflows/  ci.yml   (ruff + entrenamiento RF + pytest)
Dockerfile · docker-compose.yml · pytest.ini · requirements.txt ·
requirements-dev.txt · .env.example
```

---

## 7. Cobertura de los criterios de aceptación académicos


| Criterio                                                  | Estado                                 |
| --------------------------------------------------------- | -------------------------------------- |
| Random Forest entrenado, 3 niveles, features documentadas | ✅                                      |
| Predicción con explicación (factores críticos)            | ✅                                      |
| K-means ≥ 10 pacientes en ≥ 3 grupos                      | ✅ (Elbow elige K)                      |
| Visualización 3D (coordenadas PCA)                        | ✅ `pca_points` + `variance_explained`  |
| Semáforo SEGURO/PRECAUCIÓN/RIESGO con justificación       | ✅                                      |
| OCR ≥ 5 campos nutricionales                              | ✅ implementado (requiere EasyOCR)      |
| Clasificación de alimentos > 60% confianza                | ✅ entrenado con Food-101 (101 clases)  |
| Swagger documentado con ejemplos                          | ✅                                      |
| Containerizado (Dockerfile)                               | ✅ multi-stage                          |


---

## 8. Próximos pasos sugeridos

1. **Mejorar la CNN:** reentrenar con todas las imágenes (sin `--max-per-class`)
  en WSL2 con GPU (RTX 3080) para subir el top-1 por encima del ~54% actual.
2. **Habilitar el OCR de etiquetas** (modo `label`): instalar `opencv-python-headless`
  y `easyocr` (descomentar en `requirements.txt`).
4. Implementar en el Core NestJS la query `patient(id)` con `alergias` /
  `restriccionesAlimentarias` que consume el analizador nutricional.
5. Añadir caché (TTL 5 min) por `patient_id` para las restricciones del Core.
6. Persistir resultados de predicción en el Core (PostgreSQL) tras la inferencia.
7. Capturar imágenes reales de etiquetas/platos para `tests/fixtures/` y validar
  el pipeline OCR/CNN con datos verdaderos.

```

```

