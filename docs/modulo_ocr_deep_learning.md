# Módulo OCR + Deep Learning — Scanner Nutricional
**Caso de Uso:** CU9 — Analizar Etiquetas y Alimentos con Deep Learning  
**Actor:** A1 (Paciente) · Canal: App Móvil React Native  
**Endpoint:** `POST /api/v1/food-scan`

---

## 1. Descripción del Módulo

El Scanner Nutricional es la funcionalidad más visible del microservicio de IA. Permite al paciente fotografiar desde su celular:

1. **Una etiqueta nutricional** de un producto envasado → se activa el pipeline de **OCR**
2. **Un plato de comida** preparado → se activa el pipeline de **Deep Learning (CNN)**

En ambos casos, el sistema cruza los resultados con el perfil de alergias y restricciones del paciente para emitir un **diagnóstico de semáforo** en tiempo real.

---

## 2. Arquitectura del Pipeline

```
Imagen capturada (app móvil)
         │
         ▼
┌─────────────────────────────────────────┐
│         PREPROCESAMIENTO                │
│  - Validar formato (JPEG/PNG/WEBP)      │
│  - Validar tamaño (< 5 MB)              │
│  - Decode base64 o multipart → PIL      │
└──────────────┬──────────────────────────┘
               │
       ┌───────▼───────┐
       │  Detectar modo │
       │  "label" / "plate"│
       └───┬───────┬───┘
           │       │
    ┌──────▼──┐  ┌─▼────────────────────┐
    │  OCR    │  │  CNN Clasificación   │
    │ Pipeline│  │  Pipeline            │
    └──────┬──┘  └─┬────────────────────┘
           │       │
           └───┬───┘
               ▼
    ┌──────────────────────────┐
    │  ANALIZADOR NUTRICIONAL  │
    │  - Cruce con alergias    │
    │  - Evaluación semáforo   │
    │  - Generación advertencias│
    └──────────────────────────┘
               │
               ▼
         Respuesta JSON
```

---

## 3. Pipeline OCR — Etiquetas Nutricionales

### 3.1 Objetivo

Extraer automáticamente valores nutricionales clave de la tabla nutricional impresa en el empaque de un producto alimenticio.

### 3.2 Tecnología: EasyOCR

**Justificación de elección sobre Tesseract:**
- Soporta múltiples idiomas nativamente (español + inglés sin configuración adicional)
- Maneja mejor texto en ángulos, curvatura y fondos complejos
- API Python nativa sin dependencias de sistema operativo
- Confianza por palabra incluida en el output

```python
# Instalación
pip install easyocr
```

### 3.3 Etapas del Pipeline OCR

#### Etapa 1 — Preprocesamiento de imagen

```python
import cv2
import numpy as np
from PIL import Image

def preprocess_for_ocr(image: Image.Image) -> np.ndarray:
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Mejora de contraste adaptativa (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Umbralización adaptativa (mejor que umbral global para etiquetas)
    threshold = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Reducción de ruido
    denoised = cv2.fastNlMeansDenoising(threshold, h=10)
    return denoised
```

#### Etapa 2 — Extracción de texto con EasyOCR

```python
import easyocr

reader = easyocr.Reader(["es", "en"], gpu=False)  # GPU opcional

def extract_text(preprocessed_image: np.ndarray) -> list[dict]:
    results = reader.readtext(preprocessed_image, detail=1)
    # Retorna: [(bbox, texto, confianza), ...]
    return [
        {"text": text, "confidence": float(conf)}
        for (_, text, conf) in results
        if conf > 0.40  # filtrar detecciones con muy baja confianza
    ]
```

#### Etapa 3 — Parseo de nutrientes con regex

```python
import re
from dataclasses import dataclass

@dataclass
class NutrientData:
    calorias: float | None = None
    carbohidratos: float | None = None
    proteinas: float | None = None
    grasas_totales: float | None = None
    grasas_saturadas: float | None = None
    sodio: float | None = None
    azucares: float | None = None
    fibra: float | None = None
    ingredientes: list[str] = None

PATTERNS = {
    "calorias": r"(?:cal(?:orías?)?|energy|kcal)[^\d]*(\d+(?:[.,]\d+)?)",
    "carbohidratos": r"(?:carbohidrat(?:os?)|carbohydrate)[^\d]*(\d+(?:[.,]\d+)?)",
    "proteinas": r"(?:prote[íi]nas?|protein)[^\d]*(\d+(?:[.,]\d+)?)",
    "grasas_totales": r"(?:grasa total|total fat|lípidos)[^\d]*(\d+(?:[.,]\d+)?)",
    "grasas_saturadas": r"(?:grasa saturada|saturated fat)[^\d]*(\d+(?:[.,]\d+)?)",
    "sodio": r"(?:sodio|sodium)[^\d]*(\d+(?:[.,]\d+)?)",
    "azucares": r"(?:azúcares?|sugars?)[^\d]*(\d+(?:[.,]\d+)?)",
    "fibra": r"(?:fibra|fiber)[^\d]*(\d+(?:[.,]\d+)?)",
}

def parse_nutrients(text_blocks: list[dict]) -> NutrientData:
    full_text = " ".join(b["text"].lower() for b in text_blocks)
    data = NutrientData(ingredientes=[])

    for field, pattern in PATTERNS.items():
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            value = float(match.group(1).replace(",", "."))
            setattr(data, field, value)

    # Extraer ingredientes (texto después de "ingredientes:")
    ing_match = re.search(r"ingredientes?[:\s]+(.+?)(?:\.|$)", full_text, re.IGNORECASE)
    if ing_match:
        data.ingredientes = [i.strip() for i in ing_match.group(1).split(",")]

    return data
```

### 3.4 Cálculo de confianza OCR

```python
def calculate_ocr_confidence(text_blocks: list[dict], parsed: NutrientData) -> float:
    # Promedio de confianza de todas las palabras detectadas
    avg_word_conf = sum(b["confidence"] for b in text_blocks) / len(text_blocks) if text_blocks else 0

    # Bonus si se extrajeron campos clave
    fields_extracted = sum(1 for f in ["calorias", "proteinas", "carbohidratos", "grasas_totales"]
                           if getattr(parsed, f) is not None)
    field_bonus = fields_extracted * 0.05

    return min(1.0, avg_word_conf + field_bonus)
```

---

## 4. Pipeline CNN — Clasificación de Alimentos

### 4.1 Objetivo

Identificar los componentes alimenticios presentes en una fotografía de un plato de comida usando una red neuronal convolucional.

### 4.2 Modelo Base: MobileNetV2 (Transfer Learning)

**Arquitectura elegida:**

```
Imagen 224×224×3
        │
   MobileNetV2 (pesos ImageNet, capas base congeladas)
        │
   GlobalAveragePooling2D
        │
   Dense(256, activation='relu')
        │
   Dropout(0.3)
        │
   Dense(N_CLASES, activation='softmax')
        │
   Probabilidades por clase alimenticia
```

**Justificación:**
- Parámetros: ~3.4 M (vs ~25 M de ResNet50) → inferencia más rápida
- Accuracy competitivo en Food-101 dataset (~70-80% top-1 con fine-tuning)
- Modelo base disponible en `tf.keras.applications`

### 4.3 Dataset de Entrenamiento

Para el entorno académico se usará una combinación de:

| Dataset | Clases | Imágenes | Uso |
|---------|--------|----------|-----|
| [Food-101](https://www.kaggle.com/datasets/dansbecker/food-101) | 101 clases | 101,000 | Entrenamiento base |
| Subset personalizado | 20 clases latinoamericanas | ~2,000 | Fine-tuning |

**Clases prioritarias para VitalBite** (alimentos comunes en consultoría nutricional):

```python
FOOD_CLASSES = [
    "ensalada", "pollo_a_la_plancha", "arroz_blanco", "legumbres",
    "fruta_fresca", "pan_integral", "huevos", "pescado",
    "pasta", "vegetales_cocidos", "snacks_procesados",
    "bebidas_azucaradas", "comida_rapida", "lacteos",
    "nueces_semillas", "avena", "proteina_batido", "sopa",
    "sandwich", "pizza"
]
```

### 4.4 Script de Entrenamiento

**Archivo:** `app/models/training/train_food_classifier.py`

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def build_model(num_classes: int) -> Model:
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Fase 1: solo entrenar capas nuevas

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=base_model.input, outputs=outputs)

def train(dataset_path: str, epochs_frozen: int = 10, epochs_finetune: int = 5):
    model = build_model(num_classes=len(FOOD_CLASSES))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ... data loading, augmentation, fit ...

    # Fase 2: descongelar últimas 30 capas para fine-tuning
    model.layers[0].trainable = True
    for layer in model.layers[0].layers[:-30]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), ...)
    # ... fit fine-tuning ...

    model.save("app/models/artifacts/food_classifier.h5")
```

### 4.5 Servicio de Clasificación

```python
# app/services/food_classification_service.py
import numpy as np
import tensorflow as tf
from PIL import Image
from functools import lru_cache

@lru_cache(maxsize=1)
def _load_model():
    return tf.keras.models.load_model(settings.DL_MODEL_PATH)

def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.resize((224, 224)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

def classify_food(image: Image.Image) -> dict:
    model = _load_model()
    tensor = preprocess_image(image)
    predictions = model.predict(tensor, verbose=0)[0]

    top3_indices = predictions.argsort()[-3:][::-1]
    return {
        "top_predictions": [
            {
                "clase": FOOD_CLASSES[i],
                "probabilidad": float(predictions[i])
            }
            for i in top3_indices
        ],
        "confianza": float(predictions[top3_indices[0]])
    }
```

---

## 5. Analizador Nutricional (Cruce con Perfil del Paciente)

### 5.1 Obtención de alergias del paciente

El analizador llama internamente al Core NestJS para obtener las restricciones del paciente:

```python
# app/services/nutrition_analyzer_service.py
import httpx
from app.core.config import settings

async def get_patient_restrictions(patient_id: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.CORE_GRAPHQL_URL}",
            json={
                "query": """
                    query GetPatientRestrictions($id: ID!) {
                        patient(id: $id) {
                            alergias
                            restriccionesAlimentarias
                            condicionesClinicas
                        }
                    }
                """,
                "variables": {"id": patient_id}
            },
            headers={"Authorization": f"Bearer {settings.INTERNAL_TOKEN}"},
            timeout=5.0
        )
        data = response.json()
        return data.get("data", {}).get("patient", {})
```

### 5.2 Evaluación del semáforo de riesgo

```python
from enum import Enum

class Semaforo(str, Enum):
    SEGURO = "SEGURO"
    PRECAUCION = "PRECAUCION"
    RIESGO = "RIESGO"

def evaluate_semaforo(
    nutrients: NutrientData,
    food_classes: list[str],
    restrictions: dict
) -> tuple[Semaforo, list[str]]:
    advertencias = []

    # 1. Cruce de ingredientes con alergias
    alergias = [a.lower() for a in restrictions.get("alergias", [])]
    for ingrediente in (nutrients.ingredientes or []):
        if any(alergia in ingrediente.lower() for alergia in alergias):
            advertencias.append(f"⚠️ Contiene alérgeno: {ingrediente}")

    # 2. Valores nutricionales fuera de rango clínico
    if nutrients.sodio and nutrients.sodio > 600:
        advertencias.append("🧂 Alto contenido de sodio (>600 mg por porción)")
    if nutrients.azucares and nutrients.azucares > 25:
        advertencias.append("🍬 Alto contenido de azúcares (>25 g por porción)")
    if nutrients.grasas_saturadas and nutrients.grasas_saturadas > 10:
        advertencias.append("🛑 Grasas saturadas elevadas (>10 g por porción)")

    # 3. Categorías de alimento de riesgo
    riesgo_clases = {"comida_rapida", "snacks_procesados", "bebidas_azucaradas"}
    if any(c in riesgo_clases for c in food_classes):
        advertencias.append("⚡ Categoría de alimento de alto procesamiento")

    # Determinación final del semáforo
    if any("alérgeno" in a for a in advertencias):
        return Semaforo.RIESGO, advertencias
    elif len(advertencias) >= 2:
        return Semaforo.PRECAUCION, advertencias
    elif len(advertencias) == 1:
        return Semaforo.PRECAUCION, advertencias
    else:
        return Semaforo.SEGURO, advertencias
```

---

## 6. Schema de Request/Response

### 6.1 Request — `FoodScanRequest`

```python
from pydantic import BaseModel, Field
from enum import Enum

class ScanMode(str, Enum):
    LABEL = "label"   # etiqueta nutricional → OCR
    PLATE = "plate"   # plato de comida → CNN

class FoodScanRequest(BaseModel):
    patient_id: str = Field(..., description="UUID del paciente autenticado")
    mode: ScanMode = Field(..., description="Modo de escaneo: label o plate")
    # La imagen se recibe como multipart/form-data (campo 'image' en el form)
```

### 6.2 Response — `FoodScanResponse`

```python
class NutrientInfo(BaseModel):
    calorias: float | None = None
    carbohidratos_g: float | None = None
    proteinas_g: float | None = None
    grasas_totales_g: float | None = None
    grasas_saturadas_g: float | None = None
    sodio_mg: float | None = None
    azucares_g: float | None = None
    fibra_g: float | None = None
    ingredientes: list[str] = []

class FoodPrediction(BaseModel):
    clase: str
    probabilidad: float

class FoodScanResponse(BaseModel):
    modo: ScanMode
    semaforo: str                          # "SEGURO" | "PRECAUCION" | "RIESGO"
    advertencias: list[str]                # Lista de mensajes de alerta
    nutrientes: NutrientInfo               # Datos nutricionales extraídos
    predicciones_alimento: list[FoodPrediction] = []  # Solo para modo "plate"
    confianza: float                       # 0.0 - 1.0
    requiere_retoma: bool                  # True si confianza < umbral mínimo
    mensaje_retoma: str | None = None      # Instrucción si imagen es borrosa
```

---

## 7. Endpoint Completo

```python
# app/api/v1/endpoints/food_scan.py
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from PIL import Image
import io

from app.schemas.food_scan import FoodScanRequest, FoodScanResponse, ScanMode
from app.services.ocr_service import extract_nutritional_data
from app.services.food_classification_service import classify_food
from app.services.nutrition_analyzer_service import analyze_and_evaluate
from app.core.security import verify_api_key
from app.core.config import settings

router = APIRouter()

@router.post(
    "",
    response_model=FoodScanResponse,
    summary="Escanear alimento con OCR o Deep Learning",
    dependencies=[Depends(verify_api_key)]
)
async def scan_food(
    patient_id: str = Form(...),
    mode: ScanMode = Form(...),
    image: UploadFile = File(...),
):
    # Validar imagen
    content = await image.read()
    if len(content) > settings.IMAGE_MAX_SIZE_BYTES:
        raise HTTPException(413, "Imagen supera el tamaño máximo permitido (5 MB)")

    try:
        pil_image = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(422, "Formato de imagen no soportado. Use JPEG, PNG o WEBP")

    # Ejecutar pipeline según modo
    result = await analyze_and_evaluate(
        image=pil_image,
        mode=mode,
        patient_id=patient_id,
    )

    # Verificar umbral de confianza
    if result.confianza < settings.OCR_MIN_CONFIDENCE:
        result.requiere_retoma = True
        result.mensaje_retoma = (
            "La imagen no tiene suficiente nitidez o iluminación. "
            "Por favor, tome otra foto con mejor luz y enfoque."
        )

    return result
```

---

## 8. Consideraciones de Rendimiento

| Operación | Tiempo estimado | Estrategia de optimización |
|-----------|-----------------|---------------------------|
| Carga modelo CNN (primera vez) | 3-8 s | Precargar en evento `startup` |
| Inferencia CNN (224×224) | 80-200 ms | Usar batch_size=1, GPU opcional |
| OCR con EasyOCR | 500ms - 2s | Preprocesamiento reduce área de búsqueda |
| Llamada a Core NestJS (alergias) | 50-200 ms | Cache por `patient_id` (TTL 5 min) |
| Total pipeline completo | < 3 s | Dentro del timeout móvil aceptable |

---

## 9. Tests

**Archivo:** `tests/test_food_scan.py`

Casos de prueba a implementar:

```python
# tests/test_food_scan.py
import pytest
from httpx import AsyncClient

# Test 1: Imagen de etiqueta nutricional real → debe extraer calorías
# Test 2: Imagen borrosa → debe retornar requiere_retoma=True
# Test 3: Imagen con alérgeno conocido del paciente → semáforo RIESGO
# Test 4: Imagen limpia de ensalada → semáforo SEGURO
# Test 5: Modo plate sin CNN model cargado → error 503 descriptivo
# Test 6: Imagen > 5 MB → error 413
# Test 7: Formato inválido (PDF) → error 422
```
