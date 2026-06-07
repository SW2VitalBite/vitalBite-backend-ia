"""Servicio de clasificación de platos con CNN MobileNetV2 (CU9, modo ``plate``).

Carga el modelo Keras serializado de forma perezosa y cacheada, y expone
:func:`classify_food` que devuelve el top-3 de categorías de alimento.

Las clases NO están hardcodeadas: se leen del artefacto ``food_class_names.json``
generado por el script de entrenamiento (orden = índice de salida del modelo).
Esto permite entrenar con cualquier dataset (p. ej. Food-101, 101 clases) sin
tocar el código. Si el JSON no existe se usa una lista por defecto de respaldo.

TensorFlow es una dependencia opcional y pesada: si no está instalada o el
artefacto no existe, se lanza :class:`ModelNotLoadedError` (→ 503) en lugar de
romper el arranque del microservicio. ``preload_cnn_model`` se invoca en el
``startup`` y nunca propaga la excepción (solo registra una advertencia).
"""

import json
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import ModelNotLoadedError
from app.core.logging import get_logger
from app.schemas.food_scan import FoodPrediction

logger = get_logger("food_cnn")

# Lista de respaldo (solo si no existe food_class_names.json junto al modelo)
DEFAULT_FOOD_CLASSES = [
    "ensalada",
    "pollo_a_la_plancha",
    "arroz_blanco",
    "legumbres",
    "fruta_fresca",
    "pan_integral",
    "huevos",
    "pescado",
    "pasta",
    "vegetales_cocidos",
    "snacks_procesados",
    "bebidas_azucaradas",
    "comida_rapida",
    "lacteos",
    "nueces_semillas",
    "avena",
    "proteina_batido",
    "sopa",
    "sandwich",
    "pizza",
]

# Categorías de Food-101 consideradas de alto procesamiento / riesgo nutricional
# (fritos, ultraprocesados, repostería y azúcares). Se cruzan con las clases
# predichas para alimentar el semáforo del analizador nutricional.
RISK_FOOD_CLASSES = {
    # genéricas (respaldo)
    "comida_rapida",
    "snacks_procesados",
    "bebidas_azucaradas",
    # Food-101
    "pizza",
    "hamburger",
    "hot_dog",
    "french_fries",
    "onion_rings",
    "nachos",
    "fried_calamari",
    "chicken_wings",
    "churros",
    "donuts",
    "ice_cream",
    "chocolate_cake",
    "chocolate_mousse",
    "cheesecake",
    "red_velvet_cake",
    "carrot_cake",
    "cup_cakes",
    "strawberry_shortcake",
    "tiramisu",
    "baklava",
    "beignets",
    "waffles",
    "pancakes",
    "french_toast",
    "macaroni_and_cheese",
    "poutine",
    "grilled_cheese_sandwich",
    "club_sandwich",
    "lobster_roll_sandwich",
    "pulled_pork_sandwich",
    "apple_pie",
    "bread_pudding",
}

# Nombre del artefacto con los nombres de clase (junto al modelo)
CLASS_NAMES_FILE = "food_class_names.json"

_model = None
_class_names: list[str] | None = None


def _class_names_path() -> Path:
    return Path(settings.DL_MODEL_PATH).parent / CLASS_NAMES_FILE


def get_class_names() -> list[str]:
    """Nombres de clase en el orden de salida del modelo (cacheado)."""
    global _class_names
    if _class_names is None:
        path = _class_names_path()
        if path.exists():
            _class_names = json.loads(path.read_text(encoding="utf-8"))
            logger.info(
                "clases del clasificador cargadas",
                extra={"n_clases": len(_class_names), "path": str(path)},
            )
        else:
            _class_names = DEFAULT_FOOD_CLASSES
            logger.warning(
                "food_class_names.json ausente; usando lista por defecto",
                extra={"n_clases": len(_class_names)},
            )
    return _class_names


def is_available() -> bool:
    """True si TensorFlow está instalado y el artefacto del modelo existe."""
    if not Path(settings.DL_MODEL_PATH).exists():
        return False
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return False
    return True


def _load_model():
    global _model
    if _model is not None:
        return _model

    path = Path(settings.DL_MODEL_PATH)
    if not path.exists():
        raise ModelNotLoadedError(
            str(path), "entrene la CNN con train_food_classifier o monte el artefacto"
        )
    try:
        import tensorflow as tf  # import perezoso
    except ImportError as exc:  # pragma: no cover
        raise ModelNotLoadedError(
            str(path), "instale tensorflow para habilitar la clasificación de platos"
        ) from exc

    logger.info("cargando modelo CNN", extra={"path": str(path)})
    _model = tf.keras.models.load_model(str(path))
    get_class_names()  # precarga/valida las clases
    return _model


def preprocess_image(image):
    """Imagen PIL → tensor (1, 224, 224, 3) normalizado [0,1]."""
    import numpy as np

    img = image.resize((224, 224)).convert("RGB")
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def classify_food(image) -> tuple[list[FoodPrediction], float]:
    """Clasifica un plato y devuelve (top-3 predicciones, confianza top-1)."""
    model = _load_model()
    class_names = get_class_names()
    tensor = preprocess_image(image)
    predictions = model.predict(tensor, verbose=0)[0]

    top3 = predictions.argsort()[-3:][::-1]
    preds = [
        FoodPrediction(clase=class_names[i], probabilidad=float(predictions[i]))
        for i in top3
    ]
    confidence = float(predictions[top3[0]])
    return preds, confidence


def preload_cnn_model() -> bool:
    """Precarga el modelo en el arranque para evitar cold-start. No propaga errores."""
    try:
        _load_model()
        logger.info("modelo CNN precargado correctamente")
        return True
    except ModelNotLoadedError as exc:
        logger.warning(
            "CNN no precargada (CU9/plate quedará en 503 hasta proveer el artefacto)",
            extra={"motivo": str(exc)},
        )
        return False
