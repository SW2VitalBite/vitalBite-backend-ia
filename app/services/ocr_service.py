"""Servicio OCR para etiquetas nutricionales (CU9, modo ``label``).

Pipeline: preprocesar imagen (CLAHE + umbralización adaptativa) → extraer texto
con EasyOCR → parsear nutrientes con regex → calcular confianza.

Las dependencias pesadas (``cv2``, ``easyocr``, ``numpy``) se importan de forma
perezosa dentro de las funciones para que el microservicio arranque aunque no
estén instaladas. Si faltan, se lanza :class:`ModelNotLoadedError` (→ 503).
"""

import re

from app.core.exceptions import ModelNotLoadedError
from app.core.logging import get_logger
from app.schemas.food_scan import NutrientInfo

logger = get_logger("ocr")

# Mapea el campo del schema → patrón regex de búsqueda en el texto OCR
PATTERNS = {
    "calorias": r"(?:cal(?:or[ií]as?)?|energ[ií]a|energy|kcal)[^\d]*(\d+(?:[.,]\d+)?)",
    "carbohidratos_g": r"(?:carbohidrat(?:os?)?|carbohydrate)[^\d]*(\d+(?:[.,]\d+)?)",
    "proteinas_g": r"(?:prote[íi]nas?|protein)[^\d]*(\d+(?:[.,]\d+)?)",
    "grasas_totales_g": r"(?:grasa[s]? total(?:es)?|total fat|l[íi]pidos)[^\d]*(\d+(?:[.,]\d+)?)",
    "grasas_saturadas_g": r"(?:grasa[s]? saturada[s]?|saturated fat)[^\d]*(\d+(?:[.,]\d+)?)",
    "sodio_mg": r"(?:sodio|sodium)[^\d]*(\d+(?:[.,]\d+)?)",
    "azucares_g": r"(?:az[úu]cares?|sugars?)[^\d]*(\d+(?:[.,]\d+)?)",
    "fibra_g": r"(?:fibra|fiber|fibre)[^\d]*(\d+(?:[.,]\d+)?)",
}

# Cache del lector EasyOCR (su inicialización es costosa)
_reader = None


def _get_reader():
    """Inicializa (una vez) el lector EasyOCR en español + inglés."""
    global _reader
    if _reader is None:
        try:
            import easyocr  # import perezoso
        except ImportError as exc:  # pragma: no cover
            raise ModelNotLoadedError(
                "easyocr", "instale easyocr para habilitar el OCR de etiquetas"
            ) from exc
        logger.info("inicializando lector EasyOCR (es, en)")
        # verbose=False evita la barra de progreso (carácter █) que rompe la
        # consola cp1252 de Windows y ensucia los logs JSON.
        _reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
    return _reader


def preprocess_for_ocr(image):
    """Mejora la legibilidad de la etiqueta antes del OCR (CLAHE + threshold)."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ModelNotLoadedError(
            "opencv", "instale opencv-python-headless para el preprocesamiento OCR"
        ) from exc

    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    threshold = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    denoised = cv2.fastNlMeansDenoising(threshold, h=10)
    return denoised


def extract_text(preprocessed_image) -> list[dict]:
    """Extrae bloques de texto con su confianza usando EasyOCR."""
    reader = _get_reader()
    results = reader.readtext(preprocessed_image, detail=1)
    return [
        {"text": text, "confidence": float(conf)}
        for (_, text, conf) in results
        if conf > 0.40
    ]


def parse_nutrients(text_blocks: list[dict]) -> NutrientInfo:
    """Parsea el texto OCR a un :class:`NutrientInfo` con regex por nutriente."""
    full_text = " ".join(b["text"].lower() for b in text_blocks)
    values: dict = {}

    for field, pattern in PATTERNS.items():
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            values[field] = float(match.group(1).replace(",", "."))

    ingredientes: list[str] = []
    ing_match = re.search(r"ingredientes?[:\s]+(.+?)(?:\.|$)", full_text, re.IGNORECASE)
    if ing_match:
        ingredientes = [i.strip() for i in ing_match.group(1).split(",") if i.strip()]

    return NutrientInfo(ingredientes=ingredientes, **values)


def calculate_ocr_confidence(text_blocks: list[dict], parsed: NutrientInfo) -> float:
    """Confianza = promedio de palabras + bonus por campos clave extraídos."""
    if not text_blocks:
        return 0.0
    avg_word_conf = sum(b["confidence"] for b in text_blocks) / len(text_blocks)
    key_fields = ["calorias", "proteinas_g", "carbohidratos_g", "grasas_totales_g"]
    fields_extracted = sum(1 for f in key_fields if getattr(parsed, f) is not None)
    field_bonus = fields_extracted * 0.05
    return min(1.0, avg_word_conf + field_bonus)


def extract_nutritional_data(image) -> tuple[NutrientInfo, float]:
    """Pipeline OCR completo: imagen PIL → (NutrientInfo, confianza)."""
    preprocessed = preprocess_for_ocr(image)
    blocks = extract_text(preprocessed)
    nutrients = parse_nutrients(blocks)
    confidence = calculate_ocr_confidence(blocks, nutrients)
    logger.info(
        "ocr completado",
        extra={"bloques": len(blocks), "confianza": round(confidence, 3)},
    )
    return nutrients, confidence
