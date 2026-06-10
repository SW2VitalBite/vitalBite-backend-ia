"""Servicio OCR para etiquetas nutricionales (CU9, modo ``label``).

Pipeline: preprocesar imagen (escalado + CLAHE) → extraer texto **con su
geometría** usando EasyOCR → agrupar los bloques en filas visuales → parsear los
nutrientes por fila con regex → calcular confianza.

Clave de precisión: las etiquetas nutricionales son **tablas**. La etiqueta
("Sodio") y su valor ("120 mg") casi siempre están en la misma fila visual pero
en bloques OCR distintos. Por eso NO aplanamos todo el texto en una sola cadena
(eso mezcla el número de un nutriente con la etiqueta de otro): primero
reconstruimos las filas a partir de las coordenadas ``y`` de cada bloque y
buscamos cada nutriente dentro de su fila. El texto completo queda solo como
respaldo.

Las dependencias pesadas (``cv2``, ``easyocr``, ``numpy``) se importan de forma
perezosa dentro de las funciones para que el microservicio arranque aunque no
estén instaladas. Si faltan, se lanza :class:`ModelNotLoadedError` (→ 503).
"""

import asyncio
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from app.core.exceptions import ModelNotLoadedError
from app.core.logging import get_logger
from app.schemas.food_scan import NutrientInfo
from app.services.ocr_worker import run_ocr_blocks

logger = get_logger("ocr")

# Mapea el campo del schema → patrón regex (palabra clave + primer número).
# Importa el orden: ``grasas_saturadas_g`` se evalúa antes que ``grasas_totales_g``
# para que "Grasas saturadas" no sea capturado como grasa total.
_NUM = r"(\d+(?:[.,]\d+)?)"
PATTERNS = {
    "calorias": r"(?:valor\s+energ[eé]tico|energ[ií]a|calor[ií]as?|energy|kcal)[^\d]*" + _NUM,
    "carbohidratos_g": r"(?:carbohidratos?(?:\s+totales)?|hidratos\s+de\s+carbono|carbohydrate)[^\d]*" + _NUM,
    "proteinas_g": r"(?:prote[íi]nas?|protein)[^\d]*" + _NUM,
    "grasas_saturadas_g": r"(?:saturad\w*|saturated)[^\d]*" + _NUM,
    "grasas_totales_g": r"(?:grasas?(?:\s*totales)?|grasa\s+total|total\s+fat|l[íi]pidos)[^\d]*" + _NUM,
    "sodio_mg": r"(?:sodio|sodium)[^\d]*" + _NUM,
    "azucares_g": r"(?:az[úu]cares?|sugars?)[^\d]*" + _NUM,
    "fibra_g": r"(?:fibra|fiber|fibre)[^\d]*" + _NUM,
}

# ── Aislamiento del OCR en un subproceso ────────────────────────────────────
# EasyOCR (PyTorch) y TensorFlow no pueden convivir en el mismo proceso (SIGSEGV,
# ver app/services/ocr_worker.py). Ejecutamos el OCR en un pool de UN proceso
# hijo lanzado con ``spawn`` (intérprete limpio, sin TF). El worker queda vivo y
# reutiliza el lector EasyOCR entre peticiones.
_executor: ProcessPoolExecutor | None = None


def _get_executor() -> ProcessPoolExecutor:
    global _executor
    if _executor is None:
        # ``spawn`` (no ``fork``): el hijo NO hereda la memoria del padre, así no
        # arrastra TensorFlow. max_workers=1 serializa el OCR (1 inferencia a la
        # vez) y mantiene un único worker caliente con el modelo cargado.
        ctx = multiprocessing.get_context("spawn")
        _executor = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
    return _executor


def _reset_executor() -> None:
    """Descarta el pool tras un fallo del worker para recrearlo en la próxima."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None


def _group_into_rows(blocks: list[dict]) -> list[str]:
    """Reconstruye las filas visuales agrupando bloques por su centro ``y``.

    Devuelve el texto de cada fila ordenado de izquierda a derecha. Así "Sodio"
    y "120 mg" quedan juntos en la misma cadena y el regex los empareja bien.
    """
    if not blocks:
        return []

    heights = sorted(b["h"] for b in blocks if b["h"] > 0)
    median_h = heights[len(heights) // 2] if heights else 10.0
    tol = max(8.0, median_h * 0.6)  # tolerancia vertical para "misma fila"

    rows: list[dict] = []
    for b in sorted(blocks, key=lambda b: b["cy"]):
        target = next((r for r in rows if abs(b["cy"] - r["cy"]) <= tol), None)
        if target is None:
            rows.append({"cy": b["cy"], "blocks": [b]})
        else:
            target["blocks"].append(b)
            target["cy"] = sum(x["cy"] for x in target["blocks"]) / len(target["blocks"])

    row_texts: list[str] = []
    for row in rows:
        ordered = sorted(row["blocks"], key=lambda b: b["x0"])
        row_texts.append(" ".join(b["text"] for b in ordered))
    return row_texts


def parse_nutrients(blocks: list[dict]) -> NutrientInfo:
    """Parsea los nutrientes usando primero las filas y luego el texto completo."""
    rows = _group_into_rows(blocks)
    full_text = " ".join(b["text"] for b in blocks)
    values: dict = {}

    for field, pattern in PATTERNS.items():
        # Para grasa total ignoramos las filas de saturadas, que comparten la
        # palabra "grasa" y, si no, robarían el valor.
        candidate_rows = (
            [r for r in rows if "satur" not in r.lower()]
            if field == "grasas_totales_g"
            else rows
        )

        # Paso 1: buscar dentro de cada fila visual (empareja etiqueta↔valor).
        for row in candidate_rows:
            match = re.search(pattern, row, re.IGNORECASE)
            if match:
                values[field] = float(match.group(1).replace(",", "."))
                break

        # Paso 2 (respaldo): buscar en todo el texto si la fila no lo encontró.
        if field not in values:
            haystack = full_text
            if field == "grasas_totales_g":
                # Evita capturar el número de "grasas saturadas" en el blob.
                haystack = re.sub(
                    r"grasas?\s*saturad\w*[^a-z0-9]*\d+(?:[.,]\d+)?",
                    " ",
                    full_text,
                    flags=re.IGNORECASE,
                )
            match = re.search(pattern, haystack, re.IGNORECASE)
            if match:
                values[field] = float(match.group(1).replace(",", "."))

    # Caso "250 kcal" (el número va antes de la unidad): respaldo dedicado.
    if "calorias" not in values:
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*kcal", full_text, re.IGNORECASE)
        if match:
            values["calorias"] = float(match.group(1).replace(",", "."))

    # Ingredientes: desde "ingredientes:" hasta el siguiente punto / fin.
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


async def extract_nutritional_data(image) -> tuple[NutrientInfo, float]:
    """Pipeline OCR completo: imagen PIL → (NutrientInfo, confianza).

    El preprocesado + EasyOCR corren en el subproceso aislado; el agrupamiento en
    filas y el parseo (regex) se hacen aquí en el proceso padre.
    """
    import numpy as np

    rgb_array = np.asarray(image.convert("RGB"))
    loop = asyncio.get_event_loop()
    try:
        blocks = await loop.run_in_executor(_get_executor(), run_ocr_blocks, rgb_array)
    except BrokenProcessPool as exc:
        # El worker murió (p. ej. SIGSEGV). Se recrea el pool para la próxima y
        # se devuelve 503 sin tumbar el proceso principal (plato/RF/health siguen).
        _reset_executor()
        logger.error("el worker de OCR terminó inesperadamente", extra={"error": str(exc)})
        raise ModelNotLoadedError(
            "easyocr", "el worker de OCR terminó inesperadamente; reintente"
        ) from exc

    nutrients = parse_nutrients(blocks)
    confidence = calculate_ocr_confidence(blocks, nutrients)
    logger.info(
        "ocr completado",
        extra={"bloques": len(blocks), "confianza": round(confidence, 3)},
    )
    return nutrients, confidence
