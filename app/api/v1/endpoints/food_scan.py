"""Endpoint CU9 — POST /api/v1/food-scan (OCR + Deep Learning).

Recibe la imagen como ``multipart/form-data``. Valida tamaño/formato, ejecuta el
pipeline del analizador y aplica el umbral de confianza para sugerir retoma de
la foto. ``ModelNotLoadedError`` (TF/EasyOCR ausentes) → 503 vía handler global.
"""

import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image

from app.core.config import settings
from app.core.security import verify_api_key
from app.schemas.common import ScanMode
from app.schemas.food_scan import FoodScanResponse
from app.services.nutrition_analyzer_service import analyze_and_evaluate

router = APIRouter()


@router.post(
    "",
    response_model=FoodScanResponse,
    summary="Escanear alimento con OCR (etiqueta) o Deep Learning (plato)",
    dependencies=[Depends(verify_api_key)],
)
async def scan_food(
    patient_id: str = Form(..., description="UUID del paciente autenticado"),
    mode: ScanMode = Form(..., description="label = etiqueta (OCR) | plate = plato (CNN)"),
    image: UploadFile = File(..., description="Imagen JPEG/PNG/WEBP (< 5 MB)"),
) -> FoodScanResponse:
    content = await image.read()

    if len(content) > settings.IMAGE_MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Imagen supera el tamaño máximo permitido ({settings.IMAGE_MAX_SIZE_MB} MB).",
        )

    try:
        pil_image = Image.open(io.BytesIO(content))
        pil_image.load()
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Formato de imagen no soportado. Use JPEG, PNG o WEBP.",
        )

    result = await analyze_and_evaluate(image=pil_image, mode=mode, patient_id=patient_id)

    # Umbral de confianza: imagen borrosa / mal iluminada → sugerir retoma
    if result.confianza < settings.OCR_MIN_CONFIDENCE:
        result.requiere_retoma = True
        result.mensaje_retoma = (
            "La imagen no tiene suficiente nitidez o iluminación. "
            "Por favor, tome otra foto con mejor luz y enfoque."
        )

    return result
