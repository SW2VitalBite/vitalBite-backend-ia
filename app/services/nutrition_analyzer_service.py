"""Analizador nutricional — orquesta el pipeline CU9 y construye el semáforo.

Responsabilidades:
1. Obtener las restricciones/alergias del paciente desde el Core NestJS (GraphQL).
2. Ejecutar el pipeline correspondiente al modo (OCR para ``label``, CNN para
   ``plate``).
3. Cruzar el resultado con las alergias y valores nutricionales para emitir un
   semáforo (SEGURO / PRECAUCION / RIESGO) con advertencias personalizadas.

La llamada al Core es tolerante a fallos: si no responde, se continúa con un
perfil de restricciones vacío y se registra una advertencia (no se aborta el
escaneo, que es la funcionalidad principal para el paciente).
"""

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.common import ScanMode, Semaforo
from app.schemas.food_scan import FoodScanResponse, NutrientInfo
from app.services import food_classification_service, ocr_service

logger = get_logger("nutrition_analyzer")

_PATIENT_QUERY = """
query GetPatientRestrictions($id: ID!) {
    patient(id: $id) {
        alergias
        restriccionesAlimentarias
        condicionesClinicas
    }
}
"""


async def get_patient_restrictions(patient_id: str) -> dict:
    """Consulta alergias/restricciones del paciente. Devuelve {} si falla."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                settings.CORE_GRAPHQL_URL,
                json={"query": _PATIENT_QUERY, "variables": {"id": patient_id}},
                headers={"Authorization": f"Bearer {settings.INTERNAL_TOKEN}"},
            )
        data = response.json()
        return data.get("data", {}).get("patient") or {}
    except Exception as exc:  # red, timeout, JSON inválido, Core caído...
        logger.warning(
            "no se pudieron obtener restricciones del Core; se continúa sin ellas",
            extra={"patient_id": patient_id, "motivo": str(exc)},
        )
        return {}


def evaluate_semaforo(
    nutrients: NutrientInfo,
    food_classes: list[str],
    restrictions: dict,
) -> tuple[Semaforo, list[str]]:
    """Determina el semáforo cruzando nutrientes, categorías y alergias."""
    advertencias: list[str] = []

    # 1. Cruce de ingredientes con alergias del paciente
    alergias = [str(a).lower() for a in (restrictions.get("alergias") or [])]
    for ingrediente in nutrients.ingredientes or []:
        if any(alergia in ingrediente.lower() for alergia in alergias):
            advertencias.append(f"⚠️ Contiene alérgeno: {ingrediente}")

    # 2. Valores nutricionales fuera de rango clínico
    if nutrients.sodio_mg and nutrients.sodio_mg > 600:
        advertencias.append("🧂 Alto contenido de sodio (>600 mg por porción)")
    if nutrients.azucares_g and nutrients.azucares_g > 25:
        advertencias.append("🍬 Alto contenido de azúcares (>25 g por porción)")
    if nutrients.grasas_saturadas_g and nutrients.grasas_saturadas_g > 10:
        advertencias.append("🛑 Grasas saturadas elevadas (>10 g por porción)")

    # 3. Categorías de alimento de alto procesamiento
    if any(c in food_classification_service.RISK_FOOD_CLASSES for c in food_classes):
        advertencias.append("⚡ Categoría de alimento de alto procesamiento")

    # Determinación final
    if any("alérgeno" in a for a in advertencias):
        return Semaforo.RIESGO, advertencias
    if len(advertencias) >= 1:
        return Semaforo.PRECAUCION, advertencias
    return Semaforo.SEGURO, advertencias


async def analyze_and_evaluate(
    image,
    mode: ScanMode,
    patient_id: str,
) -> FoodScanResponse:
    """Pipeline completo CU9: routing por modo + análisis + semáforo."""
    nutrients = NutrientInfo()
    predicciones = []
    food_classes: list[str] = []

    if mode == ScanMode.LABEL:
        nutrients, confianza = ocr_service.extract_nutritional_data(image)
    else:  # ScanMode.PLATE
        predicciones, confianza = food_classification_service.classify_food(image)
        food_classes = [p.clase for p in predicciones]

    restrictions = await get_patient_restrictions(patient_id)
    semaforo, advertencias = evaluate_semaforo(nutrients, food_classes, restrictions)

    return FoodScanResponse(
        modo=mode,
        semaforo=semaforo.value,
        advertencias=advertencias,
        nutrientes=nutrients,
        predicciones_alimento=predicciones,
        confianza=confianza,
        requiere_retoma=False,
        mensaje_retoma=None,
    )
