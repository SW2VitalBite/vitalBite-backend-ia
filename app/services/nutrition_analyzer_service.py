"""Analizador nutricional — orquesta el pipeline CU9 y construye el semáforo.

Responsabilidades:
1. Ejecutar el pipeline correspondiente al modo (OCR para ``label``, CNN para
   ``plate``).
2. Emitir un semáforo (SEGURO / PRECAUCION / RIESGO) a partir de los valores
   nutricionales y la categoría del alimento.

La personalización según el plan de dieta del paciente (alineación de calorías,
sugerencias) se resuelve en la app móvil, que consulta el Core GraphQL con el
JWT del propio paciente. Por eso este servicio ya no llama al Core: el semáforo
es un análisis genérico que no requiere datos del paciente.
"""

from app.schemas.common import ScanMode, Semaforo
from app.schemas.food_scan import FoodScanResponse, NutrientInfo
from app.services import food_classification_service, ocr_service


def evaluate_semaforo(
    nutrients: NutrientInfo,
    food_classes: list[str],
) -> tuple[Semaforo, list[str]]:
    """Determina el semáforo a partir de valores nutricionales y categoría.

    Sin datos del paciente: aplica umbrales clínicos genéricos y detecta
    categorías de alto procesamiento. El número de banderas rojas define el
    nivel: 0 → SEGURO, 1 → PRECAUCION, ≥2 → RIESGO.
    """
    advertencias: list[str] = []

    # 1. Valores nutricionales fuera de rango clínico
    if nutrients.sodio_mg and nutrients.sodio_mg > 600:
        advertencias.append("🧂 Alto contenido de sodio (>600 mg por porción)")
    if nutrients.azucares_g and nutrients.azucares_g > 25:
        advertencias.append("🍬 Alto contenido de azúcares (>25 g por porción)")
    if nutrients.grasas_saturadas_g and nutrients.grasas_saturadas_g > 10:
        advertencias.append("🛑 Grasas saturadas elevadas (>10 g por porción)")

    # 2. Categorías de alimento de alto procesamiento
    if any(c in food_classification_service.RISK_FOOD_CLASSES for c in food_classes):
        advertencias.append("⚡ Categoría de alimento de alto procesamiento")

    # Determinación final por número de banderas rojas
    if len(advertencias) >= 2:
        return Semaforo.RIESGO, advertencias
    if len(advertencias) == 1:
        return Semaforo.PRECAUCION, advertencias
    return Semaforo.SEGURO, advertencias


async def analyze_and_evaluate(
    image,
    mode: ScanMode,
    patient_id: str,  # conservado por compatibilidad del endpoint; ya no se usa aquí
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

    semaforo, advertencias = evaluate_semaforo(nutrients, food_classes)

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
