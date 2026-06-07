"""Servicio de inferencia del Random Forest — Predicción de Riesgo (CU10).

Carga el modelo y el scaler (vía ``model_loader``), construye el vector de
features en el orden de entrenamiento, ejecuta ``predict_proba`` y traduce el
resultado a un nivel de riesgo textual con factores críticos y recomendación.
"""

import numpy as np

from app.core.exceptions import InsufficientDataError, ModelNotLoadedError
from app.schemas.common import RISK_LABELS
from app.schemas.risk_prediction import RiskPredictionRequest, RiskPredictionResponse
from app.services.model_loader import get_rf_importances, get_rf_model, get_rf_scaler

# Orden EXACTO de features con el que se entrenó el modelo
FEATURES_ORDER = [
    "edad",
    "sexo",
    "peso_kg",
    "talla_m",
    "imc",
    "variacion_peso_3m_kg",
    "porcentaje_grasa",
    "nivel_actividad",
    "calidad_dieta_score",
    "num_comorbilidades",
]

FEATURE_DESCRIPTIONS = {
    "imc": "Índice de Masa Corporal elevado",
    "porcentaje_grasa": "Porcentaje de grasa corporal fuera de rango",
    "variacion_peso_3m_kg": "Variación de peso significativa en los últimos 3 meses",
    "num_comorbilidades": "Presencia de comorbilidades registradas",
    "calidad_dieta_score": "Calidad de la dieta por debajo del promedio",
    "nivel_actividad": "Nivel de actividad física insuficiente",
    "edad": "Edad como factor de riesgo",
    "peso_kg": "Peso corporal fuera de rango saludable",
}


def _identify_critical_factors(features: dict, importances: dict) -> list[str]:
    """Top 3 factores legibles según la importancia global del modelo.

    Garantiza siempre al menos un factor para satisfacer la exigencia clínica de
    explicabilidad.
    """
    factors: list[str] = []
    ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    for feat_name, _ in ranked:
        desc = FEATURE_DESCRIPTIONS.get(feat_name)
        if desc and feat_name in features and desc not in factors:
            factors.append(desc)
        if len(factors) == 3:
            break

    if not factors:
        factors.append("Perfil clínico general del paciente")
    return factors


def _generate_recommendation(level: int) -> str:
    base = {
        0: "Mantener hábitos actuales y continuar monitoreo preventivo trimestral.",
        1: "Se recomienda ajuste de plan dietético y seguimiento mensual de métricas.",
        2: (
            "Atención prioritaria requerida. Revisar plan de alimentación y "
            "considerar interconsulta médica."
        ),
    }
    return base[level]


class RiskPredictionService:
    MIN_REQUIRED_FEATURES = 7  # Mínimo de features no nulas para habilitar predicción

    def predict(self, request: RiskPredictionRequest) -> RiskPredictionResponse:
        features_dict = request.features.model_dump()

        # Validación de datos mínimos
        non_null = sum(1 for v in features_dict.values() if v is not None)
        if non_null < self.MIN_REQUIRED_FEATURES:
            missing = [k for k, v in features_dict.items() if v is None]
            raise InsufficientDataError(missing_fields=missing)

        # Calcular IMC si no viene provisto pero hay peso y talla
        if (
            features_dict.get("imc") is None
            and features_dict.get("peso_kg")
            and features_dict.get("talla_m")
        ):
            features_dict["imc"] = features_dict["peso_kg"] / (
                features_dict["talla_m"] ** 2
            )

        # Vector en el orden del entrenamiento (los None se imputan a 0.0)
        vector = np.array(
            [features_dict.get(f) or 0.0 for f in FEATURES_ORDER],
            dtype=np.float64,
        ).reshape(1, -1)

        # ModelNotLoadedError se propaga y lo mapea el handler global a 503
        model = get_rf_model()
        scaler = get_rf_scaler()
        importances = get_rf_importances()

        vector_scaled = scaler.transform(vector)
        probabilities = model.predict_proba(vector_scaled)[0]
        predicted_class = int(np.argmax(probabilities))

        factores = _identify_critical_factors(features_dict, importances)

        # Mapear probabilidades a etiquetas según las clases conocidas del modelo
        classes = list(getattr(model, "classes_", [0, 1, 2]))
        prob_map = {RISK_LABELS[int(c)]: float(probabilities[i]) for i, c in enumerate(classes)}
        for label in RISK_LABELS.values():
            prob_map.setdefault(label, 0.0)

        return RiskPredictionResponse(
            nivel_riesgo=RISK_LABELS[predicted_class],
            probabilidad=float(probabilities[predicted_class]),
            probabilidades=prob_map,
            factores_criticos=factores,
            recomendacion=_generate_recommendation(predicted_class),
        )


# Re-exportar para compatibilidad con imports desde el endpoint
__all__ = ["RiskPredictionService", "InsufficientDataError", "ModelNotLoadedError"]
