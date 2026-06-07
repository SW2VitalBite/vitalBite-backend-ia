"""Servicio de inferencia del Random Forest - Prediccion de Riesgo (CU10).

Carga el modelo y el scaler, construye el vector de features en el orden de
entrenamiento, ejecuta ``predict_proba`` y traduce el resultado a un nivel de
riesgo textual con factores clinicos activos y recomendacion.
"""

import numpy as np

from app.core.exceptions import InsufficientDataError, ModelNotLoadedError
from app.schemas.common import RISK_LABELS
from app.schemas.risk_prediction import RiskPredictionRequest, RiskPredictionResponse
from app.services.model_loader import get_rf_model, get_rf_scaler

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


def _add_factor(
    factors: list[tuple[str, float]],
    description: str,
    severity: float,
) -> None:
    if severity > 0:
        factors.append((description, severity))


def _identify_critical_factors(features: dict) -> list[str]:
    """Devuelve solo factores justificados por valores reales del paciente."""
    factors: list[tuple[str, float]] = []

    imc = features.get("imc")
    if imc is not None:
        _add_factor(
            factors,
            "Indice de Masa Corporal elevado",
            max(0.0, (float(imc) - 25) / 10),
        )
        _add_factor(
            factors,
            "Indice de Masa Corporal por debajo del rango saludable",
            max(0.0, (18.5 - float(imc)) / 4),
        )

    porcentaje_grasa = features.get("porcentaje_grasa")
    sexo = features.get("sexo")
    if porcentaje_grasa is not None:
        threshold = 32 if sexo == 0 else 25
        _add_factor(
            factors,
            "Porcentaje de grasa corporal fuera de rango",
            max(0.0, (float(porcentaje_grasa) - threshold) / 10),
        )

    variacion_peso = features.get("variacion_peso_3m_kg")
    if variacion_peso is not None:
        _add_factor(
            factors,
            "Variacion de peso significativa en los ultimos 3 meses",
            max(0.0, (abs(float(variacion_peso)) - 3) / 5),
        )

    comorbilidades = features.get("num_comorbilidades")
    if comorbilidades is not None:
        _add_factor(
            factors,
            "Presencia de comorbilidades registradas",
            float(comorbilidades) / 2,
        )

    calidad_dieta = features.get("calidad_dieta_score")
    if calidad_dieta is not None:
        _add_factor(
            factors,
            "Calidad de la dieta por debajo del promedio",
            max(0.0, (6 - float(calidad_dieta)) / 3),
        )

    actividad = features.get("nivel_actividad")
    if actividad is not None:
        _add_factor(
            factors,
            "Nivel de actividad fisica insuficiente",
            max(0.0, (2 - float(actividad)) / 2),
        )

    edad = features.get("edad")
    if edad is not None:
        _add_factor(
            factors,
            "Edad como factor de seguimiento preventivo",
            max(0.0, (float(edad) - 60) / 20),
        )

    ranked = sorted(factors, key=lambda item: item[1], reverse=True)
    active = [description for description, _ in ranked[:3]]
    return active or ["Perfil clinico general del paciente"]


def _generate_recommendation(level: int) -> str:
    base = {
        0: "Mantener habitos actuales y continuar monitoreo preventivo trimestral.",
        1: "Se recomienda ajuste de plan dietetico y seguimiento mensual de metricas.",
        2: (
            "Atencion prioritaria requerida. Revisar plan de alimentacion y "
            "considerar interconsulta medica."
        ),
    }
    return base[level]


class RiskPredictionService:
    MIN_REQUIRED_FEATURES = 7

    def predict(self, request: RiskPredictionRequest) -> RiskPredictionResponse:
        features_dict = request.features.model_dump()

        non_null = sum(1 for v in features_dict.values() if v is not None)
        if non_null < self.MIN_REQUIRED_FEATURES:
            missing = [k for k, v in features_dict.items() if v is None]
            raise InsufficientDataError(missing_fields=missing)

        if (
            features_dict.get("imc") is None
            and features_dict.get("peso_kg")
            and features_dict.get("talla_m")
        ):
            features_dict["imc"] = features_dict["peso_kg"] / (
                features_dict["talla_m"] ** 2
            )

        vector = np.array(
            [features_dict.get(f) or 0.0 for f in FEATURES_ORDER],
            dtype=np.float64,
        ).reshape(1, -1)

        model = get_rf_model()
        scaler = get_rf_scaler()

        vector_scaled = scaler.transform(vector)
        probabilities = model.predict_proba(vector_scaled)[0]
        predicted_class = int(np.argmax(probabilities))

        factores = _identify_critical_factors(features_dict)

        classes = list(getattr(model, "classes_", [0, 1, 2]))
        prob_map = {
            RISK_LABELS[int(c)]: float(probabilities[i])
            for i, c in enumerate(classes)
        }
        for label in RISK_LABELS.values():
            prob_map.setdefault(label, 0.0)

        return RiskPredictionResponse(
            nivel_riesgo=RISK_LABELS[predicted_class],
            probabilidad=float(probabilities[predicted_class]),
            probabilidades=prob_map,
            factores_criticos=factores,
            recomendacion=_generate_recommendation(predicted_class),
        )


__all__ = ["RiskPredictionService", "InsufficientDataError", "ModelNotLoadedError"]
