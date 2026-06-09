"""Tests del predictor de riesgo Random Forest (CU10).

Requieren los artefactos entrenados en app/models/artifacts/. Si no existen, se
omiten los tests que llaman al modelo (se marca skip), salvo los de validación
de entrada que no dependen del modelo.
"""

import pytest

from app.services.risk_prediction_service import _identify_critical_factors
from app.services.model_loader import models_status

MODEL_AVAILABLE = models_status()["random_forest"]

requires_model = pytest.mark.skipif(
    not MODEL_AVAILABLE, reason="Modelo RF no entrenado (ejecute train_random_forest)"
)

BASE = {
    "patient_id": "p1",
    "tenant_id": "t1",
}


def _features(**overrides) -> dict:
    base = {
        "edad": 40,
        "sexo": 1,
        "peso_kg": 80,
        "talla_m": 1.75,
        "imc": 26.1,
        "variacion_peso_3m_kg": 0.0,
        "porcentaje_grasa": 25,
        "nivel_actividad": 2,
        "calidad_dieta_score": 6,
        "num_comorbilidades": 0,
    }
    base.update(overrides)
    return base


@requires_model
def test_paciente_alto_riesgo(client, auth_headers):
    payload = {**BASE, "features": _features(
        imc=35, peso_kg=105, porcentaje_grasa=40, calidad_dieta_score=2,
        num_comorbilidades=3, nivel_actividad=0, variacion_peso_3m_kg=7,
    )}
    resp = client.post("/api/v1/risk-prediction", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["nivel_riesgo"] == "Alto"


@requires_model
def test_paciente_bajo_riesgo(client, auth_headers):
    payload = {**BASE, "features": _features(
        edad=25, imc=21, peso_kg=64, porcentaje_grasa=15, calidad_dieta_score=9,
        num_comorbilidades=0, nivel_actividad=4, variacion_peso_3m_kg=0,
    )}
    resp = client.post("/api/v1/risk-prediction", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["nivel_riesgo"] == "Bajo"


def test_datos_insuficientes_retorna_422(client, auth_headers):
    payload = {**BASE, "features": {"edad": 40, "sexo": 1, "peso_kg": 80}}
    resp = client.post("/api/v1/risk-prediction", json=payload, headers=auth_headers)
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "datos_insuficientes"
    assert isinstance(body["campos_faltantes"], list)
    assert len(body["campos_faltantes"]) > 0


def test_sin_api_key_retorna_403(client):
    payload = {**BASE, "features": _features()}
    resp = client.post("/api/v1/risk-prediction", json=payload)
    assert resp.status_code in (401, 403)  # no autorizado (API Key ausente)


def test_factores_criticos_ignoran_valores_saludables():
    factors = _identify_critical_factors(
        _features(
            edad=25,
            imc=21,
            peso_kg=64,
            porcentaje_grasa=15,
            calidad_dieta_score=9,
            num_comorbilidades=0,
            nivel_actividad=4,
            variacion_peso_3m_kg=0,
        )
    )

    assert factors == ["Perfil clinico general del paciente"]
    assert "Presencia de comorbilidades registradas" not in factors
    assert "Nivel de actividad fisica insuficiente" not in factors
    assert "Calidad de la dieta por debajo del promedio" not in factors


def test_factores_criticos_reflejan_valores_de_riesgo():
    factors = _identify_critical_factors(
        _features(
            imc=35,
            porcentaje_grasa=40,
            calidad_dieta_score=5,
            num_comorbilidades=3,
            nivel_actividad=0,
            variacion_peso_3m_kg=7,
        )
    )

    assert "Presencia de comorbilidades registradas" in factors
    assert "Indice de Masa Corporal elevado" in factors
    assert len(factors) == 3


@requires_model
def test_factores_criticos_no_vacio(client, auth_headers):
    payload = {**BASE, "features": _features(imc=33, num_comorbilidades=2)}
    resp = client.post("/api/v1/risk-prediction", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()["factores_criticos"]) >= 1


@requires_model
def test_probabilidades_suman_uno(client, auth_headers):
    payload = {**BASE, "features": _features()}
    resp = client.post("/api/v1/risk-prediction", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    total = sum(resp.json()["probabilidades"].values())
    assert abs(total - 1.0) < 1e-3
