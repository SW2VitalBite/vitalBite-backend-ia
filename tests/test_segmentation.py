"""Tests de la segmentación K-means (CU11)."""

import random


def _make_patients(n: int, seed: int = 1) -> list[dict]:
    rng = random.Random(seed)
    patients = []
    for i in range(n):
        # Dos perfiles para inducir estructura de clusters
        if i % 2 == 0:
            patients.append({
                "patient_id": f"p{i}",
                "imc": rng.uniform(30, 38),
                "porcentaje_grasa": rng.uniform(35, 50),
                "masa_muscular_kg": rng.uniform(20, 30),
                "variacion_peso_3m": rng.uniform(2, 8),
                "nivel_actividad": rng.uniform(0, 1),
                "adherencia_dieta_pct": rng.uniform(20, 50),
                "asistencia_citas_pct": rng.uniform(30, 60),
            })
        else:
            patients.append({
                "patient_id": f"p{i}",
                "imc": rng.uniform(19, 24),
                "porcentaje_grasa": rng.uniform(10, 20),
                "masa_muscular_kg": rng.uniform(35, 55),
                "variacion_peso_3m": rng.uniform(-2, 1),
                "nivel_actividad": rng.uniform(3, 4),
                "adherencia_dieta_pct": rng.uniform(80, 100),
                "asistencia_citas_pct": rng.uniform(85, 100),
            })
    return patients


def test_segmentacion_valida(client, auth_headers):
    payload = {"tenant_id": "t1", "patients": _make_patients(20)}
    resp = client.post("/api/v1/segmentation", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["k_clusters"] >= 2
    assert body["total_pacientes"] == 20
    assert len(body["pca_points"]) == 20


def test_pocos_pacientes_retorna_422(client, auth_headers):
    payload = {"tenant_id": "t1", "patients": _make_patients(8)}
    resp = client.post("/api/v1/segmentation", json=payload, headers=auth_headers)
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "pacientes_insuficientes"
    assert body["minimo_requerido"] == 10
    assert body["pacientes_actuales"] == 8


def test_valores_nulos_se_imputan(client, auth_headers):
    patients = _make_patients(12)
    patients[0]["imc"] = None
    patients[1]["porcentaje_grasa"] = None
    payload = {"tenant_id": "t1", "patients": patients}
    resp = client.post("/api/v1/segmentation", json=payload, headers=auth_headers)
    assert resp.status_code == 200


def test_cluster_ids_en_rango(client, auth_headers):
    payload = {"tenant_id": "t1", "patients": _make_patients(20)}
    resp = client.post("/api/v1/segmentation", json=payload, headers=auth_headers)
    body = resp.json()
    k = body["k_clusters"]
    for point in body["pca_points"]:
        assert 0 <= point["cluster_id"] < k


def test_suma_pacientes_por_cluster(client, auth_headers):
    payload = {"tenant_id": "t1", "patients": _make_patients(20)}
    resp = client.post("/api/v1/segmentation", json=payload, headers=auth_headers)
    body = resp.json()
    suma = sum(c["total_pacientes"] for c in body["clusters"])
    assert suma == body["total_pacientes"]


def test_variance_explained_tres_valores(client, auth_headers):
    payload = {"tenant_id": "t1", "patients": _make_patients(20)}
    resp = client.post("/api/v1/segmentation", json=payload, headers=auth_headers)
    body = resp.json()
    assert len(body["variance_explained"]) == 3
    assert sum(body["variance_explained"]) <= 1.0 + 1e-6
