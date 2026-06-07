"""Tests del endpoint de salud y la raíz."""


def test_health_ok(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "modelos" in body
    assert "random_forest" in body["modelos"]


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["docs"] == "/docs"
