"""Tests del scanner nutricional (CU9).

El pipeline OCR/CNN depende de TensorFlow/EasyOCR (opcionales). Estos tests
verifican la validación de entrada (auth, tamaño, formato) y el semáforo, que no
dependen de las librerías pesadas. Las pruebas que requieren los modelos se
omiten si no están disponibles.
"""

import io

import pytest

from app.schemas.common import Semaforo
from app.schemas.food_scan import NutrientInfo
from app.services.nutrition_analyzer_service import evaluate_semaforo

try:
    from PIL import Image

    PIL_OK = True
except ImportError:  # pragma: no cover
    PIL_OK = False


def _png_bytes(size=(64, 64), color=(120, 120, 120)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- Validación de entrada (no requiere modelos) -----------------------------

def test_sin_api_key_retorna_403(client):
    resp = client.post(
        "/api/v1/food-scan",
        data={"patient_id": "p1", "mode": "label"},
        files={"image": ("x.png", b"123", "image/png")},
    )
    assert resp.status_code in (401, 403)  # no autorizado (API Key ausente)


@pytest.mark.skipif(not PIL_OK, reason="Pillow no instalado")
def test_imagen_demasiado_grande_retorna_413(client, auth_headers):
    big = b"0" * (6 * 1024 * 1024)  # 6 MB > límite 5 MB
    resp = client.post(
        "/api/v1/food-scan",
        data={"patient_id": "p1", "mode": "label"},
        files={"image": ("big.png", big, "image/png")},
        headers=auth_headers,
    )
    assert resp.status_code == 413


def test_formato_invalido_retorna_422(client, auth_headers):
    resp = client.post(
        "/api/v1/food-scan",
        data={"patient_id": "p1", "mode": "label"},
        files={"image": ("doc.pdf", b"%PDF-1.4 not an image", "application/pdf")},
        headers=auth_headers,
    )
    assert resp.status_code == 422


# --- Lógica del semáforo (unitario, sin modelos) -----------------------------

def test_semaforo_riesgo_por_multiples_banderas():
    nutrients = NutrientInfo(sodio_mg=800, azucares_g=30)
    semaforo, advertencias = evaluate_semaforo(nutrients, [])
    assert semaforo == Semaforo.RIESGO
    assert len(advertencias) >= 2


def test_semaforo_seguro_sin_problemas():
    nutrients = NutrientInfo(calorias=120, proteinas_g=10, ingredientes=["lechuga", "tomate"])
    semaforo, advertencias = evaluate_semaforo(nutrients, ["ensalada"])
    assert semaforo == Semaforo.SEGURO
    assert advertencias == []


def test_semaforo_precaucion_por_sodio_alto():
    nutrients = NutrientInfo(sodio_mg=800)
    semaforo, advertencias = evaluate_semaforo(nutrients, [])
    assert semaforo == Semaforo.PRECAUCION
    assert any("sodio" in a.lower() for a in advertencias)


def test_semaforo_precaucion_por_categoria_riesgo():
    semaforo, advertencias = evaluate_semaforo(NutrientInfo(), ["comida_rapida"])
    assert semaforo == Semaforo.PRECAUCION


# --- Clasificación CNN (requiere TensorFlow + artefacto entrenado) ------------

def test_clasificacion_plato_si_hay_modelo():
    """Si el modelo CNN está disponible, clasifica un color sólido sin error."""
    from app.services import food_classification_service as fc

    if not fc.is_available():
        pytest.skip("Modelo CNN/TensorFlow no disponible")
    if not PIL_OK:
        pytest.skip("Pillow no instalado")

    img = Image.new("RGB", (224, 224), (200, 150, 100))
    preds, conf = fc.classify_food(img)
    assert len(preds) == 3
    assert 0.0 <= conf <= 1.0
    # Las clases predichas deben pertenecer al vocabulario del modelo
    vocab = set(fc.get_class_names())
    assert all(p.clase in vocab for p in preds)
