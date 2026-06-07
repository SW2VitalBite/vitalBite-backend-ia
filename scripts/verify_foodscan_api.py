"""Smoke test del endpoint food-scan vía TestClient (debe correr en el venv).

Confirma que el modelo CNN carga y que POST /api/v1/food-scan responde 200.
Uso: python scripts/verify_foodscan_api.py
"""

import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Evita UnicodeEncodeError en la consola cp1252 de Windows al imprimir emojis.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

FOOD_DIR = r"D:\FoodNetProject\FoodNet\Food Datasets\food-101\images\pizza"


def main() -> None:
    files = glob.glob(os.path.join(FOOD_DIR, "*.jpg"))
    if not files:
        print("No hay imágenes de prueba en", FOOD_DIR)
        return
    img_path = files[0]

    with TestClient(app) as client:
        with open(img_path, "rb") as fh:
            resp = client.post(
                "/api/v1/food-scan",
                headers={"X-API-Key": settings.API_KEY},
                data={"patient_id": "demo-patient", "mode": "plate"},
                files={"image": ("pizza.jpg", fh, "image/jpeg")},
            )
    print("HTTP", resp.status_code)
    print(resp.json())


if __name__ == "__main__":
    main()
