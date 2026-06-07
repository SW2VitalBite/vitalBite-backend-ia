"""Verificación rápida del modelo de clasificación de platos (CU9).

Toma imágenes reales de Food-101 y reporta top-1 / top-3 vía el servicio.
Uso: python scripts/verify_food_model.py
"""

import glob
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from app.services import food_classification_service as fc

BASE = r"D:\FoodNetProject\FoodNet\Food Datasets\food-101\images"
TEST_CLASSES = [
    "pizza",
    "hamburger",
    "apple_pie",
    "caesar_salad",
    "sushi",
    "french_fries",
    "donuts",
    "ramen",
]


def main() -> None:
    names = fc.get_class_names()
    print("num clases:", len(names))
    random.seed(7)
    top1 = top3 = tot = 0
    for c in TEST_CLASSES:
        files = glob.glob(os.path.join(BASE, c, "*.jpg"))
        if not files:
            print(f"{c}: sin imágenes")
            continue
        img = Image.open(random.choice(files))
        preds, conf = fc.classify_food(img)
        pc = [p.clase for p in preds]
        ok1 = pc[0] == c
        ok3 = c in pc
        top1 += ok1
        top3 += ok3
        tot += 1
        print(f"{c:14s} -> top1={pc[0]:18s} conf={conf:.2f}  top1_ok={ok1} top3_ok={ok3}")
    print(f"\nTOP-1: {top1}/{tot}   TOP-3: {top3}/{tot}")


if __name__ == "__main__":
    main()
