"""Validación end-to-end del OCR de etiquetas (CU9, modo label).

Sintetiza una imagen de tabla nutricional y la pasa por el pipeline OCR real
(EasyOCR + parseo). Uso: python scripts/verify_ocr.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont

from app.services.ocr_service import extract_nutritional_data

LABEL_LINES = [
    "Informacion Nutricional",
    "Calorias 250 kcal",
    "Proteinas 5 g",
    "Carbohidratos 30 g",
    "Grasa total 12 g",
    "Grasa saturada 4 g",
    "Sodio 720 mg",
    "Azucares 18 g",
    "Fibra 3 g",
    "Ingredientes: harina, mani, azucar, sal",
]


def _make_label_image() -> Image.Image:
    img = Image.new("RGB", (700, 560), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    y = 24
    for line in LABEL_LINES:
        draw.text((28, y), line, fill="black", font=font)
        y += 50
    return img


def main() -> None:
    print("Generando etiqueta sintética y ejecutando OCR (puede tardar la 1ª vez)...")
    img = _make_label_image()
    nutrientes, confianza = extract_nutritional_data(img)
    print("\n=== NUTRIENTES EXTRAÍDOS ===")
    data = nutrientes.model_dump()
    extracted = {k: v for k, v in data.items() if v not in (None, [], "")}
    for k, v in extracted.items():
        print(f"  {k}: {v}")
    n_campos = sum(
        1
        for k, v in data.items()
        if k != "ingredientes" and v is not None
    )
    print(f"\nCampos nutricionales extraídos: {n_campos}")
    print(f"Confianza OCR: {confianza:.2f}")
    print("CRITERIO (>=5 campos):", "OK" if n_campos >= 5 else "INSUFICIENTE")


if __name__ == "__main__":
    main()
