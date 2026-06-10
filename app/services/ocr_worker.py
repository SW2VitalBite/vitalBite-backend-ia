"""Worker de OCR aislado en un subproceso (multiprocessing ``spawn``).

⚠️ CRÍTICO — POR QUÉ EXISTE ESTE MÓDULO
EasyOCR usa PyTorch. TensorFlow (la CNN del modo "plato", precargada al arranque)
y PyTorch **no pueden convivir en el mismo proceso** en esta imagen: sus
runtimes nativos chocan (símbolos/ABI) y el proceso muere con SIGSEGV (signal 11)
en cuanto el OCR ejecuta inferencia. Limitar hilos (OMP_NUM_THREADS=1) NO lo
evita.

La solución es ejecutar el OCR en un proceso hijo **separado** lanzado con el
método ``spawn`` (intérprete nuevo, espacio de memoria propio). Ese hijo importa
solo cv2 + easyocr y NUNCA TensorFlow, así que no hay convivencia → no hay crash.
Si el hijo cayera, el proceso padre (plato/RF/segmentación/health) sigue vivo.

Por eso este módulo NO debe importar —ni directa ni transitivamente— TensorFlow
ni ``food_classification_service``. Mantén cv2/easyocr/numpy como imports
perezosos dentro de las funciones.
"""

from __future__ import annotations

# Cache del lector EasyOCR **dentro del proceso hijo** (se reutiliza entre
# peticiones mientras el worker siga vivo, evitando recargar el modelo).
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr  # import perezoso (solo en el hijo)

        # verbose=False evita la barra de progreso (carácter █) que ensucia los
        # logs JSON. Los modelos vienen horneados en la imagen (ver Dockerfile).
        _reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
    return _reader


def _preprocess(rgb_array):
    """Realza la etiqueta antes del OCR: gris + escalado + contraste local (CLAHE).

    CLAHE conserva la forma de los trazos (lo que mejor lee el reconocedor de
    EasyOCR) y el escalado ayuda con los números pequeños de la tabla.
    """
    import cv2

    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest < 1400:
        scale = 1400.0 / longest
        gray = cv2.resize(
            gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
        )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def run_ocr_blocks(rgb_array) -> list[dict]:
    """Entrada al subproceso: array RGB (numpy) → bloques de texto con geometría.

    Devuelve estructuras simples (dicts) picklables para que el proceso padre las
    agrupe en filas y parsee los nutrientes (eso es regex puro, sin PyTorch).
    """
    reader = _get_reader()
    image_for_ocr = _preprocess(rgb_array)
    results = reader.readtext(image_for_ocr, detail=1)

    blocks: list[dict] = []
    for bbox, text, conf in results:
        # Umbral bajo: los números pequeños de la tabla tienen menor confianza.
        if conf < 0.30:
            continue
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        blocks.append(
            {
                "text": text,
                "confidence": float(conf),
                "x0": min(xs),
                "cy": sum(ys) / len(ys),
                "h": max(ys) - min(ys),
            }
        )
    return blocks
