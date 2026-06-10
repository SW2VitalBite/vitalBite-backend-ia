# syntax=docker/dockerfile:1

# ============================================================
# Etapa 1: Construcción de dependencias
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /install

# Dependencias del sistema necesarias para OpenCV / EasyOCR (si se habilitan)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/packages -r requirements.txt

# ============================================================
# Etapa 2: Imagen de producción
# ============================================================
FROM python:3.11-slim AS production

WORKDIR /app

# Hilos a 1 por librería numérica: TensorFlow (CNN del modo "plato") y PyTorch
# (EasyOCR del modo "etiqueta") comparten el mismo proceso y sus runtimes de
# OpenMP chocaban al paralelizar sobre la CPU → SIGSEGV (signal 11). Forzar
# single-thread evita el crash; el costo es un OCR algo más lento.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=TRUE

# Librerías de runtime para OpenCV (libGL) + curl para el healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python instaladas en la etapa builder
COPY --from=builder /install/packages /usr/local

# Pre-descarga los modelos de EasyOCR (detección CRAFT + reconocimiento latin) a
# la imagen, para que el primer escaneo de etiqueta no los baje (~100 MB) en cada
# cold start. Depende solo de los paquetes → la capa se cachea entre builds.
RUN python -c "import easyocr; easyocr.Reader(['es', 'en'], gpu=False, verbose=False)"

# Código fuente
COPY app/ ./app/
COPY .env.example ./.env

# Cloud Run inyecta la variable PORT (8080 por defecto). En local cae a 8001.
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8001}/api/v1/health || exit 1

# Forma shell con `exec` para que: (1) ${PORT} se expanda en runtime y (2) uvicorn
# herede el PID 1 y reciba SIGTERM de Cloud Run para un apagado limpio.
# WEB_CONCURRENCY=1 por defecto: cada worker carga la CNN en RAM; en Cloud Run se
# escala por instancias, no por workers (evita OOM). Override con WEB_CONCURRENCY.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001} --workers ${WEB_CONCURRENCY:-1}"]
