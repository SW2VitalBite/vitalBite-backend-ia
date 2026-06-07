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

# Librerías de runtime para OpenCV (libGL) + curl para el healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python instaladas en la etapa builder
COPY --from=builder /install/packages /usr/local

# Código fuente
COPY app/ ./app/
COPY .env.example ./.env

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8001/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
