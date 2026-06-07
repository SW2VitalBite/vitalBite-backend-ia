#!/usr/bin/env bash
# Entrena/serializa todos los artefactos de ML del microservicio de IA.
# Pensado para CI/CD o para preparar la imagen Docker antes del despliegue.
#
# Uso:
#   bash scripts/seed_models.sh
#
# Variables opcionales:
#   FOOD_DATA_DIR  Directorio del dataset de alimentos (default: data/food)
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> [1/2] Entrenando Random Forest (CU10)..."
python -m app.models.training.train_random_forest

FOOD_DATA_DIR="${FOOD_DATA_DIR:-data/food}"
if [ -d "$FOOD_DATA_DIR" ]; then
  echo "==> [2/2] Entrenando CNN de alimentos (CU9) desde $FOOD_DATA_DIR..."
  python -m app.models.training.train_food_classifier --data "$FOOD_DATA_DIR"
else
  echo "==> [2/2] Omitido: no existe el dataset en '$FOOD_DATA_DIR'."
  echo "    CU9/plate responderá 503 hasta proveer food_classifier.h5."
fi

echo "==> Listo. Artefactos en app/models/artifacts/"
