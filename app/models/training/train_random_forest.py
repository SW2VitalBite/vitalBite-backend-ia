"""Entrenamiento y serialización del Random Forest de riesgo nutricional (CU10).

Genera datos sintéticos, entrena un ``RandomForestClassifier``, evalúa con
hold-out + validación cruzada, y serializa tres artefactos en
``app/models/artifacts/``:

* ``risk_rf_model.pkl``           — el clasificador entrenado
* ``risk_scaler.pkl``            — el StandardScaler de las features
* ``risk_feature_importances.pkl`` — dict {feature: importancia}

Ejecutar (desde la raíz del proyecto, con el venv activo)::

    python -m app.models.training.train_random_forest
"""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

try:  # Soporta ejecución como módulo (-m) o como script suelto
    from app.models.training.generate_synthetic_data import (
        FEATURES,
        TARGET,
        generate_nutritional_dataset,
    )
except ImportError:  # pragma: no cover
    from generate_synthetic_data import (  # type: ignore
        FEATURES,
        TARGET,
        generate_nutritional_dataset,
    )

# Raíz del proyecto = .../vitalBite-backend-ia (sube 4 niveles desde este archivo)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "app" / "models" / "artifacts"

RISK_LABELS = {0: "Bajo", 1: "Medio", 2: "Alto"}


def train(n_samples: int = 3000) -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generando datos sintéticos...")
    df = generate_nutritional_dataset(n_samples=n_samples)

    X = df[FEATURES].values
    y = df[TARGET].values

    # Normalización (se guarda el scaler para reutilizarlo en inferencia)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Entrenando Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluación
    y_pred = model.predict(X_test)
    print("\n=== REPORTE DE CLASIFICACIÓN ===")
    print(classification_report(y_test, y_pred, target_names=list(RISK_LABELS.values())))

    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="f1_macro")
    print(f"F1 Macro (CV 5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    importances = dict(zip(FEATURES, model.feature_importances_.tolist()))
    print("\n=== IMPORTANCIA DE FEATURES ===")
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {imp:.4f}")

    joblib.dump(model, ARTIFACTS_DIR / "risk_rf_model.pkl")
    joblib.dump(scaler, ARTIFACTS_DIR / "risk_scaler.pkl")
    joblib.dump(importances, ARTIFACTS_DIR / "risk_feature_importances.pkl")
    print(f"\nModelo guardado en {ARTIFACTS_DIR}")

    return importances


if __name__ == "__main__":
    train()
