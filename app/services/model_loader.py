"""Carga perezosa (lazy + cacheada) de artefactos de ML en memoria.

Evita releer los ``.pkl`` desde disco en cada request. Los modelos sklearn se
cargan con ``joblib`` y se memorizan con ``lru_cache``. El modelo CNN de
TensorFlow se gestiona aparte en ``food_classification_service`` (import
perezoso) porque la dependencia es opcional.

También expone :func:`models_status` para el health check extendido, sin
necesidad de cargar las librerías pesadas.
"""

from functools import lru_cache
from pathlib import Path

import joblib

from app.core.config import settings
from app.core.exceptions import ModelNotLoadedError

RF_MODEL_FILE = "risk_rf_model.pkl"
RF_SCALER_FILE = "risk_scaler.pkl"
RF_IMPORTANCES_FILE = "risk_feature_importances.pkl"


def _artifact_path(filename: str) -> Path:
    return settings.models_dir_path / filename


@lru_cache(maxsize=1)
def get_rf_model():
    path = _artifact_path(RF_MODEL_FILE)
    if not path.exists():
        raise ModelNotLoadedError(str(path), "ejecute el script train_random_forest")
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_rf_scaler():
    path = _artifact_path(RF_SCALER_FILE)
    if not path.exists():
        raise ModelNotLoadedError(str(path), "ejecute el script train_random_forest")
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_rf_importances() -> dict:
    path = _artifact_path(RF_IMPORTANCES_FILE)
    return joblib.load(path) if path.exists() else {}


def models_status() -> dict[str, bool]:
    """Estado de disponibilidad de cada artefacto (existencia en disco)."""
    return {
        "random_forest": _artifact_path(RF_MODEL_FILE).exists(),
        "rf_scaler": _artifact_path(RF_SCALER_FILE).exists(),
        "food_classifier_cnn": Path(settings.DL_MODEL_PATH).exists(),
    }


def reset_cache() -> None:
    """Limpia la caché de modelos (útil en tests tras (re)entrenar)."""
    get_rf_model.cache_clear()
    get_rf_scaler.cache_clear()
    get_rf_importances.cache_clear()
