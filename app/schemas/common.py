"""Tipos compartidos entre los distintos módulos de IA."""

from enum import Enum


class NivelRiesgo(str, Enum):
    BAJO = "Bajo"
    MEDIO = "Medio"
    ALTO = "Alto"


class Semaforo(str, Enum):
    SEGURO = "SEGURO"
    PRECAUCION = "PRECAUCION"
    RIESGO = "RIESGO"


class ScanMode(str, Enum):
    LABEL = "label"  # etiqueta nutricional → OCR
    PLATE = "plate"  # plato de comida → CNN


# Etiquetas legibles para el target del Random Forest
RISK_LABELS: dict[int, str] = {0: "Bajo", 1: "Medio", 2: "Alto"}
