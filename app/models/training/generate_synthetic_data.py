"""Generación de datos sintéticos para entrenar el Random Forest (CU10).

En el entorno académico no se dispone de datos clínicos reales, por lo que se
sintetiza un dataset con reglas epidemiológicas conocidas (IMC, variación de
peso, % grasa, actividad, calidad de dieta y comorbilidades) que determinan un
``risk_score`` y, a partir de él, la clase de riesgo (0=Bajo, 1=Medio, 2=Alto).
"""

import numpy as np
import pandas as pd

FEATURES = [
    "edad",
    "sexo",
    "peso_kg",
    "talla_m",
    "imc",
    "variacion_peso_3m_kg",
    "porcentaje_grasa",
    "nivel_actividad",
    "calidad_dieta_score",
    "num_comorbilidades",
]
TARGET = "riesgo"


def generate_nutritional_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    edad = rng.integers(15, 80, n_samples).astype(float)
    sexo = rng.integers(0, 2, n_samples).astype(float)
    peso_kg = rng.normal(75, 18, n_samples).clip(40, 200)
    talla_m = rng.normal(1.68, 0.10, n_samples).clip(1.45, 2.05)
    imc = peso_kg / (talla_m**2)
    variacion_peso = rng.normal(0, 3, n_samples)
    porcentaje_grasa = rng.normal(28, 10, n_samples).clip(5, 55)
    nivel_actividad = rng.integers(0, 5, n_samples).astype(float)
    calidad_dieta = rng.uniform(0, 10, n_samples)
    num_comorbilidades = rng.integers(0, 6, n_samples).astype(float)

    # Regla de etiquetado clínico (combinación ponderada de factores de riesgo)
    risk_score = (
        (imc > 30).astype(float) * 2
        + (imc < 18.5).astype(float) * 1.5
        + (variacion_peso > 5).astype(float) * 1.2
        + (porcentaje_grasa > 35).astype(float) * 1.0
        + (nivel_actividad < 1).astype(float) * 0.8
        + (calidad_dieta < 4).astype(float) * 0.7
        + num_comorbilidades * 0.5
    )

    riesgo = np.where(risk_score >= 4, 2, np.where(risk_score >= 2, 1, 0))

    return pd.DataFrame(
        {
            "edad": edad,
            "sexo": sexo,
            "peso_kg": peso_kg,
            "talla_m": talla_m,
            "imc": imc,
            "variacion_peso_3m_kg": variacion_peso,
            "porcentaje_grasa": porcentaje_grasa,
            "nivel_actividad": nivel_actividad,
            "calidad_dieta_score": calidad_dieta,
            "num_comorbilidades": num_comorbilidades,
            "riesgo": riesgo,
        }
    )


if __name__ == "__main__":
    df = generate_nutritional_dataset()
    print(df.head())
    print("\nDistribución de clases:")
    print(df["riesgo"].value_counts().sort_index())
