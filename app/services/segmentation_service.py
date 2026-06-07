"""Servicio de segmentación de pacientes con K-means (CU11).

Pipeline: validar mínimo de pacientes → construir matriz e imputar nulos →
normalizar (MinMaxScaler) → determinar K óptimo (Elbow Method) → KMeans →
proyección PCA 3D → descripción de clusters. El modelo se re-entrena en cada
llamada (no hay artefacto persistente por tenant) ya que la cartera de
pacientes cambia constantemente y el costo es bajo (<500 ms para <500 pacientes).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from app.core.config import settings
from app.core.exceptions import InsufficientPatientsError
from app.core.logging import get_logger
from app.schemas.segmentation import (
    ClusterInfo,
    PatientClusterPoint,
    SegmentationRequest,
    SegmentationResponse,
)

logger = get_logger("segmentation")

FEATURES = [
    "imc",
    "porcentaje_grasa",
    "masa_muscular_kg",
    "variacion_peso_3m",
    "nivel_actividad",
    "adherencia_dieta_pct",
    "asistencia_citas_pct",
]

FEATURE_LABELS = {
    "imc": "IMC",
    "porcentaje_grasa": "% Grasa",
    "masa_muscular_kg": "Masa Muscular",
    "variacion_peso_3m": "Variación de Peso",
    "nivel_actividad": "Actividad Física",
    "adherencia_dieta_pct": "Adherencia a Dieta",
    "asistencia_citas_pct": "Asistencia a Citas",
}


def _generate_cluster_label(scaled_mean: np.ndarray, features: list[str]) -> str:
    """Etiqueta descriptiva a partir del centroide normalizado [0,1]."""
    fm = dict(zip(features, scaled_mean))
    imc = fm.get("imc", 0.5)
    grasa = fm.get("porcentaje_grasa", 0.5)
    actividad = fm.get("nivel_actividad", 0.5)
    adherencia = fm.get("adherencia_dieta_pct", 0.5)

    if imc > 0.7 and actividad < 0.3:
        return "Riesgo Alto — Sedentario con Sobrepeso"
    if actividad > 0.7 and grasa < 0.3:
        return "Perfil Activo — Composición Corporal Favorable"
    if adherencia > 0.7:
        return "Alta Adherencia — Progreso Constante"
    if grasa > 0.7:
        return "Riesgo Metabólico — Grasa Corporal Elevada"
    if imc < 0.2:
        return "Bajo Peso — Requiere Intervención Nutricional"
    return "Perfil Moderado — Seguimiento Regular"


class SegmentationService:
    def segment(self, request: SegmentationRequest) -> SegmentationResponse:
        patients = request.patients
        min_required = settings.MIN_PATIENTS_FOR_KMEANS

        if len(patients) < min_required:
            raise InsufficientPatientsError(len(patients), min_required)

        df = pd.DataFrame([p.model_dump() for p in patients])
        patient_ids = df["patient_id"].tolist()
        X_raw = df[FEATURES].copy()

        # Imputar nulos con la media del tenant; si una columna es toda nula → 0
        X_raw = X_raw.fillna(X_raw.mean()).fillna(0.0)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_raw.values)

        k_optimal = self._find_optimal_k(X_scaled, len(patients))

        kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X_scaled)

        # Silhouette score (calidad del clustering) — solo si hay >1 cluster real
        sil_score: float | None = None
        if len(set(labels)) > 1:
            sil_score = float(silhouette_score(X_scaled, labels))
            logger.info(
                "clustering completado",
                extra={
                    "tenant_id": request.tenant_id,
                    "k": k_optimal,
                    "silhouette": round(sil_score, 4),
                    "n_pacientes": len(patients),
                },
            )

        # Proyección PCA 3D (rellena con ceros si hay <3 dimensiones útiles)
        n_components = min(3, X_scaled.shape[1], X_scaled.shape[0])
        pca = PCA(n_components=n_components, random_state=42)
        coords = pca.fit_transform(X_scaled)
        if coords.shape[1] < 3:
            pad = np.zeros((coords.shape[0], 3 - coords.shape[1]))
            coords = np.hstack([coords, pad])
        variance_explained = pca.explained_variance_ratio_.tolist()
        while len(variance_explained) < 3:
            variance_explained.append(0.0)

        clusters_info = self._describe_clusters(
            X_raw.values, X_scaled, labels, k_optimal
        )

        points = [
            PatientClusterPoint(
                patient_id=patient_ids[i],
                cluster_id=int(labels[i]),
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                z=float(coords[i, 2]),
            )
            for i in range(len(patients))
        ]

        return SegmentationResponse(
            k_clusters=k_optimal,
            clusters=clusters_info,
            pca_points=points,
            variance_explained=variance_explained,
            silhouette_score=sil_score,
            total_pacientes=len(patients),
        )

    def _find_optimal_k(self, X: np.ndarray, n_patients: int) -> int:
        # No se puede pedir más clusters que muestras
        k_max = min(8, n_patients - 1)
        if k_max < 2:
            return 1
        k_range = list(range(2, k_max + 1))
        inertias = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            km.fit(X)
            inertias.append(km.inertia_)
        return self._elbow_point(k_range, inertias)

    @staticmethod
    def _elbow_point(ks: list[int], inertias: list[float]) -> int:
        """K óptimo por máxima curvatura (segunda derivada) de la inercia."""
        if len(ks) <= 2:
            return ks[0]
        diffs1 = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        diffs2 = [diffs1[i] - diffs1[i + 1] for i in range(len(diffs1) - 1)]
        elbow_idx = diffs2.index(max(diffs2)) + 1
        return ks[min(elbow_idx, len(ks) - 1)]

    @staticmethod
    def _describe_clusters(
        X_raw: np.ndarray,
        X_scaled: np.ndarray,
        labels: np.ndarray,
        k: int,
    ) -> list[ClusterInfo]:
        clusters: list[ClusterInfo] = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            if not mask.any():
                continue
            raw_mean = X_raw[mask].mean(axis=0)
            scaled_mean = X_scaled[mask].mean(axis=0)

            # Feature dominante = la de mayor valor normalizado (comparable entre sí)
            dominant_idx = int(scaled_mean.argmax())
            dominant_feature = FEATURE_LABELS.get(
                FEATURES[dominant_idx], FEATURES[dominant_idx]
            )

            label = _generate_cluster_label(scaled_mean, FEATURES)

            clusters.append(
                ClusterInfo(
                    cluster_id=cluster_id,
                    label=label,
                    total_pacientes=int(mask.sum()),
                    porcentaje=round(float(mask.sum()) / len(labels) * 100, 1),
                    feature_dominante=dominant_feature,
                    caracteristicas={
                        FEATURE_LABELS.get(FEATURES[i], FEATURES[i]): round(float(v), 2)
                        for i, v in enumerate(raw_mean)
                    },
                )
            )
        return clusters
