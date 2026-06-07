"""Endpoint CU11 — POST /api/v1/segmentation (K-means).

Controlador delgado: el error de dominio ``InsufficientPatientsError`` se mapea
a 422 mediante el handler global registrado en ``main.py``.
"""

from fastapi import APIRouter, Depends

from app.core.security import verify_api_key
from app.schemas.segmentation import SegmentationRequest, SegmentationResponse
from app.services.segmentation_service import SegmentationService

router = APIRouter()


def get_service() -> SegmentationService:
    return SegmentationService()


@router.post(
    "",
    response_model=SegmentationResponse,
    summary="Segmentar pacientes de un tenant con K-means",
    dependencies=[Depends(verify_api_key)],
)
def segment_patients(
    request: SegmentationRequest,
    service: SegmentationService = Depends(get_service),
) -> SegmentationResponse:
    return service.segment(request)
