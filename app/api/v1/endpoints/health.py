from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }
