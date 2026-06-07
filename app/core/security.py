"""Autenticación por API Key para los endpoints protegidos.

Todos los endpoints de inferencia exigen la cabecera ``X-API-Key`` cuyo valor
debe coincidir con ``settings.API_KEY``. Se implementa como dependencia
reutilizable de FastAPI.
"""

from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

from app.core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API Key inválida o no autorizada.",
        )
    return api_key
