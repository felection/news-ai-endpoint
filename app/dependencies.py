# app/dependencies.py
from fastapi.security import APIKeyHeader
from fastapi import Depends, HTTPException, status
from .config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(settings = Depends(get_settings)):
    if not settings.api_secret:
        return None  # No API key required
    
    async def validate_api_key(api_key: str = Depends(api_key_header)):
        if api_key != settings.api_secret:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key"
            )
        return api_key
    
    return validate_api_key

async def validate_api_key(
    api_key: str = Depends(api_key_header),
    settings=Depends(get_settings)
):
    if settings.api_secret and api_key != settings.api_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key