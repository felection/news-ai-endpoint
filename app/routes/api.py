# routes/api.py
from fastapi import APIRouter
from .health import router as health_router
from .vector_embedding import router as embedding_router
from ..config import get_settings

settings = get_settings()

router = APIRouter()
router.include_router(health_router, prefix=settings.api_v1_str, tags=["health"])
router.include_router(embedding_router, prefix=settings.api_v1_str, tags=["vector_embedding"])