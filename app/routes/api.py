# routes/api.py
from fastapi import APIRouter
from .health import router as health_router
from .vector_embedding import router as embedding_router
from .translation import router as translation_router
from .summarization import router as summarization_router
from ..config import get_settings

settings = get_settings()

router = APIRouter()
router.include_router(health_router, prefix=settings.api_v1_str, tags=["health"])
router.include_router(embedding_router, prefix=settings.api_v1_str, tags=["vector_embedding"])
router.include_router(translation_router, prefix=settings.api_v1_str, tags=["translation"])
router.include_router(summarization_router, prefix=settings.api_v1_str, tags=["summarization"])