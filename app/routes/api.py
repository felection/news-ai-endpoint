# routes/api.py
from fastapi import APIRouter
from .health import router as health_router
from .vector_embedding import router as embedding_router
from .translation import router as translation_router
from .summarization import router as summarization_router
from .named_entity_recognition import router as named_entity_recognition_router
from .emotion_extraction import router as emotion_router
from .sentiment_analysis import router as sentiment_router
from ..config import get_settings

settings = get_settings()

router = APIRouter()
router.include_router(health_router, prefix=settings.api_v1_str, tags=["Health"])
router.include_router(embedding_router, prefix=settings.api_v1_str, tags=["Vector embedding"])
router.include_router(translation_router, prefix=settings.api_v1_str, tags=["Translation"])
router.include_router(summarization_router, prefix=settings.api_v1_str, tags=["Summarization"])
router.include_router(named_entity_recognition_router, prefix=settings.api_v1_str, tags=["Named entity recognition"])
router.include_router(sentiment_router, prefix=settings.api_v1_str, tags=["Sentiment analysis"])
router.include_router(emotion_router, prefix=settings.api_v1_str, tags=["Emotion analysis"])