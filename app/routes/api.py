# routes/api.py
from fastapi import APIRouter
from .health import router as health_router
from .vector_embedding import router as embedding_router
from .translation import router as translation_router
from .summarization import router as summarization_router
from .named_entity_recognition import router as named_entity_recognition_router
from .emotion_extraction import router as emotion_router
from .sentiment_analysis import router as sentiment_router
from .text_generation import router as text_generation_router
from .gemini import router as gemini_router

router = APIRouter()

# Prefixes are now handled in app/main.py when including this main router
router.include_router(health_router, tags=["Health"])
router.include_router(embedding_router, tags=["Vector embedding"])
router.include_router(translation_router, tags=["Translation"])
router.include_router(summarization_router, tags=["Summarization"])
router.include_router(
    named_entity_recognition_router, tags=["Named entity recognition"]
)
router.include_router(sentiment_router, tags=["Sentiment analysis"])
router.include_router(emotion_router, tags=["Emotion analysis"])
router.include_router(text_generation_router, tags=["Text generation"])
router.include_router(gemini_router, tags=["Gemini API"])
