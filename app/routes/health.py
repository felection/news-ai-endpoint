# routes/health.py
from fastapi import APIRouter, Depends
from ..models.health import HealthStatus
from ..config import get_settings

router = APIRouter()
settings = get_settings()

@router.get("/", response_model=HealthStatus)
async def health_check():
    return {
        "status": "healthy",
        "endpoints": [
            {"path": f"{settings.api_v1_str}/embed", "method": "POST", "description": "Single text embedding"},
            {"path": f"{settings.api_v1_str}/embed/batch", "method": "POST", "description": "Batch text embeddings"},
            {"path": f"{settings.api_v1_str}/translate", "method": "POST", "description": "German to English translation"},
            {"path": f"{settings.api_v1_str}/summarize", "method": "POST", "description": "Text summarization"},
            {"path": f"{settings.api_v1_str}/ner", "method": "POST", "description": "Named Entity Recognition"}
        ]
    }

