# routes/health.py
from fastapi import APIRouter
from ..models.health import (
    HealthStatus,
    HealthResponse,
    EndpointDetail,
    ModelStatus,
)  # Updated models
from ..config import get_settings
from ..utils.model_manager import model_manager  # Import model_manager
import platform
import os
import psutil  # For system resource usage

router = APIRouter()
settings = get_settings()


@router.get(
    "/", response_model=HealthResponse, summary="Comprehensive Health Check"
)  # Changed response_model
async def health_check():
    """
    Provides a comprehensive health check of the API.

    Includes:
    - Overall application status
    - System information (OS, Python version)
    - Resource usage (CPU, Memory)
    - Status of loaded models
    - List of available API endpoints
    """

    # System Information
    system_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
    }

    # Resource Usage
    cpu_usage = psutil.cpu_percent(interval=0.1)  # Non-blocking CPU usage
    memory_info = psutil.virtual_memory()
    resource_usage = {
        "cpu_percent": cpu_usage,
        "memory_total_gb": round(memory_info.total / (1024**3), 2),
        "memory_used_gb": round(memory_info.used / (1024**3), 2),
        "memory_percent": memory_info.percent,
    }

    # Model Status from ModelManager
    model_statuses = []
    mm_info = model_manager.get_memory_usage()

    # Basic status for each model type known to the application
    # This could be expanded if models are registered more dynamically
    known_model_services = {
        "text_to_vector": settings.text_to_vector_model_name,
        "sentiment_analysis": "distilbert-base-uncased-finetuned-sst-2-english",  # Default from service
        "emotion_analysis": "cardiffnlp/twitter-roberta-large-emotion-latest",  # Default from service
        # Add other services/models here
    }

    for service_name, model_name_or_path in known_model_services.items():
        is_loaded = any(
            key.startswith(model_name_or_path) for key in model_manager._models.keys()
        )
        model_statuses.append(
            ModelStatus(
                model_name=model_name_or_path,
                service=service_name,
                loaded=is_loaded,
                device=str(model_manager.device) if is_loaded else "N/A",
            )
        )

    # Available Endpoints (simplified, as full list is in OpenAPI docs)
    # The prefix is now added in main.py, so we don't need settings.api_v1_str here
    endpoints = [
        EndpointDetail(
            path="/embed", method="POST", description="Single text embedding"
        ),
        EndpointDetail(
            path="/embed/batch", method="POST", description="Batch text embeddings"
        ),
        EndpointDetail(
            path="/translate",
            method="POST",
            description="German to English translation",
        ),
        EndpointDetail(
            path="/summarize", method="POST", description="Text summarization"
        ),
        EndpointDetail(
            path="/ner", method="POST", description="Named Entity Recognition"
        ),
        EndpointDetail(
            path="/sentiment", method="POST", description="Sentiment analysis (single)"
        ),
        EndpointDetail(
            path="/sentiment/batch",
            method="POST",
            description="Sentiment analysis (batch)",
        ),
        EndpointDetail(
            path="/emotions", method="POST", description="Emotion analysis (single)"
        ),
        EndpointDetail(
            path="/emotions/batch",
            method="POST",
            description="Emotion analysis (batch)",
        ),
    ]

    return HealthResponse(
        status=HealthStatus.healthy,
        system_info=system_info,
        resource_usage=resource_usage,
        model_statuses=model_statuses,
        endpoints=endpoints,
        model_manager_memory_usage=mm_info,
    )
