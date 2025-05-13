# app/models/health.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum


class HealthStatus(str, Enum):
    """Possible health statuses."""

    healthy = "healthy"
    unhealthy = "unhealthy"
    degraded = "degraded"


class EndpointDetail(BaseModel):
    """Details of an API endpoint."""

    path: str = Field(..., description="Endpoint path")
    method: str = Field(..., description="HTTP method")
    description: str = Field(..., description="Brief description of the endpoint")


class ModelStatus(BaseModel):
    """Status of a specific model."""

    model_name: str = Field(..., description="Name or path of the model")
    service: str = Field(..., description="Service using this model")
    loaded: bool = Field(..., description="Whether the model is currently loaded")
    device: str = Field(
        ..., description="Device the model is loaded on (e.g., cpu, cuda)"
    )
    # Add more details like memory usage per model if needed


class HealthResponse(BaseModel):
    """Comprehensive health check response."""

    status: HealthStatus = Field(
        ..., description="Overall health status of the application"
    )
    system_info: Dict[str, Any] = Field(
        ..., description="Information about the host system"
    )
    resource_usage: Dict[str, Any] = Field(
        ..., description="Current system resource usage"
    )
    model_statuses: List[ModelStatus] = Field(
        ..., description="Status of loaded machine learning models"
    )
    endpoints: List[EndpointDetail] = Field(
        ..., description="List of available API endpoints"
    )
    model_manager_memory_usage: Optional[Dict[str, Any]] = Field(
        None, description="Memory usage details from ModelManager"
    )
