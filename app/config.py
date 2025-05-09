# app/config.py
import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API settings
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", 8000))
    api_v1_str: str = os.getenv("API_V1_STR", "/api/v1")
    
    # Model settings
    text_to_vector_model_name: str = os.getenv(
        "TEXT_TO_VECTOR_MODEL_NAME", 
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    # Add other model names here if they need to be configurable
    # e.g., SENTIMENT_MODEL_NAME, EMOTION_MODEL_NAME
    
    # Security settings
    api_secret: Optional[str] = os.getenv("API_SECRET", None) # Allow empty for no auth
    
    # Performance and operational settings
    debug_mode: bool = os.getenv("DEBUG_MODE", False)
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper() # Default to INFO, ensure uppercase
    hf_token: str = os.getenv("HF_TOKEN", '') 
    
    # New settings for middleware
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", 180))
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", 100))
    rate_limit_window_seconds: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 60))
    
    # Model quantization (global default, can be overridden per service)
    default_model_quantization: bool = os.getenv("DEFAULT_MODEL_QUANTIZATION", True)

    class Config:
        env_file = "../.env"

def get_settings():
    """Return cached settings instance."""
    return Settings()
