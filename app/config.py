# app/config.py
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API settings
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", 8000))
    api_v1_str: str = os.getenv("API_V1_STR", "/api/v1")

    # Model settings
    text_to_vector_model_name: str = os.getenv(
        "TEXT_TO_VECTOR_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    # Add other model names here if they need to be configurable
    # e.g., SENTIMENT_MODEL_NAME, EMOTION_MODEL_NAME

    # Security settings
    api_secret: Optional[str] = os.getenv("API_SECRET", None)  # Allow empty for no auth

    # Performance and operational settings
    debug_mode: bool = os.getenv("DEBUG_MODE", False)
    log_level: str = os.getenv(
        "LOG_LEVEL", "INFO"
    ).upper()  # Default to INFO, ensure uppercase
    hf_token: str = os.getenv("HF_TOKEN", "")

    # New settings for middleware
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", 180))
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", 100))
    rate_limit_window_seconds: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 60))

    # Model quantization (global default, can be overridden per service)
    default_model_quantization: bool = os.getenv("DEFAULT_MODEL_QUANTIZATION", True)

    # Text generation model settings
    text_generation_model_repo: str = os.getenv(
        "TEXT_GENERATION_MODEL_REPO", "allenai/OLMo-2-0425-1B-Instruct-GGUF"
    )
    text_generation_model_file: str = os.getenv(
        "TEXT_GENERATION_MODEL_FILE", "OLMo-2-0425-1B-Instruct-Q4_K_M.gguf"
    )
    text_generation_context_length: int = int(
        os.getenv("TEXT_GENERATION_CONTEXT_LENGTH", 2048)
    )  # Context length for text generation

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    class Config:
        env_file = "../.env"


def get_settings():
    """Return cached settings instance."""
    return Settings()
