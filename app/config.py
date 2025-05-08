# app/config.py
import os
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", 8000))
    api_v1_str: str = os.getenv("API_V1_STR", "/api/v1")
    text_to_vector_model_name: str = os.getenv("TEXT_TO_VECTOR_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    api_secret: str = os.getenv("API_SECRET", "")
    debug_mode: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "info")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()