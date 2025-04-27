# app/main.py
from fastapi import FastAPI
from app.config import get_settings
from app.routes.api import router

settings = get_settings()

app = FastAPI(
    title="News AI endpoints",
    description="Implementation of text embedding service",
    version="0.0.1",
    docs_url=f"{settings.api_v1_str}/docs",
    redoc_url=f"{settings.api_v1_str}/redoc",
    debug=settings.debug_mode
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
        reload=settings.debug_mode
    )