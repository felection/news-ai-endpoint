# app/main.py
from fastapi import FastAPI, Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.config import get_settings
from app.routes.api import router as api_router  # Renamed to avoid conflict
from app.middleware import RateLimitMiddleware, TimeoutMiddleware
from app.utils.logging_utils import setup_logging, logger # Use our setup_logging
from app.utils.model_manager import model_manager # For health check
from contextlib import asynccontextmanager

settings = get_settings()

# Setup logging
# We get the log_level from settings now
setup_logging(settings.log_level.upper())

# Define lifespan context manager (replaces on_event handlers)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic (replaces on_event("startup"))
    logger.info("Application startup...")
    # You can pre-load critical models here if needed
    # model_manager.get_model(settings.text_to_vector_model_name)
    logger.info("Application startup complete.")
    
    yield  # This is where the app runs
    
    # Shutdown logic (replaces on_event("shutdown"))
    logger.info("Application shutdown...")
    model_manager.clear_cache()  # Clear model cache on shutdown
    logger.info("Application shutdown complete.")

app = FastAPI(
    title="News AI Endpoints",
    description="High-performance NLP services including text embedding, sentiment analysis, and more.",
    version="0.1.0",
    docs_url=f"{settings.api_v1_str}/docs",
    redoc_url=f"{settings.api_v1_str}/redoc",
    debug=settings.debug_mode,
    # Add custom exception handlers
    exception_handlers={
        StarletteHTTPException: lambda request, exc: Response(
            content=str(exc.detail), status_code=exc.status_code
        ),
        RequestValidationError: lambda request, exc: Response(
            content=str(exc), status_code=422 # Unprocessable Entity
        ),
    }
)

# Add middleware
app.add_middleware(TimeoutMiddleware, timeout=settings.request_timeout_seconds)
app.add_middleware(
    RateLimitMiddleware, 
    limit=settings.rate_limit_requests, 
    window=settings.rate_limit_window_seconds,
    exempt_paths=[f"{settings.api_v1_str}/", f"{settings.api_v1_str}/docs", f"{settings.api_v1_str}/redoc"] # Exempt health, docs, redoc
)

# Include API router
app.include_router(api_router, prefix=settings.api_v1_str) # Add prefix here for all API routes

# Root path for basic health check
@app.get("/", tags=["Root"])
async def read_root():
    """Basic health check for the root path."""
    return {"message": "News AI Endpoints are running!"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(), # Use log_level from settings
        reload=settings.debug_mode # Reload only in debug mode
    )
