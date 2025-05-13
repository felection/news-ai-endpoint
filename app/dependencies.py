# app/dependencies.py
from fastapi.security import APIKeyHeader
from fastapi import Depends, HTTPException, status
from .config import get_settings, Settings  # Import Settings for type hinting
from typing import Optional

# Define the API key header
api_key_header_scheme = APIKeyHeader(
    name="X-API-Key", auto_error=False
)  # auto_error=False to handle missing key manually


async def validate_api_key(
    api_key: Optional[str] = Depends(api_key_header_scheme),  # API key is now optional
    settings: Settings = Depends(get_settings),  # Use Settings for type hint
):
    """
    Validate the API key if `api_secret` is set in the configuration.

    If `api_secret` is not set, this dependency does nothing and allows the request.
    If `api_secret` is set, it requires a matching `X-API-Key` header.

    Raises:
        HTTPException (401): If `api_secret` is set and the key is missing or invalid.
    """
    # If no API secret is configured, then no API key is required.
    if not settings.api_secret:
        return  # Allow access

    # If API secret is configured, but no key was provided in the header
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Please include an 'X-API-Key' header.",
        )

    # If API secret is configured and a key was provided, validate it
    if api_key != settings.api_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key. Check your 'X-API-Key' header.",
        )

    return api_key  # Return the validated key (though it's not typically used directly after validation)


# Note: The previous get_api_key factory function was a bit convoluted.
# The simplified validate_api_key above is more direct for FastAPI's dependency injection.
# If you need to conditionally apply the dependency, FastAPI's `Security` or router-level dependencies are better.
