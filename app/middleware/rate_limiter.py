"""
Rate limiting middleware for FastAPI.

This module provides middleware for:
- Rate limiting requests based on client IP or API key
- Request timeouts to prevent long-running requests
- Basic metrics collection
"""

import time
from typing import Dict, Tuple, Optional, Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from ..utils.logging_utils import logger
from ..config import get_settings

settings = get_settings()


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, limit: int = 60, window: int = 60):
        """
        Initialize the rate limiter.

        Args:
            limit: Maximum number of requests allowed in the window
            window: Time window in seconds
        """
        self.limit = limit
        self.window = window
        self.requests: Dict[
            str, Tuple[int, float]
        ] = {}  # {key: (count, first_request_time)}

    def is_allowed(self, key: str) -> bool:
        """
        Check if a request is allowed based on the rate limit.

        Args:
            key: The key to check (usually IP address or API key)

        Returns:
            True if the request is allowed, False otherwise
        """
        current_time = time.time()

        if key not in self.requests:
            # First request from this key
            self.requests[key] = (1, current_time)
            return True

        count, first_request_time = self.requests[key]
        time_passed = current_time - first_request_time

        if time_passed > self.window:
            # Reset window
            self.requests[key] = (1, current_time)
            return True

        if count < self.limit:
            # Increment count
            self.requests[key] = (count + 1, first_request_time)
            return True

        return False

    def get_retry_after(self, key: str) -> int:
        """
        Get the number of seconds until the rate limit resets.

        Args:
            key: The key to check

        Returns:
            Number of seconds until the rate limit resets
        """
        if key not in self.requests:
            return 0

        _, first_request_time = self.requests[key]
        current_time = time.time()
        time_passed = current_time - first_request_time

        return max(0, int(self.window - time_passed))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests."""

    def __init__(
        self,
        app: ASGIApp,
        limit: int = 180,
        window: int = 60,
        exempt_paths: Optional[list] = None,
        get_key: Optional[Callable] = None,
    ):
        """
        Initialize the middleware.

        Args:
            app: The ASGI application
            limit: Maximum number of requests allowed in the window
            window: Time window in seconds
            exempt_paths: List of paths exempt from rate limiting
            get_key: Function to extract the key from the request
        """
        super().__init__(app)
        self.limiter = RateLimiter(limit, window)
        self.exempt_paths = exempt_paths or ["/api/v1/"]
        self.get_key = get_key or self._default_key_extractor

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process the request.

        Args:
            request: The incoming request
            call_next: The next middleware or endpoint

        Returns:
            The response
        """
        # Start timing for all requests
        start_time = time.time()

        # Skip rate limiting for exempt paths
        for path in self.exempt_paths:
            if request.url.path.startswith(path):
                response = await call_next(request)
                process_time = time.time() - start_time
                response.headers["X-Process-Time"] = f"{process_time:.6f}"
                return response

        # Get the key for rate limiting
        key = await self.get_key(request)

        # Check if the request is allowed
        if not self.limiter.is_allowed(key):
            retry_after = self.limiter.get_retry_after(key)
            logger.warning(f"Rate limit exceeded for {key}, retry after {retry_after}s")

            # Return 429 Too Many Requests
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

        # Process the request with timing
        response = await call_next(request)
        process_time = time.time() - start_time

        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)

        return response

    @staticmethod
    async def _default_key_extractor(request: Request) -> str:
        """
        Extract the key from the request.

        By default, use the client IP address.

        Args:
            request: The incoming request

        Returns:
            The key for rate limiting
        """
        # Try to get the real IP from common headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For can contain multiple IPs, use the first one
            return forwarded_for.split(",")[0].strip()

        # Fall back to the client's direct IP
        return request.client.host if request.client else "unknown"


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware for timing out long-running requests."""

    def __init__(self, app: ASGIApp, timeout: float = 30.0):
        """
        Initialize the middleware.

        Args:
            app: The ASGI application
            timeout: Maximum request processing time in seconds
        """
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process the request with a timeout.

        Args:
            request: The incoming request
            call_next: The next middleware or endpoint

        Returns:
            The response
        """
        import asyncio

        try:
            # Create a task for the request processing
            task = asyncio.create_task(call_next(request))

            # Wait for the task to complete with a timeout
            response = await asyncio.wait_for(task, timeout=self.timeout)
            return response
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {self.timeout}s: {request.url.path}")

            # Return 503 Service Unavailable
            return Response(
                content="Request timed out",
                status_code=503,
                headers={"Retry-After": "10"},
            )
