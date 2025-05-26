from google import genai
from google.genai import types
from typing import Dict, Any, Optional
from ..utils.logging_utils import logger
from ..config import get_settings
import os

settings = get_settings()
client = genai.Client(api_key=settings.gemini_api_key)


class GeminiKeyManager:
    """Manages multiple Gemini API keys and rotates them for each request."""

    def __init__(self, primary_key=settings.gemini_api_key):
        # Initialize with the main key from settings
        self.keys = [primary_key] if primary_key else []
        self.current_index = 0

        # Look for additional keys in environment variables (GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.)
        i = 1
        while True:
            key_name = f"GEMINI_API_KEY_{i}"
            key = os.environ.get(key_name)
            if key:
                self.keys.append(key)
                logger.info(f"Added additional Gemini API key: {key_name}")
                i += 1
            else:
                break

        logger.info(f"Initialized Gemini key manager with {len(self.keys)} keys")

    def get_next_key(self) -> str:
        """Get the next API key in the rotation."""
        if not self.keys:
            raise ValueError("No Gemini API keys available")
        print("Rotating keys", self.keys)
        key = self.keys[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.keys)
        return key

    def get_client(self):
        """Get a Google Genai client with the next API key."""
        return genai.Client(api_key=self.get_next_key())


key_manager = GeminiKeyManager()


class GeminiService:
    """Service for generating text using Google's Gemini API."""

    def __init__(self, default_model: str = "gemini-2.0-flash"):
        self.default_model = default_model
        logger.info(f"Gemini service initialized with default model: {default_model}")
        # self.client = client

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        max_output_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Generate text using Gemini API.

        Args:
            prompt: Text prompt to generate from
            model: Gemini model to use (defaults to the configured default)
            temperature: Controls randomness (lower is more deterministic)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            max_output_tokens: Maximum number of tokens to generate

        Returns:
            Dictionary with generated text and response metadata
        """
        try:
            model_name = model or self.default_model

            config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
            )
            # Get a fresh client with the next API key
            client = key_manager.get_client()
            print(f"Using model: {model_name} with key: {client._api_client.api_key}")

            request_kwargs = {
                "model": model_name,
                "contents": [prompt],
                "config": config,
            }

            response = client.models.generate_content(**request_kwargs)

            result = {
                "generated_text": getattr(response, "text", None),
                "model": model_name,
                "parameters": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "max_output_tokens": max_output_tokens,
                },
            }

            return result

        except Exception as e:
            logger.error(f"Error generating text with Gemini: {str(e)}")
            raise


gemini_service = GeminiService()
