from google import genai
from google.genai import types
from typing import Dict, Any, Optional
from ..utils.logging_utils import logger
from ..config import get_settings
from pydantic import BaseModel

settings = get_settings()
print("GEMINI_API_KEY:", settings.gemini_api_key)
client = genai.Client(api_key=settings.gemini_api_key)


class Recipe(BaseModel):
    recipe_name: str
    ingredients: list[str]


class GeminiService:
    """Service for generating text using Google's Gemini API."""

    def __init__(self, default_model: str = "gemini-2.0-flash"):
        self.default_model = default_model
        logger.info(f"Gemini service initialized with default model: {default_model}")
        self.client = client

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

            request_kwargs = {
                "model": model_name,
                "contents": [prompt],
                "config": config,
            }

            response = self.client.models.generate_content(**request_kwargs)

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
