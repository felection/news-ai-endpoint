"""
Text Generation service using OLMo-2-1B and other models.

This module provides functionality for text generation tasks like:
- Text rephrasing
- Translation
- Other text generation tasks
"""

from typing import Dict, Any, Optional, List
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from ..utils.logging_utils import logger
from ..config import get_settings

settings = get_settings()


class TextGenerationService:
    """Service for text generation using LLM models."""

    def __init__(
        self,
        model_repo=settings.text_generation_model_repo,
        model_file=settings.text_generation_model_file,
    ):
        """
        Initialize the text generation service.

        Args:
            model_repo: Repository ID for the model
            model_file: Filename of the model to download
        """
        self.model_repo = model_repo
        self.model_file = model_file
        self._llm = None
        logger.info(
            f"Text generation service initialized with model: {model_repo}/{model_file}"
        )

    @property
    def llm(self):
        """Lazy loading of the LLM model."""
        if self._llm is None:
            try:
                logger.info(
                    f"Loading text generation model: {self.model_repo}/{self.model_file}"
                )

                # Download the model if not already available
                model_path = hf_hub_download(
                    repo_id=self.model_repo, filename=self.model_file
                )

                # Load the model with llama-cpp
                self._llm = Llama(
                    model_path=model_path,
                    n_ctx=settings.text_generation_context_length,
                    verbose=False,
                )

                logger.info("Text generation model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading text generation model: {str(e)}")
                raise

        return self._llm

    def rephrase_text(self, text: str, max_tokens: int = 500) -> Dict[str, Any]:
        """
        Rephrase text using different wording while keeping the same meaning.

        Args:
            text: Text to rephrase
            max_tokens: Maximum number of tokens to generate

        Returns:
            Dictionary with rephrased text
        """

        try:
            prompt = f"""Rephrase the following sentence using different words but the same meaning. 
            Only return the rephrased sentence. 
            Text: "{text}" 
            Rephrased:"""

            result = self.llm(
                prompt, max_tokens=max_tokens, stop=["</s>", "\n"], echo=False
            )
            rephrased_text = result["choices"][0]["text"].strip()

            return {
                "rephrased_text": rephrased_text,
                "usage": result.get("usage", {}),
            }
        except Exception as e:
            logger.error(f"Error rephrasing text: {str(e)}")
            raise

    def translate_text(
        self, text: str, target_language: str, max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Translate text to the specified language.

        Args:
            text: Text to translate
            target_language: Target language for translation
            max_tokens: Maximum number of tokens to generate

        Returns:
            Dictionary with translated text
        """

        try:
            prompt = f"""Translate the following sentence to {target_language}. Only return the translated sentence.
            Text: {text}
            Translation:"""

            result = self.llm(prompt, max_tokens=max_tokens, echo=False)
            translated_text = result["choices"][0]["text"].strip()

            return {
                "translated_text": translated_text,
                "target_language": target_language,
                "usage": result.get("usage", {}),
            }
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            raise

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate text based on a prompt with customizable parameters.

        Args:
            prompt: Text prompt to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (lower is more deterministic)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop: List of strings to stop generation at

        Returns:
            Dictionary with generated text
        """

        try:
            stop_sequences = stop or ["</s>", "\n\n"]

            result = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop_sequences,
                echo=False,
            )

            generated_text = result["choices"][0]["text"].strip()

            return {
                "generated_text": generated_text,
                "usage": result.get("usage", {}),
                "parameters": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                },
            }
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise


# Create singleton instance
text_generation_service = TextGenerationService()
