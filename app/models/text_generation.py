from pydantic import BaseModel, Field
from typing import List, Optional


class RephraseInput(BaseModel):
    text: str = Field(..., description="Text to rephrase")
    max_tokens: int = Field(500, description="Maximum number of tokens to generate")


class RephraseResponse(BaseModel):
    rephrased_text: str = Field(..., description="Rephrased text")


class TranslationGenInput(BaseModel):
    text: str = Field(..., description="Text to translate")
    target_language: str = Field(..., description="Target language for translation")
    max_tokens: int = Field(500, description="Maximum number of tokens to generate")


class TranslationGenResponse(BaseModel):
    translated_text: str = Field(..., description="Translated text")
    target_language: str = Field(..., description="Target language")


class TextGenerationInput(BaseModel):
    prompt: str = Field(..., description="Text prompt to generate from")
    max_tokens: int = Field(500, description="Maximum number of tokens to generate")
    temperature: float = Field(
        0.7, description="Controls randomness (lower is more deterministic)"
    )
    top_p: float = Field(0.95, description="Nucleus sampling parameter")
    top_k: int = Field(40, description="Top-k sampling parameter")
    stop: Optional[List[str]] = Field(
        None, description="List of strings to stop generation at"
    )


class TextGenerationResponse(BaseModel):
    generated_text: str = Field(..., description="Generated text")
    parameters: dict = Field(..., description="Parameters used for generation")
