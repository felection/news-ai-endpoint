from pydantic import BaseModel, Field
from typing import Dict, Any, Literal


class GeminiInput(BaseModel):
    prompt: str = Field(..., description="Text prompt to generate from")
    model: Literal[
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ] = Field(
        "gemini-2.0-flash",
        description="Gemini model to use (defaults to configured default)",
    )
    temperature: float = Field(
        0.7, description="Controls randomness (lower is more deterministic)"
    )
    top_p: float = Field(0.95, description="Nucleus sampling parameter")
    top_k: int = Field(40, description="Top-k sampling parameter")
    max_output_tokens: int = Field(
        1024, description="Maximum number of tokens to generate"
    )


class GeminiResponse(BaseModel):
    generated_text: str = Field(..., description="Generated text from the model")
    model: str = Field(..., description="Model used for generation")
    parameters: Dict[str, Any] = Field(
        ..., description="Parameters used for generation"
    )
