# app/models/translation.py
from pydantic import BaseModel, Field


class TranslationInput(BaseModel):
    text: str = Field(..., description="German text to translate")
    max_chunk_size: int = Field(128, description="Maximum size of each chunk in tokens")
    direction: str = Field(
        "de-en", description="Translation direction (de-en or en-de)"
    )


class TranslationResponse(BaseModel):
    translated_text: str = Field(..., description="Translated English text")
