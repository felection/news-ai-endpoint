# app/models/summarization.py
from pydantic import BaseModel, Field


class SummarizationInput(BaseModel):
    text: str = Field(..., description="Text to summarize")
    max_length: int = Field(150, description="Maximum length of summary in tokens")
    min_length: int = Field(80, description="Minimum length of summary in tokens")


class SummarizationResponse(BaseModel):
    summary: str = Field(..., description="Generated summary text")
