# app/model/emotion_extraction.py
from pydantic import BaseModel, Field
from typing import Dict

class EmotionInput(BaseModel):
    text: str = Field(..., description="Text to analyze for emotions")

class EmotionResponse(BaseModel):
    emotions: Dict[str, float] = Field(
        ..., 
        description="Detected emotions and their confidence scores, sorted by confidence"
    )
    dominant_emotion: str = Field(
        ...,
        description="The emotion with the highest confidence score"
    )