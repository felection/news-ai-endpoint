# app/models/named_entity_recognition.py
from pydantic import BaseModel, Field
from typing import Dict, List


class Entity(BaseModel):
    text: str
    score: float


class NERInput(BaseModel):
    text: str = Field(..., description="Text to analyze for named entities")
    min_score: float = Field(
        0.70, description="Minimum score for entity classification"
    )


class NERResponse(BaseModel):
    entities: Dict[str, List[Entity]] = Field(
        ..., description="Dictionary of categorized named entities"
    )
