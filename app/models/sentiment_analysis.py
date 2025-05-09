# app/model/sentiment_analysis.py
from pydantic import BaseModel, Field
from typing import Dict, List

class SentimentInput(BaseModel):
    text: str = Field(..., description="English Text to analyze for sentiment")

class BatchSentimentInput(BaseModel):
    texts: List[str] = Field(..., description="List of English texts to analyze for sentiment")

class SentimentResponse(BaseModel):
    sentiment: str = Field(..., description="Predicted sentiment (positive or negative)")
    confidence: float = Field(..., description="Confidence score for the prediction")
    probabilities: Dict[str, float] = Field(
        ..., 
        description="Probability scores for each sentiment class"
    )

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse] = Field(..., description="List of sentiment analysis results")
