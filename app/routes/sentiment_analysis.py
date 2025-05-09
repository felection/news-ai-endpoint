# app/route/sentiment_analysis.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Union
from ..models.sentiment_analysis import SentimentInput, SentimentResponse, BatchSentimentInput, BatchSentimentResponse
from ..services.sentiment_analysis import sentiment_service
from ..dependencies import validate_api_key

router = APIRouter()

@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(
    sentiment_input: SentimentInput,
    _ = Depends(validate_api_key)
):
    """
    Analyze sentiment of English text. Positive or Negative sentiment.
    
    - **text**: Text to analyze for sentiment
    
    Returns the sentiment analysis result with confidence scores.
    """
    try:
        result = sentiment_service.analyze_text(sentiment_input.text)
        return SentimentResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing sentiment: {str(e)}"
        )

@router.post("/sentiment/batch", response_model=List[SentimentResponse])
async def analyze_sentiment_batch(
    batch_input: BatchSentimentInput,
    _ = Depends(validate_api_key)
):
    """
    Analyze sentiment of multiple English texts in a batch.
    
    - **texts**: List of texts to analyze for sentiment
    
    Returns a list of sentiment analysis results with confidence scores.
    """
    try:
        results = sentiment_service.analyze_batch(batch_input.texts)
        return [SentimentResponse(**result) for result in results]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing sentiment batch: {str(e)}"
        )
