# app/routes/emotion_extraction.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Union
from ..models.emotion_extraction import EmotionInput, EmotionResponse, BatchEmotionInput, BatchEmotionResponse
from ..services.emotion_extraction import emotion_service
from ..dependencies import validate_api_key

router = APIRouter()

@router.post("/emotions", response_model=EmotionResponse)
async def analyze_emotions(
    emotion_input: EmotionInput,
    _ = Depends(validate_api_key)
):
    """
    Analyze emotions in text.
    
    - **text**: Text to analyze for emotions
    
    Returns detected emotions with confidence scores and the dominant emotion.
    """
    try:
        result = emotion_service.detect_emotions(emotion_input.text)
        return EmotionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing emotions: {str(e)}"
        )

@router.post("/emotions/batch", response_model=List[EmotionResponse])
async def analyze_emotions_batch(
    batch_input: BatchEmotionInput,
    _ = Depends(validate_api_key)
):
    """
    Analyze emotions in multiple texts in a batch.
    
    - **texts**: List of texts to analyze for emotions
    
    Returns a list of detected emotions with confidence scores and dominant emotions.
    """
    try:
        results = emotion_service.detect_emotions_batch(batch_input.texts)
        return [EmotionResponse(**result) for result in results]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing emotions batch: {str(e)}"
        )
