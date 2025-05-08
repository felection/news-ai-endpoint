# app/routes/emotion_extraction.py
from fastapi import APIRouter, Depends, HTTPException
from ..models.emotion_extraction import EmotionInput, EmotionResponse
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
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing emotions: {str(e)}"
        )