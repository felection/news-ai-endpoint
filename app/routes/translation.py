# app/routes/translation.py
from fastapi import APIRouter, Depends
from ..models.translation import TranslationInput, TranslationResponse
from ..services.translation import translation_service
from ..dependencies import validate_api_key

router = APIRouter()

@router.post("/translate", response_model=TranslationResponse)
async def translate_text(
    translation_input: TranslationInput,
    _ = Depends(validate_api_key)
):
    """
    Translates German text to English.
    
    - **text**: German text to translate
    - **max_chunk_size**: Maximum size of each chunk in tokens (default: 128)
    
    Returns the translated English text.
    """
    translated_text = translation_service.translate_in_chunks(
        translation_input.text, 
        translation_input.max_chunk_size
    )
    return {"translated_text": translated_text}