from fastapi import APIRouter, Depends, HTTPException
from ..models.gemini import GeminiInput, GeminiResponse
from ..services.gemini_service import gemini_service
from ..dependencies import validate_api_key

router = APIRouter()


@router.post("/gemini/generate", response_model=GeminiResponse)
async def generate_text_with_gemini(
    gemini_input: GeminiInput, _=Depends(validate_api_key)
):
    """
    Generate text using Google's Gemini API.

    - **prompt**: Text prompt to generate from
    - **model**: Gemini model to use (defaults to the configured default)
    - **temperature**: Controls randomness (lower is more deterministic)
    - **top_p**: Nucleus sampling parameter
    - **top_k**: Top-k sampling parameter
    - **max_output_tokens**: Maximum number of tokens to generate

    Returns the generated text and response metadata.
    """
    try:
        result = gemini_service.generate_text(
            prompt=gemini_input.prompt,
            model=gemini_input.model,
            temperature=gemini_input.temperature,
            top_p=gemini_input.top_p,
            top_k=gemini_input.top_k,
            max_output_tokens=gemini_input.max_output_tokens,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating text with Gemini: {str(e)}"
        )
