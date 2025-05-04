from fastapi import APIRouter, Depends
from ..models.summarization import SummarizationInput, SummarizationResponse
from ..services.summarization import summarization_service
from ..dependencies import validate_api_key

router = APIRouter()

@router.post("/summarize", response_model=SummarizationResponse)
async def summarize_text(
    summarization_input: SummarizationInput,
    _ = Depends(validate_api_key)
):
    """
    Generates a summary of the input text - in german and english.
    
    - **text**: Text to summarize
    - **max_length**: Maximum length of summary in tokens (default: 150)
    - **min_length**: Minimum length of summary in tokens (default: 80)
    
    Returns the generated summary text.
    """
    summary = summarization_service.generate_summary(
        summarization_input.text,
        summarization_input.max_length,
        summarization_input.min_length
    )
    return {"summary": summary}