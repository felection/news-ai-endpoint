from fastapi import APIRouter, Depends, HTTPException
from ..models.text_generation import (
    RephraseInput,
    RephraseResponse,
    TranslationGenInput,
    TranslationGenResponse,
    TextGenerationInput,
    TextGenerationResponse,
)
from ..services.text_generation import text_generation_service
from ..dependencies import validate_api_key

router = APIRouter()


@router.post("/llm/rephrase", response_model=RephraseResponse)
async def rephrase_text(rephrase_input: RephraseInput, _=Depends(validate_api_key)):
    """
    Rephrase text using different wording while keeping the same meaning.

    - **text**: Text to rephrase
    - **max_tokens**: Maximum number of tokens to generate

    Returns the rephrased text and processing time.
    """
    try:
        result = text_generation_service.rephrase_text(
            rephrase_input.text, rephrase_input.max_tokens
        )
        return RephraseResponse(
            rephrased_text=result["rephrased_text"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rephrasing text: {str(e)}")


@router.post("/llm/translate", response_model=TranslationGenResponse)
async def translate_text(
    translation_input: TranslationGenInput, _=Depends(validate_api_key)
):
    """
    Translate text to the specified language.

    - **text**: Text to translate
    - **target_language**: Target language for translation
    - **max_tokens**: Maximum number of tokens to generate

    Returns the translated text and processing time.
    """
    try:
        result = text_generation_service.translate_text(
            translation_input.text,
            translation_input.target_language,
            translation_input.max_tokens,
        )
        return TranslationGenResponse(
            translated_text=result["translated_text"],
            target_language=result["target_language"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error translating text: {str(e)}")


@router.post("/llm/generate", response_model=TextGenerationResponse)
async def generate_text(
    generation_input: TextGenerationInput, _=Depends(validate_api_key)
):
    """
    Generate text based on a prompt with customizable parameters.

    - **prompt**: Text prompt to generate from
    - **max_tokens**: Maximum number of tokens to generate
    - **temperature**: Controls randomness (lower is more deterministic)
    - **top_p**: Nucleus sampling parameter
    - **top_k**: Top-k sampling parameter
    - **stop**: List of strings to stop generation at

    Returns the generated text.
    """
    try:
        result = text_generation_service.generate_text(
            generation_input.prompt,
            generation_input.max_tokens,
            generation_input.temperature,
            generation_input.top_p,
            generation_input.top_k,
            generation_input.stop,
        )
        return TextGenerationResponse(
            generated_text=result["generated_text"],
            parameters=result["parameters"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")
