# app/route/named_entity_recognition.py
from fastapi import APIRouter, Depends
from ..models.named_entity_recognition import NERInput, NERResponse
from ..services.named_entity_recognition import ner_service
from ..dependencies import validate_api_key

router = APIRouter()


@router.post("/ner", response_model=NERResponse)
async def extract_entities(ner_input: NERInput, _=Depends(validate_api_key)):
    """
    Extract named entities from text using dslim/bert-base-NER model .

    - **text**: Text to analyze for named entities

    Returns a dictionary of categorized entities.
    """
    entities = ner_service.process_text(ner_input.text, ner_input.min_score)
    return {"entities": entities}
