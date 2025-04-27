# routes/vector_embedding.py
from fastapi import APIRouter, Depends
from ..models.vector_embedding import TextInput, TextsInput, EmbeddingResponse, BatchEmbeddingResponse
from ..services.vector_embedding import embedding_service
from ..dependencies import validate_api_key

router = APIRouter()

@router.post("/embed", response_model=EmbeddingResponse)
async def embed_text(
    text_input: TextInput,
    _ = Depends(validate_api_key)
):
    embedding = embedding_service.generate_embeddings([text_input.text])[0]
    return {"embedding": embedding}

@router.post("/embed/batch", response_model=BatchEmbeddingResponse)
async def embed_texts_batch(
    texts_input: TextsInput,
    _ = Depends(validate_api_key)
):
    embeddings = embedding_service.generate_embeddings(texts_input.texts)
    return {"embeddings": embeddings}