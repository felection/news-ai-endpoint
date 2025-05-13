# app/models/vector_embedding.py
from pydantic import BaseModel
from typing import List


class TextInput(BaseModel):
    text: str


class TextsInput(BaseModel):
    texts: List[str]


class EmbeddingResponse(BaseModel):
    embedding: List[float]


class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
