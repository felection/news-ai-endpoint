import os
from typing import List, Union

import torch
from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Constants from environment
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_V1_STR = os.getenv("API_V1_STR", "/api/v1")
MODEL_NAME = os.getenv("TEXT_TO_VECTOR_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
API_SECRET = os.getenv("API_SECRET", "")
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Initialize FastAPI app
app = FastAPI(
    title="Text Embedding API",
    description="Simple API for text embedding generation",
    version="1.0.0",
    docs_url=f"{API_V1_STR}/docs",
    redoc_url=f"{API_V1_STR}/redoc",
    debug=DEBUG_MODE
)

# Security setup
api_key_header = APIKeyHeader(name="X-API-Key")

async def validate_api_key(api_key: str = Depends(api_key_header)):
    if API_SECRET and api_key != API_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

# Load model and tokenizer (only once at startup)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Set model to eval mode and move to GPU if available
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Request/Response models
class TextInput(BaseModel):
    text: str

class TextsInput(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    embedding: List[float]

class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

# Helper function for embedding generation
def generate_embeddings(texts: List[str]) -> List[List[float]]:
    # Tokenize and move to same device as model
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=256
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy().tolist()

# Endpoints
@app.get(API_V1_STR + "/", include_in_schema=False)
async def root():
    return {
        "message": "Text Embedding API",
        "endpoints": [
            {"path": API_V1_STR + "/embed", "method": "POST", "description": "Generate embedding for single text"},
            {"path": API_V1_STR + "/embed/batch", "method": "POST", "description": "Generate embeddings for multiple texts"}
        ]
    }

@app.post(API_V1_STR + "/embed", response_model=EmbeddingResponse)
async def embed_text(
    text_input: TextInput,
    api_key: str = Depends(validate_api_key) if API_SECRET else None
):
    """Generate embedding for a single text"""
    embedding = generate_embeddings([text_input.text])[0]
    return {"embedding": embedding}

@app.post(API_V1_STR + "/embed/batch", response_model=BatchEmbeddingResponse)
async def embed_texts_batch(
    texts_input: TextsInput,
    api_key: str = Depends(validate_api_key) if API_SECRET else None
):
    """Generate embeddings for multiple texts in a batch"""
    embeddings = generate_embeddings(texts_input.texts)
    return {"embeddings": embeddings}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        log_level="info",
        reload=DEBUG_MODE
    )