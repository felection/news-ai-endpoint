# app/services/vector_embedding.py
import torch
from transformers import AutoTokenizer, AutoModel
from ..config import get_settings


class VectorEmbeddingService:
    def __init__(self):
        settings = get_settings()
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.text_to_vector_model_name
        )
        self.model = AutoModel.from_pretrained(settings.text_to_vector_model_name)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.model.to(self.device)
        self.model.eval()

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy().tolist()


embedding_service = VectorEmbeddingService()
