"""
IntelliRAG — API-based Embeddings
Thin wrapper around NVIDIA NIM (OpenAI-compatible) embeddings endpoint.
No local model download — uses the openai SDK already in requirements.
"""

from openai import OpenAI
from backend.config import settings


class NVIDIAEmbeddings:
    """Embeddings via NVIDIA NIM API. Zero local memory — pure API calls."""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model = settings.embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
