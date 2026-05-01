"""
IntelliRAG — Configuration Management
Centralized configuration using Pydantic Settings for type safety and validation.
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # API Keys
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible API base URL (e.g., NVIDIA NIM, Ollama)",
    )
    openai_model: str = Field(default="gpt-4o-mini", description="LLM model name")

    # Embedding
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )

    # Retrieval
    chunk_size: int = Field(default=512, description="Document chunk size in characters")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")
    top_k_retrieval: int = Field(default=10, description="Number of chunks to retrieve")
    top_k_rerank: int = Field(default=5, description="Number of chunks after reranking")
    relevance_threshold: float = Field(
        default=0.3, description="Minimum relevance score for retrieval"
    )

    # Reranker
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking",
    )

    # ChromaDB
    chroma_persist_dir: str = Field(
        default="./chroma_db", description="ChromaDB persistence directory"
    )
    chroma_collection: str = Field(
        default="intellirag_docs", description="ChromaDB collection name"
    )

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    cors_origins: list[str] = Field(default=["*"])

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
