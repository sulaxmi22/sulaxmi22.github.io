"""
IntelliRAG — Document Ingestion Module
Handles document loading, smart chunking, and embedding storage.
"""

import os
import hashlib
import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

from backend.config import settings

logger = logging.getLogger(__name__)

# Supported file formats and their loaders
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
}


class DocumentIngestion:
    """Handles document ingestion: loading, chunking, embedding, and storage."""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def _get_loader(self, file_path: str):
        """Get the appropriate document loader for a file type."""
        ext = Path(file_path).suffix.lower()
        loader_class = LOADER_MAP.get(ext)
        if not loader_class:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {list(LOADER_MAP.keys())}")
        return loader_class(file_path)

    def _generate_chunk_id(self, content: str, source: str, index: int) -> str:
        """Generate a deterministic ID for a chunk to avoid duplicates."""
        raw = f"{source}:{index}:{content[:100]}"
        return hashlib.md5(raw.encode()).hexdigest()

    async def ingest_file(self, file_path: str) -> dict:
        """
        Ingest a single file: load, chunk, embed, and store.
        
        Returns:
            dict with ingestion statistics
        """
        logger.info(f"Ingesting file: {file_path}")

        # Load document
        loader = self._get_loader(file_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from {file_path}")

        # Chunk documents
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")

        # Prepare for ChromaDB
        source_name = Path(file_path).name
        ids = []
        texts = []
        metadatas = []
        embeddings_list = []

        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(chunk.page_content, source_name, i)
            ids.append(chunk_id)
            texts.append(chunk.page_content)
            metadatas.append({
                "source": source_name,
                "chunk_index": i,
                "page": chunk.metadata.get("page", 0),
                "total_chunks": len(chunks),
            })

        # Generate embeddings in batch
        embeddings_list = self.embeddings.embed_documents(texts)

        # Upsert to ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings_list,
            metadatas=metadatas,
        )

        result = {
            "file": source_name,
            "pages_loaded": len(documents),
            "chunks_created": len(chunks),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "status": "success",
        }
        logger.info(f"Ingestion complete: {result}")
        return result

    async def ingest_text(self, text: str, source_name: str = "direct_input") -> dict:
        """Ingest raw text directly."""
        from langchain_core.documents import Document

        doc = Document(page_content=text, metadata={"source": source_name})
        chunks = self.text_splitter.split_documents([doc])

        ids = []
        texts = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(chunk.page_content, source_name, i)
            ids.append(chunk_id)
            texts.append(chunk.page_content)
            metadatas.append({
                "source": source_name,
                "chunk_index": i,
                "page": 0,
                "total_chunks": len(chunks),
            })

        embeddings_list = self.embeddings.embed_documents(texts)

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings_list,
            metadatas=metadatas,
        )

        return {
            "source": source_name,
            "chunks_created": len(chunks),
            "status": "success",
        }

    def get_collection_stats(self) -> dict:
        """Get statistics about the current document collection."""
        count = self.collection.count()
        return {
            "collection": settings.chroma_collection,
            "total_chunks": count,
            "persist_dir": settings.chroma_persist_dir,
        }

    def clear_collection(self):
        """Clear all documents from the collection."""
        self.chroma_client.delete_collection(settings.chroma_collection)
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection cleared")
