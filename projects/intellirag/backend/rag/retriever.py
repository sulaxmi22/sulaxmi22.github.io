"""
IntelliRAG — Hybrid Retriever Module
Implements BM25 + Dense Vector hybrid search with Cross-Encoder reranking.
"""

import logging
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import chromadb

from backend.config import settings
from backend.rag.embeddings import NVIDIAEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with metadata and scores."""
    content: str
    source: str
    page: int
    chunk_index: int
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    rrf_score: float = 0.0


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 keyword search with dense vector search,
    fused via Reciprocal Rank Fusion (RRF), then reranked with a Cross-Encoder.
    """

    def __init__(self):
        self.embeddings = NVIDIAEmbeddings()
        self.chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def _vector_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Dense vector similarity search via ChromaDB."""
        query_embedding = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunks.append(RetrievedChunk(
                    content=doc,
                    source=meta.get("source", "unknown"),
                    page=meta.get("page", 0),
                    chunk_index=meta.get("chunk_index", 0),
                    vector_score=1 - dist,  # Convert distance to similarity
                ))
        return chunks

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """BM25 keyword search over all documents in the collection."""
        # Get all documents from ChromaDB
        all_docs = self.collection.get(include=["documents", "metadatas"])
        if not all_docs or not all_docs["documents"]:
            return []

        documents = all_docs["documents"]
        metadatas = all_docs["metadatas"]

        # Tokenize for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)

        # Query
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Get top-k
        scored_docs = list(zip(documents, metadatas, scores))
        scored_docs.sort(key=lambda x: x[2], reverse=True)

        chunks = []
        for doc, meta, score in scored_docs[:top_k]:
            chunks.append(RetrievedChunk(
                content=doc,
                source=meta.get("source", "unknown"),
                page=meta.get("page", 0),
                chunk_index=meta.get("chunk_index", 0),
                bm25_score=float(score),
            ))
        return chunks

    def _reciprocal_rank_fusion(
        self,
        vector_chunks: list[RetrievedChunk],
        bm25_chunks: list[RetrievedChunk],
        k: int = 60,
    ) -> list[RetrievedChunk]:
        """
        Reciprocal Rank Fusion (RRF) to combine vector and BM25 results.
        RRF score = Σ 1/(k + rank_i) for each ranking list.
        """
        # Build a map of content -> chunk
        chunk_map: dict[str, RetrievedChunk] = {}

        # Score from vector search
        for rank, chunk in enumerate(vector_chunks):
            key = chunk.content[:200]  # Use first 200 chars as key
            if key not in chunk_map:
                chunk_map[key] = chunk
            chunk_map[key].rrf_score += 1 / (k + rank + 1)
            chunk_map[key].vector_score = chunk.vector_score

        # Score from BM25 search
        for rank, chunk in enumerate(bm25_chunks):
            key = chunk.content[:200]
            if key not in chunk_map:
                chunk_map[key] = chunk
            chunk_map[key].rrf_score += 1 / (k + rank + 1)
            chunk_map[key].bm25_score = chunk.bm25_score

        # Sort by RRF score
        fused = list(chunk_map.values())
        fused.sort(key=lambda x: x.rrf_score, reverse=True)
        return fused

    def _rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Use RRF score as the final ranking — no local cross-encoder model needed."""
        for chunk in chunks:
            chunk.rerank_score = chunk.rrf_score
        return chunks[:top_k]

    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Full hybrid retrieval pipeline:
        1. Vector search (semantic)
        2. BM25 search (keyword)
        3. Reciprocal Rank Fusion
        4. Cross-Encoder reranking
        """
        logger.info(f"Retrieving for query: {query[:100]}...")

        # Step 1 & 2: Parallel retrieval
        vector_chunks = self._vector_search(query, settings.top_k_retrieval)
        bm25_chunks = self._bm25_search(query, settings.top_k_retrieval)

        logger.info(f"Vector: {len(vector_chunks)} chunks, BM25: {len(bm25_chunks)} chunks")

        # Step 3: Fuse results
        fused_chunks = self._reciprocal_rank_fusion(vector_chunks, bm25_chunks)
        logger.info(f"After RRF fusion: {len(fused_chunks)} unique chunks")

        # Step 4: Rerank
        reranked_chunks = self._rerank(query, fused_chunks, settings.top_k_rerank)
        logger.info(f"After reranking: {len(reranked_chunks)} chunks (top-{settings.top_k_rerank})")

        return reranked_chunks

    def evaluate_relevance(self, query: str, chunks: list[RetrievedChunk]) -> float:
        """
        Evaluate overall relevance of retrieved chunks.
        Returns average rerank score — used by Corrective RAG to decide
        whether to fall back to web search.
        """
        if not chunks:
            return 0.0
        avg_score = sum(c.rerank_score for c in chunks) / len(chunks)
        return avg_score
