"""
End-to-end Retrieval-Augmented Generation (RAG) Pipeline.
Coordinates Document Ingestion -> Deliberate Chunking -> Vector Store Indexing -> Retrieval -> LLM Generation.
"""

from typing import List, Dict, Optional
import os

from src.chunking import DeliberateChunker
from src.vector_store import ChromaVectorStore
from src.llm import LLMClient


class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 120,
        chunk_overlap: int = 30,
        collection_name: str = "quantumcorp_docs",
        top_k: int = 2,
        llm_provider: str = "local"
    ):
        self.chunker = DeliberateChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = ChromaVectorStore(collection_name=collection_name)
        self.llm = LLMClient(provider=llm_provider)
        self.top_k = top_k

    def index_document(self, text: str, source_name: str = "document") -> int:
        """Chunks and stores a document in the vector database."""
        chunks = self.chunker.chunk_document(text, source_name=source_name)
        self.vector_store.add_chunks(chunks)
        return len(chunks)

    def index_file(self, file_path: str) -> int:
        """Reads a file from disk and indexes it."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        source_name = os.path.basename(file_path)
        return self.index_document(content, source_name=source_name)

    def answer_query(self, query: str, top_k: Optional[int] = None) -> Dict:
        """
        Executes the full RAG workflow for a query:
        1. Retrieves top-k chunks with cosine similarity.
        2. Constructs grounded context.
        3. Generates base answer (without context) to demonstrate contrast.
        4. Generates RAG answer (with context).
        """
        k = top_k or self.top_k
        retrieved_chunks = self.vector_store.retrieve(query, top_k=k)

        # Build context from retrieved chunks
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_blocks.append(f"[Chunk {i} | Similarity: {chunk['similarity_score']}]:\n{chunk['text']}")
        context_str = "\n\n".join(context_blocks)

        # Baseline generation (No Retrieval)
        base_answer = self.llm.generate_base_answer(query)

        # RAG generation (With Retrieval)
        rag_answer = self.llm.generate_rag_answer(query, context_str)

        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "context_used": context_str,
            "base_answer": base_answer,
            "rag_answer": rag_answer
        }

