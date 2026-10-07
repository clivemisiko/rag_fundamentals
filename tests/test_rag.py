"""
Automated Pytest Suite for Track K: AI Engineering - K1 RAG Fundamentals.

Verifies:
1. Deliberate chunking strategy (size, overlap, and non-default constraints).
2. ChromaDB vector indexing and Cosine Similarity ranking.
3. Retrieval influence test: proves Base LLM fails alone, whereas RAG pipeline succeeds.
4. Grounded answer verification for proprietary document facts.
"""

import os
import pytest
from src.chunking import DeliberateChunker
from src.vector_store import ChromaVectorStore
from src.rag_pipeline import RAGPipeline


@pytest.fixture(scope="session")
def sample_document():
    doc_path = os.path.join(os.path.dirname(__file__), "..", "data", "quantumcorp_handbook.md")
    with open(doc_path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="session")
def rag_pipeline(sample_document):
    pipeline = RAGPipeline(
        chunk_size=120,
        chunk_overlap=30,
        collection_name="quantumcorp_test_collection",
        top_k=2
    )
    pipeline.index_document(sample_document, source_name="handbook")
    return pipeline


def test_deliberate_chunking_strategy(sample_document):
    """Verify that chunking is a deliberate choice of size and overlap, not an arbitrary default."""
    chunker = DeliberateChunker(chunk_size=120, chunk_overlap=30)
    chunks = chunker.chunk_document(sample_document, source_name="test_doc")

    assert len(chunks) > 1, "Document should be partitioned into multiple chunks"

    # Verify size constraint
    for c in chunks:
        assert c["metadata"]["word_count"] <= 120

    # Verify sliding window overlap constraint: consecutive chunks must share words
    first_chunk_words = set(chunks[0]["text"].split()[-30:])
    second_chunk_words = set(chunks[1]["text"].split()[:30])
    common_words = first_chunk_words.intersection(second_chunk_words)
    assert len(common_words) > 10, "Consecutive chunks must share words representing the deliberate overlap"

    # Verify error raised if overlap >= size
    with pytest.raises(ValueError):
        DeliberateChunker(chunk_size=50, chunk_overlap=50)


def test_vector_store_cosine_ranking(sample_document):
    """Verify ChromaDB vector store indexing and Cosine Similarity metric."""
    chunker = DeliberateChunker(chunk_size=120, chunk_overlap=30)
    chunks = chunker.chunk_document(sample_document, source_name="test_doc")

    vs = ChromaVectorStore(collection_name="test_ranking_collection")
    vs.add_chunks(chunks)
    assert vs.count() == len(chunks)

    # Query specific to Server Cluster Vega
    query = "What is the reboot passphrase for Server Cluster Vega?"
    retrieved = vs.retrieve(query, top_k=2)

    assert len(retrieved) == 2
    top_chunk = retrieved[0]

    # Verify Cosine Similarity metrics
    assert 0.0 <= top_chunk["similarity_score"] <= 1.0
    assert 0.0 <= top_chunk["distance"] <= 1.0

    # Top chunk must contain the specific section
    assert "Cluster Vega" in top_chunk["text"]
    assert "Crimson-Nightingale-88" in top_chunk["text"]


def test_retrieval_actually_influences_answer_pass_criteria(rag_pipeline):
    """
    PASS CRITERIA CORE TEST:
    Demonstrates that retrieval actually influences the answer:
    - Base LLM without retrieval gets the answer WRONG (fails alone).
    - RAG Pipeline with retrieval gets the answer RIGHT (grounded).
    """
    query = "What is the emergency manual reboot passphrase for Server Cluster Vega?"
    expected_secret = "Crimson-Nightingale-88"

    result = rag_pipeline.answer_query(query)

    base_answer = result["base_answer"]
    rag_answer = result["rag_answer"]

    # 1. Base LLM without context gets it wrong alone
    assert expected_secret.lower() not in base_answer.lower(), (
        f"Base LLM should not know private secret without retrieval, but got: '{base_answer}'"
    )

    # 2. RAG Pipeline with retrieved context correctly extracts the secret
    assert expected_secret.lower() in rag_answer.lower(), (
        f"RAG answer should contain '{expected_secret}', but got: '{rag_answer}'"
    )


def test_flux_reactor_grounding(rag_pipeline):
    """Test verifiable grounding for numerical technical specification."""
    query = "What is the maximum permissible operating pressure of the Phase-7 Flux Reactor?"
    expected_pressure = "4,850"

    result = rag_pipeline.answer_query(query)
    assert expected_pressure in result["rag_answer"] or "4850" in result["rag_answer"]

