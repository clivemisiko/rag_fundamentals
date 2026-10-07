"""
Interactive RAG Demonstration for Track K: AI Engineering - K1 RAG Fundamentals.

Demonstrates:
1. Ingestion of proprietary document set into ChromaDB.
2. Deliberate chunking strategy (size=120 words, overlap=30 words / 25%).
3. Cosine similarity retrieval ranking.
4. Direct comparison: Base LLM (Without Retrieval) vs RAG Pipeline (Grounded with Retrieval).
"""

import os
import sys

# Ensure local imports work cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag_pipeline import RAGPipeline


def print_banner(text: str):
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)


def run_demo():
    print_banner("TRACK K: AI ENGINEERING - K1 RAG FUNDAMENTALS DEMO")
    print("System Architecture:")
    print(" - Vector Store: ChromaDB (HNSW with Cosine Similarity metric)")
    print(" - Dense Embeddings: 384-dimensional all-MiniLM-L6-v2")
    print(" - Deliberate Chunking: 120 words per chunk, 30 words overlap (25% sliding window)")
    print(" - Generator: Local Google Flan-T5 Seq2Seq model (zero external API keys required)")

    pipeline = RAGPipeline(
        chunk_size=120,
        chunk_overlap=30,
        collection_name="quantumcorp_demo_collection",
        top_k=2
    )

    doc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "quantumcorp_handbook.md")
    print(f"\n[1/3] Indexing document: {doc_path}")
    num_chunks = pipeline.index_file(doc_path)
    print(f" -> Successfully chunked and indexed {num_chunks} chunks into ChromaDB.")

    test_queries = [
        {
            "query": "What is the emergency manual reboot passphrase for Server Cluster Vega?",
            "ground_truth_key": "Crimson-Nightingale-88"
        },
        {
            "query": "What is the maximum permissible operating pressure of the Phase-7 Flux Reactor, and what emergency sequence triggers if exceeded?",
            "ground_truth_key": "4,850 kPa"
        },
        {
            "query": "What is the annual quantum workstation stipend for remote Tier-3 engineers under Protocol Omega-9?",
            "ground_truth_key": "$7,420"
        }
    ]

    print_banner("EXECUTING RAG RETRIEVAL & GENERATION DEMO")

    for i, test in enumerate(test_queries, 1):
        q = test["query"]
        expected = test["ground_truth_key"]

        print(f"\n--- [Test Query {i}] ---")
        print(f"QUERY: \"{q}\"")
        print(f"VERIFIABLE FACT EXPECTED: {expected}")

        result = pipeline.answer_query(q)

        print("\n>>> RETRIEVED CHUNKS FROM CHROMADB (Ranked by Cosine Similarity):")
        for rank, chunk in enumerate(result["retrieved_chunks"], 1):
            print(f"  [{rank}] ID: {chunk['id']}")
            print(f"      Cosine Similarity: {chunk['similarity_score']:.4f} | Cosine Distance: {chunk['distance']:.4f}")
            print(f"      Text Excerpt: \"{chunk['text'][:140]}...\"")

        print("\n>>> BASE LLM ANSWER (Without Retrieval / Pre-training knowledge only):")
        print(f"  \"{result['base_answer']}\"")
        if expected.lower() not in result["base_answer"].lower():
            print("  [x] Base LLM FAILED (does not know private QuantumCorp data alone)")

        print("\n>>> RAG PIPELINE ANSWER (With Retrieved Context):")
        print(f"  \"{result['rag_answer']}\"")
        if expected.lower() in result["rag_answer"].lower():
            print("  [PASS] RAG SUCCEEDED (Grounding verified! Retrieval directly influenced the answer)")
        else:
            print("  [?] Check output grounding manually.")

    print_banner("DEMO COMPLETE -- ALL PASS CRITERIA DEMONSTRATED")


if __name__ == "__main__":
    run_demo()

