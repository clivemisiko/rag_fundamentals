"""
Vector Store module utilizing ChromaDB with dense embeddings and Cosine Similarity retrieval.

Retrieval Ranking & Similarity Explanation:
-------------------------------------------
1. Dense Vector Embeddings:
   Text chunks and queries are projected into a 384-dimensional continuous vector space
   using 'all-MiniLM-L6-v2' (via Chroma's default sentence-transformers embedding function).
   Semantic meaning is captured geometrically: texts with similar semantics map to nearby vectors.

2. Distance Metric (Cosine Distance):
   ChromaDB is configured with {"hnsw:space": "cosine"}.
   Cosine distance between query vector u and document vector v is defined as:
       D_cosine(u, v) = 1 - (u · v) / (||u||_2 * ||v||_2)
   
   Cosine Similarity is directly derived as:
       Similarity(u, v) = 1 - D_cosine(u, v) = (u · v) / (||u||_2 * ||v||_2)
   
   Unlike Euclidean (L2) distance, Cosine Similarity normalizes for vector length,
   ensuring that document chunk length differences do not artificially skew ranking.

3. Top-k Ranking Step:
   For any incoming query, ChromaDB indexes vectors using HNSW (Hierarchical Navigable Small World)
   graphs to retrieve the top-k chunks with the lowest cosine distance (highest cosine similarity).
"""

from typing import List, Dict, Optional
import chromadb
from chromadb.utils import embedding_functions


class ChromaVectorStore:
    def __init__(self, collection_name: str = "quantumcorp_docs", persist_directory: Optional[str] = None):
        """
        Initializes an in-memory or persistent ChromaDB vector store.
        """
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        # Sentence-transformers / all-MiniLM-L6-v2 embedding function
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # Initialize collection with explicit cosine distance metric
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict]):
        """Adds a list of chunk dictionaries to the vector store."""
        if not chunks:
            return

        ids = [chunk["id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Queries the vector collection and returns ranked chunks with similarity scores.

        Returns:
            List of dicts: [
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "similarity_score": round(1 - distance, 4),
                    "distance": round(distance, 4),
                    "metadata": chunk_metadata
                }
            ]
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        ranked_chunks = []
        if not results or not results["documents"] or not results["documents"][0]:
            return ranked_chunks

        docs = results["documents"][0]
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metadatas, distances):
            # In cosine space: similarity = 1 - distance
            similarity = max(0.0, min(1.0, 1.0 - dist))
            ranked_chunks.append({
                "id": chunk_id,
                "text": doc,
                "similarity_score": round(similarity, 4),
                "distance": round(dist, 4),
                "metadata": meta
            })

        return ranked_chunks

    def count(self) -> int:
        """Returns total number of chunks stored."""
        return self.collection.count()

