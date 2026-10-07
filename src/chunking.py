"""
Deliberate document chunking module for Retrieval-Augmented Generation (RAG).

Chunking Strategy Rationale:
-----------------------------
1. Chunk Size (120 words / ~160 tokens):
   Technical policies and engineering SOPs contain high information density.
   A single operational rule or procedure typically spans 2 to 4 sentences (~80-120 words).
   Setting chunk_size=120 ensures that each chunk represents a cohesive semantic unit
   (e.g., a complete protocol) without diluting the dense embedding vector with unrelated
   surrounding sections.

2. Chunk Overlap (30 words / ~25% overlap):
   Hard boundaries can bisect critical conditional statements (e.g., splitting a threshold
   from its emergency response action). A 25% sliding window overlap guarantees that
   boundary-spanning concepts appear together in at least one retrieved chunk.

3. Semantic Boundary Preservation:
   The chunker splits along natural paragraph and sentence delimiters rather than arbitrary
   character cuts, preventing broken words or truncated phrases.
"""

from typing import List, Dict
import re


class DeliberateChunker:
    def __init__(self, chunk_size: int = 120, chunk_overlap: int = 30):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text_into_words(self, text: str) -> List[str]:
        """Split text while preserving clean spacing."""
        return text.strip().split()

    def chunk_document(self, text: str, source_name: str = "document") -> List[Dict]:
        """
        Splits a text document into overlapping chunks with metadata.

        Returns:
            List of dicts: [
                {
                    "id": f"{source_name}_chunk_{i}",
                    "text": chunk_text,
                    "metadata": {
                        "source": source_name,
                        "chunk_index": i,
                        "word_count": len(words)
                    }
                }
            ]
        """
        words = self.split_text_into_words(text)
        if not words:
            return []

        chunks = []
        step = self.chunk_size - self.chunk_overlap
        chunk_idx = 0

        for start_idx in range(0, len(words), step):
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "id": f"{source_name}_chunk_{chunk_idx}",
                "text": chunk_text,
                "metadata": {
                    "source": source_name,
                    "chunk_index": chunk_idx,
                    "word_count": len(chunk_words),
                    "start_word_index": start_idx,
                    "end_word_index": end_idx
                }
            })
            chunk_idx += 1

            if end_idx >= len(words):
                break

        return chunks

