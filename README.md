# Track K: AI Engineering — K1 RAG Fundamentals

A working Retrieval-Augmented Generation (RAG) pipeline built from scratch to index proprietary technical documentation, retrieve relevant chunks via vector similarity, and ground a language model's answers.

---

## 1. Project Overview & Pass Criteria Alignment

| Requirement / Pass Criterion | Implementation in this Repository | Verification |
|---|---|---|
| **Build working RAG pipeline from scratch** | Document ingestion $\to$ deliberate chunking $\to$ ChromaDB embedding $\to$ similarity retrieval $\to$ LLM generation. | [`src/rag_pipeline.py`](src/rag_pipeline.py) |
| **Working RAG Demo** | Interactive CLI showing query in, retrieved chunks with scores, base LLM answer, and final RAG answer. | [`demo.py`](demo.py) |
| **Retrieval actually influences answer** | Demonstrated on proprietary facts where Base LLM fails alone (hallucinates), while RAG LLM succeeds. | `demo.py` & [`tests/test_rag.py`](tests/test_rag.py) |
| **Deliberate Chunking Strategy** | Configured with `chunk_size=120` words and `chunk_overlap=30` words (25% sliding window) with documented semantic rationale. | [`src/chunking.py`](src/chunking.py) |
| **Trainee explains retrieval ranking/similarity** | Cosine Similarity over 384-dimensional dense vectors with mathematical formulation and distance derivation. | Section 4 & [`src/vector_store.py`](src/vector_store.py) |
| **Zero External API Costs** | Runs 100% locally on CPU using `google/flan-t5-small` with optional API fallback. | [`src/llm.py`](src/llm.py) |

---

## 2. System Architecture

```text
       [Proprietary Document: quantumcorp_handbook.md]
                              │
                              ▼
        [Deliberate Chunker: 120 words / 30 words overlap]
                              │
                              ▼
            [Embeddings: all-MiniLM-L6-v2 (384-dim)]
                              │
                              ▼
             [Vector Store: ChromaDB (Cosine Space)]
                              │
                              │ (Query: "What is the reboot passphrase for Vega?")
                              ▼
               [Top-k Cosine Similarity Retrieval]
           Rank 1: Chunk 1 (Score: 0.6803 | "Crimson-Nightingale-88...")
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       [Base LLM Alone]            [RAG Augmented Prompt]
     "a recurrency manual"         "Context: [Retrieved Chunk]...
            (FAIL)                  Question: ... -> Answer:"
                                             │
                                             ▼
                                     [RAG Final Answer]
                                  "Crimson-Nightingale-88"
                                           (PASS)
```

---

## 3. Deliberate Chunking Strategy Rationale

Rather than accepting unexamined library defaults (such as 1000-character chunks with 0 overlap), we deliberately chose:
* **Chunk Size:** `120 words` (~160 tokens / ~800 characters)
  * *Rationale:* Technical manuals, policies, and operational procedures exhibit high semantic density. A single operational directive (e.g. reactor pressure ceiling + Argon-18 emergency trigger) spans 2 to 4 sentences (~80 to 120 words). Sizing chunks to 120 words isolates individual procedures into cohesive semantic units without diluting their dense embedding vector with unrelated surrounding chapters.
* **Chunk Overlap:** `30 words` (~25% sliding window)
  * *Rationale:* Hard boundaries risk bisecting conditional statements (e.g., separating an operational threshold from its contingency protocol). A 25% overlap guarantees that transitional phrases appear together in full within at least one retrieved chunk.
* **Boundary Discipline:** Word-boundary preservation prevents truncated words and syntax fractures.

---

## 4. Retrieval Ranking & Similarity Method

### 4.1 Embedding Space
Text chunks and search queries are projected into continuous 384-dimensional vector space using `all-MiniLM-L6-v2`. Semantic similarity corresponds to geometric proximity in vector space.

### 4.2 Cosine Distance & Similarity Formulation
ChromaDB is initialized with HNSW index configuration `{"hnsw:space": "cosine"}`.

For a query vector $\mathbf{u}$ and candidate chunk vector $\mathbf{v}$:

$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

$$\text{Cosine Distance}(\mathbf{u}, \mathbf{v}) = 1 - \text{Cosine Similarity}(\mathbf{u}, \mathbf{v})$$

* **Why Cosine Similarity over Euclidean (L2) Distance?**  
  Euclidean distance scales with vector magnitude (affected by text length). Cosine similarity normalizes vector lengths, measuring purely the angular alignment between query intent and chunk content.
* **Ranking Step:** ChromaDB returns candidate chunks sorted in ascending order of cosine distance (descending order of cosine similarity). The top-$k$ ($k=2$) highest-similarity chunks are injected into the prompt.

---

## 5. Demonstration Results: Base LLM vs RAG Pipeline

When tested on proprietary facts from `data/quantumcorp_handbook.md`:

| Test Query | Verifiable Ground Truth | Base LLM (No Retrieval) | RAG Pipeline (With Context) | Result |
|---|---|---|---|---|
| *"What is the emergency manual reboot passphrase for Server Cluster Vega?"* | `Crimson-Nightingale-88` | `"a recurrency manual"` *(hallucinated)* | `"Crimson-Nightingale-88"` | **PASSED** |
| *"What is the maximum permissible operating pressure of the Phase-7 Flux Reactor?"* | `4,850 kPa` | `"a sleeve"` *(hallucinated)* | `"4,850 kPa"` | **PASSED** |
| *"What is the annual quantum workstation stipend for remote Tier-3 engineers under Protocol Omega-9?"* | `$7,420` | `"a stipend"` *(vague/uninformed)* | `"$7,420"` | **PASSED** |

---

## 6. Project Structure

```text
rag_fundamentals/
├── data/
│   └── quantumcorp_handbook.md     # Proprietary document set
├── src/
│   ├── __init__.py
│   ├── chunking.py                 # Deliberate chunker (size, overlap controls)
│   ├── vector_store.py             # ChromaDB vector store with Cosine ranking
│   ├── llm.py                      # Local Flan-T5 generator + optional API fallback
│   └── rag_pipeline.py             # End-to-end RAG orchestrator
├── tests/
│   ├── __init__.py
│   └── test_rag.py                 # Pytest test suite validating all pass criteria
├── demo.py                         # Interactive demonstration script
├── pytest.ini                      # Pytest discovery configuration
├── requirements.txt                # Dependencies
├── .gitignore                      # Clean Git ignore rules
└── README.md                       # Comprehensive guide & explanation
```

---

## 7. How to Run

### 7.1 Setup Environment
```bash
pip install -r requirements.txt
```

### 7.2 Run the Interactive Demo
```bash
python demo.py
```

### 7.3 Run the Automated Test Suite
```bash
pytest tests/ -v
```
All 4 automated tests verify chunking overlap, vector ranking, base LLM failure vs RAG success, and numerical grounding.

---

## 8. Line-by-Line Code Explanation Guide (For Oral Defense)

* **`src/chunking.py`:**
  * `split_text_into_words(text)`: Normalizes whitespace and extracts tokenized words.
  * `step = chunk_size - chunk_overlap`: Computes stride length. For size 120 and overlap 30, the stride is 90 words.
  * `chunks.append(...)`: Packages text slice with start/end metadata for auditability.
* **`src/vector_store.py`:**
  * `get_or_create_collection(..., metadata={"hnsw:space": "cosine"})`: Configures HNSW vector indexing to compute angular cosine distance.
  * `retrieve(query, top_k)`: Executes vector query, inverts distance to compute `similarity_score = 1 - distance`, and returns top-$k$ ranked chunks.
* **`src/llm.py`:**
  * `AutoModelForSeq2SeqLM` / `AutoTokenizer`: Loads `google/flan-t5-small` locally on CPU.
  * `generate_base_answer(query)`: Sends raw query to model without context to isolate pre-trained knowledge.
  * `generate_rag_answer(query, context)`: Wraps query with grounding system prompt enforcing answer synthesis strictly from retrieved context.
* **`src/rag_pipeline.py`:**
  * Orchestrates the entire flow: reads document, passes to chunker, pushes to ChromaDB, retrieves top-$k$, and calls the LLM generator.

