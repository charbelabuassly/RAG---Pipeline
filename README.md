# 🔍 Custom RAG Pipeline

**A from-scratch Retrieval-Augmented Generation system built with raw Python — no LangChain, no LlamaIndex. Just tokenizers, tensors, vector math, and a local LLM.**

<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/🤗%20Transformers-FFD21E?style=for-the-badge" alt="Hugging Face Transformers"/>
  <img src="https://img.shields.io/badge/Qdrant-DC244C?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/LM%20Studio-6E56CF?style=for-the-badge" alt="LM Studio"/>
  <img src="https://img.shields.io/badge/Qwen2.5--Coder-000000?style=for-the-badge" alt="Qwen"/>
  <img src="https://img.shields.io/badge/OpenAI%20SDK-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI SDK"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Architecture-v1.0-blue?style=flat-square" alt="Architecture Version"/>
  <img src="https://img.shields.io/badge/Vector%20Dim-384-orange?style=flat-square" alt="Vector Dimension"/>
  <img src="https://img.shields.io/badge/Chunking-400%20tok%20%2F%2050%20overlap-9cf?style=flat-square" alt="Chunking Config"/>
</p>

---

## 📖 Overview

Most RAG tutorials hide the mechanics behind a framework. This project does the opposite: it implements every core primitive of Retrieval-Augmented Generation by hand — document loading, token-aware chunking, embedding, mean pooling, vector storage, similarity search, and grounded prompt construction — to build a clear, composable, single-responsibility pipeline on top of raw Hugging Face, PyTorch, and Qdrant APIs.

The system is split into two decoupled paths:

- **📥 Indexing Pipeline (offline)** — ingests raw files, chunks them by token count, embeds them, and stores the resulting vectors + payloads in Qdrant.
- **📤 Query Pipeline (online)** — embeds an incoming question into the same vector space, retrieves the top-K most similar chunks, and feeds a grounded prompt to a locally-hosted Qwen model for generation.

## 🏗️ Architecture

```
┌─────────────────────────── INDEXING PIPELINE (OFFLINE) ───────────────────────────┐
│                                                                                     │
│  Document Files  ──►  DocumentLoader  ──►  Chunker  ──►  BGE Embedder  ──►  Qdrant │
│  (PDF/MD/TXT)         (format extraction) (400 tok /     (bge-small-en)   (vectors │
│                                             50 overlap)                     + text  │
│                                                                             payload)│
└─────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────── QUERY PIPELINE (ONLINE) ───────────────────────────────┐
│                                                                                     │
│  User Question  ──►  BGE Embedder  ──►  Retriever  ──►  Generator  ──►  Local Qwen │
│  (query string)      (same vector       (cosine        (prompt +       (LM Studio  │
│                       space)             top-K search)   context)       API)       │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### The Three-Phase Mental Model

| Phase | Input | Operations | Output |
|---|---|---|---|
| **Indexing** | Raw documents (PDF, Markdown, TXT) | Load → Chunk (token-based) → Embed (BGE-small) → Store | Qdrant vector points + payloads |
| **Retrieval** | User natural-language query | Tokenize → Model forward pass → Mean pool → Cosine search | Top-K most similar chunks |
| **Generation** | Original query + retrieved chunks | Merge context → Build grounded prompt → Local Qwen inference | Contextually grounded answer |

## 📦 Modules

| File | Responsibility |
|---|---|
| `loader.py` | `DocumentLoader` — single entry point for ingestion; dispatches to the correct reader by file extension |
| `txt_reader.py` | Reads plaintext `.txt` files directly into a unified string |
| `md_reader.py` | Reads `.md` files as plaintext (structure-aware parsing reserved for v2) |
| `pdf_reader.py` | Extracts text from `.pdf` files page-by-page via `pypdf` |
| `chunker.py` | `Chunker` — token-based sliding-window text splitter using the BGE tokenizer |
| `embedding_model.py` | `Embedder` — wraps `BAAI/bge-small-en-v1.5` for tokenization + forward-pass encoding |
| `retriever.py` | `Retriever` — embeds a query, mean-pools it, and runs a cosine similarity search against Qdrant |
| `docs.py` | `QdrantDB` — thin wrapper around the Qdrant client: collection creation, point upserts, similarity search |
| `generator.py` | `QwenGenerator` — sends grounded prompts to a local Qwen model served through LM Studio's OpenAI-compatible API |
| `rag.py` | `RAG` — orchestrates the full query path: retrieve → build prompt → generate |

## 🧩 Document Loading & Chunking

The `DocumentLoader` abstracts away file-format differences behind a common `.read()` interface, so the rest of the pipeline never has to know whether a document originated as a PDF, Markdown file, or plain text file.

| Reader | Format | Strategy |
|---|---|---|
| `TXTReader` | UTF-8 plaintext | Direct disk read |
| `MarkdownReader` | `.md` | Plaintext read (structure-aware tags planned for v2) |
| `PDFReader` | Binary PDF | Iterates pages via `pypdf`, concatenates extracted text |

### Token-Based Sliding Chunker

Character- or word-count chunkers can silently violate a transformer's real capacity, since models operate on tokens, not characters. `Chunker` sizes chunks using the actual BGE tokenizer and slides a window across the token stream to avoid truncating context at chunk borders.

| Parameter | Value | Rationale |
|---|---|---|
| Chunk size | 400 tokens | Maximum tokens per chunk |
| Overlap | 50 tokens | Carries trailing context into the next chunk |
| Sliding step | 350 tokens (400 − 50) | Start-index increment of the sliding window |
| Small documents | Adaptive | Text under 400 tokens is returned as a single chunk |

> **Why overlap matters:** without it, a concept split across a chunk boundary gets bisected — both halves become semantically incomplete on their own. A 50-token overlap ensures boundary concepts are fully captured in at least one chunk.

## 🧠 Embedding & Mean Pooling

Embeddings are produced with `BAAI/bge-small-en-v1.5`, a 384-dimensional dense retrieval transformer loaded through Hugging Face `transformers`. The **same embedder** is used for both indexing and querying, guaranteeing documents and questions live in the same vector space.

A transformer forward pass returns one 384-dim vector *per token* — shape `[batch, sequence_length, hidden_size]`. To collapse that into a single vector per chunk, the pipeline applies **attention-masked mean pooling**:

```
Pooled Vector = Sum(Token Vectors × Attention Mask) / Sum(Attention Mask)
```

Multiplying by the attention mask before summing ensures padding tokens (mask = 0) don't distort the average — only real tokens (mask = 1) contribute.

| Execution flag | Purpose |
|---|---|
| `model.eval()` | Disables training-only behavior (e.g. dropout) for deterministic inference |
| `torch.no_grad()` | Skips gradient-graph construction, cutting memory usage significantly |

There's no classification head involved — the pooled vector *is* the semantic coordinate, used directly for similarity search.

## 🗄️ Vector Storage — Qdrant

[Qdrant](https://qdrant.tech/) stores and indexes the embedding vectors, run locally in Docker to keep the database process isolated from the Python application:

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

| Entity | Role |
|---|---|
| **Collection** | Logical group of vectors sharing one config — 384 dimensions, Cosine distance |
| **Point** | One record: an ID, its embedding vector, and a payload containing the original chunk text |

> **Text preservation policy:** embedding vectors are one-way — they cannot be decoded back into readable text. The original chunk text is always stored in the point's payload so it can be recovered at retrieval time to build the generation prompt.

## ✨ Generation

`QwenGenerator` talks to a Qwen model served locally through **LM Studio's** OpenAI-compatible HTTP API (`qwen/qwen2.5-coder-14b`), using the standard `openai` Python SDK pointed at `http://127.0.0.1:1234/v1`. The retrieved chunks and the user's question are merged into a grounded prompt that instructs the model to answer strictly from the supplied context:

```
Context Chunks: [Extracted Text 1, Extracted Text 2, ...]
User Question: [Question]
Instruction: Answer using ONLY the context above. If you cannot find the
answer, state that clearly.
```

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install torch transformers qdrant-client pypdf openai
```

### 2. Start Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### 3. Start the local LLM in LM Studio

Load `qwen/qwen2.5-coder-14b` in [LM Studio](https://lmstudio.ai/) and start its local server on `127.0.0.1:1234`.

### 4. Index your documents

```python
from loader import DocumentLoader
from chunker import Chunker
from embedding_model import Embedder
from docs import QdrantDB

loader = DocumentLoader()
embedder = Embedder()
chunker = Chunker(tokenizer=embedder.tokenizer)
db = QdrantDB()
db.create_collection()

text = loader.load("docs/handbook.pdf")
chunks = chunker.chunk(text)

for i, chunk in enumerate(chunks):
    inputs = embedder.tokenize(chunk)
    output = embedder.encode(inputs)
    vector = mean_pool(output, inputs["attention_mask"])  # see Embedding & Mean Pooling
    db.add(point_id=i, vector=vector, payload={"text": chunk})
```

### 5. Ask a question

```python
from rag import RAG

rag = RAG()
answer = rag.ask("What is the refund policy?")
print(answer)
```

## 🧰 Tech Stack

| Tool | Purpose |
|---|---|
| ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) | Core language |
| ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) | Tensor ops & model inference |
| ![Transformers](https://img.shields.io/badge/-🤗%20Transformers-FFD21E?style=flat-square) | Tokenizer + `BAAI/bge-small-en-v1.5` embedding model |
| ![Qdrant](https://img.shields.io/badge/-Qdrant-DC244C?style=flat-square&logo=qdrant&logoColor=white) | Vector storage & cosine similarity search |
| ![Docker](https://img.shields.io/badge/-Docker-2496ED?style=flat-square&logo=docker&logoColor=white) | Isolated Qdrant deployment |
| ![pypdf](https://img.shields.io/badge/-pypdf-4B8BBE?style=flat-square) | PDF text extraction |
| ![LM Studio](https://img.shields.io/badge/-LM%20Studio-6E56CF?style=flat-square) | Local LLM serving with an OpenAI-compatible API |
| ![OpenAI SDK](https://img.shields.io/badge/-OpenAI%20SDK-412991?style=flat-square&logo=openai&logoColor=white) | Client used to talk to the local Qwen server |
| ![Qwen](https://img.shields.io/badge/-Qwen2.5--Coder--14B-000000?style=flat-square) | Local generation model |

## 🗺️ Version 2 Roadmap

| Upgrade | Impact |
|---|---|
| **Metadata & document objects** | Move from raw strings to structured objects carrying page numbers and source metadata, enabling precise citations |
| **Batch embedding** | Replace one-by-one chunk embedding with batched processing — up to **8x** faster ingestion |
| **Reranking layer** | Add a second-stage cross-encoder to re-score candidate chunks and improve context quality |
| **Hybrid search** | Combine semantic cosine search with lexical BM25 to catch exact keyword matches alongside semantic concepts |
| **Structured RAG evaluation** | Automated measurement of retrieval quality (precision/recall) and generation quality (faithfulness, grounding, hallucination rate) |

---
<p align="center"><i>Built to expose the raw mechanics of RAG — chunking, embedding, vector search, and grounded generation — without a framework in the way.</i></p>
