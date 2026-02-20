# AI RAG System

A production-grade **Retrieval-Augmented Generation** system built with FastAPI, ChromaDB, and Groq LLM.

Upload documents (PDF, DOCX, TXT, Markdown), ask questions in natural language, and get accurate answers **grounded in your own files with per-claim citations** — not hallucinated content.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT (REST API)                              │
│                    Swagger UI / curl / any HTTP client                       │
└────────┬──────────────────────┬──────────────────────┬──────────────────────┘
         │                      │                      │
    POST /upload           POST /query           POST /compare
         │                      │                      │
┌────────▼──────────────────────▼──────────────────────▼──────────────────────┐
│                          FastAPI  (Routes Layer)                             │
│              documents.py      query.py       compare.py                    │
└────────┬──────────────────────┬──────────────────────┬──────────────────────┘
         │                      │                      │
┌────────▼──────────────────────▼──────────────────────▼──────────────────────┐
│                     RAG Orchestration  (rag_service.py)                      │
│                                                                             │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│   │  INGESTION   │    │    QUERY     │    │   COMPARE    │                  │
│   │              │    │              │    │              │                  │
│   │ Parse file   │    │ Embed query  │    │ Embed topic  │                  │
│   │ Chunk text   │    │ Vector search│    │ Search doc1  │                  │
│   │ Embed chunks │    │ Re-rank(opt) │    │ Search doc2  │                  │
│   │ Store vectors│    │ LLM answer   │    │ LLM compare  │                  │
│   │              │    │ Verify (opt) │    │ JSON / text  │                  │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                  │
└──────────┼───────────────────┼───────────────────┼──────────────────────────┘
           │                   │                   │
     ┌─────▼─────┐      ┌─────▼─────┐       ┌─────▼─────┐
     │ EMBEDDING │      │   LLM     │       │  RERANK   │
     │ SERVICE   │      │  SERVICE  │       │  SERVICE  │
     │           │      │           │       │           │
     │ ONNX      │      │ Groq API  │       │ BM25 +    │
     │ MiniLM-L6 │      │ LLaMA 3.3 │       │ Vector    │
     │ (local)   │      │ 70B       │       │ hybrid    │
     └─────┬─────┘      └─────┬─────┘       └───────────┘
           │                   │
     ┌─────▼─────┐      ┌─────▼─────┐
     │ CHROMADB  │      │  CACHE    │
     │           │      │  SERVICE  │
     │ Persistent│      │           │
     │ cosine    │      │ Embed 24h │
     │ similarity│      │ Query 1h  │
     └───────────┘      └───────────┘
```

### Data Flow

```
UPLOAD:  File → Parse (PDF/DOCX/TXT/MD) → Sentence-Aware Chunking → ONNX Embed → ChromaDB

QUERY:   Question → Embed → Top-K Search → [Re-rank] → LLM + Citations → [Verify] → Answer

COMPARE: Topic → Embed → Search Doc1 + Doc2 → LLM Comparison → Structured JSON / Text
```

---

## Project Structure

```
ai-rag-system/
├── app/
│   ├── __init__.py                # FastAPI app (CORS, routes, lifespan, client UI)
│   ├── config/
│   │   ├── database.py            # ChromaDB settings
│   │   └── llm.py                 # Groq API & embedding config
│   ├── routes/
│   │   ├── documents.py           # POST /upload, GET /list, DELETE /{id}
│   │   ├── query.py               # POST /query
│   │   └── compare.py             # POST /compare
│   ├── services/
│   │   ├── embeddings/
│   │   │   └── embedding_service.py   # ONNX embeddings (all-MiniLM-L6-v2)
│   │   ├── llm/
│   │   │   └── llm_service.py         # Groq chat completions + prompts
│   │   └── rag/
│   │       ├── rag_service.py         # RAG orchestrator (ingest/query/compare)
│   │       └── rerank_service.py      # BM25 keyword re-ranker
│   ├── vectorstore/
│   │   └── vector_store_service.py    # ChromaDB CRUD operations
│   └── utils/
│       ├── cache_service.py           # In-memory TTL cache
│       ├── chunking_service.py        # Sentence-aware text chunking
│       ├── document_parser.py         # File parsing (PDF/DOCX/TXT/MD)
│       ├── error_handler.py           # Custom exception classes
│       └── validators.py             # Pydantic request models
├── client.html            # Web UI (single-file, no build step)
├── main.py                # Server entry point (Uvicorn)
├── requirements.txt       # Python dependencies
├── .env                   # API keys & config (not committed)
└── README.md
```

---

## Features

| #   | Feature                     | Description                                                                                           |
| --- | --------------------------- | ----------------------------------------------------------------------------------------------------- |
| 1   | **Document Upload**         | Upload PDF, DOCX, TXT, or Markdown files via multipart form                                           |
| 2   | **Intelligent Chunking**    | Sentence-boundary-aware splitting with configurable size (100–10,000 chars) and overlap (0–500 chars) |
| 3   | **Embeddings**              | 384-dim vectors via ONNX runtime (`all-MiniLM-L6-v2`) — local, free, ~100 MB RAM                      |
| 4   | **Vector Storage**          | ChromaDB PersistentClient with cosine similarity, batched insertion                                   |
| 5   | **Query with Citations**    | Top-K retrieval → LLM answer with `[Source: filename, Chunk N]` citations + confidence score          |
| 6   | **Document Comparison**     | Compare two documents on any topic — plain text or structured JSON                                    |
| 7   | **Structured Output**       | JSON with similarities, differences, unique points, agreement level                                   |
| 8   | **Hallucination Reduction** | 3-layer strategy: strict prompt + low temperature + citation enforcement + optional verification      |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root and set your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
PORT=3000
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

Get a free API key at [console.groq.com](https://console.groq.com)

### 3. Run the server

```bash
python main.py
```

Open **http://localhost:3000/** for the **Web UI** (client.html) — upload documents, ask questions, and compare docs from your browser.

Open **http://localhost:3000/docs** for the interactive Swagger UI (API reference).

---

## API Endpoints

| Method   | Endpoint                | Description                             |
| -------- | ----------------------- | --------------------------------------- |
| `GET`    | `/health`               | Health check + vector store stats       |
| `POST`   | `/api/documents/upload` | Upload a document (multipart form-data) |
| `GET`    | `/api/documents`        | List all uploaded documents             |
| `GET`    | `/api/documents/stats`  | Collection statistics                   |
| `DELETE` | `/api/documents/{id}`   | Delete a document and its chunks        |
| `POST`   | `/api/query`            | Ask a question against the corpus       |
| `POST`   | `/api/compare`          | Compare two documents on a topic        |

### Web UI (client.html)

The project includes a **single-file web client** served automatically at the root URL (`/`). No build tools, no npm — just open your browser.

| Feature       | Description                                                                                                             |
| ------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Dashboard** | Live health status + stats (documents, chunks, cached queries)                                                          |
| **Upload**    | Drag & drop or browse — supports PDF, DOCX, TXT, Markdown with configurable chunk size/overlap                          |
| **Documents** | List all uploaded documents, view metadata, delete with one click                                                       |
| **Query**     | Ask questions with adjustable Top-K, Re-rank toggle, Verification toggle; see answer + confidence bar + source previews |
| **Compare**   | Select two documents, pick a topic, toggle structured JSON — rendered as a rich comparison view                         |

### Example: Upload a document

```bash
curl -X POST http://localhost:3000/api/documents/upload \
  -F "file=@report.pdf" \
  -F "chunk_size=500" \
  -F "chunk_overlap=100"
```

### Example: Query

```bash
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key findings?",
    "top_k": 5,
    "verify": true,
    "rerank": true
  }'
```

### Example: Compare two documents

```bash
curl -X POST http://localhost:3000/api/compare \
  -H "Content-Type: application/json" \
  -d '{
    "document_ids": ["uuid-1", "uuid-2"],
    "topic": "methodology",
    "structured": true
  }'
```

---

## Hallucination Reduction Strategy

```
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │   LAYER 1    │     │   LAYER 2    │     │   LAYER 3    │     │   LAYER 4    │
  │              │     │              │     │              │     │  (optional)  │
  │ Strict       │────→│ Low          │────→│ Citation     │────→│ Verification │
  │ System       │     │ Temperature  │     │ Enforcement  │     │ Pass         │
  │ Prompt       │     │ (0.1)        │     │              │     │              │
  │              │     │              │     │ [Source: file │     │ Second LLM   │
  │ "Answer ONLY │     │ Near-        │     │  Chunk N]    │     │ call checks  │
  │  from context│     │ deterministic│     │              │     │ each claim   │
  │  provided"   │     │ output       │     │ Confidence   │     │ against      │
  │              │     │              │     │ score 1-10   │     │ source text  │
  └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

| Layer                    | Mechanism                                                   | Effect                             |
| ------------------------ | ----------------------------------------------------------- | ---------------------------------- |
| **Strict Prompt**        | System prompt forces LLM to use only provided context       | Prevents reliance on training data |
| **Low Temperature**      | `temperature=0.1` for near-deterministic output             | Reduces creative gap-filling       |
| **Citation Enforcement** | Every claim must cite `[Source: filename, Chunk N]`         | Makes unsupported claims visible   |
| **Confidence Score**     | LLM self-rates 1–10 with reasoning                          | Flags weak answers (score < 5)     |
| **Verification**         | Optional second LLM pass fact-checks claims against sources | Catches remaining hallucinations   |

---

## Tech Stack

| Component         | Technology                      | Why                                                  |
| ----------------- | ------------------------------- | ---------------------------------------------------- |
| **API Framework** | FastAPI + Uvicorn               | Async, auto-generated Swagger docs, high performance |
| **LLM**           | Groq API (LLaMA 3.3 70B)        | Free tier, fast inference, OpenAI-compatible         |
| **Embeddings**    | ONNX Runtime (all-MiniLM-L6-v2) | ~100 MB RAM vs ~2 GB for PyTorch, same quality       |
| **Vector DB**     | ChromaDB (PersistentClient)     | Embedded, no server process, cosine similarity       |
| **Re-ranking**    | Custom BM25 + Vector hybrid     | Lightweight keyword overlap improves retrieval       |
| **Caching**       | In-memory TTL cache             | Embeddings 24h, queries 1h — avoids redundant work   |
| **Validation**    | Pydantic v2                     | Type-safe request/response models                    |
| **HTTP Client**   | httpx (async)                   | Non-blocking LLM API calls with retry logic          |

---

## Environment Variables

| Variable             | Default                          | Description                            |
| -------------------- | -------------------------------- | -------------------------------------- |
| `GROQ_API_KEY`       | —                                | Groq API key (**required**)            |
| `GROQ_API_URL`       | `https://api.groq.com/openai/v1` | Groq API base URL                      |
| `GROQ_MODEL`         | `llama-3.3-70b-versatile`        | LLM model name                         |
| `EMBEDDING_MODEL`    | `all-MiniLM-L6-v2`               | Local embedding model                  |
| `PORT`               | `3000`                           | Server port                            |
| `CHUNK_SIZE`         | `500`                            | Characters per chunk                   |
| `CHUNK_OVERLAP`      | `100`                            | Overlap between chunks                 |
| `TEMPERATURE`        | `0.1`                            | LLM temperature (lower = more factual) |
| `MAX_TOKENS`         | `2048`                           | Max LLM response tokens                |
| `CHROMA_PERSIST_DIR` | `./chroma_data`                  | ChromaDB storage path                  |
| `CHROMA_COLLECTION`  | `rag_documents`                  | ChromaDB collection name               |

---
