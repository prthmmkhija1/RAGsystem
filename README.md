# 🚀 Production-Grade RAG System

> **Retrieval-Augmented Generation** system built with **Grok AI** and **ChromaDB** for grounding LLM responses in your private documents.

[![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org/)
[![Express](https://img.shields.io/badge/Express-4.x-blue.svg)](https://expressjs.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg)](https://www.trychroma.com/)
[![Grok AI](https://img.shields.io/badge/Grok-xAI-purple.svg)](https://x.ai/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Hallucination Reduction Strategy](#-hallucination-reduction-strategy)
- [Retrieval Design Choices](#-retrieval-design-choices)
- [Caching Strategy](#-caching-strategy)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

This RAG system solves the **LLM hallucination problem** by grounding AI responses in your private documents. Instead of letting the AI "guess," the system:

1. **Retrieves** relevant information from your documents
2. **Augments** the prompt with that context
3. **Generates** an answer using only the provided information

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG SYSTEM OVERVIEW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   📄 Documents ──► 🔪 Chunking ──► 🧮 Embeddings ──► 📦 Vector Store    │
│                                                                          │
│   ❓ Query ──► 🧮 Embed ──► 🔍 Search ──► 📝 Context ──► 🤖 LLM ──► ✅   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component            | Technology                 | Purpose                      |
| -------------------- | -------------------------- | ---------------------------- |
| **Runtime**          | Node.js 18+                | Server runtime               |
| **Framework**        | Express.js                 | REST API                     |
| **LLM & Embeddings** | Grok AI (xAI)              | Text generation & embeddings |
| **Vector Database**  | ChromaDB                   | Similarity search            |
| **Document Parsing** | pdf-parse, mammoth, marked | PDF, DOCX, MD support        |
| **Validation**       | Joi                        | Request validation           |
| **Re-ranking**       | Transformers.js            | Cross-encoder re-ranking     |

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│                         (REST API Consumers)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API GATEWAY                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   /upload   │    │   /query    │    │  /compare   │    │  /documents │  │
│  │   (POST)    │    │   (POST)    │    │   (POST)    │    │ (GET/DELETE)│  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CONTROLLER LAYER                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │documentController│  │ queryController  │  │compareController │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SERVICE LAYER                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         RAG Service                                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │  Document   │  │  Embedding  │  │   Rerank    │  │    LLM     │ │   │
│  │  │   Parser    │  │   Service   │  │   Service   │  │  Service   │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌──────────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│    Vector Store      │ │    Grok API      │ │    Cache Layer       │
│    (ChromaDB)        │ │    (xAI)         │ │   (node-cache)       │
└──────────────────────┘ └──────────────────┘ └──────────────────────┘
```

### Document Ingestion Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DOCUMENT INGESTION FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

   ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
   │  Upload  │      │  Parse   │      │  Chunk   │      │  Embed   │
   │   File   │ ───► │ Document │ ───► │   Text   │ ───► │  Chunks  │
   │          │      │          │      │          │      │          │
   └──────────┘      └──────────┘      └──────────┘      └──────────┘
        │                 │                 │                 │
        │                 │                 │                 │
        ▼                 ▼                 ▼                 ▼
   ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
   │ Validate │      │ Extract  │      │ Sentence │      │ Batched  │
   │ Format   │      │ Raw Text │      │ Boundary │      │   API    │
   │ (PDF,    │      │ (UTF-8)  │      │ Aware    │      │  Calls   │
   │ DOCX,TXT)│      │          │      │ Overlap  │      │ (Cached) │
   └──────────┘      └──────────┘      └──────────┘      └──────────┘
                                                              │
                                                              ▼
                                                         ┌──────────┐
                                                         │  Store   │
                                                         │  Vectors │
                                                         │ ChromaDB │
                                                         └──────────┘

   Supported Formats: PDF │ DOCX │ TXT │ Markdown
   Chunk Size: 1000 words (configurable)
   Overlap: 50 words (configurable)
```

### Query Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          QUERY PROCESSING FLOW                               │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │    User     │
                              │   Query     │
                              └──────┬──────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                                 ▼
             ┌─────────────┐                  ┌─────────────┐
             │ Check Query │                  │   Embed     │
             │   Cache     │                  │   Query     │
             └──────┬──────┘                  └──────┬──────┘
                    │                                │
              (HIT) │ (MISS)                        │
                    │    └──────────────────────────┤
                    │                               ▼
                    │                        ┌─────────────┐
                    │                        │   Vector    │
                    │                        │   Search    │
                    │                        │  (Top-K×3)  │
                    │                        └──────┬──────┘
                    │                               │
                    │                               ▼
                    │                        ┌─────────────┐
                    │                        │  Re-Rank    │
                    │                        │(Cross-Enc.) │
                    │                        │  → Top-K    │
                    │                        └──────┬──────┘
                    │                               │
                    │                               ▼
                    │                        ┌─────────────┐
                    │                        │   Build     │
                    │                        │  Context    │
                    │                        │   Block     │
                    │                        └──────┬──────┘
                    │                               │
                    │                               ▼
                    │                        ┌─────────────┐
                    │                        │    LLM      │
                    │                        │  Generate   │
                    │                        │   Answer    │
                    │                        └──────┬──────┘
                    │                               │
                    │                               ▼
                    │                        ┌─────────────┐
                    │                        │   Parse     │
                    │                        │ Confidence  │
                    │                        └──────┬──────┘
                    │                               │
                    │     ┌─────────────────────────┤
                    │     │ (if verify=true)        │
                    │     ▼                         │
                    │  ┌─────────────┐              │
                    │  │   Verify    │              │
                    │  │   Answer    │              │
                    │  └──────┬──────┘              │
                    │         │                     │
                    │         └──────────┬──────────┘
                    │                    │
                    ▼                    ▼
               ┌─────────────────────────────┐
               │      Cache & Return         │
               │         Response            │
               └─────────────────────────────┘
```

### Comparison Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DOCUMENT COMPARISON FLOW                               │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐    ┌──────────┐                     ┌──────────┐    ┌──────────┐
  │Document 1│    │Document 2│                     │   Doc 1  │    │   Doc 2  │
  │    ID    │    │    ID    │                     │  Chunks  │    │  Chunks  │
  └────┬─────┘    └────┬─────┘                     └────┬─────┘    └────┬─────┘
       │               │                                │               │
       └───────┬───────┘                                └───────┬───────┘
               │                                                │
               ▼                                                ▼
        ┌─────────────┐                                  ┌─────────────┐
        │   Embed     │                                  │   Merge &   │
        │   Topic     │                                  │   Format    │
        └──────┬──────┘                                  │   Context   │
               │                                         └──────┬──────┘
               ▼                                                │
        ┌─────────────┐                                         ▼
        │  Parallel   │                                  ┌─────────────┐
        │   Search    │ ─────────────────────────────►   │    LLM      │
        │ Both Docs   │                                  │  Compare    │
        └─────────────┘                                  └──────┬──────┘
                                                                │
                                         ┌──────────────────────┴───────┐
                                         │                              │
                                  (structured=false)            (structured=true)
                                         │                              │
                                         ▼                              ▼
                                  ┌─────────────┐               ┌─────────────┐
                                  │ Markdown    │               │   JSON      │
                                  │ Response    │               │ Structured  │
                                  └─────────────┘               └─────────────┘
```

---

## ✨ Features

### Core Features (Required)

| Feature                 | Status | Description                                                         |
| ----------------------- | ------ | ------------------------------------------------------------------- |
| Document Upload         | ✅     | PDF, DOCX, TXT, Markdown support                                    |
| Intelligent Chunking    | ✅     | Configurable size (1000 words) + overlap (50 words), sentence-aware |
| Embeddings              | ✅     | Grok API compatible (OpenAI/HuggingFace compatible)                 |
| Vector Storage          | ✅     | ChromaDB with cosine similarity                                     |
| Query Endpoint          | ✅     | Top-K retrieval + LLM answer + citations                            |
| Compare Endpoint        | ✅     | Two-document comparison with structured differences                 |
| Hallucination Reduction | ✅     | 3-layer strategy (see below)                                        |

### Advanced Features

| Feature                  | Status | Description                              |
| ------------------------ | ------ | ---------------------------------------- |
| Confidence Scoring       | ✅     | 1-10 score with explanation per response |
| Answer Verification      | ✅     | Second-pass claim verification           |
| Cross-Encoder Re-ranking | ✅     | Improved relevance with MiniLM model     |
| Caching Layer            | ✅     | Embeddings (24h) + queries (1h)          |
| Structured Comparison    | ✅     | JSON output with parsed sections         |

---

## 📁 Project Structure

```
/ai-rag-system
├── server.js                          # Express server entry point
├── package.json                       # Dependencies & scripts
├── .env                               # Environment variables
├── .gitignore                         # Git ignore rules
├── README.md                          # This documentation
│
└── /src
    ├── /config
    │   ├── database.js                # ChromaDB configuration
    │   └── llm.js                     # Grok API configuration
    │
    ├── /routes
    │   ├── documents.js               # Document upload/list/delete routes
    │   ├── query.js                   # Query route
    │   └── compare.js                 # Compare route
    │
    ├── /controllers
    │   ├── documentController.js      # Document ingestion logic
    │   ├── queryController.js         # Query handling logic
    │   └── compareController.js       # Comparison logic
    │
    ├── /services
    │   ├── /rag
    │   │   ├── ragService.js          # RAG orchestration (main pipeline)
    │   │   └── rerankService.js       # Cross-encoder re-ranking
    │   ├── /embeddings
    │   │   └── embeddingService.js    # Grok embedding generation
    │   └── /llm
    │       └── llmService.js          # Grok chat completions
    │
    ├── /vectorstore
    │   └── vectorStoreService.js      # ChromaDB operations
    │
    └── /utils
        ├── documentParser.js          # PDF/DOCX/TXT/MD parsing
        ├── chunkingService.js         # Intelligent text chunking
        ├── cacheService.js            # In-memory caching layer
        ├── errorHandler.js            # Error handling middleware
        └── validators.js              # Joi validation schemas
```

---

## 🛠️ Installation

### Prerequisites

- **Node.js** 18+
- **Python** 3.8+ (for ChromaDB)
- **Grok API Key** (from [x.ai](https://x.ai/api))

### Step 1: Clone & Install Dependencies

```bash
git clone <repository-url>
cd ai-rag-system
npm install
```

### Step 2: Install & Start ChromaDB

```bash
# Install ChromaDB
pip install chromadb

# Start ChromaDB server (terminal 1)
chroma run --host localhost --port 8000
```

### Step 3: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# Grok API
GROK_API_KEY=your_grok_api_key_here
GROK_API_URL=https://api.x.ai/v1
GROK_MODEL=grok-3-mini-fast
GROK_EMBEDDING_MODEL=grok-embedding-public

# ChromaDB
CHROMA_HOST=http://localhost:8000
CHROMA_COLLECTION=rag_documents

# Server
PORT=3000

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=50

# LLM Settings
MAX_TOKENS=1024
TEMPERATURE=0.1

# Cache TTL (seconds)
EMBEDDING_CACHE_TTL=86400
QUERY_CACHE_TTL=3600
```

### Step 4: Start the Server

```bash
# Development (with hot reload)
npm run dev

# Production
npm start
```

---

## ⚙️ Configuration

### Chunking Configuration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CHUNKING STRATEGY                                   │
└─────────────────────────────────────────────────────────────────────────────┘

   Document Text
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5. Sentence 6.│
   │ Sentence 7. Sentence 8. Sentence 9. Sentence 10. Sentence 11. ...      │
   └─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
   ┌────────────────────────────┐    Overlap    ┌────────────────────────────┐
   │        CHUNK 1             │◄─────────────►│        CHUNK 2             │
   │ Sentence 1-6               │    50 words   │ Sentence 5-10              │
   │ (~1000 words)              │               │ (~1000 words)              │
   └────────────────────────────┘               └────────────────────────────┘

   Key Features:
   ├── Sentence boundary awareness (never splits mid-sentence)
   ├── Configurable chunk size (CHUNK_SIZE env var)
   ├── Configurable overlap (CHUNK_OVERLAP env var)
   └── Force-splits very long sentences when needed
```

| Parameter       | Default | Description                      |
| --------------- | ------- | -------------------------------- |
| `CHUNK_SIZE`    | 1000    | Maximum words per chunk          |
| `CHUNK_OVERLAP` | 50      | Overlapping words between chunks |

### LLM Configuration

| Parameter              | Default               | Description                                        |
| ---------------------- | --------------------- | -------------------------------------------------- |
| `TEMPERATURE`          | 0.1                   | Lower = more deterministic (reduces hallucination) |
| `MAX_TOKENS`           | 1024                  | Maximum response length                            |
| `GROK_MODEL`           | grok-3-mini-fast      | LLM model for generation                           |
| `GROK_EMBEDDING_MODEL` | grok-embedding-public | Model for embeddings                               |

---

## 📚 API Reference

### Base URL

```
http://localhost:3000/api
```

### Endpoints Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            API ENDPOINTS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  POST   /api/upload        Upload and ingest a document                     │
│  POST   /api/query         Query documents with natural language            │
│  POST   /api/compare       Compare two documents on a topic                 │
│  GET    /api/documents     List all ingested documents                      │
│  DELETE /api/documents/:id Delete a document                                │
│  GET    /api/stats         Get vector store statistics                      │
│  GET    /health            Health check with cache stats                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### POST /api/upload

Upload and ingest a document into the vector store.

**Request:**

```bash
curl -X POST http://localhost:3000/api/upload \
  -F "document=@./my-document.pdf" \
  -F "chunkSize=1000" \
  -F "chunkOverlap=50"
```

**Response:**

```json
{
  "success": true,
  "data": {
    "documentId": "550e8400-e29b-41d4-a716-446655440000",
    "filename": "my-document.pdf",
    "chunkCount": 15,
    "characterCount": 45230,
    "processingTime": "2.34s",
    "message": "Successfully ingested \"my-document.pdf\" into 15 chunks"
  }
}
```

---

### POST /api/query

Query the document corpus with natural language.

**Request:**

```json
{
  "query": "What are the main features of the product?",
  "topK": 5,
  "documentId": null,
  "rerank": true,
  "verify": false
}
```

| Parameter     | Type    | Default  | Description                       |
| ------------- | ------- | -------- | --------------------------------- |
| `query`       | string  | required | The question to ask               |
| `topK`        | number  | 5        | Number of chunks to retrieve      |
| `documentId`  | string  | null     | Limit search to specific document |
| `rerank`      | boolean | false    | Enable cross-encoder re-ranking   |
| `verify`      | boolean | false    | Enable answer verification        |
| `temperature` | number  | 0.1      | LLM temperature override          |

**Response:**

```json
{
  "success": true,
  "data": {
    "answer": "Based on the documentation, the main features include...",
    "confidence": {
      "score": 8,
      "reason": "Direct support found in multiple chunks",
      "level": "high"
    },
    "sources": [
      {
        "filename": "product-docs.pdf",
        "chunkIndex": 3,
        "documentId": "550e8400-...",
        "similarityScore": 0.8934,
        "crossEncoderScore": 0.9234,
        "originalRank": 2,
        "preview": "The product includes automatic scaling..."
      }
    ],
    "verification": {
      "isVerified": true,
      "overallScore": 9,
      "claims": [...],
      "unsupportedClaims": [],
      "summary": "All claims are supported"
    },
    "query": "What are the main features of the product?",
    "topK": 5,
    "reranked": true,
    "chunksUsed": 5,
    "processingTime": "1.23s"
  }
}
```

---

### POST /api/compare

Compare two documents on a specific topic.

**Request:**

```json
{
  "documentIds": [
    "550e8400-e29b-41d4-a716-446655440000",
    "660f9500-f39c-52e5-b827-557766551111"
  ],
  "topic": "pricing strategies",
  "topK": 5,
  "structured": true
}
```

| Parameter     | Type    | Default  | Description                      |
| ------------- | ------- | -------- | -------------------------------- |
| `documentIds` | array   | required | Exactly 2 document UUIDs         |
| `topic`       | string  | required | What to compare the documents on |
| `topK`        | number  | 5        | Chunks per document to analyze   |
| `structured`  | boolean | false    | Return structured JSON           |

**Response (structured=true):**

```json
{
  "success": true,
  "data": {
    "comparison": {
      "similarities": [
        {
          "point": "Both documents emphasize customer value",
          "doc1Evidence": { "quote": "...", "source": "doc1.pdf", "chunk": 2 },
          "doc2Evidence": { "quote": "...", "source": "doc2.pdf", "chunk": 1 }
        }
      ],
      "differences": [
        {
          "aspect": "Pricing model",
          "doc1Position": "Subscription-based",
          "doc2Position": "One-time purchase",
          "doc1Source": { "source": "doc1.pdf", "chunk": 5 },
          "doc2Source": { "source": "doc2.pdf", "chunk": 3 }
        }
      ],
      "uniqueToDoc1": [...],
      "uniqueToDoc2": [...],
      "summary": {
        "overallAssessment": "Documents take contrasting approaches",
        "agreementLevel": "low",
        "keyTakeaway": "Fundamental difference in pricing philosophy"
      },
      "metadata": {
        "doc1ChunksAnalyzed": 5,
        "doc2ChunksAnalyzed": 5,
        "comparisonConfidence": 8
      }
    },
    "structured": true,
    "doc1Sources": [...],
    "doc2Sources": [...],
    "topic": "pricing strategies",
    "documentsCompared": ["550e8400-...", "660f9500-..."],
    "processingTime": "3.45s"
  }
}
```

---

### GET /api/documents

List all ingested documents.

**Response:**

```json
{
  "success": true,
  "data": {
    "count": 3,
    "documents": [
      {
        "documentId": "550e8400-...",
        "filename": "product-docs.pdf",
        "chunkCount": 15,
        "uploadedAt": "2024-01-15T10:30:00.000Z"
      }
    ]
  }
}
```

---

### DELETE /api/documents/:id

Delete a document and all its chunks.

**Response:**

```json
{
  "success": true,
  "data": {
    "documentId": "550e8400-...",
    "message": "Document deleted successfully"
  }
}
```

---

### GET /health

Health check with cache statistics.

**Response:**

```json
{
  "status": "ok",
  "service": "RAG System with Grok + ChromaDB",
  "uptime": "45.2 min",
  "vectorStore": {
    "totalDocuments": 3,
    "totalChunks": 47
  },
  "cache": {
    "embeddings": {
      "keys": 150,
      "hits": 1234,
      "misses": 89,
      "hitRate": "93.3%"
    },
    "queries": {
      "keys": 12,
      "hits": 45,
      "misses": 23,
      "hitRate": "66.2%"
    }
  }
}
```

---

## 🛡️ Hallucination Reduction Strategy

The system implements a **3-layer hallucination reduction strategy**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HALLUCINATION REDUCTION LAYERS                            │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LAYER 1: SYSTEM PROMPT ENGINEERING                                      │
  │                                                                          │
  │  "You MUST answer ONLY using the provided context chunks.               │
  │   Do NOT use prior knowledge. If the answer is not in the              │
  │   context, say: 'I cannot answer this question based on                │
  │   the provided documents.'"                                             │
  │                                                                          │
  │  ✓ Explicit instruction to use ONLY provided context                   │
  │  ✓ Clear guidance when information is insufficient                     │
  │  ✓ Citation enforcement for every claim                                │
  └─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LAYER 2: LOW TEMPERATURE (0.1)                                          │
  │                                                                          │
  │  Temperature Scale:                                                      │
  │  ├── 0.0-0.2: Highly deterministic, factual (WE USE THIS)              │
  │  ├── 0.3-0.5: Balanced creativity                                       │
  │  ├── 0.6-0.8: More creative, varied                                     │
  │  └── 0.9-1.0: Maximum creativity                                        │
  │                                                                          │
  │  Low temperature = Less random/creative = Fewer hallucinations          │
  └─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LAYER 3: CITATION ENFORCEMENT                                           │
  │                                                                          │
  │  Every response includes:                                               │
  │  ├── Source citations: [Source: filename.pdf, Chunk 3]                 │
  │  ├── Confidence score: 1-10 with explanation                           │
  │  └── Source previews: Original text snippets                           │
  │                                                                          │
  │  Users can verify claims against original documents                     │
  └─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  BONUS: ANSWER VERIFICATION (Optional)                                   │
  │                                                                          │
  │  When verify=true:                                                       │
  │  ├── Second LLM pass fact-checks the answer                            │
  │  ├── Each claim marked: supported/partially_supported/unsupported      │
  │  ├── Evidence quotes provided                                           │
  │  └── Overall verification score                                         │
  └─────────────────────────────────────────────────────────────────────────┘
```

### Confidence Score Interpretation

| Score | Level     | Meaning                                        |
| ----- | --------- | ---------------------------------------------- |
| 9-10  | Very High | Direct, explicit support from multiple sources |
| 7-8   | High      | Good support with minor inference              |
| 5-6   | Medium    | Partial support, some inference needed         |
| 3-4   | Low       | Weak support, significant inference            |
| 1-2   | Very Low  | Minimal or no support from context             |

---

## 🔍 Retrieval Design Choices

### Why Top-K = 5?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TOP-K SELECTION RATIONALE                            │
└─────────────────────────────────────────────────────────────────────────────┘

   K=3: Too few
   ├── May miss relevant context
   └── Simple questions only

   K=5: Sweet spot (DEFAULT)  ◄──── RECOMMENDED
   ├── Good balance of context & precision
   ├── Handles most question complexities
   └── Reasonable LLM context usage

   K=10+: Too many
   ├── Noise from irrelevant chunks
   ├── Confuses the LLM
   └── Higher API costs
```

### Why Re-ranking?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BI-ENCODER vs CROSS-ENCODER                             │
└─────────────────────────────────────────────────────────────────────────────┘

   BI-ENCODER (Embedding Similarity):
   ┌─────────────┐     ┌─────────────┐
   │   Query     │     │  Document   │
   │  Embedding  │     │  Embedding  │
   └──────┬──────┘     └──────┬──────┘
          │                   │
          └─────────┬─────────┘
                    │
               Cosine Sim
                    │
               ✗ FAST but less accurate
               ✗ Independent encodings miss nuance

   CROSS-ENCODER (Joint Scoring):
   ┌─────────────────────────────────┐
   │    Query [SEP] Document         │
   │         Together                │
   └──────────────┬──────────────────┘
                  │
            Relevance Score
                  │
               ✓ SLOWER but more accurate
               ✓ Sees query-document interaction

   OUR APPROACH:
   1. Fast retrieval: Get top K×3 with embeddings (fast)
   2. Re-rank: Score top K×3 with cross-encoder (accurate)
   3. Return: Best K after re-ranking
```

### Model Used for Re-ranking

- **Model**: `Xenova/ms-marco-MiniLM-L-6-v2`
- **Type**: Cross-encoder trained on MS MARCO
- **Quantized**: Yes (faster inference)
- **First-load**: ~30 seconds (cached afterward)

---

## 💾 Caching Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CACHING ARCHITECTURE                                │
└─────────────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │                      EMBEDDING CACHE                                  │
   │                                                                       │
   │  Purpose: Avoid re-calling Grok API for seen text                    │
   │  TTL: 24 hours                                                        │
   │  Max Keys: 10,000                                                     │
   │  Key: MD5(text)                                                       │
   │                                                                       │
   │  Hit Rate Target: >90% after warm-up                                 │
   └──────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │                       QUERY CACHE                                     │
   │                                                                       │
   │  Purpose: Instant responses for repeated queries                     │
   │  TTL: 1 hour                                                          │
   │  Max Keys: 1,000                                                      │
   │  Key: MD5(query + topK + documentId + rerank)                        │
   │                                                                       │
   │  Invalidation: On document add/delete                                │
   └──────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │                     DOCUMENT CACHE                                    │
   │                                                                       │
   │  Purpose: Store document metadata                                    │
   │  TTL: 30 minutes                                                      │
   │  Max Keys: 500                                                        │
   └──────────────────────────────────────────────────────────────────────┘
```

### Cache Invalidation

| Event                 | Action                                   |
| --------------------- | ---------------------------------------- |
| New document uploaded | Clear all query cache                    |
| Document deleted      | Clear query cache + document cache entry |
| Manual clear          | All caches flushed                       |

---

## 🧪 Testing

### Test Document Upload

```bash
curl -X POST http://localhost:3000/api/upload \
  -F "document=@./test.pdf"
```

### Test Query

```bash
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic?",
    "topK": 5,
    "rerank": true
  }'
```

### Test Comparison

```bash
curl -X POST http://localhost:3000/api/compare \
  -H "Content-Type: application/json" \
  -d '{
    "documentIds": ["uuid-1", "uuid-2"],
    "topic": "key differences",
    "structured": true
  }'
```

### Test with Verification

```bash
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the system requirements?",
    "verify": true
  }'
```

---

## 🔧 Troubleshooting

### ChromaDB Connection Failed

```bash
# Ensure ChromaDB is running
chroma run --host localhost --port 8000

# Check if port is in use
netstat -an | findstr 8000
```

### Grok API Errors

| Error            | Solution                         |
| ---------------- | -------------------------------- |
| 401 Unauthorized | Check `GROK_API_KEY` in `.env`   |
| 429 Rate Limited | System auto-retries with backoff |
| 500 Server Error | System auto-retries 3 times      |

### Out of Memory (Re-ranking)

The cross-encoder model downloads on first use (~100MB). Ensure sufficient memory:

```bash
# Check memory
node --max-old-space-size=4096 server.js
```

### Slow First Query

First query loads the cross-encoder model (~30s). Subsequent queries are fast.

---

## 🙏 Acknowledgments

- [Grok AI (xAI)](https://x.ai/) - LLM and Embeddings
- [ChromaDB](https://www.trychroma.com/) - Vector Database
- [Hugging Face](https://huggingface.co/) - Transformers.js
- [Express.js](https://expressjs.com/) - Web Framework
