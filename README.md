# Production-Grade RAG System - Implementation Guide

## Grok AI + ChromaDB

## 🎯 Project Overview

This is a **Retrieval-Augmented Generation (RAG)** system built with **Grok AI** and **ChromaDB** to solve the LLM hallucination problem by grounding AI responses in your private documents. Instead of letting the AI "guess," the system retrieves relevant information from your documents and instructs the AI to answer using **only** that information.

### The Problem

Standard LLMs (like ChatGPT) might not know about your private company data, or they might hallucinate when asked about specific obscure topics.

### The Solution

Your RAG system:

1. Looks up relevant information from your documents
2. Feeds that info to the AI
3. Says: "Answer the user's question using **only** this information"

### Technology Stack

- **LLM & Embeddings:** Grok AI (xAI's Grok API)
- **Vector Database:** ChromaDB
- **Backend:** Node.js + Express
- **Document Processing:** PDF, DOCX, TXT, Markdown support

---

## 📁 Project Structure

```
/ai-rag-system
├── server.js                          # Express server entry point
├── package.json                       # Dependencies
├── .env.example                       # Environment variables template
├── .gitignore
├── README.md                          # This file
└── /src
    ├── /config
    │   ├── database.js               # ChromaDB configuration
    │   └── llm.js                    # Grok API configuration
    ├── /routes
    │   ├── documents.js              # Document upload routes
    │   ├── query.js                  # Query routes
    │   └── compare.js                # Compare routes
    ├── /controllers
    │   ├── documentController.js     # Document ingestion logic
    │   ├── queryController.js        # Query handling logic
    │   └── compareController.js      # Comparison logic
    ├── /services
    │   ├── /rag
    │   │   └── ragService.js         # RAG orchestration
    │   ├── /embeddings
    │   │   └── embeddingService.js   # Grok embedding generation
    │   └── /llm
    │       └── llmService.js         # Grok LLM API calls
    ├── /vectorstore
    │   └── vectorStoreService.js     # ChromaDB operations
    └── /utils
        ├── documentParser.js         # Parse PDF/DOCX/TXT/MD
        ├── chunkingService.js        # Intelligent text chunking
        ├── errorHandler.js           # Error handling
        └── validators.js             # Input validation
```

    ├── errorHandler.js               # Error handling
    └── validators.js                 # Input validation

```

        ├── errorHandler.js           # Error handling
        └── validators.js             # Input validation

```

---

## 🚀 Implementation Roadmap

### Phase 1: Setup & Dependencies

#### 1.1 Install Core Dependencies

```bash
npm install express cors dotenv multer pdf-parse mammoth marked chromadb axios uuid joi
```

**Dependencies breakdown:**

- `express` - Web server framework
- `cors` - Enable CORS
- `dotenv` - Environment variables
- `multer` - File upload handling
- `pdf-parse` - PDF parsing
- `mammoth` - DOCX parsing
- `marked` - Markdown parsing
- `chromadb` - ChromaDB vector database client
- `axios` - HTTP requests for Grok API
- `uuid` - Generate unique IDs
- `joi` - Input validation

#### 1.2 Install Development Dependencies

```bash
npm install --save-dev nodemon
```

#### 1.3 Setup ChromaDB Server

**Install ChromaDB:**

```bash
pip install chromadb
```

**Run ChromaDB server:**

```bash
chroma run --host localhost --port 8000
```

#### 1.4 Configure Environment Variables

Copy `.env.example` to `.env` and fill in your Grok API key:

```bash
GROK_API_KEY=your_grok_api_key_here
GROK_API_URL=https://api.x.ai/v1
GROK_MODEL=grok-beta
CHROMA_HOST=http://localhost:8000
PORT=3000
CHUNK_SIZE=1000
CHUNK_OVERLAP=50
```

---

## 📦 Module 1: Document Ingestion Engine

### Goal

Build an API route that accepts file uploads (PDF, DOCX, TXT, Markdown) and processes them for storage.

### Implementation Steps

#### Step 1.1: Document Parser Service (`src/utils/documentParser.js`)

**Purpose:** Extract raw text from different file formats.

**Key Functions:**

- `parseDocument(file)` - Main entry point, detects file type and delegates
- `parsePDF(buffer)` - Uses `pdf-parse` to extract text from PDFs
- `parseDOCX(buffer)` - Uses `mammoth` to extract text from Word docs
- `parseTXT(buffer)` - Simple UTF-8 text extraction
- `parseMarkdown(buffer)` - Parse markdown files (optionally convert to plain text)

**Implementation Tips:**

```javascript
const pdfParse = require("pdf-parse");
const mammoth = require("mammoth");

async function parsePDF(buffer) {
  const data = await pdfParse(buffer);
  return data.text;
}

async function parseDOCX(buffer) {
  const result = await mammoth.extractRawText({ buffer });
  return result.value;
}

// Error handling: Wrap in try-catch, return meaningful errors
```

---

#### Step 1.2: Chunking Service (`src/utils/chunkingService.js`)

**Purpose:** Split long documents into smaller, overlapping chunks.

**Why Chunking?**

- You cannot feed a 100-page PDF to an LLM all at once (token limits)
- Smaller chunks = more precise retrieval
- Overlap ensures context isn't lost at boundaries

**Configuration (from requirements):**

- **Chunk Size:** 1000 words (configurable via `CHUNK_SIZE` in `.env`)
- **Overlap:** 50 words (configurable via `CHUNK_OVERLAP` in `.env`)

**Key Function:**

- `chunkText(text, chunkSize, overlap)` - Returns array of text chunks

**Implementation Strategy:**

```javascript
function chunkText(text, chunkSize = 1000, overlap = 50) {
  const words = text.split(/\s+/);
  const chunks = [];

  for (let i = 0; i < words.length; i += chunkSize - overlap) {
    const chunk = words.slice(i, i + chunkSize).join(" ");
    chunks.push({
      text: chunk,
      startIndex: i,
      endIndex: Math.min(i + chunkSize, words.length),
    });
  }

  return chunks;
}
```

**Advanced Chunking (Optional):**

- Respect paragraph boundaries
- Don't split sentences mid-way
- Use sliding window with sentence detection

---

#### Step 1.3: Embedding Service (`src/services/embeddings/embeddingService.js`)

**Purpose:** Convert text chunks into numerical vectors (embeddings) using Grok AI.

**Key Function:**

- `generateEmbedding(text)` - Returns vector array using Grok's embedding model

**Grok API Implementation:**

```javascript
const axios = require("axios");
const config = require("../../config/llm");

async function generateEmbedding(text) {
  try {
    const response = await axios.post(
      `${config.grok.apiUrl}/embeddings`,
      {
        model: config.grok.embeddingModel,
        input: text,
      },
      {
        headers: {
          Authorization: `Bearer ${config.grok.apiKey}`,
          "Content-Type": "application/json",
        },
      },
    );
    return response.data.data[0].embedding;
  } catch (error) {
    throw new Error(`Grok embedding failed: ${error.message}`);
  }
}

module.exports = { generateEmbedding };
```

**Batch Processing (Important):**

- Process multiple chunks together when possible
- Add retry logic with exponential backoff
- Cache embeddings to avoid redundant API calls

---

#### Step 1.4: Vector Store Service (`src/vectorstore/vectorStoreService.js`)

**Purpose:** Store and retrieve vector embeddings from ChromaDB.

**Key Functions:**

- `initialize()` - Setup database connection/collection
- `storeChunks(documentId, chunks, embeddings, metadata)` - Save vectors
- `searchSimilar(queryVector, topK)` - Find most similar chunks
- `deleteDocument(documentId)` - Remove document vectors

**ChromaDB Example:**

```javascript
const { ChromaClient } = require("chromadb");

class VectorStoreService {
  constructor() {
    this.client = new ChromaClient({
      path: process.env.CHROMA_HOST || "http://localhost:8000",
    });
    this.collection = null;
  }

  async initialize() {
    this.collection = await this.client.getOrCreateCollection({
      name: "documents",
      metadata: { "hnsw:space": "cosine" },
    });
  }

  async storeChunks(documentId, chunks, embeddings, metadata) {
    const ids = chunks.map((_, i) => `${documentId}_chunk_${i}`);
    await this.collection.add({
      ids,
      embeddings,
      documents: chunks,
      metadatas: chunks.map((_, i) => ({
        documentId,
        chunkIndex: i,
        ...metadata,
      })),
    });
  }

  async searchSimilar(queryVector, topK = 5) {
    const results = await this.collection.query({
      queryEmbeddings: [queryVector],
      nResults: topK,
    });
    return results;
  }

  async deleteDocument(documentId) {
    // Implementation to delete all chunks for a document
    // Query by metadata filter and delete
  }
}

module.exports = new VectorStoreService();
```

---

#### Step 1.5: Document Controller (`src/controllers/documentController.js`)

**Purpose:** Orchestrate the entire ingestion pipeline.

**Upload Flow:**

1. Receive file upload (via `multer`)
2. Parse document → extract text
3. Chunk text into smaller pieces
4. Generate embeddings for each chunk
5. Store chunks + embeddings + metadata in vector database
6. Return success response with document ID

**Key Function:**

```javascript
const { v4: uuidv4 } = require("uuid");
const documentParser = require("../utils/documentParser");
const chunkingService = require("../utils/chunkingService");
const embeddingService = require("../services/embeddings/embeddingService");
const vectorStoreService = require("../vectorstore/vectorStoreService");

async function uploadDocument(req, res) {
  try {
    // 1. Get uploaded file
    const file = req.file;

    if (!file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    // 2. Parse document
    const text = await documentParser.parseDocument(file);

    // 3. Chunk text
    const chunkSize = parseInt(process.env.CHUNK_SIZE) || 1000;
    const chunkOverlap = parseInt(process.env.CHUNK_OVERLAP) || 50;
    const chunks = chunkingService.chunkText(text, chunkSize, chunkOverlap);

    // 4. Generate embeddings (batch process)
    const embeddings = await Promise.all(
      chunks.map((chunk) => embeddingService.generateEmbedding(chunk.text)),
    );

    // 5. Store in ChromaDB
    const documentId = uuidv4();
    await vectorStoreService.storeChunks(
      documentId,
      chunks.map((c) => c.text),
      embeddings,
      { filename: file.originalname, uploadDate: new Date() },
    );

    // 6. Return response
    res.json({
      success: true,
      documentId,
      filename: file.originalname,
      chunksCount: chunks.length,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}
```

---

#### Step 1.6: Document Routes (`src/routes/documents.js`)

**Endpoint:** `POST /upload`

**Setup:**

```javascript
const express = require("express");
const multer = require("multer");
const documentController = require("../controllers/documentController");

const router = express.Router();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB limit
});

router.post(
  "/upload",
  upload.single("file"),
  documentController.uploadDocument,
);

module.exports = router;
```

---

## 🔍 Module 2: Retrieval API (Query Endpoint)

### Goal

Build a `POST /query` endpoint that:

1. Takes user's question
2. Retrieves top-K most relevant chunks
3. Sends question + chunks to LLM
4. Returns answer with citations

### Implementation Steps

#### Step 2.1: Query Controller (`src/controllers/queryController.js`)

**Query Flow:**

1. Receive user question
2. Generate embedding for the question using Grok
3. Search ChromaDB for top-K similar chunks (e.g., K=3 or K=5)
4. Build prompt with retrieved chunks
5. Send to Grok LLM with strict instructions (hallucination reduction)
6. Parse response and extract citations
7. Return structured response

**Key Implementation:**

```javascript
const embeddingService = require('../services/embeddings/embeddingService');
const vectorStoreService = require('../vectorstore/vectorStoreService');
const llmService = require('../services/llm/llmService');
const { validateQuery } = require('../utils/validators');

async function handleQuery(req, res) {
  try {
    // Validate input
    const { error, value } = validateQuery(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { question, topK = 5 } = value;

    // 1. Generate query embedding
    const queryVector = await embeddingService.generateEmbedding(question);

    // 2. Retrieve top-K similar chunks from ChromaDB
    const results = await vectorStoreService.searchSimilar(queryVector, topK);

    // 3. Extract chunks and metadata
    const context = results.documents[0];
    const metadata = results.metadatas[0];

    // 4. Build prompt with strict instructions
    const prompt = buildRAGPrompt(question, context, metadata);

    // 5. Get LLM response
    const answer = await llmService.generateAnswer(prompt);

    // 6. Return response with citations
    res.json({
    // 4. Build prompt with strict instructions
    const answer = await llmService.generateAnswer(question, context, metadata);

    // 5. Return response with citations
    res.json({
      question,
      answer: answer.text,
      citations: metadata.map((m) => ({
        documentId: m.documentId,
        filename: m.filename,
        chunkIndex: m.chunkIndex,
      })),
      retrievedChunks: context.length,
    });
  } catch (error) {
    console.error('Query error:', error);
    res.status(500).json({ error: error.message });
  }
}

module.exports = { handleQuery };
```

---

#### Step 2.2: LLM Service (`src/services/llm/llmService.js`)

**Purpose:** Interface with Grok API for answer generation.

**Hallucination Reduction Strategy:**
Use a **strict system prompt** that enforces context-only answering:

```javascript
const axios = require("axios");
const config = require("../../config/llm");

function buildRAGPrompt(question, contextChunks, metadata) {
  const context = contextChunks
    .map((chunk, i) => `[Document ${i + 1}: ${metadata[i].filename}]\n${chunk}`)
    .join("\n\n---\n\n");

  return {
    systemPrompt: `You are a helpful assistant that answers questions based ONLY on the provided context. 

CRITICAL RULES:
1. If the answer is not in the context, respond: "I don't have enough information to answer this question."
2. Never make up information or use external knowledge.
3. Always cite which document your answer came from.
4. Be concise and direct.`,

    userPrompt: `Context:\n${context}\n\nQuestion: ${question}\n\nAnswer:`,
  };
}

async function generateAnswer(question, contextChunks, metadata) {
  try {
    const prompt = buildRAGPrompt(question, contextChunks, metadata);

    const response = await axios.post(
      `${config.grok.apiUrl}/chat/completions`,
      {
        model: config.grok.model,
        messages: [
          { role: "system", content: prompt.systemPrompt },
          { role: "user", content: prompt.userPrompt },
        ],
        temperature: 0.1, // Low temperature = less creative, more factual
        max_tokens: 500,
      },
      {
        headers: {
          Authorization: `Bearer ${config.grok.apiKey}`,
          "Content-Type": "application/json",
        },
      },
    );

    return {
      text: response.data.choices[0].message.content,
      usage: response.data.usage,
    };
  } catch (error) {
    throw new Error(`Grok API error: ${error.message}`);
  }
}

module.exports = { generateAnswer };
```

````

---

### API Endpoints

**Query Endpoint:** `POST /query`

**Request Body:**

```json
{
  "question": "What are the key features of our product?",
  "topK": 5
}
````

**Response:**

```json
{
  "question": "What are the key features of our product?",
  "answer": "Based on the product documentation, the key features include...",
  "citations": [
    {
      "documentId": "uuid-123",
      "filename": "product_spec.pdf",
      "chunkIndex": 3
    }
  ],
  "retrievedChunks": 5
}
```

---

## 🔄 Module 3: Comparison API

### Goal

Build a `POST /compare` endpoint that compares two specific documents on a given topic.

### Implementation Steps

#### Step 3.1: Comparison Controller (`src/controllers/compareController.js`)

**Comparison Flow:**

1. Receive topic + two document IDs
2. Retrieve all chunks from both documents
3. Filter chunks relevant to the topic (semantic search within each document)
4. Send both documents' relevant sections to Grok LLM
5. Ask LLM to compare and contrast
6. Return structured comparison (JSON or table format)

**Key Implementation:**

```javascript
const embeddingService = require("../services/embeddings/embeddingService");
const vectorStoreService = require("../vectorstore/vectorStoreService");
const llmService = require("../services/llm/llmService");
const { validateComparison } = require("../utils/validators");

async function compareDocuments(req, res) {
  try {
    // Validate input
    const { error, value } = validateComparison(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { topic, documentId1, documentId2 } = value;

    // 1. Generate topic embedding
    const topicVector = await embeddingService.generateEmbedding(topic);

    // 2. Retrieve relevant chunks from both documents
    const doc1Chunks = await vectorStoreService.searchByDocument(
      documentId1,
      topicVector,
      5,
    );
    const doc2Chunks = await vectorStoreService.searchByDocument(
      documentId2,
      topicVector,
      5,
    );

    // 3. Get LLM comparison
    const comparison = await llmService.generateComparison(
      topic,
      doc1Chunks,
      doc2Chunks,
    );

    // 4. Return structured response
    res.json({
      topic,
      document1: {
        id: documentId1,
        filename: doc1Chunks.metadatas[0][0].filename,
      },
      document2: {
        id: documentId2,
        filename: doc2Chunks.metadata[0].filename,
      },
      comparison: comparison.structured,
      differences: comparison.differences,
      similarities: comparison.similarities,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}
```

---

#### Step 3.2: Comparison Prompt Engineering

**Strategy:** Ask the LLM to return structured JSON comparing the documents.

```javascript
function buildComparisonPrompt(topic, doc1Chunks, doc2Chunks) {
  const doc1Context = doc1Chunks.documents.join("\n\n");
  const doc2Context = doc2Chunks.documents.join("\n\n");

  return {
    systemPrompt: `You are a document comparison expert. Compare two documents on a specific topic and return a structured JSON response.

Response format:
{
  "summary": "Brief comparison summary",
  "differences": [
    {"aspect": "...", "document1": "...", "document2": "..."}
  ],
  "similarities": ["..."],
  "conclusion": "..."
}`,

    userPrompt: `Topic: ${topic}

Document 1:
${doc1Context}

Document 2:
${doc2Context}

Compare these documents on the topic "${topic}" and return a structured comparison.`,
  };
}
```

---

#### Step 3.3: Comparison Routes (`src/routes/compare.js`)

**Endpoint:** `POST /compare`

**Request Body:**

```json
{
  "topic": "pricing structure",
  "documentId1": "uuid-123",
  "documentId2": "uuid-456"
}
```

**Response:**

```json
{
  "topic": "pricing structure",
  "document1": {
    "id": "uuid-123",
    "filename": "2024_pricing.pdf"
  },
  "document2": {
    "id": "uuid-456",
    "filename": "2023_pricing.pdf"
  },
  "comparison": {
    "summary": "The 2024 pricing shows a 15% increase...",
    "differences": [
      {
        "aspect": "Base pricing",
        "document1": "$99/month",
        "document2": "$85/month"
      }
    ],
    "similarities": [
      "Both offer free trial",
      "Same enterprise discount structure"
    ]
  }
}
```

---

## 🛠️ Module 4: Server Setup & Configuration

### Step 4.1: Main Server (`server.js`)

**Already implemented!** The server is configured with:

- Express web server
- CORS enabled
- File upload handling (multer)
- Three main endpoints: `/upload`, `/query`, `/compare`
- ChromaDB initialization
- Error handling middleware

Refer to the [server.js](server.js) file for the complete implementation.

---

### Step 4.2: Database Configuration (`src/config/database.js`)

**Already configured!** ChromaDB settings are in place.

---

### Step 4.3: LLM Configuration (`src/config/llm.js`)

**Already configured!** Grok API settings are ready.

---

### Step 4.4: Error Handler (`src/utils/errorHandler.js`)

```javascript
module.exports = (err, req, res, next) => {
  console.error("Error:", err);

  const statusCode = err.statusCode || 500;
  const message = err.message || "Internal server error";

  res.status(statusCode).json({
    error: {
      message,
      status: statusCode,
      ...(process.env.NODE_ENV === "development" && { stack: err.stack }),
    },
  });
};
```

---

### Step 4.5: Input Validators (`src/utils/validators.js`)

```javascript
const Joi = require("joi");

const querySchema = Joi.object({
  question: Joi.string().required().min(3).max(500),
  topK: Joi.number().integer().min(1).max(20).default(5),
});

const compareSchema = Joi.object({
  topic: Joi.string().required().min(3).max(200),
  documentId1: Joi.string().required(),
  documentId2: Joi.string().required(),
});

function validateQuery(data) {
  return querySchema.validate(data);
}

function validateComparison(data) {
  return compareSchema.validate(data);
}

module.exports = {
  validateQuery,
  validateComparison,
};
```

---

## 🎓 Design Decisions & Justifications

### 1. Why Chunk Size = 1000 words?

**Reasoning:**

- **Too small (e.g., 100 words):** Context is fragmented, answers lack coherence
- **Too large (e.g., 5000 words):** Retrieval becomes imprecise, too much noise
- **1000 words (~1500 tokens):** Sweet spot for most LLMs
  - Provides sufficient context
  - Stays well within token limits
  - Allows embedding models to capture semantic meaning effectively

**Adjustments:**

- Technical documentation: 500-800 words (more precise retrieval)
- Narrative/legal documents: 1500-2000 words (more context needed)

---

### 2. Why 50-word Overlap?

**Reasoning:**

- Prevents loss of context at chunk boundaries
- Ensures sentences/paragraphs aren't artificially split
- ~5% overlap is standard practice in RAG systems
- Minimal storage overhead, maximum context preservation

**Example:**

```
Chunk 1: [words 0-1000]
Chunk 2: [words 950-1950] ← 50 words overlap with Chunk 1
Chunk 3: [words 1900-2900] ← 50 words overlap with Chunk 2
```

---

### 3. Why ChromaDB?

**ChromaDB** is our chosen vector database for this RAG system.

**Advantages:**

- ✅ Easy local setup and development
- ✅ Free and open-source
- ✅ Python and JavaScript SDKs
- ✅ Built-in embedding support
- ✅ SQL-like query interface
- ✅ Persistent storage

**Setup:**

```bash
pip install chromadb
chroma run --host localhost --port 8000
```

**Production Considerations:**

- For production at scale, consider:
  - Docker deployment for ChromaDB
  - Load balancing for high traffic
  - Backup strategies for vector collections
  - Monitoring and alerting

---

### 4. Why Grok AI?

**Grok** (from xAI) provides both embeddings and LLM capabilities.

**Advantages:**

- ✅ Single API for both embeddings and text generation
- ✅ Competitive pricing
- ✅ High-quality embeddings
- ✅ Strong reasoning capabilities
- ✅ API compatible with OpenAI format

**Setup:**

- Sign up at https://x.ai/api
- Get your API key
- Add to `.env` file: `GROK_API_KEY=your_key_here`

---

### 5. Hallucination Reduction Strategy

**Problem:** LLMs will confidently make up answers when they don't know something.

**Solution: 3-Layer Approach**

**Layer 1: System Prompt Engineering**

```
"You MUST answer ONLY based on the provided context.
If the answer is not in the context, say 'I don't have enough information.'"
```

**Layer 2: Low Temperature Setting**

- `temperature: 0.1` = Less creative, more deterministic
- Reduces random/hallucinated responses

**Layer 3: Citation Enforcement**

- Force LLM to cite sources
- Return chunk metadata with every response
- Users can verify answers against original documents

**Advanced Techniques (Optional):**

- **Confidence scoring:** Ask LLM to rate its confidence (0-10)
- **Multi-query retrieval:** Rephrase user question, retrieve multiple times, merge results
- **Re-ranking:** Use cross-encoder model to re-rank retrieved chunks

---

### 6. Top-K Retrieval: How Many Chunks?

**Default: K=5**

**Reasoning:**

- K=3: Often too few, misses relevant context
- K=5: Good balance for most questions
- K=10+: Too much noise, confuses LLM, higher costs

**Dynamic K (Advanced):**

```javascript
// Adjust K based on question complexity
function determineTopK(question) {
  const wordCount = question.split(" ").length;
  if (wordCount > 20) return 7; // Complex question
  if (wordCount > 10) return 5; // Medium question
  return 3; // Simple question
}
```

---

## 🧪 Testing Your System

### Test 1: Upload a Document

```bash
curl -X POST http://localhost:3000/upload \
  -F "file=@./test_document.pdf"
```

Expected response:

```json
{
  "success": true,
  "documentId": "uuid-123",
  "filename": "test_document.pdf",
  "chunksCount": 45
}
```

---

### Test 2: Query the System

```bash
curl -X POST http://localhost:3000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main features?",
    "topK": 5
  }'
```

Expected response:

```json
{
  "question": "What are the main features?",
  "answer": "Based on the documentation in test_document.pdf, the main features include...",
  "citations": [
    {
      "documentId": "uuid-123",
      "filename": "test_document.pdf",
      "chunkIndex": 3
    }
  ],
  "retrievedChunks": 5
}
```

---

### Test 3: Compare Two Documents

```bash
curl -X POST http://localhost:3000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "pricing",
    "documentId1": "uuid-123",
    "documentId2": "uuid-456"
  }'
```

---

## 🚀 Deployment Checklist

### Before Production:

- [ ] Add authentication/authorization (JWT tokens)
- [ ] Rate limiting (prevent abuse)
- [ ] Input sanitization (prevent injection attacks)
- [ ] Logging system (Winston/Morgan)
- [ ] Monitoring (Prometheus/New Relic)
- [ ] Containerization (Docker)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] API documentation (Swagger/OpenAPI)
- [ ] Load testing (Artillery/k6)
- [ ] Backup strategy for vector database

---

## 📚 Additional Features (Future Enhancements)

1. **Document Management:**
   - List all uploaded documents
   - Delete documents (and their vectors)
   - Update documents (re-process and re-embed)

2. **Query History:**
   - Store user queries for analytics
   - Track popular questions
   - Feedback loop (thumbs up/down)

3. **Multi-tenancy:**
   - Separate collections per user/organization
   - Access control per document

4. **Hybrid Search:**
   - Combine semantic (vector) search with keyword (BM25) search
   - Often improves retrieval quality

5. **Advanced RAG Techniques:**
   - **HyDE (Hypothetical Document Embeddings):** Generate hypothetical answer, embed it, search with that
   - **Reranking:** Use cross-encoder to re-score retrieved chunks
   - **Query decomposition:** Break complex questions into sub-questions

---

## 🆘 Troubleshooting

### Issue: Embeddings API timeout

**Solution:** Batch your embedding requests, add retry logic with exponential backoff.

### Issue: Vector search returns irrelevant results

**Solution:**

- Try different embedding models
- Adjust chunk size/overlap
- Increase top-K and let LLM filter
- Implement re-ranking

### Issue: LLM still hallucinates

**Solution:**

- Lower temperature (try 0.0)
- Strengthen system prompt
- Add examples of correct behavior (few-shot prompting)
- Increase context quality (better chunking strategy)

### Issue: Slow response times

**Solution:**

- Cache frequent queries and embeddings
- Batch embedding requests
- Reduce max_tokens in LLM call
- Optimize ChromaDB (add indexes, tune parameters)
- Consider async processing for large documents

---

## 📖 Key Resources

- [Grok API Documentation](https://x.ai/api)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [ChromaDB GitHub](https://github.com/chroma-core/chroma)
- [RAG Best Practices](https://arxiv.org/abs/2312.10997)
- [Advanced RAG Techniques](https://python.langchain.com/docs/use_cases/question_answering/)

---

## 🎯 Summary: Implementation Order

1. ✅ **Setup** (Phase 1): Install dependencies, setup ChromaDB server, configure Grok API
2. 🔧 **Document Ingestion** (Module 1): Parser → Chunking → Grok Embeddings → ChromaDB Storage
3. 🔍 **Query System** (Module 2): Query embedding → ChromaDB Retrieval → Grok LLM → Response
4. 🔄 **Comparison** (Module 3): Multi-document retrieval → Comparison prompt → Structured output
5. 🚀 **Server** (Module 4): Express server, controllers, error handling
6. 🧪 **Testing**: Test all endpoints thoroughly
7. 📦 **Production**: Deploy with proper security and monitoring

---

## 🏗️ Quick Start Guide

1. **Install dependencies:**

   ```bash
   npm install
   ```

2. **Setup ChromaDB:**

   ```bash
   pip install chromadb
   chroma run --host localhost --port 8000
   ```

3. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env and add your GROK_API_KEY
   ```

4. **Start the server:**

   ```bash
   npm start
   # or for development:
   npm run dev
   ```

5. **Test the system:**
   - Upload a document: `POST /upload`
   - Query: `POST /query`
   - Compare: `POST /compare`

---

**Good luck building your production-grade RAG system with Grok AI + ChromaDB! 🚀**

If you encounter issues, refer to the troubleshooting section or consult the key resources listed above.
