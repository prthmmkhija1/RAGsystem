/**
 * RAG Orchestration Service
 * Ties together: embedding → vector search → reranking → LLM answer generation.
 */
const embeddingService = require('../embeddings/embeddingService');
const llmService = require('../llm/llmService');
const rerankService = require('./rerankService');
const vectorStoreService = require('../../vectorstore/vectorStoreService');
const cacheService = require('../../utils/cacheService');
const { parseDocument } = require('../../utils/documentParser');
const { chunkText, createChunkMetadata } = require('../../utils/chunkingService');
const { v4: uuidv4 } = require('uuid');

// ──────────────────────────────────────────
// Document Ingestion Pipeline
// ──────────────────────────────────────────

/**
 * Full upload pipeline: parse → chunk → embed → store.
 * @param {Object} file - Multer file object { originalname, buffer, mimetype }
 * @param {Object} opts - { chunkSize, chunkOverlap }
 * @returns {Promise<Object>} { documentId, filename, chunkCount, message }
 */
async function ingestDocument(file, opts = {}) {
  const documentId = uuidv4();
  const filename = file.originalname;
  const startTime = Date.now();

  console.log(`[RAG] Ingesting "${filename}" (id: ${documentId})`);

  // 1. Parse the document into raw text
  const rawText = await parseDocument(file);
  console.log(`[RAG]   Parsed: ${rawText.length} characters`);

  // 2. Chunk the text
  const chunks = chunkText(rawText, {
    chunkSize: opts.chunkSize,
    chunkOverlap: opts.chunkOverlap
  });
  console.log(`[RAG]   Chunked: ${chunks.length} chunks`);

  if (chunks.length === 0) {
    throw new Error('Document produced zero chunks after parsing');
  }

  // 3. Create chunk metadata
  const chunkMeta = createChunkMetadata(documentId, filename, chunks);

  // 4. Generate embeddings for all chunks (batched)
  const embeddings = await embeddingService.generateEmbeddingsBatched(
    chunks,
    20 // batch size
  );
  console.log(`[RAG]   Embedded: ${embeddings.length} vectors`);

  // 5. Store in ChromaDB
  await vectorStoreService.storeChunks(chunkMeta, embeddings);

  // 6. Invalidate query cache since document corpus changed
  cacheService.invalidateQueries();

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
  console.log(`[RAG]   Ingestion complete in ${elapsed}s`);

  return {
    documentId,
    filename,
    chunkCount: chunks.length,
    characterCount: rawText.length,
    processingTime: `${elapsed}s`,
    message: `Successfully ingested "${filename}" into ${chunks.length} chunks`
  };
}

// ──────────────────────────────────────────
// Query Pipeline
// ──────────────────────────────────────────

/**
 * Full query pipeline: embed query → search → rerank → generate answer.
 * @param {string} query - User question
 * @param {Object} opts - { topK, documentId, temperature, includeMetadata, verify, rerank, skipCache }
 * @returns {Promise<Object>} { answer, sources, query, processingTime, verification? }
 */
async function queryDocuments(query, opts = {}) {
  const topK = opts.topK || 5;
  const shouldVerify = opts.verify === true;
  const shouldRerank = opts.rerank === true;
  const skipCache = opts.skipCache === true || shouldVerify; // Don't cache verified queries
  const startTime = Date.now();

  // Check query cache first (only for non-verify requests)
  if (!skipCache) {
    const cacheOpts = { topK, documentId: opts.documentId, rerank: shouldRerank };
    const cached = cacheService.getQueryResult(query, cacheOpts);
    if (cached) {
      console.log(`[RAG] Query served from cache: "${query.substring(0, 50)}..."`);
      return {
        ...cached,
        processingTime: '0.00s (cached)'
      };
    }
  }

  console.log(`[RAG] Query: "${query.substring(0, 80)}..." (topK=${topK}, verify=${shouldVerify}, rerank=${shouldRerank})`);

  // 1. Generate embedding for the query
  const queryEmbedding = await embeddingService.generateEmbedding(query);

  // 2. Search for relevant chunks (fetch more if reranking)
  const fetchK = shouldRerank ? Math.min(topK * 3, 20) : topK;
  let results;
  if (opts.documentId) {
    results = await vectorStoreService.searchByDocument(queryEmbedding, opts.documentId, fetchK);
  } else {
    results = await vectorStoreService.searchSimilar(queryEmbedding, fetchK);
  }

  if (results.length === 0) {
    return {
      answer: 'No relevant documents found for your query. Please upload documents first or refine your question.',
      confidence: { score: 0, reason: 'No relevant documents found', level: 'none' },
      sources: [],
      query,
      reranked: false,
      processingTime: `${((Date.now() - startTime) / 1000).toFixed(2)}s`
    };
  }

  // 3. Optionally re-rank using cross-encoder
  if (shouldRerank && results.length > 1) {
    console.log(`[RAG]   Re-ranking ${results.length} chunks...`);
    results = await rerankService.rerankChunks(query, results, { topN: topK });
  } else if (results.length > topK) {
    // Trim to topK if we fetched extra but didn't rerank
    results = results.slice(0, topK);
  }

  // 4. Generate answer with LLM using the retrieved context
  const rawAnswer = await llmService.generateAnswer(query, results, {
    temperature: opts.temperature
  });

  // 5. Parse confidence from the response
  const { answer, confidence } = llmService.parseConfidence(rawAnswer);

  // 6. Optionally verify the answer against sources
  let verification = null;
  if (shouldVerify) {
    console.log(`[RAG]   Running answer verification...`);
    verification = await llmService.verifyAnswer(answer, results);
    console.log(`[RAG]   Verification complete: ${verification.isVerified ? 'PASSED' : 'FAILED'} (${verification.overallScore}/10)`);
  }

  // 7. Build source citations
  const sources = results.map(r => ({
    filename: r.metadata?.filename,
    chunkIndex: r.metadata?.chunkIndex,
    documentId: r.metadata?.documentId,
    similarityScore: parseFloat((r.similarityScore || 0).toFixed(4)),
    ...(r.crossEncoderScore !== undefined && {
      crossEncoderScore: r.crossEncoderScore,
      originalRank: r.originalRank
    }),
    ...(opts.includeMetadata !== false && {
      preview: r.text?.substring(0, 200) + (r.text?.length > 200 ? '...' : '')
    })
  }));

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
  console.log(`[RAG]   Query answered in ${elapsed}s using ${results.length} chunks (confidence: ${confidence.score}/10)`);

  const response = {
    answer,
    confidence,
    sources,
    query,
    topK,
    reranked: shouldRerank,
    chunksUsed: results.length,
    processingTime: `${elapsed}s`
  };

  // Include verification results if requested
  if (verification) {
    response.verification = verification;
  }

  // Cache the result (only if not skipping cache)
  if (!skipCache) {
    const cacheOpts = { topK, documentId: opts.documentId, rerank: shouldRerank };
    cacheService.setQueryResult(query, cacheOpts, response);
  }

  return response;
}

// ──────────────────────────────────────────
// Compare Pipeline
// ──────────────────────────────────────────

/**
 * Compare two documents on a given topic.
 * @param {string[]} documentIds - Exactly 2 document IDs
 * @param {string} topic - What to compare
 * @param {Object} opts - { topK, structured }
 * @returns {Promise<Object>} { comparison, doc1Sources, doc2Sources, topic, processingTime }
 */
async function compareDocuments(documentIds, topic, opts = {}) {
  const topK = opts.topK || 5;
  const useStructured = opts.structured === true;
  const startTime = Date.now();

  console.log(`[RAG] Comparing docs [${documentIds.join(', ')}] on topic: "${topic.substring(0, 60)}" (structured=${useStructured})`);

  // 1. Generate embedding for the comparison topic
  const topicEmbedding = await embeddingService.generateEmbedding(topic);

  // 2. Search each document for relevant chunks
  const [doc1Results, doc2Results] = await Promise.all([
    vectorStoreService.searchByDocument(topicEmbedding, documentIds[0], topK),
    vectorStoreService.searchByDocument(topicEmbedding, documentIds[1], topK)
  ]);

  if (doc1Results.length === 0 && doc2Results.length === 0) {
    const emptyStructured = {
      similarities: [],
      differences: [],
      uniqueToDoc1: [],
      uniqueToDoc2: [],
      summary: {
        overallAssessment: 'No relevant content found in either document for the given topic.',
        agreementLevel: 'none',
        keyTakeaway: 'Upload documents with relevant content and try again.'
      },
      metadata: {
        doc1ChunksAnalyzed: 0,
        doc2ChunksAnalyzed: 0,
        topic,
        comparisonConfidence: 0
      }
    };

    return {
      comparison: useStructured ? emptyStructured : 'No relevant content found in either document for the given topic.',
      structured: useStructured,
      doc1Sources: [],
      doc2Sources: [],
      topic,
      processingTime: `${((Date.now() - startTime) / 1000).toFixed(2)}s`
    };
  }

  // 3. Generate comparison with LLM (structured or plain text)
  let comparison;
  if (useStructured) {
    comparison = await llmService.generateStructuredComparison(topic, doc1Results, doc2Results);
  } else {
    comparison = await llmService.generateComparison(topic, doc1Results, doc2Results);
  }

  // 4. Build source citations for each document
  const buildSources = (results) =>
    results.map(r => ({
      filename: r.metadata?.filename,
      chunkIndex: r.metadata?.chunkIndex,
      documentId: r.metadata?.documentId,
      similarityScore: parseFloat((r.similarityScore || 0).toFixed(4)),
      preview: r.text?.substring(0, 200) + (r.text?.length > 200 ? '...' : '')
    }));

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
  console.log(`[RAG]   Comparison generated in ${elapsed}s`);

  return {
    comparison,
    structured: useStructured,
    doc1Sources: buildSources(doc1Results),
    doc2Sources: buildSources(doc2Results),
    topic,
    documentsCompared: documentIds,
    processingTime: `${elapsed}s`
  };
}

// ──────────────────────────────────────────
// Document Management
// ──────────────────────────────────────────

/**
 * Delete a document and all its chunks from the vector store.
 */
async function deleteDocument(documentId) {
  await vectorStoreService.deleteDocument(documentId);
  
  // Invalidate query cache since document corpus changed
  cacheService.invalidateDocumentQueries(documentId);
  
  return { documentId, message: 'Document deleted successfully' };
}

/**
 * List all ingested documents.
 */
async function listDocuments() {
  return vectorStoreService.listDocuments();
}

/**
 * Get vector store statistics.
 */
async function getStats() {
  return vectorStoreService.getStats();
}

module.exports = {
  ingestDocument,
  queryDocuments,
  compareDocuments,
  deleteDocument,
  listDocuments,
  getStats
};
