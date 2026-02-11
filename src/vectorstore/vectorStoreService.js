/**
 * Vector Store Service
 * Manages ChromaDB collection: store, search, delete embeddings.
 */
const { ChromaClient } = require('chromadb');
const dbConfig = require('../config/database');
const { ExternalServiceError } = require('../utils/errorHandler');

let client = null;
let collection = null;

/**
 * Initialize the ChromaDB client and ensure the collection exists.
 */
async function initialize() {
  try {
    client = new ChromaClient({ path: dbConfig.chromaHost });

    // Get or create the collection with cosine distance
    collection = await client.getOrCreateCollection({
      name: dbConfig.collectionName,
      metadata: { 'hnsw:space': dbConfig.distanceFunction }
    });

    const count = await collection.count();
    console.log(`[VectorStore] Connected to ChromaDB. Collection "${dbConfig.collectionName}" has ${count} vectors.`);
    return collection;
  } catch (err) {
    throw new ExternalServiceError('ChromaDB', `Initialization failed: ${err.message}`);
  }
}

/**
 * Ensure the collection is ready; initialize lazily if needed.
 */
async function getCollection() {
  if (!collection) {
    await initialize();
  }
  return collection;
}

/**
 * Store document chunk embeddings in ChromaDB.
 * @param {Object[]} chunkMetadata - Array of { chunkId, documentId, filename, chunkIndex, totalChunks, text, wordCount }
 * @param {number[][]} embeddings - Corresponding embedding vectors
 */
async function storeChunks(chunkMetadata, embeddings) {
  const col = await getCollection();
  const batchSize = dbConfig.batchSize || 100;

  for (let i = 0; i < chunkMetadata.length; i += batchSize) {
    const batchMeta = chunkMetadata.slice(i, i + batchSize);
    const batchEmb = embeddings.slice(i, i + batchSize);

    try {
      await col.add({
        ids: batchMeta.map(c => c.chunkId),
        embeddings: batchEmb,
        documents: batchMeta.map(c => c.text),
        metadatas: batchMeta.map(c => ({
          documentId: c.documentId,
          filename: c.filename,
          chunkIndex: c.chunkIndex,
          totalChunks: c.totalChunks,
          wordCount: c.wordCount,
          uploadedAt: new Date().toISOString()
        }))
      });
    } catch (err) {
      throw new ExternalServiceError('ChromaDB', `Failed to store chunks: ${err.message}`);
    }
  }

  console.log(`[VectorStore] Stored ${chunkMetadata.length} chunks for document ${chunkMetadata[0]?.documentId}`);
}

/**
 * Search for similar chunks across all documents.
 * @param {number[]} queryEmbedding - The query vector
 * @param {number} topK - Number of results to return
 * @param {Object} whereFilter - Optional metadata filter (e.g., { documentId: '...' })
 * @returns {Promise<Object[]>} Array of { text, metadata, distance }
 */
async function searchSimilar(queryEmbedding, topK = 5, whereFilter = null) {
  const col = await getCollection();

  try {
    const queryParams = {
      queryEmbeddings: [queryEmbedding],
      nResults: topK,
      include: ['documents', 'metadatas', 'distances']
    };

    if (whereFilter && Object.keys(whereFilter).length > 0) {
      queryParams.where = whereFilter;
    }

    const results = await col.query(queryParams);

    // Flatten the nested arrays ChromaDB returns
    if (!results.ids || !results.ids[0]) return [];

    return results.ids[0].map((id, idx) => ({
      chunkId: id,
      text: results.documents[0][idx],
      metadata: results.metadatas[0][idx],
      distance: results.distances[0][idx],
      // Convert distance to a 0-1 similarity score (cosine distance → similarity)
      similarityScore: 1 - results.distances[0][idx]
    }));
  } catch (err) {
    throw new ExternalServiceError('ChromaDB', `Search failed: ${err.message}`);
  }
}

/**
 * Search similar chunks scoped to a specific document.
 */
async function searchByDocument(queryEmbedding, documentId, topK = 5) {
  return searchSimilar(queryEmbedding, topK, { documentId });
}

/**
 * Delete all chunks belonging to a document.
 * @param {string} documentId
 */
async function deleteDocument(documentId) {
  const col = await getCollection();

  try {
    // ChromaDB supports where-based deletion
    await col.delete({ where: { documentId } });
    console.log(`[VectorStore] Deleted all chunks for document ${documentId}`);
  } catch (err) {
    throw new ExternalServiceError('ChromaDB', `Delete failed: ${err.message}`);
  }
}

/**
 * List all unique documents stored in the collection.
 * @returns {Promise<Object[]>} Array of { documentId, filename, chunkCount, uploadedAt }
 */
async function listDocuments() {
  const col = await getCollection();

  try {
    // Get all metadata from the collection
    const all = await col.get({ include: ['metadatas'] });

    if (!all.ids || all.ids.length === 0) return [];

    // Group by documentId
    const docMap = new Map();
    for (const meta of all.metadatas) {
      if (!docMap.has(meta.documentId)) {
        docMap.set(meta.documentId, {
          documentId: meta.documentId,
          filename: meta.filename,
          chunkCount: 0,
          uploadedAt: meta.uploadedAt
        });
      }
      docMap.get(meta.documentId).chunkCount++;
    }

    return Array.from(docMap.values());
  } catch (err) {
    throw new ExternalServiceError('ChromaDB', `List failed: ${err.message}`);
  }
}

/**
 * Get collection statistics.
 */
async function getStats() {
  const col = await getCollection();
  const count = await col.count();
  const docs = await listDocuments();
  return {
    totalChunks: count,
    totalDocuments: docs.length,
    documents: docs
  };
}

module.exports = {
  initialize,
  getCollection,
  storeChunks,
  searchSimilar,
  searchByDocument,
  deleteDocument,
  listDocuments,
  getStats
};
