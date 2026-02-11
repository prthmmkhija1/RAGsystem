/**
 * Cache Service
 * Provides in-memory caching for embeddings, query results, and other expensive operations.
 * 
 * Cache Types:
 * - Embedding cache: Stores text → embedding vector mappings
 * - Query cache: Stores complete query → response mappings
 * - Document cache: Stores document metadata/chunk counts
 */
const NodeCache = require('node-cache');
const crypto = require('crypto');

// ── Cache Configuration ──────────────────────────────────
const CACHE_CONFIG = {
  // Embedding cache - longer TTL since embeddings are deterministic
  embeddings: {
    ttl: parseInt(process.env.EMBEDDING_CACHE_TTL, 10) || 86400, // 24 hours
    maxKeys: parseInt(process.env.EMBEDDING_CACHE_MAX, 10) || 10000,
    checkperiod: 600 // Check for expired keys every 10 min
  },
  // Query cache - shorter TTL, can be invalidated by new docs
  queries: {
    ttl: parseInt(process.env.QUERY_CACHE_TTL, 10) || 3600, // 1 hour
    maxKeys: parseInt(process.env.QUERY_CACHE_MAX, 10) || 1000,
    checkperiod: 120
  },
  // Document metadata cache
  documents: {
    ttl: parseInt(process.env.DOC_CACHE_TTL, 10) || 1800, // 30 min
    maxKeys: 500,
    checkperiod: 300
  }
};

// ── Initialize Caches ────────────────────────────────────
const embeddingCache = new NodeCache({
  stdTTL: CACHE_CONFIG.embeddings.ttl,
  maxKeys: CACHE_CONFIG.embeddings.maxKeys,
  checkperiod: CACHE_CONFIG.embeddings.checkperiod,
  useClones: false // Embeddings are read-only, avoid cloning overhead
});

const queryCache = new NodeCache({
  stdTTL: CACHE_CONFIG.queries.ttl,
  maxKeys: CACHE_CONFIG.queries.maxKeys,
  checkperiod: CACHE_CONFIG.queries.checkperiod,
  useClones: true // Query results should be cloned
});

const documentCache = new NodeCache({
  stdTTL: CACHE_CONFIG.documents.ttl,
  maxKeys: CACHE_CONFIG.documents.maxKeys,
  checkperiod: CACHE_CONFIG.documents.checkperiod
});

// ── Helper Functions ─────────────────────────────────────

/**
 * Generate a hash key for a given input.
 * @param {string} input - Input text to hash
 * @returns {string} MD5 hash
 */
function hashKey(input) {
  return crypto.createHash('md5').update(input).digest('hex');
}

/**
 * Generate a cache key for a query with options.
 * @param {string} query - The query text
 * @param {Object} opts - Query options affecting the result
 * @returns {string} Cache key
 */
function buildQueryCacheKey(query, opts = {}) {
  const keyParts = [
    query,
    opts.topK || 5,
    opts.documentId || 'all',
    opts.rerank ? 'rerank' : 'norerank'
  ];
  return `q:${hashKey(keyParts.join('|'))}`;
}

// ── Embedding Cache ──────────────────────────────────────

/**
 * Get cached embedding for text.
 * @param {string} text - Input text
 * @returns {number[]|null} Cached embedding or null
 */
function getEmbedding(text) {
  const key = `emb:${hashKey(text)}`;
  return embeddingCache.get(key) || null;
}

/**
 * Cache an embedding for text.
 * @param {string} text - Input text
 * @param {number[]} embedding - Embedding vector
 * @param {number} ttl - Optional custom TTL in seconds
 */
function setEmbedding(text, embedding, ttl = null) {
  const key = `emb:${hashKey(text)}`;
  if (ttl) {
    embeddingCache.set(key, embedding, ttl);
  } else {
    embeddingCache.set(key, embedding);
  }
}

/**
 * Get multiple cached embeddings (returns array with nulls for misses).
 * @param {string[]} texts - Array of input texts
 * @returns {{ hits: Map<string, number[]>, misses: string[] }}
 */
function getEmbeddingsBatch(texts) {
  const hits = new Map();
  const misses = [];

  for (const text of texts) {
    const cached = getEmbedding(text);
    if (cached) {
      hits.set(text, cached);
    } else {
      misses.push(text);
    }
  }

  return { hits, misses };
}

/**
 * Cache multiple embeddings.
 * @param {Map<string, number[]>} embeddings - Map of text → embedding
 */
function setEmbeddingsBatch(embeddings) {
  for (const [text, embedding] of embeddings) {
    setEmbedding(text, embedding);
  }
}

// ── Query Cache ──────────────────────────────────────────

/**
 * Get cached query result.
 * @param {string} query - Query text
 * @param {Object} opts - Query options
 * @returns {Object|null} Cached result or null
 */
function getQueryResult(query, opts = {}) {
  const key = buildQueryCacheKey(query, opts);
  const cached = queryCache.get(key);
  if (cached) {
    // Mark as cache hit
    cached._fromCache = true;
    cached._cachedAt = cached._cachedAt || new Date().toISOString();
  }
  return cached || null;
}

/**
 * Cache a query result.
 * @param {string} query - Query text
 * @param {Object} opts - Query options
 * @param {Object} result - Query result to cache
 * @param {number} ttl - Optional custom TTL in seconds
 */
function setQueryResult(query, opts, result, ttl = null) {
  const key = buildQueryCacheKey(query, opts);
  const toCache = {
    ...result,
    _cachedAt: new Date().toISOString()
  };
  
  if (ttl) {
    queryCache.set(key, toCache, ttl);
  } else {
    queryCache.set(key, toCache);
  }
}

// ── Document Cache ───────────────────────────────────────

/**
 * Get cached document info.
 * @param {string} documentId
 * @returns {Object|null}
 */
function getDocumentInfo(documentId) {
  return documentCache.get(`doc:${documentId}`) || null;
}

/**
 * Cache document info.
 * @param {string} documentId
 * @param {Object} info
 */
function setDocumentInfo(documentId, info) {
  documentCache.set(`doc:${documentId}`, info);
}

// ── Cache Invalidation ───────────────────────────────────

/**
 * Invalidate all query cache entries (e.g., when new documents are added).
 */
function invalidateQueries() {
  queryCache.flushAll();
  console.log('[Cache] Query cache invalidated');
}

/**
 * Invalidate query cache entries for a specific document.
 * Note: This is approximate since keys are hashed.
 */
function invalidateDocumentQueries(documentId) {
  // Full invalidation for now - could be optimized with tagging
  queryCache.flushAll();
  documentCache.del(`doc:${documentId}`);
  console.log(`[Cache] Invalidated queries for document ${documentId}`);
}

/**
 * Clear all caches.
 */
function clearAll() {
  embeddingCache.flushAll();
  queryCache.flushAll();
  documentCache.flushAll();
  console.log('[Cache] All caches cleared');
}

// ── Statistics ───────────────────────────────────────────

/**
 * Get cache statistics.
 * @returns {Object} Stats for all caches
 */
function getStats() {
  return {
    embeddings: {
      keys: embeddingCache.keys().length,
      hits: embeddingCache.getStats().hits,
      misses: embeddingCache.getStats().misses,
      hitRate: calculateHitRate(embeddingCache.getStats())
    },
    queries: {
      keys: queryCache.keys().length,
      hits: queryCache.getStats().hits,
      misses: queryCache.getStats().misses,
      hitRate: calculateHitRate(queryCache.getStats())
    },
    documents: {
      keys: documentCache.keys().length,
      hits: documentCache.getStats().hits,
      misses: documentCache.getStats().misses
    }
  };
}

function calculateHitRate(stats) {
  const total = stats.hits + stats.misses;
  if (total === 0) return '0%';
  return `${((stats.hits / total) * 100).toFixed(1)}%`;
}

module.exports = {
  // Embedding cache
  getEmbedding,
  setEmbedding,
  getEmbeddingsBatch,
  setEmbeddingsBatch,
  
  // Query cache
  getQueryResult,
  setQueryResult,
  
  // Document cache
  getDocumentInfo,
  setDocumentInfo,
  
  // Invalidation
  invalidateQueries,
  invalidateDocumentQueries,
  clearAll,
  
  // Stats
  getStats,
  
  // Key helpers
  hashKey,
  buildQueryCacheKey
};
