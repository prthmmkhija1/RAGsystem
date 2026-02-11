/**
 * Embedding Service
 * Generates vector embeddings using the Grok (xAI) Embeddings API.
 * Includes caching layer for performance optimization.
 */
const axios = require('axios');
const llmConfig = require('../../config/llm');
const cacheService = require('../../utils/cacheService');
const { ExternalServiceError } = require('../../utils/errorHandler');

/**
 * Generate an embedding vector for a single text string (with caching).
 * @param {string} text - Input text to embed
 * @param {boolean} skipCache - Skip cache lookup (default false)
 * @returns {Promise<number[]>} Embedding vector
 */
async function generateEmbedding(text, skipCache = false) {
  // Check cache first
  if (!skipCache) {
    const cached = cacheService.getEmbedding(text);
    if (cached) {
      return cached;
    }
  }

  const embeddings = await generateEmbeddings([text]);
  const embedding = embeddings[0];

  // Cache the result
  cacheService.setEmbedding(text, embedding);

  return embedding;
}

/**
 * Generate embeddings for an array of text strings (batch).
 * Includes retry logic with exponential back-off.
 * @param {string[]} texts - Array of text strings
 * @returns {Promise<number[][]>} Array of embedding vectors
 */
async function generateEmbeddings(texts) {
  if (!texts || texts.length === 0) {
    return [];
  }

  let lastError;

  for (let attempt = 1; attempt <= llmConfig.retryAttempts; attempt++) {
    try {
      const response = await axios.post(
        `${llmConfig.apiUrl}/embeddings`,
        {
          model: llmConfig.embeddingModel,
          input: texts,
          encoding_format: 'float'
        },
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${llmConfig.apiKey}`
          },
          timeout: 60000 // 60 s per request
        }
      );

      // Sort by index to maintain order
      const sorted = response.data.data.sort((a, b) => a.index - b.index);
      return sorted.map(item => item.embedding);
    } catch (err) {
      lastError = err;
      const status = err.response?.status;

      // Rate-limited → respect Retry-After header
      if (status === 429) {
        const retryAfter =
          parseInt(err.response.headers['retry-after'], 10) || llmConfig.retryDelay / 1000;
        console.warn(`Grok rate limit hit, retrying in ${retryAfter}s (attempt ${attempt}/${llmConfig.retryAttempts})`);
        await sleep(retryAfter * 1000);
        continue;
      }

      // Transient server errors → retry with back-off
      if (status >= 500 && attempt < llmConfig.retryAttempts) {
        const delay = llmConfig.retryDelay * Math.pow(2, attempt - 1);
        console.warn(`Grok server error ${status}, retrying in ${delay}ms (attempt ${attempt}/${llmConfig.retryAttempts})`);
        await sleep(delay);
        continue;
      }

      // Non-retriable error → throw immediately
      break;
    }
  }

  const msg = lastError.response?.data?.error?.message || lastError.message;
  throw new ExternalServiceError('Grok Embeddings', msg);
}

/**
 * Generate embeddings in batches to avoid payload limits.
 * Uses caching to avoid re-generating embeddings for previously seen text.
 * @param {string[]} texts - All texts to embed
 * @param {number} batchSize - Texts per API call (default 20)
 * @param {boolean} skipCache - Skip cache lookup (default false)
 * @returns {Promise<number[][]>}
 */
async function generateEmbeddingsBatched(texts, batchSize = 20, skipCache = false) {
  if (!texts || texts.length === 0) {
    return [];
  }

  // Check cache for existing embeddings
  const results = new Array(texts.length);
  let textsToGenerate = [];
  let indicesToFill = [];

  if (!skipCache) {
    const { hits, misses } = cacheService.getEmbeddingsBatch(texts);
    
    // Fill in cached results
    texts.forEach((text, idx) => {
      if (hits.has(text)) {
        results[idx] = hits.get(text);
      } else {
        textsToGenerate.push(text);
        indicesToFill.push(idx);
      }
    });

    if (textsToGenerate.length === 0) {
      console.log(`[Embeddings] All ${texts.length} embeddings served from cache`);
      return results;
    }

    console.log(`[Embeddings] Cache: ${hits.size} hits, ${misses.length} misses`);
  } else {
    textsToGenerate = texts;
    indicesToFill = texts.map((_, i) => i);
  }

  // Generate embeddings for cache misses
  const generatedEmbeddings = [];
  for (let i = 0; i < textsToGenerate.length; i += batchSize) {
    const batch = textsToGenerate.slice(i, i + batchSize);
    const embeddings = await generateEmbeddings(batch);
    generatedEmbeddings.push(...embeddings);

    // Small delay between batches to be polite to the API
    if (i + batchSize < textsToGenerate.length) {
      await sleep(200);
    }
  }

  // Fill in results and cache new embeddings
  const toCache = new Map();
  generatedEmbeddings.forEach((emb, i) => {
    const originalIdx = indicesToFill[i];
    results[originalIdx] = emb;
    toCache.set(textsToGenerate[i], emb);
  });

  // Cache all new embeddings
  cacheService.setEmbeddingsBatch(toCache);

  return results;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

module.exports = {
  generateEmbedding,
  generateEmbeddings,
  generateEmbeddingsBatched
};
