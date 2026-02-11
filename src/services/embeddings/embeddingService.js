/**
 * Embedding Service
 * Generates vector embeddings using the Grok (xAI) Embeddings API.
 */
const axios = require('axios');
const llmConfig = require('../../config/llm');
const { ExternalServiceError } = require('../../utils/errorHandler');

/**
 * Generate an embedding vector for a single text string.
 * @param {string} text - Input text to embed
 * @returns {Promise<number[]>} Embedding vector
 */
async function generateEmbedding(text) {
  const embeddings = await generateEmbeddings([text]);
  return embeddings[0];
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
 * @param {string[]} texts - All texts to embed
 * @param {number} batchSize - Texts per API call (default 20)
 * @returns {Promise<number[][]>}
 */
async function generateEmbeddingsBatched(texts, batchSize = 20) {
  const allEmbeddings = [];

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const embeddings = await generateEmbeddings(batch);
    allEmbeddings.push(...embeddings);

    // Small delay between batches to be polite to the API
    if (i + batchSize < texts.length) {
      await sleep(200);
    }
  }

  return allEmbeddings;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

module.exports = {
  generateEmbedding,
  generateEmbeddings,
  generateEmbeddingsBatched
};
