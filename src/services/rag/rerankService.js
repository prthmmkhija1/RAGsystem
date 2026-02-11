/**
 * Re-ranking Service
 * Uses a cross-encoder model to re-rank retrieved chunks for better relevance.
 * 
 * Cross-encoders score query-document pairs directly, providing more accurate
 * relevance scores than bi-encoder (embedding) similarity.
 */
const { pipeline, env } = require('@xenova/transformers');

// Configure transformers.js to use local cache
env.cacheDir = './.cache/transformers';
env.allowRemoteModels = true;

let rerankPipeline = null;
let isLoading = false;
let loadPromise = null;

// Default model - a lightweight cross-encoder trained on MS MARCO
const DEFAULT_MODEL = 'Xenova/ms-marco-MiniLM-L-6-v2';

/**
 * Initialize the cross-encoder pipeline (lazy loading).
 * @returns {Promise<Function>}
 */
async function getReranker() {
  if (rerankPipeline) return rerankPipeline;
  
  if (isLoading) {
    // Wait for the existing load to complete
    return loadPromise;
  }

  isLoading = true;
  console.log('[Reranker] Loading cross-encoder model (first run may take a minute)...');
  
  loadPromise = pipeline('text-classification', DEFAULT_MODEL, {
    quantized: true // Use quantized model for faster inference
  });

  try {
    rerankPipeline = await loadPromise;
    console.log('[Reranker] Model loaded successfully');
    return rerankPipeline;
  } catch (err) {
    isLoading = false;
    loadPromise = null;
    throw new Error(`Failed to load reranker model: ${err.message}`);
  }
}

/**
 * Re-rank chunks using cross-encoder scores.
 * @param {string} query - The search query
 * @param {Object[]} chunks - Array of { text, metadata, similarityScore, ... }
 * @param {Object} opts - { topN: number of results to return (default: all) }
 * @returns {Promise<Object[]>} Re-ranked chunks with crossEncoderScore added
 */
async function rerankChunks(query, chunks, opts = {}) {
  if (!chunks || chunks.length === 0) {
    return [];
  }

  // If only 1 chunk, no need to re-rank
  if (chunks.length === 1) {
    return chunks.map(c => ({ ...c, crossEncoderScore: 1.0 }));
  }

  const reranker = await getReranker();
  const startTime = Date.now();

  // Build query-document pairs for the cross-encoder
  const pairs = chunks.map(chunk => ({
    text: `${query} [SEP] ${chunk.text.substring(0, 512)}` // Truncate to model max length
  }));

  // Score all pairs
  const scores = [];
  for (let i = 0; i < pairs.length; i++) {
    try {
      const result = await reranker(pairs[i].text);
      // The model outputs a relevance score
      const score = result[0]?.score || 0;
      scores.push({ index: i, score });
    } catch (err) {
      console.warn(`[Reranker] Failed to score chunk ${i}: ${err.message}`);
      scores.push({ index: i, score: 0 });
    }
  }

  // Sort by cross-encoder score (descending)
  scores.sort((a, b) => b.score - a.score);

  // Rebuild chunks array with cross-encoder scores
  const rerankedChunks = scores.map(s => ({
    ...chunks[s.index],
    crossEncoderScore: parseFloat(s.score.toFixed(4)),
    originalRank: s.index + 1
  }));

  // Optionally limit to topN
  const topN = opts.topN || rerankedChunks.length;
  const result = rerankedChunks.slice(0, topN);

  const elapsed = Date.now() - startTime;
  console.log(`[Reranker] Re-ranked ${chunks.length} chunks in ${elapsed}ms`);

  return result;
}

/**
 * Calculate a combined score from embeddings and cross-encoder.
 * @param {number} embeddingScore - Similarity from vector search (0-1)
 * @param {number} crossEncoderScore - Score from cross-encoder (typically 0-1)
 * @param {number} embeddingWeight - Weight for embedding score (default 0.3)
 * @returns {number} Combined score
 */
function combineScores(embeddingScore, crossEncoderScore, embeddingWeight = 0.3) {
  const ceWeight = 1 - embeddingWeight;
  return (embeddingScore * embeddingWeight) + (crossEncoderScore * ceWeight);
}

/**
 * Check if the reranker is ready (model loaded).
 */
function isReady() {
  return rerankPipeline !== null;
}

/**
 * Preload the model (useful at startup).
 */
async function preload() {
  await getReranker();
}

module.exports = {
  rerankChunks,
  combineScores,
  isReady,
  preload
};
