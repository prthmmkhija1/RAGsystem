/**
 * Grok (xAI) LLM Configuration
 */
module.exports = {
  apiKey: process.env.GROK_API_KEY,
  apiUrl: process.env.GROK_API_URL || 'https://api.x.ai/v1',
  model: process.env.GROK_MODEL || 'grok-3-mini-fast',
  embeddingModel: process.env.GROK_EMBEDDING_MODEL || 'grok-embedding-public',
  maxTokens: parseInt(process.env.MAX_TOKENS, 10) || 1024,
  temperature: parseFloat(process.env.TEMPERATURE) || 0.1, // low = less hallucination
  retryAttempts: 3,
  retryDelay: 1000 // ms between retries
};
