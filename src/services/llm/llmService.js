/**
 * LLM Service
 * Communicates with the Grok (xAI) Chat Completions API.
 * Implements the 3-layer hallucination-reduction strategy:
 *   1. Strict system prompt constraining the model to provided context
 *   2. Low temperature (0.1) for deterministic outputs
 *   3. Citation enforcement — model must reference source chunks
 */
const axios = require('axios');
const llmConfig = require('../../config/llm');
const { ExternalServiceError } = require('../../utils/errorHandler');

// ──────────────────────────────────────────
// System prompts
// ──────────────────────────────────────────

const QUERY_SYSTEM_PROMPT = `You are a precise, factual assistant.
RULES — follow ALL of them strictly:
1. Answer ONLY using the provided context chunks. Do NOT use prior knowledge.
2. If the context does not contain enough information, respond with:
   "I cannot answer this question based on the provided documents."
3. For every claim you make, include a citation in the format [Source: <filename>, Chunk <chunkIndex>].
4. Keep answers concise and structured. Use bullet points or numbered lists when appropriate.
5. If multiple chunks support a claim, cite all of them.
6. Never fabricate information. If unsure, say so explicitly.`;

const COMPARE_SYSTEM_PROMPT = `You are a precise, analytical document comparison assistant.
RULES — follow ALL of them strictly:
1. Compare ONLY using the provided context chunks from two documents. Do NOT use prior knowledge.
2. Structure your response with these sections:
   - **Similarities**: Points both documents agree on
   - **Differences**: Points where documents diverge or contradict
   - **Unique to Document 1**: Information only in the first document
   - **Unique to Document 2**: Information only in the second document
   - **Summary**: A brief overall comparison conclusion
3. For every point, include a citation in the format [Source: <filename>, Chunk <chunkIndex>].
4. Be objective and factual. Do not add interpretation beyond what the text says.
5. If context is insufficient for comparison, state what is missing.`;

// ──────────────────────────────────────────
// Core API call with retry
// ──────────────────────────────────────────

/**
 * Send a chat completion request to Grok.
 * @param {Object[]} messages - Array of { role, content }
 * @param {Object} opts - Override options (temperature, maxTokens, etc.)
 * @returns {Promise<string>} The assistant's response text
 */
async function chatCompletion(messages, opts = {}) {
  const temperature = opts.temperature ?? llmConfig.temperature;
  const maxTokens = opts.maxTokens ?? llmConfig.maxTokens;
  let lastError;

  for (let attempt = 1; attempt <= llmConfig.retryAttempts; attempt++) {
    try {
      const response = await axios.post(
        `${llmConfig.apiUrl}/chat/completions`,
        {
          model: llmConfig.model,
          messages,
          temperature,
          max_tokens: maxTokens
        },
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${llmConfig.apiKey}`
          },
          timeout: 120000 // 2 min for longer completions
        }
      );

      return response.data.choices[0].message.content;
    } catch (err) {
      lastError = err;
      const status = err.response?.status;

      if (status === 429) {
        const retryAfter =
          parseInt(err.response.headers['retry-after'], 10) || llmConfig.retryDelay / 1000;
        console.warn(`Grok rate limit, retrying in ${retryAfter}s (attempt ${attempt}/${llmConfig.retryAttempts})`);
        await sleep(retryAfter * 1000);
        continue;
      }

      if (status >= 500 && attempt < llmConfig.retryAttempts) {
        const delay = llmConfig.retryDelay * Math.pow(2, attempt - 1);
        console.warn(`Grok server error ${status}, retrying in ${delay}ms`);
        await sleep(delay);
        continue;
      }

      break;
    }
  }

  const msg = lastError.response?.data?.error?.message || lastError.message;
  throw new ExternalServiceError('Grok LLM', msg);
}

// ──────────────────────────────────────────
// High-level helpers
// ──────────────────────────────────────────

/**
 * Generate an answer to a user query using retrieved context chunks.
 * @param {string} query - The user's question
 * @param {Object[]} contextChunks - Array of { text, metadata }
 * @param {Object} opts
 * @returns {Promise<string>} The answer with citations
 */
async function generateAnswer(query, contextChunks, opts = {}) {
  const contextBlock = formatContextChunks(contextChunks);

  const messages = [
    { role: 'system', content: QUERY_SYSTEM_PROMPT },
    {
      role: 'user',
      content: `Context:\n${contextBlock}\n\n---\nQuestion: ${query}`
    }
  ];

  return chatCompletion(messages, opts);
}

/**
 * Generate a structured comparison of two documents on a given topic.
 * @param {string} topic - What to compare the documents on
 * @param {Object[]} doc1Chunks - Chunks from document 1 (with metadata)
 * @param {Object[]} doc2Chunks - Chunks from document 2 (with metadata)
 * @param {Object} opts
 * @returns {Promise<string>} Structured comparison text
 */
async function generateComparison(topic, doc1Chunks, doc2Chunks, opts = {}) {
  const doc1Context = formatContextChunks(doc1Chunks, 'Document 1');
  const doc2Context = formatContextChunks(doc2Chunks, 'Document 2');

  const messages = [
    { role: 'system', content: COMPARE_SYSTEM_PROMPT },
    {
      role: 'user',
      content: `Document 1 Context:\n${doc1Context}\n\n---\nDocument 2 Context:\n${doc2Context}\n\n---\nComparison Topic: ${topic}`
    }
  ];

  // Allow slightly more tokens for comparison responses
  return chatCompletion(messages, { maxTokens: opts.maxTokens || 2048, ...opts });
}

// ──────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────

/**
 * Format context chunks into a numbered text block for the prompt.
 */
function formatContextChunks(chunks, label = null) {
  return chunks
    .map((chunk, idx) => {
      const filename = chunk.metadata?.filename || 'unknown';
      const chunkIndex = chunk.metadata?.chunkIndex ?? idx;
      const header = label
        ? `[${label} | ${filename}, Chunk ${chunkIndex}]`
        : `[${filename}, Chunk ${chunkIndex}]`;
      return `${header}\n${chunk.text}`;
    })
    .join('\n\n');
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

module.exports = {
  chatCompletion,
  generateAnswer,
  generateComparison,
  formatContextChunks,
  QUERY_SYSTEM_PROMPT,
  COMPARE_SYSTEM_PROMPT
};
