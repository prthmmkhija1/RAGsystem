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
6. Never fabricate information. If unsure, say so explicitly.
7. At the END of your response, on a new line, provide a confidence assessment in EXACTLY this format:
   [CONFIDENCE: X/10 | REASON: brief explanation]
   Where X is 1-10 based on how well the context supports your answer:
   - 9-10: Direct, explicit support from multiple sources
   - 7-8: Good support with minor inference
   - 5-6: Partial support, some inference needed
   - 3-4: Weak support, significant inference
   - 1-2: Minimal or no support from context`;

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

const COMPARE_STRUCTURED_SYSTEM_PROMPT = `You are a precise, analytical document comparison assistant.
You MUST respond with ONLY valid JSON (no markdown, no code blocks, no explanation before or after).

Compare the provided context chunks from two documents and return this exact JSON structure:

{
  "similarities": [
    {
      "point": "description of what both documents agree on",
      "doc1Evidence": { "quote": "relevant quote", "source": "filename", "chunk": 0 },
      "doc2Evidence": { "quote": "relevant quote", "source": "filename", "chunk": 0 }
    }
  ],
  "differences": [
    {
      "aspect": "the topic/aspect being compared",
      "doc1Position": "what document 1 says",
      "doc2Position": "what document 2 says",
      "doc1Source": { "source": "filename", "chunk": 0 },
      "doc2Source": { "source": "filename", "chunk": 0 }
    }
  ],
  "uniqueToDoc1": [
    {
      "point": "information only in document 1",
      "source": { "source": "filename", "chunk": 0 },
      "quote": "supporting quote"
    }
  ],
  "uniqueToDoc2": [
    {
      "point": "information only in document 2",
      "source": { "source": "filename", "chunk": 0 },
      "quote": "supporting quote"
    }
  ],
  "summary": {
    "overallAssessment": "brief summary of how the documents relate",
    "agreementLevel": "high|medium|low|none",
    "keyTakeaway": "the most important finding from the comparison"
  },
  "metadata": {
    "doc1ChunksAnalyzed": 0,
    "doc2ChunksAnalyzed": 0,
    "comparisonConfidence": 1-10
  }
}

RULES:
1. Use ONLY information from the provided context chunks
2. If you cannot find information for a section, use an empty array []
3. Be objective and factual
4. Include actual quotes where possible
5. Respond with ONLY the JSON object, nothing else`;

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

/**
 * Generate a structured JSON comparison of two documents on a given topic.
 * @param {string} topic - What to compare the documents on
 * @param {Object[]} doc1Chunks - Chunks from document 1 (with metadata)
 * @param {Object[]} doc2Chunks - Chunks from document 2 (with metadata)
 * @param {Object} opts
 * @returns {Promise<Object>} Structured comparison object
 */
async function generateStructuredComparison(topic, doc1Chunks, doc2Chunks, opts = {}) {
  const doc1Context = formatContextChunks(doc1Chunks, 'Document 1');
  const doc2Context = formatContextChunks(doc2Chunks, 'Document 2');

  const messages = [
    { role: 'system', content: COMPARE_STRUCTURED_SYSTEM_PROMPT },
    {
      role: 'user',
      content: `Document 1 Context:\n${doc1Context}\n\n---\nDocument 2 Context:\n${doc2Context}\n\n---\nComparison Topic: ${topic}`
    }
  ];

  const response = await chatCompletion(messages, { 
    maxTokens: opts.maxTokens || 2500,
    temperature: 0.1, // Low temperature for consistent JSON
    ...opts 
  });

  try {
    // Try to parse as JSON
    const cleaned = response.replace(/```json\n?|\n?```/g, '').trim();
    const parsed = JSON.parse(cleaned);
    
    // Add metadata about the comparison
    return {
      ...parsed,
      metadata: {
        ...parsed.metadata,
        doc1ChunksAnalyzed: doc1Chunks.length,
        doc2ChunksAnalyzed: doc2Chunks.length,
        topic
      }
    };
  } catch (err) {
    // If parsing fails, return a structured error with the raw response
    console.warn('[LLM] Failed to parse structured comparison JSON:', err.message);
    return {
      similarities: [],
      differences: [],
      uniqueToDoc1: [],
      uniqueToDoc2: [],
      summary: {
        overallAssessment: 'Comparison generated but could not be parsed as structured JSON',
        agreementLevel: 'unknown',
        keyTakeaway: 'Please see rawComparison field for the text response'
      },
      metadata: {
        doc1ChunksAnalyzed: doc1Chunks.length,
        doc2ChunksAnalyzed: doc2Chunks.length,
        topic,
        comparisonConfidence: 0,
        parseError: true
      },
      rawComparison: response
    };
  }
}

// ──────────────────────────────────────────
// Answer Verification
// ──────────────────────────────────────────

const VERIFICATION_SYSTEM_PROMPT = `You are a precise fact-checker. Your job is to verify if an answer is supported by the provided source context.

TASK: Analyze the answer and determine which claims are supported by the context.

OUTPUT FORMAT (respond ONLY with this JSON structure, no markdown):
{
  "isVerified": true/false,
  "overallScore": 1-10,
  "claims": [
    {
      "claim": "the specific claim text",
      "status": "supported" | "partially_supported" | "unsupported" | "unverifiable",
      "evidence": "quote or reference from context that supports/refutes this",
      "sourceChunk": "filename, Chunk X" or null
    }
  ],
  "unsupportedClaims": ["list of claims not found in context"],
  "summary": "brief verification summary"
}

RULES:
1. Extract each factual claim from the answer
2. For each claim, search the context for supporting evidence
3. Mark as "unsupported" if no evidence found in context
4. Mark as "partially_supported" if only weak/indirect evidence
5. Be strict — if context doesn't explicitly state something, it's not supported
6. Ignore citations in the answer — verify the actual content`;

/**
 * Verify if an answer is supported by the source chunks.
 * @param {string} answer - The generated answer to verify
 * @param {Object[]} contextChunks - The source chunks used to generate the answer
 * @returns {Promise<Object>} Verification result
 */
async function verifyAnswer(answer, contextChunks) {
  const contextBlock = formatContextChunks(contextChunks);

  const messages = [
    { role: 'system', content: VERIFICATION_SYSTEM_PROMPT },
    {
      role: 'user',
      content: `SOURCE CONTEXT:\n${contextBlock}\n\n---\n\nANSWER TO VERIFY:\n${answer}\n\n---\nVerify this answer against the source context.`
    }
  ];

  const response = await chatCompletion(messages, { temperature: 0.1, maxTokens: 1500 });

  try {
    // Try to parse as JSON
    const cleaned = response.replace(/```json\n?|\n?```/g, '').trim();
    return JSON.parse(cleaned);
  } catch {
    // If parsing fails, return a structured error
    return {
      isVerified: false,
      overallScore: 0,
      claims: [],
      unsupportedClaims: [],
      summary: 'Verification parsing failed',
      rawResponse: response
    };
  }
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

/**
 * Parse confidence score from LLM response.
 * Expected format: [CONFIDENCE: X/10 | REASON: explanation]
 * @param {string} response - Full LLM response
 * @returns {{ answer: string, confidence: { score: number, reason: string, level: string } }}
 */
function parseConfidence(response) {
  const confidenceRegex = /\[CONFIDENCE:\s*(\d+)\/10\s*\|\s*REASON:\s*(.+?)\]$/i;
  const match = response.match(confidenceRegex);

  let answer = response;
  let confidence = {
    score: null,
    reason: 'Confidence not provided',
    level: 'unknown'
  };

  if (match) {
    // Remove the confidence line from the answer
    answer = response.replace(confidenceRegex, '').trim();
    const score = parseInt(match[1], 10);
    const reason = match[2].trim();

    // Determine confidence level
    let level;
    if (score >= 9) level = 'very_high';
    else if (score >= 7) level = 'high';
    else if (score >= 5) level = 'medium';
    else if (score >= 3) level = 'low';
    else level = 'very_low';

    confidence = { score, reason, level };
  }

  return { answer, confidence };
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

module.exports = {
  chatCompletion,
  generateAnswer,
  generateComparison,
  generateStructuredComparison,
  verifyAnswer,
  formatContextChunks,
  parseConfidence,
  QUERY_SYSTEM_PROMPT,
  COMPARE_SYSTEM_PROMPT,
  COMPARE_STRUCTURED_SYSTEM_PROMPT,
  VERIFICATION_SYSTEM_PROMPT
};
