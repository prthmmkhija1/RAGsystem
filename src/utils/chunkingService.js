/**
 * Chunking Service
 * Splits documents into overlapping chunks with sentence-boundary awareness.
 */

const DEFAULT_CHUNK_SIZE = parseInt(process.env.CHUNK_SIZE, 10) || 1000;   // words
const DEFAULT_CHUNK_OVERLAP = parseInt(process.env.CHUNK_OVERLAP, 10) || 50; // words

/**
 * Split text into overlapping, sentence-aware chunks.
 * @param {string} text - The full document text
 * @param {Object} opts
 * @param {number} opts.chunkSize   - Max words per chunk (default 1000)
 * @param {number} opts.chunkOverlap - Overlap in words (default 50)
 * @returns {string[]} Array of text chunks
 */
function chunkText(text, opts = {}) {
  const chunkSize = opts.chunkSize || DEFAULT_CHUNK_SIZE;
  const chunkOverlap = opts.chunkOverlap || DEFAULT_CHUNK_OVERLAP;

  if (!text || text.trim().length === 0) {
    return [];
  }

  // Split text into sentences first for boundary awareness
  const sentences = splitIntoSentences(text);

  const chunks = [];
  let currentChunk = [];
  let currentWordCount = 0;

  for (const sentence of sentences) {
    const sentenceWords = countWords(sentence);

    // If a single sentence exceeds chunk size, force-split it by words
    if (sentenceWords > chunkSize) {
      // Flush what we have
      if (currentChunk.length > 0) {
        chunks.push(currentChunk.join(' '));
        currentChunk = getOverlapSentences(currentChunk, chunkOverlap);
        currentWordCount = countWords(currentChunk.join(' '));
      }
      // Split the long sentence by words
      const forcedChunks = forceSplitByWords(sentence, chunkSize, chunkOverlap);
      chunks.push(...forcedChunks);
      currentChunk = [];
      currentWordCount = 0;
      continue;
    }

    // If adding this sentence would exceed chunk size, start a new chunk
    if (currentWordCount + sentenceWords > chunkSize && currentChunk.length > 0) {
      chunks.push(currentChunk.join(' '));
      // Keep overlap from the end of the current chunk
      currentChunk = getOverlapSentences(currentChunk, chunkOverlap);
      currentWordCount = countWords(currentChunk.join(' '));
    }

    currentChunk.push(sentence);
    currentWordCount += sentenceWords;
  }

  // Flush remaining
  if (currentChunk.length > 0) {
    const remaining = currentChunk.join(' ').trim();
    if (remaining.length > 0) {
      chunks.push(remaining);
    }
  }

  return chunks;
}

/**
 * Split text into sentences using common delimiters.
 */
function splitIntoSentences(text) {
  // Split on sentence-ending punctuation followed by whitespace
  const raw = text.split(/(?<=[.!?])\s+/);
  return raw.map(s => s.trim()).filter(s => s.length > 0);
}

/**
 * Count words in a string.
 */
function countWords(text) {
  if (!text || text.trim().length === 0) return 0;
  return text.trim().split(/\s+/).length;
}

/**
 * Get sentences from the end of a chunk that fit within the overlap word count.
 */
function getOverlapSentences(sentences, overlapWords) {
  const result = [];
  let wordCount = 0;

  for (let i = sentences.length - 1; i >= 0; i--) {
    const sentWords = countWords(sentences[i]);
    if (wordCount + sentWords > overlapWords) break;
    result.unshift(sentences[i]);
    wordCount += sentWords;
  }

  return result;
}

/**
 * Force-split a very long sentence by word count.
 */
function forceSplitByWords(text, chunkSize, overlap) {
  const words = text.split(/\s+/);
  const chunks = [];
  let start = 0;

  while (start < words.length) {
    const end = Math.min(start + chunkSize, words.length);
    chunks.push(words.slice(start, end).join(' '));
    start = end - overlap;
    if (start >= words.length) break;
    if (end === words.length) break;
  }

  return chunks;
}

/**
 * Create chunk metadata objects for storage.
 * @param {string} documentId - Unique document identifier
 * @param {string} filename - Original filename
 * @param {string[]} chunks - Array of text chunks
 * @returns {Object[]} Array of { chunkId, documentId, filename, chunkIndex, totalChunks, text, wordCount }
 */
function createChunkMetadata(documentId, filename, chunks) {
  return chunks.map((text, index) => ({
    chunkId: `${documentId}_chunk_${index}`,
    documentId,
    filename,
    chunkIndex: index,
    totalChunks: chunks.length,
    text,
    wordCount: countWords(text)
  }));
}

module.exports = {
  chunkText,
  createChunkMetadata,
  countWords,
  splitIntoSentences,
  DEFAULT_CHUNK_SIZE,
  DEFAULT_CHUNK_OVERLAP
};
