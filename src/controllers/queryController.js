/**
 * Query Controller
 * Handles user queries using the RAG pipeline.
 */
const ragService = require('../services/rag/ragService');

/**
 * POST /query
 * Accept a user question, retrieve relevant context, and return an LLM-generated answer.
 */
async function queryDocuments(req, res, next) {
  try {
    const { query, topK, documentId, includeMetadata, temperature } = req.body;

    const result = await ragService.queryDocuments(query, {
      topK,
      documentId,
      includeMetadata,
      temperature
    });

    res.json({
      success: true,
      data: result
    });
  } catch (err) {
    next(err);
  }
}

module.exports = {
  queryDocuments
};
