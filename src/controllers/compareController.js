/**
 * Compare Controller
 * Handles document comparison requests.
 */
const ragService = require('../services/rag/ragService');

/**
 * POST /compare
 * Compare two documents on a given topic.
 */
async function compareDocuments(req, res, next) {
  try {
    const { documentIds, topic, topK, structured } = req.body;

    const result = await ragService.compareDocuments(documentIds, topic, { topK, structured });

    res.json({
      success: true,
      data: result
    });
  } catch (err) {
    next(err);
  }
}

module.exports = {
  compareDocuments
};
