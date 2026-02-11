/**
 * Document Controller
 * Handles file upload / ingestion, listing, and deletion.
 */
const ragService = require('../services/rag/ragService');
const { ValidationError, NotFoundError } = require('../utils/errorHandler');
const { SUPPORTED_EXTENSIONS } = require('../utils/documentParser');
const path = require('path');

/**
 * POST /upload
 * Upload and ingest a document.
 */
async function uploadDocument(req, res, next) {
  try {
    // Validate that a file was provided
    if (!req.file) {
      throw new ValidationError('No file uploaded. Please attach a file.', {
        supportedFormats: SUPPORTED_EXTENSIONS
      });
    }

    // Validate file extension
    const ext = path.extname(req.file.originalname).toLowerCase();
    if (!SUPPORTED_EXTENSIONS.includes(ext)) {
      throw new ValidationError(
        `Unsupported file type: "${ext}"`,
        { supportedFormats: SUPPORTED_EXTENSIONS }
      );
    }

    // Optional body params
    const opts = {};
    if (req.body.chunkSize) opts.chunkSize = parseInt(req.body.chunkSize, 10);
    if (req.body.chunkOverlap) opts.chunkOverlap = parseInt(req.body.chunkOverlap, 10);

    // Run ingestion pipeline
    const result = await ragService.ingestDocument(req.file, opts);

    res.status(201).json({
      success: true,
      data: result
    });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /documents
 * List all ingested documents.
 */
async function listDocuments(req, res, next) {
  try {
    const documents = await ragService.listDocuments();

    res.json({
      success: true,
      data: {
        count: documents.length,
        documents
      }
    });
  } catch (err) {
    next(err);
  }
}

/**
 * DELETE /documents/:id
 * Delete a document and all its chunks.
 */
async function deleteDocument(req, res, next) {
  try {
    const { id } = req.params;
    const result = await ragService.deleteDocument(id);

    res.json({
      success: true,
      data: result
    });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /stats
 * Get vector store statistics.
 */
async function getStats(req, res, next) {
  try {
    const stats = await ragService.getStats();

    res.json({
      success: true,
      data: stats
    });
  } catch (err) {
    next(err);
  }
}

module.exports = {
  uploadDocument,
  listDocuments,
  deleteDocument,
  getStats
};
