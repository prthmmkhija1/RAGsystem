/**
 * Document Routes
 * POST   /upload        – Upload & ingest a document
 * GET    /documents     – List all documents
 * DELETE /documents/:id – Delete a document
 * GET    /stats         – Vector store statistics
 */
const express = require('express');
const multer = require('multer');
const documentController = require('../controllers/documentController');
const { asyncHandler } = require('../utils/errorHandler');
const { validateParams, documentIdSchema } = require('../utils/validators');

const router = express.Router();

// Multer config: memory storage, 50 MB limit
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 } // 50 MB
});

// Upload a document
router.post('/upload', upload.single('document'), asyncHandler(documentController.uploadDocument));

// List all documents
router.get('/documents', asyncHandler(documentController.listDocuments));

// Delete a document
router.delete(
  '/documents/:id',
  validateParams(documentIdSchema),
  asyncHandler(documentController.deleteDocument)
);

// Vector store stats
router.get('/stats', asyncHandler(documentController.getStats));

module.exports = router;
