/**
 * Query Routes
 * POST /query – Ask a question against the ingested documents
 */
const express = require('express');
const queryController = require('../controllers/queryController');
const { asyncHandler } = require('../utils/errorHandler');
const { validate, querySchema } = require('../utils/validators');

const router = express.Router();

// Query the RAG pipeline
router.post('/query', validate(querySchema), asyncHandler(queryController.queryDocuments));

module.exports = router;
