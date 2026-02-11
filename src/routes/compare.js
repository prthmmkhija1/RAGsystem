/**
 * Compare Routes
 * POST /compare – Compare two documents on a given topic
 */
const express = require('express');
const compareController = require('../controllers/compareController');
const { asyncHandler } = require('../utils/errorHandler');
const { validate, compareSchema } = require('../utils/validators');

const router = express.Router();

// Compare two documents
router.post('/compare', validate(compareSchema), asyncHandler(compareController.compareDocuments));

module.exports = router;
