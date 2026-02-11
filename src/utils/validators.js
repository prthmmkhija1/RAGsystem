/**
 * Request Validators
 * Joi schemas for validating incoming requests.
 */
const Joi = require('joi');

// ---- /upload validation ----
const uploadSchema = Joi.object({
  // File is validated via Multer; this covers optional body fields
  chunkSize: Joi.number().integer().min(100).max(10000).optional(),
  chunkOverlap: Joi.number().integer().min(0).max(500).optional(),
  metadata: Joi.object().optional()
});

// ---- /query validation ----
const querySchema = Joi.object({
  query: Joi.string().trim().min(1).max(5000).required()
    .messages({ 'string.empty': 'Query text is required' }),
  topK: Joi.number().integer().min(1).max(20).default(5),
  documentId: Joi.string().uuid().optional(),
  includeMetadata: Joi.boolean().default(true),
  temperature: Joi.number().min(0).max(1).optional()
});

// ---- /compare validation ----
const compareSchema = Joi.object({
  documentIds: Joi.array()
    .items(Joi.string().uuid())
    .length(2)
    .required()
    .messages({
      'array.length': 'Exactly two document IDs are required for comparison'
    }),
  topic: Joi.string().trim().min(1).max(2000).required()
    .messages({ 'string.empty': 'Comparison topic is required' }),
  topK: Joi.number().integer().min(1).max(20).default(5)
});

// ---- /documents/:id param validation ----
const documentIdSchema = Joi.object({
  id: Joi.string().uuid().required()
});

/**
 * Express middleware factory: validates req.body against the given schema.
 * @param {Joi.ObjectSchema} schema
 * @returns {Function} Express middleware
 */
function validate(schema) {
  return (req, res, next) => {
    const { error, value } = schema.validate(req.body, {
      abortEarly: false,
      stripUnknown: true,
      convert: true
    });

    if (error) {
      const details = error.details.map(d => ({
        field: d.path.join('.'),
        message: d.message
      }));
      return res.status(400).json({
        success: false,
        error: {
          message: 'Validation failed',
          details
        }
      });
    }

    // Replace body with validated & converted values
    req.body = value;
    next();
  };
}

/**
 * Validate request params.
 */
function validateParams(schema) {
  return (req, res, next) => {
    const { error, value } = schema.validate(req.params, {
      abortEarly: false,
      stripUnknown: true
    });

    if (error) {
      const details = error.details.map(d => ({
        field: d.path.join('.'),
        message: d.message
      }));
      return res.status(400).json({
        success: false,
        error: {
          message: 'Validation failed',
          details
        }
      });
    }

    req.params = value;
    next();
  };
}

module.exports = {
  uploadSchema,
  querySchema,
  compareSchema,
  documentIdSchema,
  validate,
  validateParams
};
