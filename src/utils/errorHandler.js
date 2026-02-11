/**
 * Centralized Error Handling
 * Custom error classes + Express error-handling middleware.
 */

// ------- Custom Error Classes -------

class AppError extends Error {
  constructor(message, statusCode = 500, details = null) {
    super(message);
    this.name = this.constructor.name;
    this.statusCode = statusCode;
    this.details = details;
    this.isOperational = true; // distinguishes expected errors from bugs
    Error.captureStackTrace(this, this.constructor);
  }
}

class ValidationError extends AppError {
  constructor(message, details = null) {
    super(message, 400, details);
  }
}

class NotFoundError extends AppError {
  constructor(resource = 'Resource') {
    super(`${resource} not found`, 404);
  }
}

class ExternalServiceError extends AppError {
  constructor(service, message) {
    super(`${service} error: ${message}`, 502);
  }
}

class RateLimitError extends AppError {
  constructor(retryAfter = 60) {
    super('Rate limit exceeded, please try again later', 429);
    this.retryAfter = retryAfter;
  }
}

// ------- Express Error-Handling Middleware -------

/**
 * Global error handler — must be registered LAST in the middleware chain.
 * Signature: (err, req, res, next)
 */
function errorHandler(err, req, res, _next) {
  // Log every error with timestamp
  const timestamp = new Date().toISOString();
  console.error(`[${timestamp}] ERROR:`, {
    name: err.name,
    message: err.message,
    statusCode: err.statusCode || 500,
    path: req.originalUrl,
    method: req.method,
    ...(process.env.NODE_ENV !== 'production' && { stack: err.stack })
  });

  // Operational errors → send structured response
  if (err.isOperational) {
    return res.status(err.statusCode).json({
      success: false,
      error: {
        message: err.message,
        ...(err.details && { details: err.details }),
        ...(err.retryAfter && { retryAfter: err.retryAfter })
      }
    });
  }

  // Multer file-size error
  if (err.code === 'LIMIT_FILE_SIZE') {
    return res.status(413).json({
      success: false,
      error: { message: 'File size exceeds the 50 MB limit' }
    });
  }

  // Joi validation error (if thrown directly)
  if (err.isJoi || err.name === 'ValidationError') {
    return res.status(400).json({
      success: false,
      error: {
        message: 'Validation failed',
        details: err.details || err.message
      }
    });
  }

  // Unknown / programming errors → generic 500
  res.status(500).json({
    success: false,
    error: {
      message:
        process.env.NODE_ENV === 'production'
          ? 'Internal server error'
          : err.message
    }
  });
}

/**
 * Wrap an async route handler so thrown errors reach the error middleware.
 * Usage: router.post('/upload', asyncHandler(controller.upload));
 */
function asyncHandler(fn) {
  return (req, res, next) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
}

module.exports = {
  AppError,
  ValidationError,
  NotFoundError,
  ExternalServiceError,
  RateLimitError,
  errorHandler,
  asyncHandler
};
