require('dotenv').config();
const express = require('express');
const cors = require('cors');

// Route modules
const documentRoutes = require('./src/routes/documents');
const queryRoutes = require('./src/routes/query');
const compareRoutes = require('./src/routes/compare');

// Middleware
const { errorHandler } = require('./src/utils/errorHandler');
const vectorStoreService = require('./src/vectorstore/vectorStoreService');
const cacheService = require('./src/utils/cacheService');

const app = express();
const PORT = process.env.PORT || 3000;

// ── Global Middleware ────────────────────────────────────
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Request logging (simple)
app.use((req, _res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl}`);
  next();
});

// ── Routes ───────────────────────────────────────────────
app.use('/api', documentRoutes);  // POST /api/upload, GET /api/documents, DELETE /api/documents/:id, GET /api/stats
app.use('/api', queryRoutes);     // POST /api/query
app.use('/api', compareRoutes);   // POST /api/compare

// Health check (no /api prefix)
app.get('/health', async (req, res) => {
  try {
    const stats = await vectorStoreService.getStats();
    const cacheStats = cacheService.getStats();
    res.json({
      status: 'ok',
      service: 'RAG System with Grok + ChromaDB',
      uptime: `${(process.uptime() / 60).toFixed(1)} min`,
      vectorStore: {
        totalDocuments: stats.totalDocuments,
        totalChunks: stats.totalChunks
      },
      cache: cacheStats
    });
  } catch {
    res.json({
      status: 'degraded',
      service: 'RAG System with Grok + ChromaDB',
      vectorStore: 'unavailable'
    });
  }
});

// ── Error Handling (must be LAST) ────────────────────────
app.use(errorHandler);

// ── Start Server ─────────────────────────────────────────
async function startServer() {
  try {
    // Initialize ChromaDB before accepting requests
    await vectorStoreService.initialize();
    console.log('ChromaDB connected');
  } catch (err) {
    console.error('ChromaDB connection failed:', err.message);
    console.warn('Server starting anyway — ChromaDB will retry on first request');
  }

  app.listen(PORT, () => {
    console.log(`\nRAG System running on http://localhost:${PORT}`);
    console.log(`  Upload:     POST http://localhost:${PORT}/api/upload`);
    console.log(`  Query:      POST http://localhost:${PORT}/api/query`);
    console.log(`  Compare:    POST http://localhost:${PORT}/api/compare`);
    console.log(`  Documents:  GET  http://localhost:${PORT}/api/documents`);
    console.log(`  Stats:      GET  http://localhost:${PORT}/api/stats`);
    console.log(`  Health:     GET  http://localhost:${PORT}/health\n`);
  });
}

startServer();
