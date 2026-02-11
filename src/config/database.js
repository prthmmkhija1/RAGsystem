/**
 * ChromaDB Configuration
 */
module.exports = {
  chromaHost: process.env.CHROMA_HOST || 'http://localhost:8000',
  collectionName: process.env.CHROMA_COLLECTION || 'rag_documents',
  distanceFunction: 'cosine', // cosine | l2 | ip
  batchSize: 100 // max chunks to insert per batch
};
