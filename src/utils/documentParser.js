/**
 * Document Parser Service
 * Extracts raw text from PDF, DOCX, TXT, and Markdown files.
 */
const pdfParse = require('pdf-parse');
const mammoth = require('mammoth');
const { marked } = require('marked');
const path = require('path');

// Supported file extensions
const SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.txt', '.md', '.markdown'];

/**
 * Main entry point - detects file type and delegates to the appropriate parser.
 * @param {Object} file - Multer file object { originalname, buffer, mimetype }
 * @returns {Promise<string>} Extracted plain text
 */
async function parseDocument(file) {
  if (!file || !file.buffer) {
    throw new Error('Invalid file: no buffer provided');
  }

  const ext = path.extname(file.originalname).toLowerCase();

  if (!SUPPORTED_EXTENSIONS.includes(ext)) {
    throw new Error(
      `Unsupported file type: "${ext}". Supported formats: ${SUPPORTED_EXTENSIONS.join(', ')}`
    );
  }

  let text;

  switch (ext) {
    case '.pdf':
      text = await parsePDF(file.buffer);
      break;
    case '.docx':
      text = await parseDOCX(file.buffer);
      break;
    case '.txt':
      text = parseTXT(file.buffer);
      break;
    case '.md':
    case '.markdown':
      text = parseMarkdown(file.buffer);
      break;
    default:
      throw new Error(`No parser available for: ${ext}`);
  }

  // Clean up the extracted text
  text = cleanText(text);

  if (!text || text.trim().length === 0) {
    throw new Error('Document appears to be empty or could not be parsed');
  }

  return text;
}

/**
 * Extract text from a PDF buffer.
 */
async function parsePDF(buffer) {
  try {
    const data = await pdfParse(buffer);
    return data.text;
  } catch (err) {
    throw new Error(`PDF parsing failed: ${err.message}`);
  }
}

/**
 * Extract text from a DOCX buffer.
 */
async function parseDOCX(buffer) {
  try {
    const result = await mammoth.extractRawText({ buffer });
    if (result.messages && result.messages.length > 0) {
      console.warn('DOCX parser warnings:', result.messages);
    }
    return result.value;
  } catch (err) {
    throw new Error(`DOCX parsing failed: ${err.message}`);
  }
}

/**
 * Extract text from a plain text buffer.
 */
function parseTXT(buffer) {
  return buffer.toString('utf-8');
}

/**
 * Extract text from a Markdown buffer (strips markdown syntax).
 */
function parseMarkdown(buffer) {
  const raw = buffer.toString('utf-8');
  // Convert markdown to HTML, then strip HTML tags to get plain text
  const html = marked(raw);
  return html
    .replace(/<[^>]*>/g, ' ')   // strip HTML tags
    .replace(/&[a-z]+;/gi, ' ') // strip HTML entities
    .replace(/\s+/g, ' ')       // collapse whitespace
    .trim();
}

/**
 * Clean extracted text: normalize whitespace, remove control characters.
 */
function cleanText(text) {
  return text
    .replace(/\r\n/g, '\n')         // normalize line endings
    .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g, '') // remove control chars (keep \n, \t)
    .replace(/\n{3,}/g, '\n\n')     // collapse excessive newlines
    .replace(/[ \t]+/g, ' ')        // collapse spaces/tabs
    .trim();
}

module.exports = {
  parseDocument,
  parsePDF,
  parseDOCX,
  parseTXT,
  parseMarkdown,
  cleanText,
  SUPPORTED_EXTENSIONS
};
