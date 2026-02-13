"""
Document Parser
Extracts raw text from PDF, DOCX, TXT, and Markdown files.
"""
import os
import re
from typing import Union

import PyPDF2
import docx
import markdown

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


async def parse_document(filename: str, content: bytes) -> str:
    """
    Detect file type and extract plain text.

    Args:
        filename: Original filename (used for extension detection).
        content: Raw file bytes.

    Returns:
        Extracted plain text.

    Raises:
        ValueError: Unsupported file type or empty document.
    """
    if not content:
        raise ValueError("Invalid file: no content provided")

    size_kb = len(content) / 1024
    print(f'[Parser] Processing "{filename}" ({size_kb:.1f} KB)')

    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f'Unsupported file type: "{ext}". '
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".pdf":
        text = _parse_pdf(content)
    elif ext == ".docx":
        text = _parse_docx(content)
    elif ext == ".txt":
        text = content.decode("utf-8", errors="replace")
    elif ext in (".md", ".markdown"):
        text = _parse_markdown(content)
    else:
        raise ValueError(f"No parser for: {ext}")

    text = _clean_text(text)
    if not text.strip():
        raise ValueError("Document appears to be empty or could not be parsed")
    return text


# ─── Individual Parsers ───────────────────────────────────

def _parse_pdf(content: bytes) -> str:
    """Extract text from a PDF buffer."""
    import io
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as e:
        raise ValueError(f"PDF parsing failed: {e}")


def _parse_docx(content: bytes) -> str:
    """Extract text from a DOCX buffer."""
    import io
    try:
        doc = docx.Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        raise ValueError(f"DOCX parsing failed: {e}")


def _parse_markdown(content: bytes) -> str:
    """Convert Markdown to plain text (strip tags)."""
    raw = content.decode("utf-8", errors="replace")
    html = markdown.markdown(raw)
    text = re.sub(r"<[^>]+>", " ", html)       # strip HTML tags
    text = re.sub(r"&[a-z]+;", " ", text, flags=re.I)  # strip entities
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove control characters."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
