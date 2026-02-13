"""
LLM Service
Communicates with the Groq Chat Completions API (OpenAI-compatible).

Implements a 3-layer hallucination-reduction strategy:
  1. Strict system prompt constraining model to provided context only
  2. Low temperature (0.1) for deterministic outputs
  3. Citation enforcement — model must reference source chunks
"""
import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.config import llm as llm_config
from app.utils.error_handler import ExternalServiceError

# ─── System Prompts ───────────────────────────────────────

QUERY_SYSTEM_PROMPT = """You are a precise, factual assistant.
RULES — follow ALL of them strictly:
1. Answer ONLY using the provided context chunks. Do NOT use prior knowledge.
2. If the context does not contain enough information, respond with:
   "I cannot answer this question based on the provided documents."
3. For every claim you make, include a citation in the format [Source: <filename>, Chunk <chunkIndex>].
4. Keep answers concise and structured. Use bullet points or numbered lists when appropriate.
5. If multiple chunks support a claim, cite all of them.
6. Never fabricate information. If unsure, say so explicitly.
7. At the END of your response, on a new line, provide a confidence assessment in EXACTLY this format:
   [CONFIDENCE: X/10 | REASON: brief explanation]
   Where X is 1-10 based on how well the context supports your answer:
   - 9-10: Direct, explicit support from multiple sources
   - 7-8: Good support with minor inference
   - 5-6: Partial support, some inference needed
   - 3-4: Weak support, significant inference
   - 1-2: Minimal or no support from context"""

COMPARE_SYSTEM_PROMPT = """You are a precise, analytical document comparison assistant.
RULES — follow ALL of them strictly:
1. Compare ONLY using the provided context chunks from two documents. Do NOT use prior knowledge.
2. Structure your response with these sections:
   - **Similarities**: Points both documents agree on
   - **Differences**: Points where documents diverge or contradict
   - **Unique to Document 1**: Information only in the first document
   - **Unique to Document 2**: Information only in the second document
   - **Summary**: A brief overall comparison conclusion
3. For every point, include a citation in the format [Source: <filename>, Chunk <chunkIndex>].
4. Be objective and factual. Do not add interpretation beyond what the text says.
5. If context is insufficient for comparison, state what is missing."""

COMPARE_STRUCTURED_SYSTEM_PROMPT = """You are a precise, analytical document comparison assistant.
You MUST respond with ONLY valid JSON (no markdown, no code blocks, no explanation before or after).

Compare the provided context chunks from two documents and return this exact JSON structure:

{
  "similarities": [
    {
      "point": "description of what both documents agree on",
      "doc1Evidence": { "quote": "relevant quote", "source": "filename", "chunk": 0 },
      "doc2Evidence": { "quote": "relevant quote", "source": "filename", "chunk": 0 }
    }
  ],
  "differences": [
    {
      "aspect": "the topic/aspect being compared",
      "doc1Position": "what document 1 says",
      "doc2Position": "what document 2 says",
      "doc1Source": { "source": "filename", "chunk": 0 },
      "doc2Source": { "source": "filename", "chunk": 0 }
    }
  ],
  "uniqueToDoc1": [
    { "point": "information only in document 1", "source": { "source": "filename", "chunk": 0 }, "quote": "supporting quote" }
  ],
  "uniqueToDoc2": [
    { "point": "information only in document 2", "source": { "source": "filename", "chunk": 0 }, "quote": "supporting quote" }
  ],
  "summary": {
    "overallAssessment": "brief summary of how the documents relate",
    "agreementLevel": "high|medium|low|none",
    "keyTakeaway": "the most important finding from the comparison"
  },
  "metadata": {
    "doc1ChunksAnalyzed": 0,
    "doc2ChunksAnalyzed": 0,
    "comparisonConfidence": 1
  }
}

RULES:
1. Use ONLY information from the provided context chunks
2. If you cannot find information for a section, use an empty array []
3. Be objective and factual
4. Include actual quotes where possible
5. Respond with ONLY the JSON object, nothing else"""

VERIFICATION_SYSTEM_PROMPT = """You are a precise fact-checker. Verify if an answer is supported by the source context.

OUTPUT FORMAT (respond ONLY with this JSON structure, no markdown):
{
  "isVerified": true,
  "overallScore": 8,
  "claims": [
    {
      "claim": "the specific claim text",
      "status": "supported",
      "evidence": "quote from context",
      "sourceChunk": "filename, Chunk X"
    }
  ],
  "unsupportedClaims": [],
  "summary": "brief verification summary"
}

RULES:
1. Extract each factual claim from the answer
2. Search the context for supporting evidence for each claim
3. Mark as "unsupported" if no evidence found
4. Mark as "partially_supported" if only weak/indirect evidence
5. Be strict — if context doesn't explicitly state something, it's not supported"""


# ─── HTTP client ──────────────────────────────────────────

_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=120.0)
    return _client


# ─── Core Chat Completion ─────────────────────────────────

async def chat_completion(
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Send a chat completion request to Groq with retry logic."""
    temp = temperature if temperature is not None else llm_config.TEMPERATURE
    tokens = max_tokens or llm_config.MAX_TOKENS
    last_error: Optional[Exception] = None

    for attempt in range(1, llm_config.RETRY_ATTEMPTS + 1):
        try:
            client = _get_client()
            resp = await client.post(
                f"{llm_config.API_URL}/chat/completions",
                json={
                    "model": llm_config.MODEL,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": tokens,
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {llm_config.API_KEY}",
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as exc:
            last_error = exc
            status = exc.response.status_code
            if status == 429:
                retry_after = int(exc.response.headers.get("retry-after", llm_config.RETRY_DELAY))
                print(f"[LLM] Rate limit, retrying in {retry_after}s (attempt {attempt})")
                await asyncio.sleep(retry_after)
                continue
            if status >= 500 and attempt < llm_config.RETRY_ATTEMPTS:
                delay = llm_config.RETRY_DELAY * (2 ** (attempt - 1))
                print(f"[LLM] Server error {status}, retrying in {delay}s")
                await asyncio.sleep(delay)
                continue
            break

        except Exception as exc:
            last_error = exc
            if attempt < llm_config.RETRY_ATTEMPTS:
                await asyncio.sleep(llm_config.RETRY_DELAY * (2 ** (attempt - 1)))
                continue
            break

    raise ExternalServiceError("Groq LLM", str(last_error))


# ─── High-level Helpers ───────────────────────────────────

def _format_context(chunks: List[Dict], label: Optional[str] = None) -> str:
    parts = []
    for idx, chunk in enumerate(chunks):
        fn = chunk.get("metadata", {}).get("filename", "unknown")
        ci = chunk.get("metadata", {}).get("chunk_index", idx)
        header = f"[{label} | {fn}, Chunk {ci}]" if label else f"[{fn}, Chunk {ci}]"
        parts.append(f"{header}\n{chunk.get('text', '')}")
    return "\n\n".join(parts)


async def generate_answer(
    query: str,
    context_chunks: List[Dict],
    temperature: Optional[float] = None,
) -> str:
    """Generate an answer using retrieved context."""
    ctx = _format_context(context_chunks)
    messages = [
        {"role": "system", "content": QUERY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{ctx}\n\n---\nQuestion: {query}"},
    ]
    return await chat_completion(messages, temperature=temperature)


async def generate_comparison(
    topic: str,
    doc1_chunks: List[Dict],
    doc2_chunks: List[Dict],
) -> str:
    """Generate a plain-text comparison of two documents."""
    ctx1 = _format_context(doc1_chunks, "Document 1")
    ctx2 = _format_context(doc2_chunks, "Document 2")
    messages = [
        {"role": "system", "content": COMPARE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Document 1 Context:\n{ctx1}\n\n---\n"
            f"Document 2 Context:\n{ctx2}\n\n---\n"
            f"Comparison Topic: {topic}"
        )},
    ]
    return await chat_completion(messages, max_tokens=2048)


async def generate_structured_comparison(
    topic: str,
    doc1_chunks: List[Dict],
    doc2_chunks: List[Dict],
) -> Dict[str, Any]:
    """Generate a JSON-structured comparison."""
    ctx1 = _format_context(doc1_chunks, "Document 1")
    ctx2 = _format_context(doc2_chunks, "Document 2")
    messages = [
        {"role": "system", "content": COMPARE_STRUCTURED_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Document 1 Context:\n{ctx1}\n\n---\n"
            f"Document 2 Context:\n{ctx2}\n\n---\n"
            f"Comparison Topic: {topic}"
        )},
    ]
    response = await chat_completion(messages, temperature=0.1, max_tokens=2500)

    try:
        cleaned = re.sub(r"```json\n?|\n?```", "", response).strip()
        parsed = json.loads(cleaned)
        parsed.setdefault("metadata", {})
        parsed["metadata"].update({
            "doc1ChunksAnalyzed": len(doc1_chunks),
            "doc2ChunksAnalyzed": len(doc2_chunks),
            "topic": topic,
        })
        return parsed
    except (json.JSONDecodeError, ValueError):
        return {
            "similarities": [], "differences": [],
            "uniqueToDoc1": [], "uniqueToDoc2": [],
            "summary": {
                "overallAssessment": "Comparison generated but JSON parsing failed",
                "agreementLevel": "unknown",
                "keyTakeaway": "See rawComparison field",
            },
            "metadata": {
                "doc1ChunksAnalyzed": len(doc1_chunks),
                "doc2ChunksAnalyzed": len(doc2_chunks),
                "topic": topic, "parseError": True,
            },
            "rawComparison": response,
        }


async def verify_answer(answer: str, context_chunks: List[Dict]) -> Dict:
    """Verify an answer against source chunks."""
    ctx = _format_context(context_chunks)
    messages = [
        {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"SOURCE CONTEXT:\n{ctx}\n\n---\n"
            f"ANSWER TO VERIFY:\n{answer}\n\n---\n"
            "Verify this answer against the source context."
        )},
    ]
    response = await chat_completion(messages, temperature=0.1, max_tokens=1500)
    try:
        cleaned = re.sub(r"```json\n?|\n?```", "", response).strip()
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return {
            "isVerified": False, "overallScore": 0,
            "claims": [], "unsupportedClaims": [],
            "summary": "Verification parsing failed",
            "rawResponse": response,
        }


# ─── Confidence Parsing ───────────────────────────────────

def parse_confidence(response: str) -> Tuple[str, Dict]:
    """
    Extract [CONFIDENCE: X/10 | REASON: ...] from the LLM response.
    Returns (cleaned_answer, confidence_dict).
    """
    pattern = r"\[CONFIDENCE:\s*(\d+)/10\s*\|\s*REASON:\s*(.+?)\]\s*$"
    match = re.search(pattern, response, re.IGNORECASE)

    if match:
        answer = re.sub(pattern, "", response, flags=re.IGNORECASE).strip()
        score = int(match.group(1))
        reason = match.group(2).strip()
        if score >= 9:
            level = "very_high"
        elif score >= 7:
            level = "high"
        elif score >= 5:
            level = "medium"
        elif score >= 3:
            level = "low"
        else:
            level = "very_low"
        return answer, {"score": score, "reason": reason, "level": level}

    return response, {"score": None, "reason": "Confidence not provided", "level": "unknown"}
