"""Ask the facts-only assistant over HTTP.

POST /ask {"question": "..."} -> the answer payload, or a fixed refusal.

The handler is a thin pipeline, and the order is the design:

    guardrails.evaluate  ->  retrieve  ->  generate.answer  ->  validator

Each stage can end the request, and each end is a *deliberate* outcome rather
than an error path:

- **guard refused** (off-topic, advice, PII, out of scope) -> 200 with the
  fixed refusal payload. 200, not 4xx: the user did nothing wrong, and the
  guardrail *is* the product working. Nothing downstream runs and no tokens are
  spent.
- **retrieval found nothing above MIN_SCORE** -> 200 with "I don't have that in
  my sources". Again no LLM call (ST-1).
- **Groq missing, 429, 5xx or timed out** -> 200 with the degraded reply
  (ST-2). The alternative - surfacing a 500 - would tell the user nothing and
  would be a worse product than saying "the assistant is briefly unavailable".
- **anything unexpected** -> 500 with a generic message. Deliberately vague:
  an exception string can contain a file path or a key fragment.

`?debug=true` adds the retrieved chunk scores and ids, which is what you want
when a real answer looks wrong. It is off by default because the chunk excerpts
are source text, and this endpoint is meant to be callable by anything.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, is_dataclass

from pydantic import BaseModel, Field

from . import generate, guardrails as g
from .retrieve import retrieve

logger = logging.getLogger("app.api")

MAX_QUESTION_CHARS = 500


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        max_length=MAX_QUESTION_CHARS,
        description="The user's question, in their own words.",
    )
    debug: bool = Field(
        False,
        description=(
            "Include retrieval diagnostics (chunk ids, scores, metadata) in the "
            "response. No source text is returned even when true."
        ),
    )


def _plain(value):
    """Recursively convert dataclasses so the payload is JSON-serialisable."""
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def ask(payload: AskRequest) -> tuple[dict, int]:
    """Run the pipeline for one question. Returns (body, http_status)."""
    question = (payload.question or "").strip()
    if not question:
        return {
            "answer": "Please enter a question.",
            "refused": True,
            "intent": "empty",
            "citations": [],
        }, 400

    # --- 1. guardrails ----------------------------------------------------
    verdict = g.evaluate(question)
    if verdict.get("refused"):
        return _plain(verdict), 200

    # --- 2. retrieval -----------------------------------------------------
    started = time.perf_counter()
    try:
        found = retrieve(question, scheme=verdict.get("scheme"))
    except Exception as exc:  # noqa: BLE001 - the index failing is our fault, not the user's
        logger.exception("retrieval failed")
        return {
            "answer": "I could not search the source documents just now. Please try again.",
            "refused": True,
            "intent": "retrieval_error",
            "citations": [],
            "error": type(exc).__name__,
        }, 503

    # --- 3. generation + validation --------------------------------------
    try:
        result = generate.answer(found, question)
    except Exception:  # noqa: BLE001
        # generate.answer already handles every expected failure internally; if
        # something escapes it, degrade rather than propagate.
        logger.exception("generation failed")
        return {
            "answer": "I could not prepare an answer just now. Please try again.",
            "refused": True,
            "intent": "degraded",
            "citations": [],
        }, 503

    body = _plain({k: v for k, v in result.items() if k != "retrieval"})
    body["elapsed_s"] = round(time.perf_counter() - started, 2)

    if payload.debug:
        body["retrieval"] = {
            "n_chunks": len(found.chunks),
            "n_candidates": found.n_candidates,
            "min_score": found.min_score,
            "index_missing": found.index_missing,
            "chunks": [
                {
                    "id": c.id,
                    "score": round(c.score, 4),
                    "rank_score": round(c.rank_score, 4),
                    "lexical": round(c.lexical, 4),
                    "scheme": c.scheme,
                    "page_type": c.page_type,
                    "as_of": c.as_of,
                    "n_tokens": c.metadata.get("n_tokens"),
                    "url": c.url,
                }
                for c in found.chunks
            ],
        }
    return body, 200
