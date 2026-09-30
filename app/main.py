"""FastAPI entrypoint.

`/healthz` (Phase 1) and `POST /ask` (Phase 5). The web UI arrives in Phase 6.

Memory is a hard constraint on the deployment target (Render free tier = 512 MB,
docs/architecture.md 9.1), so resident memory is reported from the first commit
(implementation.md Phase 1, check 1.2) and every later phase is measured against it.
"""

from __future__ import annotations

import platform

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .api import AskRequest, ask
from .config import get_settings
from .meminfo import current_rss_mb, memory_mb, peak_rss_mb

app = FastAPI(
    title="Mutual Fund Facts-Only FAQ Assistant",
    version="0.5.0",
    description=(
        "Facts-only Q&A over official HDFC Mutual Fund, SEBI and AMFI pages. "
        "Not investment advice."
    ),
)


@app.get("/healthz")
def healthz() -> dict:
    """Liveness/readiness probe. Also surfaces config and memory for debugging.

    A missing API key or a missing index is reported here rather than crashing:
    neither is a reason for the process to be down.
    """
    settings = get_settings()
    return {
        "status": "ok",
        "phase": "5 (retrieval + LLM answer) -- /ask is live, UI lands in Phase 6",
        "python": platform.python_version(),
        "memory_mb": memory_mb(),
        "current_rss_mb": current_rss_mb(),
        "peak_rss_mb": peak_rss_mb(),
        "groq_configured": settings.groq_configured,
        "index_present": settings.index_present,
        "config": settings.public_dict(),
    }


@app.post("/ask")
def post_ask(payload: AskRequest) -> dict:
    """Answer one question, or explain why not.

    Returns 200 for every *product* outcome, including a refusal and a degraded
    reply: those are the guardrails doing their job, not errors. 4xx/5xx is
    reserved for a malformed request or a broken index. See `app/api.py` for
    the full status contract.
    """
    body, status = ask(payload)
    if status >= 400:
        raise HTTPException(status_code=status, detail=body)
    return body


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    """Placeholder landing page. The real UI ships in Phase 6."""
    settings = get_settings()
    ready = settings.groq_configured and settings.index_present
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>MF Facts-Only FAQ Assistant</title></head>
<body>
  <h1>Mutual Fund Facts-Only FAQ Assistant</h1>
  <p><strong>Status:</strong> Phase 5 complete. <code>POST /ask</code> is live;
     the browser interface arrives in Phase 6.</p>
  <p>Try it over HTTP:</p>
  <pre>curl -s localhost:8000/ask -H 'content-type: application/json' \\
  -d '{{"question":"Is there an exit load on HDFC Flexi Cap Fund?","debug":true}}'</pre>
  <p>Or in the terminal, where you can also see the retrieved chunks:
     <code>.venv/bin/python -m app.ask</code></p>
  <p><strong>Groq API key configured:</strong> {settings.groq_configured}</p>
  <p><strong>Vector index present:</strong> {settings.index_present}</p>
  <p><strong>Memory:</strong> {memory_mb()} MB
     (Render free tier budget: 512 MB)</p>
  <p>API docs: <a href="/docs">/docs</a> &middot; Health: <a href="/healthz">/healthz</a></p>
  <hr>
  <p><em>Facts-only. No investment advice.</em></p>
</body>
</html>
"""
