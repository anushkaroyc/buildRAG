"""FastAPI entrypoint.

`/healthz` (Phase 1), `POST /ask` (Phase 5) and the browser UI at `/` (Phase 6).

The UI is server-rendered HTML plus a dependency-free `static/app.js` that calls
`/ask` - no framework, no build step (architecture 6.1). It adds no answering
logic of its own: the guardrails cannot be bypassed by "just calling the model
from the UI", which is the property that makes the Streamlit surface in
`app/ui.py` a display layer rather than a second pipeline.

Memory is a hard constraint on the deployment target (Render free tier = 512 MB,
docs/architecture.md 9.1), so resident memory is reported from the first commit
(implementation.md Phase 1, check 1.2) and every later phase is measured against it.
"""

from __future__ import annotations

import platform
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .api import AskRequest, ask
from .config import get_settings
from .meminfo import current_rss_mb, memory_mb, peak_rss_mb

REPO = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO / "app" / "templates"
STATIC_DIR = REPO / "static"

app = FastAPI(
    title="Mutual Fund Facts-Only FAQ Assistant",
    version="0.6.0",
    description=(
        "Facts-only Q&A over official HDFC Mutual Fund, SEBI and AMFI pages. "
        "Not investment advice."
    ),
)

# Server-rendered HTML rather than a JS framework: architecture 6.1 picks this
# so the Render image carries no Node toolchain. `jinja2` is already a pinned
# dependency. Mounted from the repo root rather than relative to the CWD so the
# page still renders if the service is started from another directory.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


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
def root(request: Request):
    """The browser interface (Phase 6).

    Renders the chat page; `static/app.js` then drives it against `POST /ask`.
    Deliberately server-rendered: the page is useful with no JavaScript at all
    (it explains the API and links `/docs`), and the same URL is what a reviewer
    opens, so it must never be a blank shell that only a working bundle can fill.

    `request` comes first because Starlette 1.7 removed the older
    `(name, {"request": ...})` form. fastapi 0.142 resolves Starlette 1.7, and the
    old signature fails with `TypeError: unhashable type: 'dict'` raised from
    inside Jinja's template cache - an error naming neither the cause nor the
    call site.
    """
    return templates.TemplateResponse(request, "index.html")
