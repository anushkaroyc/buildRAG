"""FastAPI entrypoint.

Phase 1 scope: boot and health reporting only. `/ask` and the UI land in Phases 5-6.

Memory is a hard constraint on the deployment target (Render free tier = 512 MB,
docs/architecture.md 9.1), so resident memory is reported from the first commit
(implementation.md Phase 1, check 1.2) and every later phase is measured against it.
"""

from __future__ import annotations

import platform
import resource
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .config import get_settings

app = FastAPI(
    title="Mutual Fund Facts-Only FAQ Assistant",
    version="0.1.0",
    description=(
        "Facts-only Q&A over official HDFC Mutual Fund, SEBI and AMFI pages. "
        "Not investment advice."
    ),
)


def current_rss_mb() -> float | None:
    """Current resident set size in MB.

    Reads /proc, which exists on Linux (including the Render container) but not on
    macOS. Returns None where unavailable -- callers must handle that.
    """
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return round(int(line.split()[1]) / 1024, 1)
    return None


def peak_rss_mb() -> float:
    """Peak resident set size in MB.

    `ru_maxrss` is reported in bytes on macOS and kilobytes on Linux.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    return round(peak / divisor, 1)


def memory_mb() -> float:
    """Best available memory figure, preferring current RSS over peak."""
    current = current_rss_mb()
    return current if current is not None else peak_rss_mb()


@app.get("/healthz")
def healthz() -> dict:
    """Liveness/readiness probe. Also surfaces config and memory for debugging.

    A missing API key or a missing index is reported here rather than crashing:
    neither is a reason for the process to be down.
    """
    settings = get_settings()
    return {
        "status": "ok",
        "phase": "1 (setup) -- no /ask route yet",
        "python": platform.python_version(),
        "memory_mb": memory_mb(),
        "current_rss_mb": current_rss_mb(),
        "peak_rss_mb": peak_rss_mb(),
        "config": settings.public_dict(),
    }


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
  <p><strong>Status:</strong> Phase 1 (project setup) complete.
     The question-answering interface arrives in Phase 6.</p>
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
