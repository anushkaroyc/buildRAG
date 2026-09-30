"""Fetch allowlisted sources to `data/raw/`, content-hashed and idempotent.

Design notes
------------
- **The allowlist is enforced here, in code.** A URL whose host is not in
  `ALLOWED_DOMAINS` raises before any socket is opened. This is gate 2.2, so it
  must be impossible to bypass by editing `sources.csv` alone.
- **Idempotency is by content hash.** Each raw file gets a sidecar
  `<id>.meta.json` recording url/fetched_at/status/sha256/content_type. A re-run
  whose sidecar hash matches the file on disk skips the download entirely
  (architecture 4.1). This is what makes gate 2.8 and 2.10 possible.
- **Failures are collected, not swallowed.** A dead source is recorded with its
  reason and surfaced in the run summary; `build_index` exits non-zero unless
  `--allow-missing` is passed. Silently ingesting 20 of 25 sources would let
  gate 2.1 pass unnoticed.

Why `www.hdfcfund.com` is NOT in the allowlist
----------------------------------------------
It returns Akamai `403 Access Denied` to any non-browser client after a handful
of requests, and the pages are Next.js-rendered, so static fetch yields no
extractable text (architecture 3.6). Its static document CDN,
`files.hdfcfund.com`, serves the same factsheets/SIDs/KIMs as plain PDFs with
no WAF, so the corpus is PDF-first. See docs/architecture.md 12.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

# ── Allowlist ────────────────────────────────────────────────────────────────
# PRD 3.3 / gate 2.2. Only official AMC / SEBI / AMFI hosts.
ALLOWED_DOMAINS: set[str] = {
    "files.hdfcfund.com",  # HDFC AMC static document CDN (factsheets, SIDs, KIMs)
    "www.amfiindia.com",  # AMFI official
    "www.sebi.gov.in",  # SEBI official
}

# Deliberately absent: `www.hdfcfund.com` (WAF-blocked, JS-rendered),
# `groww.in` and every other third party (PRD 7.1 - discovery only, never cited),
# `hdfcmf.com` (a parked domain for sale that returns HTTP 200 - a citation trap),
# `hdfcmf.in` (does not resolve).

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

RETRY_ATTEMPTS = 2
POLITE_DELAY_SECONDS = 1.0
TIMEOUT_SECONDS = 60.0

_EXT_BY_TYPE = {
    # `sources.csv` uses short forms; responses use MIME types. Accept both.
    "pdf": "pdf",
    "application/pdf": "pdf",
    "html": "html",
    "text/html": "html",
    "text": "txt",
    "text/plain": "txt",
    "application/xml": "xml",
}


class AllowlistError(RuntimeError):
    """Raised when a source URL is not on the approved domain list."""


@dataclass(frozen=True)
class Source:
    """One row of `data/sources.csv`."""

    url: str
    title: str
    scheme: str
    page_type: str
    content_type: str
    notes: str = ""

    @property
    def host(self) -> str:
        return urlparse(self.url).netloc.lower()

    @property
    def sid(self) -> str:
        """Stable, filesystem-safe id derived from the URL."""
        digest = hashlib.sha256(self.url.encode("utf-8")).hexdigest()[:12]
        return f"{self.page_type or 'src'}-{digest}"

    def check_allowed(self) -> None:
        if self.host not in ALLOWED_DOMAINS:
            raise AllowlistError(
                f"host {self.host!r} is not on the allowlist "
                f"{sorted(ALLOWED_DOMAINS)}\n  url: {self.url}"
            )


@dataclass
class FetchRecord:
    """Outcome of fetching one source."""

    url: str
    sid: str
    status: int | None = None
    ok: bool = False
    reason: str = ""
    sha256: str = ""
    content_type: str = ""
    bytes: int = 0
    fetched_at: str = ""
    raw_path: str = ""
    from_cache: bool = False
    http_date: str = ""
    extras: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Loading the allowlist ────────────────────────────────────────────────────


def load_sources(sources_csv: Path) -> list[Source]:
    """Read and validate `data/sources.csv`. Raises on any disallowed host."""
    if not sources_csv.exists():
        raise FileNotFoundError(f"allowlist not found: {sources_csv}")

    sources: list[Source] = []
    with sources_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            url = (row.get("url") or "").strip()
            if not url or url.startswith("#"):
                continue
            src = Source(
                url=url,
                title=(row.get("title") or "").strip(),
                scheme=(row.get("scheme") or "").strip(),
                page_type=(row.get("page_type") or "").strip(),
                content_type=(row.get("content_type") or "").strip(),
                notes=(row.get("notes") or "").strip(),
            )
            src.check_allowed()  # gate 2.2, enforced here
            sources.append(src)

    if not sources:
        raise ValueError(f"{sources_csv} contains no usable rows")
    return sources


# ── Cache helpers ───────────────────────────────────────────────────────────


def _meta_path(raw_path: Path) -> Path:
    return raw_path.with_suffix(raw_path.suffix + ".meta.json")


def _cached(raw_path: Path) -> FetchRecord | None:
    """Return the sidecar record if the raw file is present and self-consistent."""
    meta = _meta_path(raw_path)
    if not raw_path.exists() or not meta.exists():
        return None
    try:
        rec = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not rec.get("ok"):
        return None
    # Re-hash the file: guards against a truncated or hand-edited cache.
    actual = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if actual != rec.get("sha256"):
        return None
    rec["from_cache"] = True
    rec["raw_path"] = str(raw_path)
    return FetchRecord(**{k: v for k, v in rec.items() if k in FetchRecord.__annotations__})


def _write_meta(raw_path: Path, rec: FetchRecord) -> None:
    payload = {k: v for k, v in rec.to_json().items() if k != "from_cache"}
    _meta_path(raw_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ── Fetching ────────────────────────────────────────────────────────────────


def fetch(
    source: Source,
    raw_dir: Path,
    *,
    force: bool = False,
    client: httpx.Client | None = None,
) -> FetchRecord:
    """Download one source into `raw_dir`, or reuse a valid cached copy.

    Retries transient failures up to `RETRY_ATTEMPTS` times with linear backoff.
    Never raises for network/HTTP problems - those become `ok=False` records so
    the caller can report every dead source at once.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    ext = _EXT_BY_TYPE.get(source.content_type, "bin")
    raw_path = raw_dir / f"{source.sid}.{ext}"

    if not force:
        hit = _cached(raw_path)
        if hit:
            return hit

    owns_client = client is None
    client = client or httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        timeout=TIMEOUT_SECONDS,
        follow_redirects=True,
    )

    record = FetchRecord(url=source.url, sid=source.sid, raw_path=str(raw_path))
    last_error = ""
    try:
        for attempt in range(RETRY_ATTEMPTS + 1):
            try:
                resp = client.get(source.url)
                record.status = resp.status_code
                record.content_type = resp.headers.get("content-type", "").split(";")[0].strip()
                record.http_date = resp.headers.get("last-modified", "") or resp.headers.get("date", "")

                if resp.status_code != 200:
                    last_error = f"HTTP {resp.status_code}"
                    if resp.status_code in (429, 500, 502, 503, 504) and attempt < RETRY_ATTEMPTS:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    record.reason = last_error
                    return record

                body = resp.content
                if not body:
                    last_error = "empty response body"
                    if attempt < RETRY_ATTEMPTS:
                        time.sleep(1.0)
                        continue
                    record.reason = last_error
                    return record

                raw_path.write_bytes(body)
                record.ok = True
                record.sha256 = hashlib.sha256(body).hexdigest()
                record.bytes = len(body)
                record.fetched_at = _now()
                _write_meta(raw_path, record)
                return record
            except Exception as exc:  # network, TLS, timeout, DNS
                last_error = f"{type(exc).__name__}: {str(exc)[:120]}"
                if attempt < RETRY_ATTEMPTS:
                    time.sleep(2.0 * (attempt + 1))
                    continue
    finally:
        if owns_client:
            client.close()

    record.reason = last_error or "unknown fetch failure"
    return record


def fetch_all(
    sources: list[Source],
    raw_dir: Path,
    *,
    force: bool = False,
    delay: float = POLITE_DELAY_SECONDS,
) -> list[FetchRecord]:
    """Fetch every source, politely, reusing one connection pool."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[FetchRecord] = []
    with httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        timeout=TIMEOUT_SECONDS,
        follow_redirects=True,
    ) as client:
        for i, src in enumerate(sources):
            if i and delay:
                time.sleep(delay)
            rec = fetch(src, raw_dir, force=force, client=client)
            flag = "ok " if rec.ok else "FAIL"
            note = "cache" if rec.from_cache else f"{rec.bytes // 1024}KB"
            print(f"  [{flag}] {src.page_type:<21} {note:>7}  {src.title[:52]}")
            if not rec.ok:
                print(f"         -> {rec.reason}")
            records.append(rec)
    return records


def summarise(records: list[FetchRecord]) -> dict:
    ok = [r for r in records if r.ok]
    return {
        "total": len(records),
        "ok": len(ok),
        "failed": len(records) - len(ok),
        "bytes": sum(r.bytes for r in ok),
        "from_cache": sum(1 for r in ok if r.from_cache),
        "failures": [{"url": r.url, "status": r.status, "reason": r.reason} for r in records if not r.ok],
    }
