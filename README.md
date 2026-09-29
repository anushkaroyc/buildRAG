# Mutual Fund Facts-Only FAQ Assistant

A RAG chatbot that answers **factual** questions about five HDFC Mutual Fund equity
schemes, using only official public pages, with **one source link on every answer**.

> **Facts-only. No investment advice.**
> It does not recommend buying, selling, or holding any scheme, and does not compute
> or compare returns.

A class demo built to a written brief. Documents:

| Document | Purpose |
| --- | --- |
| [`docs/PRD.md`](docs/PRD.md) | Goals, users, scope, success criteria, constraints |
| [`docs/architecture.md`](docs/architecture.md) | Components, data flow, tech stack, trade-offs |
| [`docs/implementation.md`](docs/implementation.md) | Six build phases with verification gates |
| [`docs/ProblemStatement.txt`](docs/ProblemStatement.txt) | The original brief |

## Status

**Phase 1 (project setup) complete. Phases 2–6 not started.**

| Phase | Brief RAG stage | Status |
| --- | --- | --- |
| 1 | — | ✅ done |
| 2 | Loading → Chunking | ⬜ not started |
| 3 | Embedding (+ store) | ⬜ not started |
| 4 | — (guardrails) | ⬜ not started |
| 5 | Similarity Search (+ generate) | ⬜ not started |
| 6 | — (UI, deploy, deliverables) | ⬜ not started |

The app boots and serves `/` and `/healthz`. There is no `/ask` route yet.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then add GROQ_API_KEY
```

Run it:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Landing page: <http://localhost:8000>
- Health + config: <http://localhost:8000/healthz>
- API docs: <http://localhost:8000/docs>

Run the tests (no network or API key needed):

```bash
pytest
```

## Configuration

All settings are read from the environment / `.env` via `app/config.py`. See
[`.env.example`](.env.example) for the annotated list. The ones that matter most:

| Variable | Default | Notes |
| --- | --- | --- |
| `GROQ_API_KEY` | — | Required to *answer*. Absent ⇒ the app boots and says so; it never guesses. |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Fixed by the brief. |
| `TOP_K` | `5` | Chunks sent to the LLM. |
| `MIN_SCORE` | `0.35` | Below this ⇒ "not in my corpus". |

## ⚠️ Two constraints that are easy to break

**1. Never add `torch` or `sentence-transformers` to `requirements.txt`.**
The brief mandates the *embedding model* `sentence-transformers/all-MiniLM-L6-v2`, but
not the PyTorch *library*. We load that model through ONNX (`fastembed`) instead.
Installing the library pulls in `torch` at ~660–930 MB peak RSS, which exceeds Render's
512 MB free-tier limit and breaks the deploy. `tests/test_config.py` guards this.
See [`docs/architecture.md` §9.1](docs/architecture.md).

**2. No chunk may exceed 220 tokens.**
The model truncates at 256 tokens, silently — no error, no warning. A longer chunk
loses its tail, which is exactly where exit-load slabs and lock-in terms live, so the
system would return confident answers that are quietly wrong. See
[`docs/architecture.md` §3.4](docs/architecture.md).

## Scope

One AMC, five schemes:

| Scheme | Category |
| --- | --- |
| HDFC Top 100 Fund – Direct Growth | Large Cap |
| HDFC Flexi Cap Fund – Direct Growth | Flexi Cap |
| HDFC ELSS Tax Saver Fund – Direct Growth | ELSS / Tax Saver |
| HDFC Mid Cap Fund – Direct Growth | Mid Cap |
| HDFC NIFTY 50 Index Fund – Direct Growth | Index |

Corpus: 15–25 public pages from HDFC AMC, SEBI and AMFI.

## Known limits

- **Not usable yet.** No question-answering; `/ask` arrives in Phase 5.
- **Demo-scale capacity.** Groq's free tier allows ~1,000 requests/day and ~8K
  tokens/minute, which works out to roughly 4–6 questions/minute. Fine for a demo,
  not for load testing.
- **Render free tier sleeps** when idle, so the first request after a quiet period
  takes a few seconds to wake up.
- **Stale figures.** AUM, expense ratios and NAVs go stale. Answers are stamped with
  the source's own "as of" date; they are not live data.

## Open questions

Still blocking, tracked in [`docs/PRD.md` §7](docs/PRD.md):

- Is **Groww** an acceptable citation source, or are HDFC/SEBI/AMFI the only ones?
  *(This blocks Phase 2 — the source allowlist cannot be written without it.)*
- 10 URLs (deliverables) vs. 15–25 pages (corpus requirement)?
- Are NAV questions in scope, or dropped as too time-sensitive?
