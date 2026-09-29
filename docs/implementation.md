# Implementation Plan — Mutual Fund Facts-Only FAQ Assistant

**Status:** Draft · **Last updated:** 2026-09-30
**Derived from:** `docs/architecture.md` · `docs/PRD.md` · `docs/ProblemStatement.txt`
**Code:** none written yet. This document specifies *what to build and how to know each
stage works* before any of it exists.

---

## How to use this plan

Six phases, strictly sequential. **Do not start a phase until the previous phase's
verification gate passes.** Each gate is a falsifiable check, not a feeling.

Every phase lists: **files to create**, **what it does**, **how to verify**, and
**failure modes to expect**.

### Phase → brief RAG stage mapping

The brief requires following every RAG stage. This plan satisfies that explicitly:

| Phase | Brief RAG stage | Deliverable |
| --- | --- | --- |
| 1 | — | Project scaffolding, config, dependency lock |
| 2 | **Loading → Chunking** | Corpus fetched, parsed, chunked |
| 3 | **Embedding** (+ store) | Vectors in ChromaDB |
| 4 | — (compliance layer) | PII + advice guardrails |
| 5 | **Similarity Search** (+ generate) | Cited, ≤3-sentence answers |
| 6 | — (presentation) | Tiny demo UI + Render deploy |

### Rules that apply to every phase

1. **`torch` must never enter `requirements.txt`.** The embedding *model* is
   `sentence-transformers/all-MiniLM-L6-v2` (mandated by the brief); the *library* is
   `fastembed` + ONNX. Installing `sentence-transformers` breaks the Render deploy
   (architecture §9.1).
2. **No chunk may exceed 220 tokens** (model truncates at 256 — architecture §3.4).
3. **No module may fabricate a fact when a dependency is missing.** Fail closed.
4. **Every new module gets its test file in the same phase.** Not later.
5. Do not write `code` for a later phase while debugging an earlier one.

### Blocking prerequisites (resolve before Phase 2)

| # | Prerequisite | Blocks | Status |
| --- | --- | --- | --- |
| P1 | **Groww: allowed citation source or not?** (PRD §7 item 1) | `data/sources.csv` — the allowlist cannot be written without it | ☐ open |
| P2 | **Free Groq API key** in `.env` | Phase 5 only (Phase 1–4 work without it) | ☐ open |
| P3 | **10 URLs vs. 15–25 pages** (PRD §7 item 1) | Final `sources.csv` size; `data/chroma/` size | ☐ open |
| P4 | **NAV questions in or out?** (PRD §7 item 4) | Chunking test cases | ☐ open |

P1 is the hard one. If unanswered, Phase 2 cannot start — there is no allowlist to
fetch. Everything up to and including Phase 1 can proceed in the meantime.

---

## Phase 1 — Project setup

**Goal:** a clean, reproducible skeleton where `python -m app.main` boots and the
dependency set is provably safe for Render.

### Files to create

| File | Purpose |
| --- | --- |
| `requirements.txt` | Pinned dependencies (pin **after** first successful install) |
| `.env.example` | `GROQ_API_KEY=` placeholder, documented, no real secret |
| `.gitignore` | `.env`, `data/parsed/`, `__pycache__/`, `.venv/` |
| `app/__init__.py` | Empty package marker |
| `app/config.py` | Typed settings via `pydantic-settings`: paths, model IDs, `TOP_K`, `MIN_SCORE`, `PORT` |
| `app/main.py` | FastAPI app skeleton: `GET /` and `GET /healthz` only |
| `tests/test_config.py` | Settings load, defaults correct, no secret leaked |
| `README.md` (skeleton) | Setup steps stub — full version is a Phase 6 deliverable |

### What this phase does

- Creates the virtualenv and installs the dependency set.
- Establishes **one** source of truth for configuration, read from the environment.
- Proves the app boots and reports health **before** any RAG logic exists — so later
  failures are unambiguously RAG failures.
- `/healthz` reports process RSS from day one, because memory is a hard constraint that
  must be tracked continuously rather than discovered at deploy time.

### Verify before moving on

Run each of these; all must pass.

| # | Check | Pass condition |
| --- | --- | --- |
| 1.1 | App boots | `uvicorn app.main:app` starts; `GET /healthz` → 200 |
| 1.2 | **Memory baseline** | `/healthz` reports RSS **< 120 MB** for a bare app. This is the number every later phase is measured against |
| 1.3 | **`torch` is absent** | `pip show torch` fails **and** `grep -ri torch requirements.txt` finds nothing |
| 1.4 | Config is env-driven | Change `TOP_K` in `.env`; `/healthz` reflects it. No hard-coded constants in modules |
| 1.5 | No secret in repo | `git status` shows no `.env`; `.env.example` has an empty value |
| 1.6 | Tests run offline | `pytest tests/ -q` passes with network disabled |

**Exit gate:** all six pass. If 1.3 fails, stop and fix `requirements.txt` — every later
phase inherits this dependency set.

---

## Phase 2 — Loading & chunking

**Goal:** a reproducible, inspectable corpus on disk. **No embeddings yet.** This is the
brief's *Loading → Chunking* stage.

> This is normally the slowest and most unpredictable phase. Real AMC sites have
> inconsistent markup, JS-rendered pages, and PDFs. Budget the most time here, and do not
> let it slide into Phase 3.

### Files to create

| File | Purpose |
| --- | --- |
| `data/sources.csv` | Allowlist. Columns: `url, title, scheme, page_type, content_type, notes` |
| `app/ingest/__init__.py` | Package marker |
| `app/ingest/fetch.py` | Download → cache raw bytes to `data/raw/`, record `fetched_at`, status, sha256, content type |
| `app/ingest/parse.py` | Bytes → clean text. **PDF path** (`pypdf`, `pdfplumber` for tables) and **HTML path** (BeautifulSoup + `lxml`) |
| `app/ingest/build_index.py` | Orchestrates fetch → parse → chunk → `data/parsed/`. *(Writes no vectors in this phase.)* |
| `app/chunking.py` | Token-aware splitter: ≤220 tokens, ~40 overlap, paragraph-aware, tables kept whole |
| `tests/test_chunking.py` | Ceiling, overlap, table integrity, metadata propagation |
| `tests/fixtures/` | 2–3 saved raw pages (1 HTML, 1 PDF) so tests run **without network** |

### What this phase does

- Fetches every URL in `sources.csv` to `data/raw/`, content-hashed so re-runs skip
  unchanged pages (architecture §4.1 idempotency).
- Converts each page to clean text, capturing the **factsheet "as of" date** — this
  becomes the `Last updated from sources:` value, so losing it breaks SC-8.
- Splits text into ≤220-token chunks, each carrying
  `{id, text, url, title, page_type, scheme, content_type, page_no, as_of, fetched_at}`.
- **Decides the chunking strategy from the data**, as the brief requires — the 220/40
  values in architecture §3.4 are a *starting hypothesis*, not a decision. This phase is
  where that decision gets made and recorded.
- Excludes unusable sources (image-only PDF, JS-only page) and logs them with a reason.

### Verify before moving on

| # | Check | Pass condition |
| --- | --- | --- |
| 2.1 | **Corpus size** | `data/raw/` holds **≥15** usable pages (PRD SC-11), each with HTTP 200 |
| 2.2 | **Allowlist compliance** | Every URL's domain is on the approved list. **Zero** blogs/news/forums/broker pages (PRD §3.3) |
| 2.3 | **Text quality** | Open 3 random parsed pages: no nav menus, no cookie banners, no repeated footers. The fee/charge text is present and readable |
| 2.4 | **Token ceiling** | A test asserts **no chunk > 220 tokens**. Zero violations. *(If any exist, fix before continuing — architecture §3.4)* |
| 2.5 | **Table survival** | An exit-load or expense-ratio table is intact in `data/parsed/` — no row split mid-sentence |
| 2.6 | **As-of dates captured** | ≥1 factsheet has a non-null `as_of`. Without it, SC-8 cannot pass later |
| 2.7 | **Metadata complete** | 100% of chunks have non-empty `url`, `scheme`, `page_type`. No nulls |
| 2.8 | **Idempotent** | Re-run `build_index` → no re-downloads, identical chunk count |
| 2.9 | **Chunking decision recorded** | Chosen chunk size/overlap and the *reason* (from 2.3/2.5 inspection) written into `README.md` — the brief asks for a data-driven strategy |
| 2.10 | **Offline reproducible** | `rm -rf data/parsed/` then rebuild from `data/raw/` → identical output, no network |
| 2.11 | Unit tests offline | `pytest tests/ -q` passes with network disabled |

**Exit gate:** 2.1–2.11 pass. **2.4 is non-negotiable** — a ceiling violation silently
corrupts every downstream answer.

---

## Phase 3 — Embedding & vector store

**Goal:** every chunk embedded with the brief-mandated model and persisted in ChromaDB.
This phase contains the project's **go/no-go memory checkpoint**.

### Files to create

| File | Purpose |
| --- | --- |
| `app/embeddings.py` | `fastembed` + `sentence-transformers/all-MiniLM-L6-v2` (ONNX, prefer int8 build). Lazy singleton — loaded once, never per-call |
| `app/store.py` | Chroma `PersistentClient` at `data/chroma/`, single collection, cosine space, metadata passthrough |
| `app/ingest/build_index.py` *(extend)* | Add the embed + store stages, now completing the full pipeline |
| `data/manifest.json` | `n_pages`, `n_chunks`, `model`, `built_at`, sources list |
| `tests/test_store.py` | Query returns expected chunk + metadata round-trip |

### What this phase does

- Embeds all chunks (batched, CPU) into **384-dim normalized vectors** using
  `sentence-transformers/all-MiniLM-L6-v2` — the model the brief names.
- Writes vectors + metadata to ChromaDB's `PersistentClient`.
- Records the model ID in `manifest.json` so a model/index mismatch is detectable later.
- Confirms the runtime fits Render's **512 MB** ceiling — *before* any UI work is done, so
  a stack problem surfaces while it is still cheap to change.

### Verify before moving on

| # | Check | Pass condition |
| --- | --- | --- |
| 3.1 | **Dimension** | Vectors are exactly **384-dim**, unit-normalized |
| 3.2 | **Model identity** | `manifest.json` records `sentence-transformers/all-MiniLM-L6-v2`, matching `EMBED_MODEL` |
| 3.3 | **Count matches** | Collection count == chunk count from `data/parsed/`. No silent drops |
| 3.4 | **Metadata survives** | A test query returns chunks with `url`, `scheme`, `page_type`, `as_of` all intact |
| 3.5 | **Metadata filter works** | Filtering `where={"scheme": "<elss scheme>"}` returns **only** that scheme's chunks |
| 3.6 | **Sanity retrieval** | Query *"HDFC ELSS lock-in period"* → top hit is an ELSS chunk mentioning lock-in, in the top 3 |
| 3.7 | **Schema filter** | Query *"exit load slabs"* → top hits are fee/charge pages, not landing pages |
| 3.8 | 🚦 **GO/NO-GO: memory** | RSS after loading model + Chroma + a query is **< 400 MB**. See below |
| 3.9 | **Cold-start time** | App restart → model loaded and first query answered in **< 10 s** (Render cold-start UX) |
| 3.10 | **Persistence** | Restart the process; the same query returns the same top hit without rebuilding |
| 3.11 | **No `torch`** | Re-check `pip show torch` → still absent |

#### 🚦 Check 3.8 is the project's go/no-go gate

Measure peak RSS **with the model loaded, Chroma open, and a query executed** — not the
import baseline from Phase 1.

- **< 400 MB** → proceed. Leave ~110 MB headroom for request handling.
- **400–480 MB** → proceed, but try the int8 ONNX build and `OMP_NUM_THREADS=2` first.
- **> 480 MB or OOM** → **stop and escalate.** Do not push forward to Phase 6 and
  discover this at deploy time. The options, in order: int8 ONNX build → drop Chroma's
  default embedding function (load weights once, share) → drop to Chroma's built-in
  `ONNXMiniLM_L6_V2` EF → accept a Render paid tier.

**This is the most likely place for the project to hit a wall.** Measuring it in Phase 3
rather than Phase 6 is deliberate.

**Exit gate:** 3.1–3.11 pass, with 3.8 firmly in the green.

---

## Phase 4 — Guardrails

**Goal:** PII and advice detection that run **before** any LLM call (architecture §5), so
refusals are cheap and cannot be paraphrased into answers.

### Files to create

| File | Purpose |
| --- | --- |
| `app/guardrails.py` | `check_pii()` regex set; `classify_intent()` via Groq `llama-prompt-guard-2-22m`; local keyword fallback |
| `app/prompts.py` | Refusal templates, educational-link map, and the system prompt (stubbed here, completed in Phase 5) |
| `tests/golden_questions.json` | 20 factual + 10 advice + 4 PII questions (PRD §4) — the eval backbone |
| `tests/test_guardrails.py` | Every PII probe refused; every advice probe classified; **false-positive check on the 20 factual questions** |

### What this phase does

- Detects PAN, Aadhaar, account numbers, OTPs, emails, phone numbers with deterministic
  regex — no LLM call, no network, sub-millisecond.
- Classifies *factual* vs *advice/injection* with the 22M-param guard model on Groq.
  Falls back to a local keyword heuristic when Groq is unreachable, and **fails closed**
  (refuse when unsure).
- **Scrubs PII from logs.** The raw query must never reach a log line or be persisted
  (PRD §3.2 item 8, SC-6).
- Returns a polite, fixed refusal plus a relevant educational link — never a generated
  opinion.

### Verify before moving on

| # | Check | Pass condition |
| --- | --- | --- |
| 4.1 | **All PII probes refused** | 4/4 from PRD §4.3 (SC-6) |
| 4.2 | **PII not echoed** | Response contains **no** digit sequence from the submitted PAN/Aadhaar/OTP |
| 4.3 | **PII not logged** | Grep the log output for the test PAN → **zero hits** |
| 4.4 | **All advice probes refused** | 10/10 from PRD §4.2, each with an educational link (SC-5) |
| 4.5 | **Refusals are polite and fixed** | Identical template every time; no LLM-generated opinion text |
| 4.6 | 🚦 **False-positive check** | **All 20 factual questions classified FACTUAL.** *This is the one most likely to fail — see below* |
| 4.7 | **Fail-closed** | With Groq unreachable, an ambiguous question is **refused**, not answered |
| 4.8 | **Zero LLM cost on refusal** | Refusals cost 0 generation tokens; the 22M guard is the only call |
| 4.9 | Tests offline | PII and fallback paths pass with network disabled |

#### Check 4.6 is the subtle one

The guard model is trained to catch harmful requests. Questions like *"What is the
lock-in period for my ELSS?"* contain mild first-person framing and may trip it. If
factual questions get refused, the assistant looks broken.

- Tune the system framing and threshold until all 20 pass.
- Keep the local keyword fallback **narrow** — it should catch clear advice asks
  (*"should I buy"*, *"best performing"*, *"allocate my money"*) and nothing else.
- If you cannot clear 20/20 without losing advice coverage, **drop the classifier** and
  rely on the keyword list. A simpler guard that works beats a smarter one that refuses
  legitimate questions. SC-5 (100% refusal) and 4.6 (0% false refusal) are both hard
  requirements — if they conflict, the keyword list is the fallback.

**Exit gate:** 4.1–4.9 pass, **including 20/20 on 4.6**.

---

## Phase 5 — Retrieval + LLM answer

**Goal:** end-to-end cited answers. The brief's *Similarity Search* stage plus grounded
generation.

### Files to create

| File | Purpose |
| --- | --- |
| `app/retrieve.py` | Embed query → Chroma search → scheme filter → `MIN_SCORE` threshold → dedupe by URL → ≤5 chunks |
| `app/generate.py` | Groq chat (temp 0); citation assembled **from chunk metadata**; answer validator with one repair retry |
| `app/prompts.py` *(complete)* | System prompt: ≤3 sentences, one citation, no advice, no return claims |
| `app/main.py` *(extend)* | `POST /ask` wiring guardrails → retrieve → generate → validate |
| `tests/test_validator.py` | Sentence count, link count, freshness stamp, return-word scan |
| `tests/eval.py` | Scores SC-1…SC-9 against the golden set |

### What this phase does

- Retrieves grounded context: top-k, filtered to the named scheme, thresholded, deduped
  so five chunks from one page do not crowd out a second source.
- If every score falls below `MIN_SCORE`, answers *"I don't have that in my corpus"* and
  generates **nothing** (ST-1).
- Generates a **≤3-sentence** answer grounded strictly in retrieved context.
- Attaches the citation from the winning chunk's `url` field — **the LLM never writes the
  URL** (architecture §9.4), which makes SC-2 and SC-3 structurally true.
- Validates the output: sentence count, exactly one link, `Last updated from sources:`
  present, no return/comparison vocabulary. One repair retry, then a safe fallback.
- Degrades correctly on Groq 429/5xx/timeout and on a missing key (ST-2).

### Verify before moving on

| # | Check | Pass condition |
| --- | --- | --- |
| 5.1 | **Grounded correctness** | **≥18/20** golden questions match the cited page (SC-1) |
| 5.2 | **One citation, always** | Exactly 1 link in 20/20 (SC-2) |
| 5.3 | **Citation is official** | Domain on the allowlist in 20/20 (SC-3) |
| 5.4 | **Citation supports the claim** | The linked page actually contains the fact, **≥19/20** (SC-4) |
| 5.5 | **Brevity** | ≤3 sentences in 20/20 (SC-7) |
| 5.6 | **Freshness stamp** | `Last updated from sources:` in 20/20, and the date matches the source's `as_of` (SC-8) |
| 5.7 | **No performance claims** | Zero return/comparison statements across the whole run (SC-9) |
| 5.8 | **Out-of-corpus** | *"What is the expense ratio of a Kotak Flexi Cap Fund?"* → "not in my corpus", no answer (ST-1) |
| 5.9 | **Citation is real** | Spot-check 5 links in a browser; every one loads and shows the claim |
| 5.10 | 🚦 **ST-2 degradation** | With `GROQ_API_KEY` unset **and** network blocked → safe message, **never** an ungrounded answer |
| 5.11 | **Latency** | p50 < 4 s, p95 < 8 s over 20 runs (SC-12) |
| 5.12 | **Schema disambiguation** | Two different schemes' expense-ratio questions get **different** correct answers |
| 5.13 | **No cross-scheme bleed** | An ELSS question never returns a Flexi Cap figure |

#### If 5.1 lands below 18/20

Diagnose before touching the prompt — the usual causes, in order:

1. **Retrieval, not generation.** Dump the retrieved chunks for a failing question. If
   the fact is not in the top 5, the fix is `MIN_SCORE`, `TOP_K`, or chunking — *not* the
   prompt. This is the most common misdiagnosis.
2. **Chunking.** A fact split across a chunk boundary is invisible. Re-tune size/overlap
   per the brief's data-driven instruction (Phase 2, check 2.9).
3. **Only then** the prompt or a re-ranking step.

**Exit gate:** 5.1–5.13 pass, or 5.1 at 17/20 with a written, understood cause.

---

## Phase 6 — UI, deliverables & deploy

**Goal:** the demo surface, the deliverables, and a live Render URL.

### Files to create

| File | Purpose |
| --- | --- |
| `app/ui.py` | Jinja2 template: welcome line, 3 example questions, disclaimer, answer + citation |
| `app/templates/index.html` | Single page |
| `static/style.css`, `static/app.js` | Styling; `fetch()` to `/ask` |
| `samples/sample_qa.md` | 5–10 queries with the assistant's answers + links **(deliverable)** |
| `render.yaml` | `type: web`, `plan: free`, `/healthz` check, `GROQ_API_KEY` env var |
| `README.md` *(complete)* | Setup, scope (AMC + 5 schemes), known limits, free-tier caveats **(deliverable)** |
| `.github/workflows/ci.yml` | Tests + a memory regression check |

### What this phase does

- Renders the **tiny** UI the brief asks for: welcome line, 3 example questions, the
  persistent *"Facts-only. No investment advice."* note (SC-10), a question box, and the
  answer with its single citation.
- Shows a loading state, because Render free tier sleeps when idle and cold starts are
  expected — a reviewer hitting a blank page concludes the demo is broken.
- Finalizes the deliverables: source list, sample Q&A, README, disclaimer snippet.
- Deploys to Render with the prebuilt index committed — the build command must **not**
  run `build_index` (architecture §11).
- Adds a CI memory check so a future dependency change cannot silently reintroduce
  `torch` and blow the 512 MB budget.

### Verify before moving on

| # | Check | Pass condition |
| --- | --- | --- |
| 6.1 | **Disclaimer visible** | On the welcome screen **and** every answer view (SC-10) |
| 6.2 | **3 example questions** | All three are clickable and return cited answers (PRD §3.2 item 7) |
| 6.3 | **Cold start** | First load after ~15 min idle shows a loading state, then works |
| 6.4 | **Mobile-width layout** | Readable at 375 px |
| 6.5 | **Deploys to Render** | Live URL loads, `plan: free`, `/healthz` green |
| 6.6 | 🚦 **Live memory** | RSS on Render < 512 MB, no OOM restarts in logs |
| 6.7 | **Index committed** | `data/chroma/` is in the repo; the build command does **not** rebuild it |
| 6.8 | **No secrets** | `GROQ_API_KEY` is a Render env var; nothing in git history |
| 6.9 | **Full golden run on live URL** | SC-1…SC-10 pass against the deployed app, not just locally |
| 6.10 | **Deliverables complete** | Source list, sample Q&A, README, disclaimer snippet all present |
| 6.11 | **Demo script** | PRD §5 acceptance test rehearsed end-to-end under 3 minutes |
| 6.12 | **Demo video recorded** | ≤3 min, as the fallback deliverable |

**Exit gate:** 6.1–6.12 pass. Then record the video while the live URL is fresh.

---

## Effort and sequencing

| Phase | Rough effort | Notes |
| --- | --- | --- |
| 1 · Setup | 1–2 h | Mostly mechanical. Install friction possible. |
| 2 · Loading & chunking | **4–8 h** | **Highest variance.** Real sites, PDFs, JS pages, 256-token ceiling. Start the `sources.csv` research first. |
| 3 · Embed & store | 2–3 h | Mechanical once Phase 2 is clean. **3.8 is the go/no-go gate.** |
| 4 · Guardrails | 3–5 h | **4.6 (false positives) is the real work.** |
| 5 · Retrieval + LLM | 4–6 h | Iterative. Expect to tune `MIN_SCORE` and chunking, not the prompt. |
| 6 · UI + deploy | 3–4 h | Plus waiting on Render deploys and the first cold start. |
| **Total** | **~17–28 h** | Add time for the open decisions (P1–P4) and re-ingest cycles. |

### Critical path

```
P1 (Groww decision) ──► Phase 2 ──► Phase 3 ──🚦 3.8 go/no-go ──► Phases 4–6
                                                                    │
                                        P2 (Groq key) ─────────────┘ (needed by Ph 5)
```

Phase 1 can start immediately. **P1 blocks Phase 2.** If P1 is not answered, resolve it
before doing anything else — everything downstream depends on the allowlist.

### Highest-risk items, ranked

| Risk | Phase | Mitigation |
| --- | --- | --- |
| RSS exceeds Render's 512 MB | 3 | ONNX int8 build; measured at 3.8, not at deploy |
| Factsheets are PDFs, not HTML | 2 | PDF path built in from the start (architecture §3.6) |
| Guard refuses legitimate questions | 4 | Keyword-list fallback; 4.6 gate |
| Chunks silently truncated at 256 | 2 | 220-token ceiling + 2.4 test gate |
| Corpus sources are JS-only | 2 | Log and exclude; pick alternative official pages |
| Groq free-tier quota exhausted mid-demo | 5, 6 | ST-2 message; documented in README (§6.1) |

---

## Definition of done

- [ ] All 6 phases' exit gates passed
- [ ] PRD §5 success criteria: SC-1…SC-12 and ST-1, ST-2 measured and met
- [ ] Live Render URL on the free tier, no OOM restarts
- [ ] Deliverables: source list, sample Q&A, README, disclaimer snippet
- [ ] ≤3-min demo video recorded
- [ ] No `torch` in the dependency tree
- [ ] Every open question in PRD §7 either resolved or documented as a known limit
