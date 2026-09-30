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

**Phases 1–4 complete. Phase 5 code is complete and wired, but its exit gate is
not met and has never been measured passing.**

| Phase | Brief RAG stage | Status |
| --- | --- | --- |
| 1 | — | ✅ done |
| 2 | Loading → Chunking | ✅ done — see [Chunking](#chunking-phase-2) |
| 3 | Embedding (+ store) | ✅ done |
| 4 | — (guardrails) | ✅ done — gates 4.1–4.9 pass |
| 5 | Similarity Search (+ generate) | ⚠️ **code done, gate 5.1 not met** |
| 6 | — (UI, deploy, deliverables) | ⬜ not started |

`POST /ask` is live and works end to end — guardrails → retrieve → generate →
validate. Ask it over HTTP:

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"Is there an exit load on HDFC Flexi Cap Fund?","debug":true}'
```

…or in the terminal, where you can also see the retrieved chunks:
`.venv/bin/python -m app.ask`

### Why Phase 5 is not marked done

`tests/eval.py` scores the PRD §5 criteria. The last run put **SC-1 at 0/20**,
because Groq's free tier had exhausted its 200,000-token daily budget and every
generation returned a 429. The pipeline then degraded exactly as ST-2 requires —
a safe message, no ungrounded answer, zero generation tokens — but that means
**no passing end-to-end run is on record**, and gates 5.1–5.9 and 5.11–5.13 are
unmeasured rather than met.

Two structural ceilings also cap SC-1 below its 90% target, independent of the
API quota:

| Golden questions | Cause | State |
| --- | --- | --- |
| F03, F14 — minimum SIP amount | **Genuinely absent from the corpus.** No installment minimum appears anywhere in `data/parsed/`; SIP appears only as a facility name. The nearest numbers are `Minimum Redemption Rs. 100` and `Minimum Application Rs. 100` — i.e. exactly the figures `generate.qualifier_conflict` exists to stop being misreported as a SIP minimum | **Unfixed.** Needs a source that states installment minimums |
| F19 — NAV of Flexi Cap | Was unreachable: the AMFI feed held all five NAVs in one chunk with `scheme: ""`, which matches no scheme filter | ✅ **Fixed.** One chunk per scheme; now ranks 1st, score 0.744, value in context |
| F08 — NAV of Top 100 | Same root cause, fixed the same way, but the chunk now lands at cosine rank 10 and loses the top-5 cut to five distinct source URLs | ⚠️ **Reachable, not retrieved.** Needs a retrieval decision tuned against a measured SC-1 run |

So the realistic ceiling today is **17/20 (85%)** — still short of gate 5.1's
18/20. The two SIP questions are the binding constraint: no prompt or rerank change
can answer a fact the corpus does not contain.

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

Check the Phase 4 guardrail gates on their own — offline, no API key, prints
the 4.1–4.9 table and exits non-zero on any failure:

```bash
python -m app.guardrails
```

## Configuration

All settings are read from the environment / `.env` via `app/config.py`. See
[`.env.example`](.env.example) for the annotated list. The ones that matter most:

| Variable | Default | Notes |
| --- | --- | --- |
| `GROQ_API_KEY` | — | Required to *answer*. Absent ⇒ the app boots and says so; it never guesses. |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Answer-generation model. `LLM_MODEL` still accepted as a fallback. |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Fixed by the brief. |
| `TOP_K` | `5` | Chunks sent to the LLM. |
| `MIN_SCORE` | `0.35` | Below this ⇒ "I don't have that in my sources". |
| `GUARD_MODE` | `auto` | How the advice guard decides. See below. |
| `GUARD_FAIL_CLOSED` | `true` | If unsure and the guard is down, refuse. |

## Guardrails (Phase 4)

Three questions get a fast, fixed answer before any LLM call. The ordering is the
design — cheapest and most certain first, and every refuse path is terminal, so a
refused question never reaches the generator:

| # | Question | What happens | Why |
| --- | --- | --- | --- |
| 1 | *"My PAN is ABCDE1234F — update my details"* | Refused, regex-detected, **0 LLM calls**. The digits are never echoed and the question is never logged. | Deterministic regex: sub-millisecond, no network, cannot be argued with |
| 2 | *"Should I buy HDFC ELSS Tax Saver Fund?"* | Polite refusal from a fixed template + one educational link | SC-5. A *generated* refusal varies in tone and sometimes implies a view on the fund |
| 3 | *"What is the weather in Mumbai?"* | Off-topic refusal, naming what it does answer | An off-topic answer from a facts-only bot is a hallucination |
| 4 | *"What is the expense ratio of a Kotak Flexi Cap Fund?"* | "I don't have that in my sources" | Outside the five in-scope schemes (ST-1) |
| 5 | A question whose retrieved pages don't state the fact | "I couldn't confirm that…" | Retrieved-but-unsupported is the failure mode that produces *confident, wrong* answers |

Notes on the design, because two of these choices are not obvious:

- **The 22M guard model detects injection, not advice — and that's measured, not
  assumed.** `llama-prompt-guard-2-22m` returns a bare float P(unsafe), not the words
  "safe"/"unsafe". Live scores: a real injection attempt **0.9986**, a benign
  expense-ratio question **0.0007**, and *"Should I buy HDFC ELSS Tax Saver Fund?"*
  **0.0007** — identical to the benign one, because nothing about "should I buy" is
  harmful. So it cannot be the advice detector. Deterministic patterns decide advice,
  scope and off-topic; the model is used only to escalate an unplaceable question to
  INJECTION, and can never clear one into "factual". Full table in
  `docs/architecture.md` §6.2.
- **A refusal costs nothing.** With a live key and `GUARD_MODE=auto`, all 45 golden
  probes were classified with **0 guard-model calls** — the deterministic layer places
  every one. `app.guardrails.guard_calls()` makes that measurable instead of assumed.
- **"I don't know" is decided by tested code, not by asking the model nicely.**
  `guardrails.grounding_decision()` is a pure function over retrieval hits and
  returns `ANSWERABLE` / `NO_CONTEXT` / `UNSUPPORTED`. Phase 5 wires it in.

Verified against `tests/golden_questions.json` (20 factual + 10 advice + 4 PII +
8 off-topic + 3 out-of-scope):

```
  4.1    PASS  all PII probes refused             4/4
  4.2    PASS  PII not echoed                     4/4 clean
  4.3    PASS  PII not logged                     4 log line(s), no question text
  4.4    PASS  advice refused with a link         10/10 refused, 10/10 with link
  4.5    PASS  refusals are fixed text            1 template
  4.6    PASS  factual questions not refused      20/20 FACTUAL
  4.7    PASS  fail closed when unsure            3/3 refused
  4.8    PASS  zero API calls on refusal          0 model call(s) for 14 refusals
  4.9    PASS  offline, deterministic path        no network used
  ST-1   PASS  unsupported context says I don't know UNSUPPORTED
```

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

## Chunking (Phase 2)

Gate 2.9 asks for the chunking decision and the *reason* it came from the data.
The decision is **220-token ceiling, ~40-token overlap, packed on block
boundaries**, and the numbers below are measured on the real 1,926-chunk build,
not assumed.

Two things about that decision were wrong when first written down. Both surfaced
from writing the tests this phase was supposed to have from the start
(`tests/test_chunking.py`), and both are now regression-tested.

**The overlap was inert.** Units are whole blocks — a paragraph or a table — and
the **median unit measures 170 tokens** against a 40-token overlap budget, with
**90% of units larger than the entire budget**. Seeding the next chunk with whole
units therefore returned an *empty* seed at ~90% of boundaries: only **5.0%** of
adjacent chunks in the 1,888-chunk build shared any text. The overlap existed only
on paper.

`_seed_units` now trims prose to a sentence boundary to fill the budget, and skips
table units entirely so a row is never cut to buy overlap (gate 2.5). Measured seam
coverage on the rebuilt 1,926-chunk corpus: **5.0% → 51.3%**.

The remaining ~49% is arithmetic, not a bug. A ~33-token seed leaves 187 tokens of
the 220 ceiling, and units larger than that cannot share it — `_split_long_unit`
packs a multi-sentence block up to exactly the ceiling, so the incoming unit is
often full-size. Closing the gap means re-tuning unit granularity, which is a fresh
gate 2.9 decision that invalidates the measured Phase 3 and Phase 5 recall numbers
— so it is left as a documented limit rather than changed silently.

**An over-ceiling table row was a silent gate-2.4 violation.** In `_split_long_unit`
the oversized-row flag was only applied to the *final* row group, so a row too long
to fit the ceiling at any position but last was emitted as an ordinary chunk and
counted as a ceiling violation rather than the flagged unit it is. It never fired on
the real corpus — no row is that long — which is exactly why it survived until the
path was tested. Every flush now applies the flag.

`validate_chunks` enforced the ceiling at build time throughout, so the corpus was
never silently truncated; what was missing was a test asserting the enforcement.

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

- **Not verifiable yet.** `POST /ask` answers questions, but the answer-quality
  gates are unmeasured rather than passing — see [Status](#why-phase-5-is-not-marked-done).
  The guardrail gates (4.1–4.9) are measured and pass.
- **Minimum SIP amount cannot be answered from this corpus.** See
  [Status](#why-phase-5-is-not-marked-done). Declining it is the correct
  behaviour; answering it from a redemption minimum would not be.
- **Refusal links are landing pages, not deep links.** HDFC MF's site answers
  every path with an SPA shell, so a deep link cannot be verified as reachable;
  SEBI and AMFI deep links are. All refusal links are verified HTTP 200 and
  allowlisted, but they are generic — a reviewer gets "Investor education"
  rather than the exact sub-page on their question.
- **Demo-scale capacity, and the real ceiling is tokens, not requests.** Groq's
  Free Plan for `openai/gpt-oss-20b` is 30 req/min, 1,000 req/day, **8,000
  tokens/min, 200,000 tokens/day**. A RAG prompt with 5 retrieved chunks costs
  ~1.5–2K input tokens, so that's ~4–6 questions/minute — but the *daily* number is
  the one that binds: **~100 questions/day**, not 1,000. Fine for a demo, not for load
  testing. A 429 must surface ST-2, never an ungrounded answer.
- **`gpt-oss-20b` is a reasoning model, and it will silently return an empty answer.**
  Its `reasoning` tokens bill against the same budget as `content`, and on a
  one-sentence question measured 98 completion tokens by default vs **46** with
  `reasoning_effort="low"` — so that parameter is worth ~2× the daily question count
  and is mandatory, not an optimisation. Worse, with `max_tokens=80` the *entire* budget
  was consumed by the reasoning trace and `content` came back as an **empty string**.
  Phase 5 must check `finish_reason` before treating an empty answer as "nothing to
  say".
- **The model choice is not a preference you need to revisit.** `openai/gpt-oss-20b`
  *is* on the Free Plan at $0 — the price in Groq's model table is the Developer
  Plan rate, which is why it looks paid. Note that the commonly recommended
  `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` are **no longer free-tier
  eligible** (they now read "Enterprise — Contact Sales").
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
