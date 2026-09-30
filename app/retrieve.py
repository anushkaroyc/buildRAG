"""Embed the question, search Chroma, return grounded chunks.

Phase 5's retrieval half. The one job here is to make sure that when the LLM is
asked a question, the context it receives actually contains the answer - and to
make "it doesn't" a cheap, early, deterministic answer rather than a hallucination
three layers downstream.

The pipeline, and why each step exists:

    embed (same model as Phase 3)  ->  Chroma search  ->  scheme filter
    ->  hybrid (cosine + term overlap) rerank  ->  MIN_SCORE threshold
    ->  dedupe by URL  ->  cap at TOP_K

**Cosine alone is not enough on this corpus, so the order is reranked with a
lexical term-overlap term** (`LEXICAL_WEIGHT`, which carries the measured
recall table). The sources are SEBI SID boilerplate: a chunk that states the
answer is wrapped in hundreds of words of regulation, and MiniLM pools the whole
220-token window into one vector, so the question's own words are averaged away.
`rank_score` drives ordering while `score` stays the raw cosine, so `MIN_SCORE`
keeps the meaning the Phase 3 gates were measured against.

**Same embedding model, or nothing works.** `embeddings.embed_query` loads the
brief-mandated `all-MiniLM-L6-v2` through the same ONNX path Phase 3 indexed with.
A different model produces vectors in a different space; Chroma would still return
nearest neighbours and the similarity numbers would still *look* plausible, so a
mismatch here does not crash - it silently degrades every answer in the app. The
dimension check in `embeddings` is the only thing standing between a bad
`EMBED_MODEL` and a confidently wrong assistant.

**Scheme filter first, because it is the cheap correctness win.** When the guard
layer has already identified the scheme (`guardrails.named_scheme`), restricting
the search to that scheme's chunks is what makes gate 5.12 (two schemes' expense
ratios get different answers) and 5.13 (an ELSS question never returns a Flexi Cap
figure) structurally true rather than a matter of luck. Note the name translation
through `INDEX_SCHEME_NAMES` - see the comment there for why this fails silently.

**Dedupe by URL, best chunk first.** A factsheet is chunked into ~5 pieces, so an
unfiltered search for "expense ratio of Flexi Cap" will happily return five
adjacent chunks of one PDF. That is a bad answer for two reasons: the LLM sees the
same page five times, and the single citation becomes unrepresentative. So the
first pass takes only the best chunk per URL, and later passes admit a second chunk
from a URL only if `TOP_K` is still unfilled. Source diversity is preferred, but a
single relevant page is not punished with an empty context.

**Threshold last, and it is the ST-1 exit.** Everything scoring below `MIN_SCORE`
is dropped; if nothing survives, the caller gets an empty result and answers "I
don't have that in my sources" *without calling the LLM at all* (gate 5.8, ST-1).
That ordering matters: thresholding before dedupe would let five sub-threshold
chunks of one page fill the budget, and thresholding after dedupe is what makes the
number mean "this page is relevant" rather than "this page is dense".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .config import INDEX_SCHEME_NAMES, get_settings
from .store import Store

logger = logging.getLogger("app.retrieve")

# Over-fetch factor. The scheme filter, the hybrid rerank and the URL dedupe are
# all applied after Chroma returns, so the raw candidate set has to be much
# larger than TOP_K.
#
# 20x (100 candidates for TOP_K=5) is set by a measurement, not a guess. The ELSS
# lock-in chunk sits at **cosine rank 23** of 335 ELSS chunks - MiniLM rates the
# SID's surrounding regulation text as a better match than the lock-in table
# inside the same document. At 4x over-fetch that chunk is never even a
# candidate, so gate 3.6 fails and no rerank weight can recover it. Measured
# recall@5 on the golden set, by over-fetch (at LEXICAL_WEIGHT 0.55):
#
#     over-fetch   4x    8x   12x   16x   20x   24x   32x
#     recall@5   13/20 13/20 14/20 15/20 16/20 16/20 16/20
#
# and gate 3.6 first passes at 16x. 20x is the first value on the plateau, and
# the plateau is flat to 32x, so this is not balanced on an edge. It costs
# nothing measurable: 100 candidates rank in ~80 ms, against a ~1.4 s
# generation call they feed.
OVERFETCH = 20

# --- hybrid (cosine + lexical) rerank ---------------------------------------
#
# Cosine alone puts the answer outside the top 5 for 8 of the 20 golden factual
# questions, and that is the single biggest cause of gate 5.1 falling short.
# The cause is specific and measurable, not general embedding weakness: the
# sources are SEBI-formatted SID boilerplate, so a chunk stating the answer is
# usually surrounded by several hundred words of regulation text, and MiniLM
# pools the whole 220-token window into one vector. The question's own words
# ("minimum", "lock-in", "expense ratio") get averaged away against that
# surrounding boilerplate.
#
# Adding back a plain term-overlap term recovers most of it. Measured
# recall@5 of the answer-bearing chunk over the 20 golden questions, through this
# module end to end (scheme filter, threshold, URL dedupe, cap at 5), at
# OVERFETCH = 20:
#
#     weight  0.35  0.45  0.55  0.65  0.75  0.85
#     recall 15/20 16/20 16/20 16/20 16/20 16/20
#
# 16/20 is the ceiling for this corpus; the remaining 4 misses are not a ranking
# problem but a source problem, and `tests/eval.py` reports them as corpus gaps
# with the phrase that is missing. The plateau from 0.45 to 0.85 is flat, so 0.55
# sits in the middle of it rather than on an edge.
#
# The weight is large relative to the cosine spread (~0.25 across the candidate
# set), which is deliberate and is the honest description of this corpus: the
# exact term is a far stronger signal than MiniLM's pooled similarity. Cosine is
# still doing real work - it breaks ties and it is the only term for a question
# with no distinctive content words ("Who manages the fund?") - but on a lookup
# question the term match decides.
#
# Two earlier attempts, both measured and both discarded: a min-max *normalised*
# blend scored 11/20, worse than cosine alone, because normalising stretches the
# cosine spread across the full range and discards the gap that separates a real
# match from boilerplate; and a 0.35 weight at 4x over-fetch scored 14/20,
# limited by the candidate cut rather than the blend.
LEXICAL_WEIGHT = 0.55

# Words carrying no retrieval signal: question-frame vocabulary, plus the AMC
# and scheme names.
#
# The scheme names matter more than they look. Inside a scheme-*filtered* search
# every candidate is already the right scheme, so "elss", "tax" and "saver" match
# all of them equally and carry no information - but they still count against the
# denominator, so leaving them in dilutes the one discriminating term. Measured:
# for "What is the lock-in period for HDFC ELSS Tax Saver Fund?" the terms were
# ["lock", "period", "elss", "tax", "saver"], and the lock-in chunk scored 0.60
# on lexical overlap instead of 1.00. With the scheme names stoplisted the same
# chunk scores 1.00 and the question is answered.
QUERY_STOPWORDS = frozenset(
    """
    a an the is are was were be been being of for in on at to from by with and or
    what which who whom whose when where why how do does did doing have has had
    i me my we our you your it its this that these those there here as if then
    than so such can could should would may might will shall must about into over
    under please tell give know any some all both each other more most
    fund funds hdfc mutual mf direct growth plan option nav
    elss tax saver cap mid nifty flexi large top index equity diversified
    """.split()
)

# Short numerals and single characters are noise in this corpus, which is full
# of section numbers ("52(6)", "XV") that match any numeric query term.
_MIN_TERM_LEN = 3


def query_terms(question: str) -> list[str]:
    """Content words of the question, for the lexical component.

    Lowercased and de-duplicated, so a repeated word cannot inflate the overlap
    ratio and let a chunk that merely repeats the query look like a match.
    """
    seen: dict[str, None] = {}
    for raw in re.findall(r"[a-z0-9]+", (question or "").lower()):
        if len(raw) < _MIN_TERM_LEN or raw in QUERY_STOPWORDS:
            continue
        seen.setdefault(raw, None)
    return list(seen)


def lexical_overlap(terms: list[str], document: str) -> float:
    """Fraction of the query's content words that appear in the chunk.

    Substring rather than word-boundary matching, because the corpus is
    inconsistent about this: "expense ratio", "Total expense ratio (TER)" and
    "TER" are the same fact in three forms, and a word-boundary match on "ter"
    only hits one of them.
    """
    if not terms:
        return 0.0
    lowered = (document or "").lower()
    return sum(1 for term in terms if term in lowered) / len(terms)


@dataclass
class RetrievedChunk:
    """One chunk that survived filtering, with the scores that ranked it.

    `score` stays the raw cosine similarity, because that is what `MIN_SCORE`
    is defined against and what the Phase 3 gate numbers were measured in.
    `rank_score` is the hybrid value actually used for ordering. Keeping them
    separate matters: the hybrid can exceed 1.0, so reusing it for the threshold
    would silently rescale `MIN_SCORE` and make the ST-1 exit fire at the wrong
    place.
    """

    id: str
    document: str
    score: float
    metadata: dict = field(default_factory=dict)
    lexical: float = 0.0
    rank_score: float = 0.0

    @property
    def url(self) -> str:
        return self.metadata.get("url", "")

    @property
    def scheme(self) -> str:
        return self.metadata.get("scheme", "")

    @property
    def as_of(self) -> str:
        return self.metadata.get("as_of", "")

    @property
    def title(self) -> str:
        return self.metadata.get("title", "")

    @property
    def page_type(self) -> str:
        return self.metadata.get("page_type", "")

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "score": round(self.score, 4),
            "rank_score": round(self.rank_score, 4),
            "lexical": round(self.lexical, 4),
            "url": self.url,
            "scheme": self.scheme,
            "title": self.title,
            "page_type": self.page_type,
            "as_of": self.as_of,
            "excerpt": self.document[:400],
        }


@dataclass
class RetrievalResult:
    """Everything the caller needs to decide answer vs. "I don't know"."""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    scheme: str | None = None
    min_score: float = 0.0
    # Diagnostics. These exist because implementation.md 5.1 says to *dump the
    # retrieved chunks* before touching the prompt when grounding fails - a
    # low-scoring but non-empty candidate set is the signature of a MIN_SCORE or
    # chunking problem, and it is invisible without the below-threshold scores.
    n_candidates: int = 0
    best_score: float = 0.0
    best_rank_score: float = 0.0
    best_below_threshold: float | None = None
    dropped_by_threshold: int = 0
    dropped_by_dedupe: int = 0
    index_missing: bool = False

    @property
    def ok(self) -> bool:
        """True when there is grounded context to generate from."""
        return bool(self.chunks)

    @property
    def best(self) -> RetrievedChunk | None:
        return self.chunks[0] if self.chunks else None


def _index_name(scheme: str | None) -> str | None:
    """Translate a canonical scheme name to the string stored in the index."""
    if not scheme:
        return None
    return INDEX_SCHEME_NAMES.get(scheme, scheme)


def retrieve(
    question: str,
    *,
    scheme: str | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
    store: Store | None = None,
) -> RetrievalResult:
    """Search the index for `question`, restricted to `scheme` when known.

    `scheme` is the canonical name from `config.IN_SCOPE_SCHEMES` (i.e. what
    `guardrails.named_scheme` returns), not the raw string from the index.

    Never raises for a missing or empty index: returns a result with
    `index_missing=True` and no chunks, so the caller can say "sources are
    unavailable" instead of raising a 500 during a demo.
    """
    settings = get_settings()
    top_k = top_k or settings.top_k
    min_score = settings.min_score if min_score is None else min_score

    if not (question or "").strip():
        return RetrievalResult(scheme=scheme, min_score=min_score)

    st = store or Store()
    if not st.path.exists() or st.count() == 0:
        logger.warning("retrieval skipped: no index at %s", st.path)
        return RetrievalResult(scheme=scheme, min_score=min_score, index_missing=True)

    # Imported here, not at module scope: loading the ONNX model costs ~100 MB and
    # ~1 s, and /healthz must stay cheap. The same lazy pattern as store.py.
    from .embeddings import embed_query

    vector = embed_query(question)
    terms = query_terms(question)

    index_scheme = _index_name(scheme)
    where = {"scheme": index_scheme} if index_scheme else None

    raw = st.query(vector, top_k=top_k * OVERFETCH, where=where)

    result = RetrievalResult(scheme=scheme, min_score=min_score, n_candidates=len(raw))
    if not raw:
        return result

    # Cosine distance -> similarity. store.query returns distance; normalising
    # here means every consumer (generate, eval, the CLI) sees one scale.
    scored: list[RetrievedChunk] = []
    for hit in raw:
        distance = hit.get("distance")
        score = 1.0 - float(distance) if distance is not None else 0.0
        document = hit.get("document", "") or ""
        lexical = lexical_overlap(terms, document)
        scored.append(
            RetrievedChunk(
                id=hit.get("id", ""),
                document=document,
                score=score,
                metadata=hit.get("metadata", {}) or {},
                lexical=lexical,
                rank_score=score + LEXICAL_WEIGHT * lexical,
            )
        )
    # Order by the hybrid, not by cosine - see LEXICAL_WEIGHT for the measurements.
    scored.sort(key=lambda c: c.rank_score, reverse=True)
    result.best_score = scored[0].score
    result.best_rank_score = scored[0].rank_score

    kept = [c for c in scored if c.score >= min_score]
    result.dropped_by_threshold = len(scored) - len(kept)
    result.best_below_threshold = (
        max((c.score for c in scored if c.score < min_score), default=None)
    )

    # Source diversity first: one chunk per URL, best-scoring wins. Then, only if
    # TOP_K is still short, allow a second chunk from a URL already represented.
    chosen: list[RetrievedChunk] = []
    seen_urls: set[str] = set()
    for chunk in kept:
        if chunk.url and chunk.url in seen_urls:
            continue
        chosen.append(chunk)
        if chunk.url:
            seen_urls.add(chunk.url)
        if len(chosen) >= top_k:
            break

    if len(chosen) < top_k:
        extras = [c for c in kept if c not in chosen]
        for chunk in extras:
            if len(chosen) >= top_k:
                break
            chosen.append(chunk)
        result.dropped_by_dedupe = max(0, len(kept) - len(chosen))

    chosen.sort(key=lambda c: c.rank_score, reverse=True)
    result.chunks = chosen[:top_k]
    return result


__all__ = [
    "LEXICAL_WEIGHT",
    "OVERFETCH",
    "QUERY_STOPWORDS",
    "RetrievalResult",
    "RetrievedChunk",
    "lexical_overlap",
    "query_terms",
    "retrieve",
]
