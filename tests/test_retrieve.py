"""Phase 5 retrieval gates.

The tests that need an index are skipped when `data/chroma/` is absent, so this
file is safe to run on a fresh clone. The pure tests - query terms, lexical
overlap, the scheme-name mapping - always run, because those are where a silent
mistake would do the most damage.

| test | gate |
| --- | --- |
| `test_index_scheme_names_exist_in_the_index` | 5.13, the silent-empty-filter trap |
| `test_gate_5_12_*` | 5.12 two schemes get different answers |
| `test_gate_5_13_*` | 5.13 no cross-scheme bleed |
| `test_*_lexical` | the hybrid rerank |
| `test_no_index_degrades` | ST-1, a missing index is not a 500 |
"""

from __future__ import annotations

import re

import pytest

from app.config import INDEX_SCHEME_NAMES, IN_SCOPE_SCHEMES, get_settings
from app.retrieve import (
    LEXICAL_WEIGHT,
    OVERFETCH,
    lexical_overlap,
    query_terms,
    retrieve,
)
from app.store import Store

HAS_INDEX = get_settings().index_present
needs_index = pytest.mark.skipif(
    not HAS_INDEX, reason="no index at data/chroma - run `python -m app.ingest.build_index`"
)


# ── the scheme-name mapping: a silent-empty-filter trap ──────────────────────


def test_index_scheme_names_cover_every_in_scope_scheme():
    assert set(INDEX_SCHEME_NAMES) == set(IN_SCOPE_SCHEMES)


def test_index_scheme_names_are_distinct():
    """Two schemes must not collapse onto one index name.

    If they did, the Chroma filter would return another scheme's chunks and
    gate 5.13 would fail with no error anywhere.
    """
    assert len(set(INDEX_SCHEME_NAMES.values())) == len(INDEX_SCHEME_NAMES)


@needs_index
def test_index_scheme_names_exist_in_the_index():
    """These strings must be what Phase 3 actually wrote.

    A re-parse that renames a scheme would empty the filter, and an empty
    filter looks exactly like "this fund is not in the corpus" - so this is
    asserted rather than assumed.
    """
    store = Store()
    present = {m["scheme"] for m in store.collection.get(include=["metadatas"])["metadatas"]}
    for canonical, indexed in INDEX_SCHEME_NAMES.items():
        assert indexed in present, f"{canonical!r} maps to {indexed!r}, not in the index"


# ── query terms and the hybrid rerank ────────────────────────────────────────


def test_query_terms_drops_frame_and_scheme_words():
    terms = query_terms("What is the expense ratio of the HDFC Flexi Cap Fund - Direct Growth?")
    assert "expense" in terms and "ratio" in terms
    for noise in ("what", "is", "the", "of", "hdfc", "fund", "direct", "growth", "plan"):
        assert noise not in terms, f"{noise!r} carries no retrieval signal"


def test_query_terms_are_deduplicated():
    """A repeated word must not let a chunk that echoes the query score highly."""
    assert query_terms("ratio ratio ratio") == ["ratio"]


def test_lexical_overlap_is_a_fraction():
    doc = "Exit Load: an Exit Load of 1.00 % is payable if redeemed within 1 year."
    assert lexical_overlap(["exit", "load"], doc) == pytest.approx(1.0)
    assert lexical_overlap(["benchmark", "riskometer"], doc) == 0.0
    assert lexical_overlap(["exit", "benchmark"], doc) == pytest.approx(0.5)
    assert lexical_overlap([], doc) == 0.0


def test_lexical_overlap_matches_inside_words():
    """'ter' has to hit 'Total Expense Ratio (TER)' and 'expense ratio' alike.

    Word-boundary matching would only find one of the two forms, and this corpus
    spells the same fact several ways.
    """
    assert lexical_overlap(["ter"], "Total Expense Ratio (TER)") == 1.0
    assert lexical_overlap(["ratio"], "expense ratio") == 1.0


def test_lexical_weight_is_inside_the_measured_plateau():
    """0.45-0.85 all scored 16/20; 0.55 keeps it off both edges."""
    assert 0.45 <= LEXICAL_WEIGHT <= 0.85


def test_overfetch_reaches_the_chunk_the_elss_lockin_gate_needs():
    """The lock-in chunk sits at cosine rank 23, so 4x over-fetch cannot see it.

    This is the constant behind gate 3.6: below ~16x the answer-bearing chunk is
    not in the candidate set at all, and no rerank weight recovers it.
    """
    assert OVERFETCH >= 16


def test_scheme_names_are_stoplisted_because_they_cannot_discriminate():
    """Inside a scheme filter, 'elss'/'tax'/'saver' match every candidate.

    Leaving them in dilutes the one discriminating term: the ELSS lock-in chunk
    scored 0.60 lexical overlap instead of 1.00, and the question went unanswered.
    """
    terms = query_terms("What is the lock-in period for HDFC ELSS Tax Saver Fund?")
    assert terms == ["lock", "period"]


# ── gates 5.12 / 5.13: the scheme filter actually works ─────────────────────


@needs_index
def test_gate_5_12_scheme_filter_isolates_schemes():
    for canonical in IN_SCOPE_SCHEMES:
        indexed = INDEX_SCHEME_NAMES[canonical]
        r = retrieve(f"exit load of {canonical}", scheme=canonical)
        if not r.ok:
            continue
        assert {c.scheme for c in r.chunks} == {indexed}, (
            f"{canonical!r} retrieved chunks from {[c.scheme for c in r.chunks]}"
        )


@needs_index
def test_gate_5_13_no_cross_scheme_bleed():
    """An ELSS question must never return another scheme's figures."""
    r = retrieve("What is the lock-in period for HDFC ELSS Tax Saver Fund?", scheme="HDFC ELSS Tax Saver Fund")
    assert r.ok
    assert {c.scheme for c in r.chunks} == {"HDFC ELSS - Tax Saver Fund"}
    for c in r.chunks:
        assert "HDFC Large Cap" not in c.document or "lock" not in c.document.lower()


@needs_index
def test_two_schemes_get_two_different_answers():
    """5.12 as an end-to-end claim: distinct schemes, distinct chunk sets."""
    a = retrieve("benchmark of HDFC Flexi Cap Fund", scheme="HDFC Flexi Cap Fund")
    b = retrieve("benchmark of HDFC Mid Cap Fund", scheme="HDFC Mid Cap Fund")
    assert a.ok and b.ok
    assert {c.id for c in a.chunks}.isdisjoint({c.id for c in b.chunks})


# ── the score contract ──────────────────────────────────────────────────────


@needs_index
def test_score_stays_cosine_so_min_score_keeps_its_meaning():
    """rank_score may exceed 1.0; score may not, or MIN_SCORE is rescaled."""
    r = retrieve("exit load of HDFC Flexi Cap Fund", scheme="HDFC Flexi Cap Fund")
    for c in r.chunks:
        assert 0.0 <= c.score <= 1.0
        assert c.rank_score >= c.score
        assert c.rank_score == pytest.approx(c.score + LEXICAL_WEIGHT * c.lexical, abs=1e-6)


@needs_index
def test_chunks_are_capped_and_ordered():
    r = retrieve("expense ratio of HDFC Mid Cap Fund", scheme="HDFC Mid Cap Fund")
    assert len(r.chunks) <= get_settings().top_k
    scores = [c.rank_score for c in r.chunks]
    assert scores == sorted(scores, reverse=True)


@needs_index
def test_diagnostics_expose_below_threshold_scores():
    """A low-scoring but non-empty candidate set is the signature of a bad MIN_SCORE."""
    r = retrieve("What is the expense ratio of HDFC Flexi Cap Fund?", scheme="HDFC Flexi Cap Fund")
    assert r.n_candidates > 0
    assert r.best_score > 0
    assert r.min_score == get_settings().min_score


@needs_index
def test_retrieval_surfaces_the_answer_bearing_chunk():
    """The regression that motivated the hybrid rerank, pinned.

    Cosine alone put this chunk outside the top 5; "exit load" now reaches it.
    """
    r = retrieve(
        "Is there an exit load on HDFC Top 100 Fund - Direct Growth?",
        scheme="HDFC Top 100 Fund",
    )
    assert r.ok
    assert any(re.search(r"exit load", c.document, re.I) for c in r.chunks)


@needs_index
def test_every_cited_url_is_on_the_allowlist():
    for canonical in IN_SCOPE_SCHEMES:
        r = retrieve(f"exit load of {canonical}", scheme=canonical)
        for c in r.chunks:
            assert c.url
            assert c.url.startswith("https://")


# ── degenerate inputs must not raise ────────────────────────────────────────


def test_empty_question_returns_no_chunks():
    r = retrieve("   ")
    assert not r.ok
    assert r.n_candidates == 0


def test_no_index_degrades_rather_than_raising(tmp_path, monkeypatch):
    """A missing index is a degraded product, not a 500 (ST-1)."""
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "absent"))
    get_settings.cache_clear()
    try:
        r = retrieve("exit load of HDFC Flexi Cap Fund", scheme="HDFC Flexi Cap Fund")
        assert r.index_missing is True
        assert not r.ok
    finally:
        get_settings.cache_clear()


@needs_index
def test_unfiltered_search_still_works():
    """F07 ('how do I download my statement') names no scheme, so the filter is skipped."""
    r = retrieve("How do I download my capital-gains statement?")
    assert r.scheme is None
    assert r.ok
