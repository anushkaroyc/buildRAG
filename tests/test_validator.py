"""Phase 5 validator and grounding gates.

Covers the checks in implementation.md that the answer text itself must pass:
sentence count, exactly one link, the freshness stamp, and the return-word scan -
plus the grounding rule (ST-1) and the degradation contract (ST-2).

**Every test here is offline and spends no tokens.** `validate` and `is_grounded`
are pure functions by design, and the autouse fixture blocks sockets so that any
accidental network call inside them fails loudly instead of quietly costing
money. The tests that need to cover `generate.answer` monkeypatch `call_groq`
instead, which is also why the model call is a single module-level function
rather than being inlined in `answer`.

| test | gate |
| --- | --- |
| `test_gate_5_6_sentence_count` | SC-7, at most 3 sentences |
| `test_gate_5_7_single_citation` | SC-2, exactly one link, from metadata |
| `test_gate_5_8_no_generation_without_context` | ST-1, 5.8 |
| `test_gate_5_9_no_return_words` | SC-9 |
| `test_gate_5_10_freshness_stamp` | SC-8 |
| `test_gate_5_11_degrades_without_network` | ST-2 |
"""

from __future__ import annotations

import socket

import pytest

from app import generate, prompts
from app.config import INDEX_SCHEME_NAMES
from app.generate import (
    LONG_ANSWER_TERMS,
    MIN_TERM_OVERLAP,
    build_context,
    call_groq,
    citation_for,
    content_terms,
    count_sentences,
    degraded,
    freshness_stamp,
    is_grounded,
    qualifiers_in,
    validate,
)
from app.retrieve import RetrievalResult, RetrievedChunk


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """No sockets: the validator is pure, and tests must not spend tokens."""

    def _blocked(*args, **kwargs):
        raise AssertionError("the validator must not touch the network in tests")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


def chunk(text: str, **meta) -> RetrievedChunk:
    """A RetrievedChunk with safe defaults, so each test states only what matters."""
    base = {
        "url": "https://www.hdfcfund.com/faqs",
        "title": "HDFC Flexi Cap Fund - Key Information Memorandum",
        "scheme": "HDFC Flexi Cap Fund",
        "page_type": "kim",
        "as_of": "2025-05-30",
    }
    base.update(meta)
    return RetrievedChunk(
        id="c1", document=text, score=0.7, metadata=base, lexical=0.5, rank_score=0.9
    )


def result(*chunks: RetrievedChunk, **kwargs) -> RetrievalResult:
    return RetrievalResult(chunks=list(chunks), **kwargs)


# ── gate 5.6 / SC-7: sentence count ─────────────────────────────────────────


def test_gate_5_6_sentence_count():
    assert count_sentences("") == 0
    assert count_sentences("One sentence.") == 1
    assert count_sentences("One. Two. Three.") == 3
    # A period inside a decimal or an abbreviation must not split the sentence,
    # or a correct 2-sentence answer gets rejected for being 4 sentences long.
    assert count_sentences("The expense ratio is 1.35 % p.a.") == 1
    assert count_sentences("Direct Growth costs 0.77 % versus Regular at 1.40 %.") == 1
    assert count_sentences("No. The scheme has no exit load.") == 2

    assert validate("A valid one sentence answer with 1.00 %.").ok
    assert not validate("One. Two. Three. Four sentences here now.").ok
    assert any("too long" in p for p in validate("A. B. C. D.").problems)


# ── gate 5.7 / SC-2 + SC-3: exactly one citation, from metadata ─────────────


def test_gate_5_7_single_citation():
    # The model must never write the link. A URL in the answer means it invented
    # or duplicated a source, so the answer is rejected rather than shipped.
    for text in (
        "The expense ratio is 1.35 %. https://example.com",
        "See [the factsheet](https://files.hdfcfund.com/x.pdf) for 1.35 %.",
    ):
        verdict = validate(text)
        assert not verdict.ok
        assert any("citation" in p or "link" in p for p in verdict.problems)

    c = citation_for(chunk("irrelevant"))
    assert c["url"] == "https://www.hdfcfund.com/faqs"
    assert c["domain_ok"] is True
    assert c["host"] == "www.hdfcfund.com"


def test_gate_5_7_citation_flags_a_non_allowlisted_domain():
    """SC-3 is checked, not assumed: a re-crawl that adds a source must show up."""
    bad = citation_for(chunk("x", url="https://random-blog.example.com/fund-facts"))
    assert bad["domain_ok"] is False
    assert prompts.is_allowlisted(bad["url"]) is False


def test_answer_carries_exactly_one_citation(monkeypatch):
    seen: dict = {}

    def fake_call(messages, max_tokens):
        seen["messages"] = messages
        return "The exit load is 1.00 % within one year.", {
            "completion_tokens": 20,
            "total_tokens": 200,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(generate, "call_groq", fake_call)
    monkeypatch.setattr(generate, "get_settings", lambda: type("S", (), {"groq_configured": True})())

    res = generate.answer(result(chunk("Exit Load: 1.00 % if redeemed within 1 year.")), "exit load?")
    assert res["intent"] == "answered"
    assert len(res["citations"]) == 1
    # The user message carries the chunk text but no URL the model could echo.
    assert "hdfcfund.com" not in seen["messages"][-1]["content"]


# ── gate 5.8 / ST-1: no generation without supporting context ───────────────


def test_gate_5_8_no_generation_without_context(monkeypatch):
    """Nothing above MIN_SCORE means zero API calls, not a hopeful guess."""
    monkeypatch.setattr(
        generate,
        "call_groq",
        lambda *a, **k: pytest.fail("the model must not be called without context"),
    )
    res = generate.answer(result(n_candidates=20, min_score=0.35), "anything")
    assert res["intent"] in ("no_context", "unsupported")
    assert res["citations"] == []
    assert res["answer"] == prompts.NO_CONTEXT_REPLY


def test_grounding_rejects_a_fabricated_figure():
    ctx = "Exit Load: 1.00 % if redeemed within 1 year of allotment."
    assert is_grounded("The exit load is 1.00 % within one year.", ctx)[0] is True
    # The classic failure: a plausible number that is nowhere in the sources.
    ok, why = is_grounded("The expense ratio is 1.87 % p.a.", ctx)
    assert ok is False
    assert "1.87" in why


def test_grounding_accepts_a_name_only_answer():
    """A correct answer may contain no digits at all.

    This is why the grounding check is not "does the answer contain a number":
    that proxy rejected the correct answer to "who is the fund manager?".
    """
    ctx = "The Fund is managed by Dhruv Muchhal, Fund Manager, since 2013."
    assert is_grounded("Dhruv Muchhal.", ctx)[0] is True


def test_grounding_rejects_fluent_ungrounded_prose():
    ctx = "Exit Load: 1.00 % if redeemed within 1 year of allotment."
    ok, why = is_grounded(
        "This scheme is widely considered a strong performer with consistent returns.",
        ctx,
    )
    assert ok is False
    assert "content words" in why


def test_grounding_is_vacuous_without_numbers():
    """A figure that appears in the context is a supported figure, ceiling or not.

    "The ceiling is 2.25 %" paraphrases a source that says "Maximum Total
    Expense Ratio under Regulation 52(6)", so it scores 0% on word overlap. A
    short answer is held to a lower bar precisely so that this is not refused;
    the number check above is what carries it.
    """
    ctx = "Maximum Total Expense Ratio under Regulation 52(6): 2.25 % p.a."
    assert is_grounded("The ceiling is 2.25 %.", ctx)[0] is True


def test_term_overlap_threshold_is_reachable_but_not_trivial():
    assert 0.0 < MIN_TERM_OVERLAP < 1.0
    # A short answer is not held to MIN_TERM_OVERLAP; a long one is.
    assert content_terms("the scheme's ongoing charge is 1.35 %") == ["ongoing", "charge"]
    long_answer = "This fund is widely regarded as a strong and consistent performer"
    assert len(content_terms(long_answer)) >= LONG_ANSWER_TERMS
    # Scheme/plan words are kept here, unlike in retrieve: an answer should be
    # allowed to name its own scheme.
    assert "hdfc" in content_terms("HDFC Flexi Cap Fund")


# ── gate 5.9 / SC-9: no return or comparison claims ─────────────────────────


def test_gate_5_9_no_return_words():
    for text in (
        "The fund outperformed its benchmark in 2024 with 24 % returns.",
        "HDFC Flexi Cap is the best performing fund and is safe for you.",
        "You should buy this scheme now; it is worth buying.",
        "I would recommend the Mid Cap Fund for a long horizon.",
        "The scheme guarantees 12 % annual returns.",
    ):
        verdict = validate(text)
        assert not verdict.ok, f"should have been rejected: {text!r}"
        assert verdict.problems


def test_gate_5_9_allows_a_documented_exit_load_return():
    """'Return' in 'exit load return' is a fact, not a performance claim.

    The scan is a banned-phrase list, not a banned word, precisely so that a
    legitimate disclosure is not refused.
    """
    assert validate("The exit load return is nil after 1 year, as 1.00 % applies before.") is not None


def test_gate_5_9_rejects_markdown_and_lists():
    assert not validate("- expense ratio: 1.35 %\n- exit load: 1.00 %").ok
    assert not validate("## Fees\nThe expense ratio is 1.35 %.").ok


# ── gate 5.10 / SC-8: the freshness stamp ───────────────────────────────────


def test_gate_5_10_freshness_stamp():
    assert freshness_stamp(chunk("x", as_of="2026-03-31")) == (
        "Last updated from sources: 2026-03-31"
    )
    # No date in the metadata means no stamp, never a guess.
    assert freshness_stamp(chunk("x", as_of="")) == ""


def test_gate_5_10_stamp_matches_the_cited_document(monkeypatch):
    """The stamp must not claim a newer source than the one actually linked.

    A maximum over all retrieved chunks is wrong here: with exactly one citation
    the two dates have to be the same number, or the reader is told a Nov-2024
    page is more current than a Jun-2026 one.
    """
    monkeypatch.setattr(
        generate,
        "call_groq",
        lambda m, max_tokens: ("The exit load is 1.00 % within one year.", {"finish_reason": "stop"}),
    )
    monkeypatch.setattr(generate, "get_settings", lambda: type("S", (), {"groq_configured": True})())

    old = chunk("Exit Load 1.00 %.", as_of="2024-11-21", url="https://www.sebi.gov.in/a.pdf")
    new = chunk("Exit Load 1.00 %.", as_of="2026-06-30", url="https://files.hdfcfund.com/b.pdf")
    res = generate.answer(result(old, new), "exit load?")

    assert res["citations"][0]["as_of"] == "2024-11-21"
    assert "Last updated from sources: 2024-11-21" in res["answer"]
    assert "2026-06-30" not in res["answer"]


# ── gate 5.11 / ST-2: degradation ───────────────────────────────────────────


def test_gate_5_11_degrades_without_network(monkeypatch):
    def boom(messages, max_tokens):
        raise RuntimeError("Connection error.")

    monkeypatch.setattr(generate, "call_groq", boom)
    monkeypatch.setattr(generate, "get_settings", lambda: type("S", (), {"groq_configured": True})())

    res = generate.answer(result(chunk("Exit Load 1.00 %.")), "exit load?")
    assert res["intent"] == "degraded"
    assert res["answer"] == prompts.DEGRADED_REPLY
    assert res["citations"] == []
    assert "RuntimeError" in res["degraded_reason"]


def test_degrades_on_a_missing_key(monkeypatch):
    monkeypatch.setattr(generate, "get_settings", lambda: type("S", (), {"groq_configured": False})())
    monkeypatch.setattr(generate, "call_groq", lambda *a, **k: pytest.fail("no key, no call"))
    res = generate.answer(result(chunk("Exit Load 1.00 %.")), "exit load?")
    assert res["intent"] == "degraded"
    assert "no API key" in res["degraded_reason"]


def test_degraded_payload_carries_no_figures():
    """A degraded reply must not leak a number, even in the reason field."""
    payload = degraded("HTTP 429 after 3 retries")
    assert payload["answer"] == prompts.DEGRADED_REPLY
    assert not any(ch.isdigit() for ch in payload["answer"])


def test_empty_completion_is_treated_as_failure(monkeypatch):
    """finish_reason=length with empty content means reasoning ate the budget."""

    class _Msg:
        content = ""

    class _Choice:
        message = _Msg()
        finish_reason = "length"

    class _Usage:
        prompt_tokens = 100
        completion_tokens = 180
        total_tokens = 280

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()

    import sys
    import types

    fake = types.ModuleType("groq")

    class _Client:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **k: _Resp())
            )

    fake.Groq = _Client
    monkeypatch.setitem(sys.modules, "groq", fake)
    monkeypatch.setattr(
        generate,
        "get_settings",
        lambda: type(
            "S",
            (),
            {"groq_api_key": "x", "groq_model": "openai/gpt-oss-20b", "guard_timeout_s": 2},
        )(),
    )

    with pytest.raises(RuntimeError, match="empty completion"):
        call_groq([{"role": "user", "content": "hi"}], 180)


# ── the context block: scheme renames must reach the model ──────────────────


def test_build_context_names_both_sides_of_a_scheme_rename():
    """HDFC Top 100 Fund is HDFC Large Cap Fund in the sources.

    The model refuses to answer across that gap, so retrieval's mapping has to
    travel into the prompt too - otherwise the knowledge stops at the Chroma
    filter and the answer is lost.
    """
    ctx = build_context([chunk("Exit Load 1.00 %.", scheme="HDFC Large Cap Fund")])
    assert "HDFC Large Cap Fund" in ctx
    assert "HDFC Top 100 Fund" in ctx
    assert "same scheme" in ctx


def test_build_context_does_not_invent_an_alias_for_unrenamed_schemes():
    ctx = build_context([chunk("x", scheme="HDFC Flexi Cap Fund")])
    assert "also called" not in ctx
    assert INDEX_SCHEME_NAMES["HDFC Flexi Cap Fund"] == "HDFC Flexi Cap Fund"


def test_build_context_labels_every_source_with_its_date():
    ctx = build_context(
        [
            chunk("a", as_of="2026-03-31", page_type="factsheet"),
            chunk("b", as_of="2025-05-30", page_type="sid"),
        ]
    )
    assert "source 1" in ctx and "source 2" in ctx
    assert "as of 2026-03-31" in ctx and "as of 2025-05-30" in ctx
    assert "document=factsheet" in ctx


# ── qualifier misattribution: the failures that got past the number check ───


def test_gate_5_12_rejects_a_regular_figure_reported_as_direct():
    """Live failure: "0.41 % per annum" for a *Direct Growth* question.

    The source says "Total Expense Ratio Regular - 0.41 % p.a." and, in the same
    breath, that Direct "shall have a lower expense ratio than Regular Plan".
    0.41 really is in the cited chunk, so the number check passes; only reading
    the label in front of the number catches it.
    """
    ctx = (
        "Total Expense Ratio Regular – 0.41 % p.a. Direct Plan under the Scheme "
        "shall have a lower expense ratio than Regular Plan."
    )
    q = "What is the expense ratio of HDFC Nifty 50 Index Fund - Direct Growth?"
    ok, why = is_grounded("The expense ratio is 0.41 % per annum.", ctx, q)
    assert ok is False
    assert "direct" in why


def test_gate_5_12_rejects_a_redemption_minimum_reported_as_sip():
    """Live failure: "Rs. 100" is the redemption minimum, not the SIP minimum."""
    ctx = (
        "minimum amount /units for redemption / switch-out of Units under each "
        "plan / option would be Rs. 100 and multiples thereof."
    )
    q = "What is the minimum SIP amount for HDFC Mid Cap Fund - Direct Growth?"
    ok, why = is_grounded("Minimum SIP amount is Rs. 100.", ctx, q)
    assert ok is False
    assert "100" in why


def test_gate_5_12_accepts_a_correct_figure_from_a_multi_plan_table():
    """The one-directional rule must not reject sequential plan listings.

    A symmetric window around 0.35 % in "Direct Growth: 0.35 % p.a. Regular:
    1.40 % p.a." reaches the neighbouring word "Regular" and used to reject a
    correct answer.
    """
    ctx = "Direct Growth: expense ratio 0.35 % p.a. Regular: 1.40 % p.a."
    assert is_grounded(
        "The Direct Growth expense ratio is 0.35 %.",
        ctx,
        "expense ratio of HDFC Nifty 50 Index Fund - Direct Growth?",
    )[0] is True
    assert is_grounded(
        "The Regular expense ratio is 1.40 %.",
        ctx,
        "expense ratio of HDFC Nifty 50 Index Fund - Regular Plan?",
    )[0] is True


def test_qualifier_check_is_skipped_when_the_question_names_none():
    """Most questions name no plan, and those must not be second-guessed."""
    assert is_grounded("The exit load is 1.00 %.", "Exit Load: 1.00 % if in 1 year.")[0] is True
    assert qualifiers_in("is there an exit load?") == frozenset()
    assert "direct" in qualifiers_in("HDFC Flexi Cap Fund - Direct Growth")


def test_qualifier_check_borrows_nothing_across_families():
    """'growth' must not be satisfied by the word 'direct'."""
    ctx = "Direct Growth: 0.35 % p.a."
    ok, _ = is_grounded("The IDCW option is 0.35 %.", ctx, "expense ratio - IDCW option?")
    assert ok is False


# ── the NOT IN CONTEXT contract ─────────────────────────────────────────────


def test_not_in_context_is_recognised_in_any_casing():
    for text in ("NOT IN CONTEXT", "not in context", "  Not In Context  "):
        assert validate(text).said_not_in_context is True


def test_sentinel_never_reaches_the_user(monkeypatch):
    """The model may decline, but it does not get to phrase the refusal."""
    monkeypatch.setattr(
        generate, "call_groq", lambda m, max_tokens: ("NOT IN CONTEXT", {"finish_reason": "stop"})
    )
    monkeypatch.setattr(generate, "get_settings", lambda: type("S", (), {"groq_configured": True})())
    res = generate.answer(result(chunk("irrelevant text")), "unanswerable?")
    assert res["intent"] == "unsupported"
    assert res["answer"] == prompts.UNSUPPORTED_REPLY
    assert "NOT IN CONTEXT" not in res["answer"]


def test_repair_retry_runs_once_then_gives_up(monkeypatch):
    """A second bad answer is dropped, not shipped with a warning attached."""
    calls: list[int] = []

    def always_listy(messages, max_tokens):
        calls.append(1)
        return "- expense ratio 1.35 %\n- exit load 1.00 %", {"finish_reason": "stop"}

    monkeypatch.setattr(generate, "call_groq", always_listy)
    monkeypatch.setattr(generate, "get_settings", lambda: type("S", (), {"groq_configured": True})())

    res = generate.answer(result(chunk("Exit Load 1.00 %.")), "expense ratio?")
    assert len(calls) == 2, "exactly one repair retry"
    assert res["intent"] == "unsupported"
    assert res["validator_problems"]
