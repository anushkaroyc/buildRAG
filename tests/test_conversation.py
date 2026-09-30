"""Tests for conversation memory and follow-up question rewriting."""
from __future__ import annotations

from app.conversation import HISTORY_MAXLEN, History, resolve_question, rewrite_question
from app.guardrails import evaluate, named_scheme


# -- the history window --------------------------------------------------------


def test_history_truncates_to_ten_messages():
    h = History()
    for i in range(12):
        h.append("user", f"Q{i}")
        h.append("assistant", f"A{i}")
    msgs = h.messages
    assert HISTORY_MAXLEN == 10
    assert len(msgs) == 10
    # 24 messages pushed through a maxlen-10 deque keeps the *last* 10.
    assert msgs[0].content == "Q7"
    assert msgs[-1].content == "A11"


def test_history_ignores_blank_messages():
    h = History()
    h.append("user", "   ")
    h.append("user", "real question")
    assert len(h) == 1
    assert h.last("user") == "real question"


def test_referent_is_the_scheme_without_the_plan_suffix():
    """`named_scheme` returns the scheme; the plan suffix is not part of it.

    Retrieval applies its own plan handling, and embedding the plan into the
    follow-up's lexical terms would pull in plan-specific boilerplate that does
    not answer a scheme-level question.
    """
    h = History()
    h.append("user", "What is the benchmark of HDFC Flexi Cap Fund - Direct Growth?")
    assert h.referent() == "HDFC Flexi Cap Fund"


def test_referent_is_none_when_no_in_scope_scheme_was_named():
    h = History()
    h.append("user", "What is the weather in Mumbai?")
    assert h.referent() is None


# -- rewriting ------------------------------------------------------------------


def test_rewrite_is_a_noop_when_the_question_names_a_scheme():
    h = History()
    h.append("user", "What is the benchmark of HDFC Flexi Cap Fund?")
    q = "What is the exit load of HDFC Mid Cap Fund - Direct Growth?"
    assert rewrite_question(q, h.messages) == q


def test_rewrite_resolves_the_its_case():
    h = History()
    h.append("user", "What is the benchmark of HDFC Flexi Cap Fund - Direct Growth?")
    h.append("assistant", "NIFTY 500 Index (TRI).")
    out = rewrite_question("What about its fees?", h.messages)
    assert "HDFC Flexi Cap Fund" in out
    assert "its" not in out.lower()
    assert named_scheme(out) == "HDFC Flexi Cap Fund"


def test_rewrite_resolves_that_fund():
    h = History()
    h.append("user", "Tell me about HDFC Nifty 50 Index Fund.")
    out = rewrite_question("What is the expense ratio for that fund?", h.messages)
    assert named_scheme(out) == "HDFC Nifty 50 Index Fund"
    assert "that fund" not in out.lower()


def test_rewrite_uses_the_most_recent_scheme():
    h = History()
    h.append("user", "Benchmark of HDFC Flexi Cap Fund?")
    h.append("assistant", "NIFTY 500.")
    h.append("user", "What is the exit load on HDFC Mid Cap Fund - Direct Growth?")
    out = rewrite_question("What about its NAV?", h.messages)
    assert named_scheme(out) == "HDFC Mid Cap Fund"


def test_rewrite_is_a_noop_without_a_referent_in_history():
    h = History()
    h.append("user", "Hello there")
    q = "What about its fees?"
    assert rewrite_question(q, h.messages) == q


def test_rewrite_is_a_noop_without_history_at_all():
    assert rewrite_question("What about its fees?") == "What about its fees?"


def test_rewrite_leaves_a_question_with_no_backreference_alone():
    h = History()
    h.append("user", "What is the benchmark of HDFC Flexi Cap Fund?")
    q = "Is the weather nice in Mumbai this week?"
    assert rewrite_question(q, h.messages) == q


def test_rewrite_does_not_redirect_a_question_that_names_another_fund():
    """"that other fund" names a competitor, so it must not resolve to ours.

    Rewriting it to the previously-discussed scheme would answer a question the
    user did not ask, and it would be answered from an in-scope source, so the
    out-of-scope guard would never see the competitor's name.
    """
    h = History()
    h.append("user", "What is the benchmark of HDFC Flexi Cap Fund?")
    q = "What is the exit load of Kotak Emerging Equity Fund?"
    assert rewrite_question(q, h.messages) == q


def test_rewrite_output_is_always_a_question_for_a_pronoun_only_followup():
    h = History()
    h.append("user", "Tell me about HDFC ELSS Tax Saver Fund.")
    out = rewrite_question("What about that?", h.messages)
    assert named_scheme(out) == "HDFC ELSS Tax Saver Fund"
    assert out.endswith("?")


# -- the guardrail gate on rewriting ------------------------------------------


def test_resolve_rescues_an_unsure_pronoun_followup():
    """The reason the feature exists: the guard refuses this as unsure."""
    h = History()
    h.append("user", "What is the benchmark of HDFC Flexi Cap Fund?")
    assert evaluate("What about its fees?").get("intent") == "unsure"
    asked, note = resolve_question("What about its fees?", h.messages)
    assert note
    assert named_scheme(asked) == "HDFC Flexi Cap Fund"


def test_resolve_does_not_let_a_rewrite_rescue_an_off_topic_followup():
    """The safety case. A scheme suffix flips off_topic to factual.

    "And what is the weather in Mumbai? for HDFC ELSS Tax Saver Fund" is
    classified FACTUAL, because the guard stops checking for an off-topic topic
    once a scheme is named. So the rewrite must be refused outright, or asking
    about the weather mid-conversation would get an answer sourced from a
    mutual-fund document.
    """
    h = History()
    h.append("user", "Tell me about HDFC ELSS Tax Saver Fund.")
    q = "And what is the weather in Mumbai?"
    assert evaluate(q).get("intent") == "off_topic"
    assert evaluate(f"{q} for HDFC ELSS Tax Saver Fund").get("intent") != "off_topic"

    asked, note = resolve_question(q, h.messages)
    assert asked == q, "the off-topic follow-up must be left exactly as typed"
    assert note == ""
    assert evaluate(asked).get("intent") == "off_topic"


def test_resolve_does_not_let_a_rewrite_rescue_pii_or_advice():
    """Same gate, for the other policy refusals. Each has a pronoun in it."""
    h = History()
    h.append("user", "What is the exit load on HDFC Flexi Cap Fund?")
    for q, intent in (
        ("And should I buy it?", "advice"),
        ("Is it safe for my retirement corpus?", "advice"),
    ):
        asked, note = resolve_question(q, h.messages)
        assert asked == q, f"{intent} follow-up must not be rewritten"
        assert evaluate(asked).get("intent") == intent


def test_resolve_leaves_a_policy_refusal_alone_even_with_a_referent():
    """No referent is needed for the gate; the typed words decide."""
    h = History()
    h.append("user", "What is the NAV of HDFC Flexi Cap Fund?")
    h.append("assistant", "NAV is 68.4321 as of 2026-06-30.")
    for q in ("Who won the IPL final last night?", "My PAN is ABCDE1234F, update my details"):
        asked, note = resolve_question(q, h.messages)
        assert asked == q
        assert note == ""
