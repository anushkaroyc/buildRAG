"""Lightweight conversation memory and follow-up question rewriting.

Keeps the last N messages in memory and rewrites short follow-up questions
(e.g. "what about its fees?") so the referent is named before the question
reaches the guard, the retriever and the model.

Two deliberate design choices:

* **The rewriter is rule-based, not an LLM call.** A rewrite that can be wrong
  in a way we cannot check would corrupt the query *before* the guardrails see
  it, and a follow-up like "its fees" carrying the wrong scheme name would
  answer a question the user never asked. Keeping it deterministic and
  inspectable means the CLI can always print exactly which question was
  searched.
* **It reuses `guardrails.named_scheme` rather than owning a scheme list.** That
  function already knows the alias table - "HDFC Top 100 Fund" and
  "HDFC Large Cap Fund" are the same scheme - and a second, looser pattern
  list here is how the two drift apart and the rewriter starts resolving "its"
  to a competitor's fund.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Deque, Sequence

from .guardrails import OFF_TOPIC, PII, ADVICE, INJECTION, UNSURE, evaluate, named_scheme

# Intents that are policy refusals rather than ambiguities. A rewrite may
# rescue an UNSURE question - "what about its fees?" is unsure only because it
# has not said which fund - but it must never rescue one of these.
#
# This distinction is the whole safety argument for the feature. The rewrite
# happens *before* the guard, because a pronoun-only follow-up is refused as
# unsure and would otherwise never be answered. But rewriting first also means
# the guard sees the rewritten text, and appending a scheme name to a question
# flips the off-topic check: "And what is the weather in Mumbai?" is off_topic,
# while "And what is the weather in Mumbai? for HDFC ELSS Tax Saver Fund" is
# factual, because the guard stops looking for an off-topic topic once a scheme
# is named. Without this gate, asking a weather question as a follow-up would
# get it answered from a mutual-fund document.
#
# A refusal is a decision about the user's actual words, and those words are
# what the user typed. So: if the raw question is refused for a policy reason,
# that verdict stands and no rewrite is attempted.
_REFUSAL_INTENTS = frozenset({OFF_TOPIC, PII, ADVICE, INJECTION})

# How many messages to keep. Enough to disambiguate "its/that fund" after a
# couple of turns of back-and-forth, small enough to stay predictable.
HISTORY_MAXLEN = 10

# Noun phrases that name a scheme without naming it, e.g. "that fund's exit load".
_REF_PHRASES = (
    r"that\s+fund",
    r"this\s+fund",
    r"that\s+scheme",
    r"this\s+scheme",
    r"same\s+fund",
    r"same\s+scheme",
)

# Bare pronouns that point back at the scheme under discussion. "that" and
# "this" are included because "what about that?" is a common follow-up shape;
# they are only substituted when no scheme is named in the question itself, so
# a demonstrative pointing at something else ("that other fund") is not
# silently redirected - the competitor name in it will be found first.
_PRONOUNS = re.compile(r"\b(it|its|them|their|theirs|that|this|these|those)\b", re.I)

# Demonstratives that are not back-references at all. "this week", "that year",
# "these days" are time idioms, and rewriting them to "HDFC Flexi Cap Fund
# week" turned a perfectly clear question into nonsense. Checked before
# _PRONOUNS, and the whole phrase is left alone.
_TIME_IDIOMS = re.compile(
    r"\b(?:this|that|these|those)\s+"
    r"(?:week|month|year|quarter|day|time|morning|evening|afternoon|"
    r"manner|way|is|was|are|were|has|have|had|will|would|should|can|"
    r"doesn't|does|isn't|aren't)\b",
    re.I,
)

# Terse follow-ups that carry no noun for the pronoun to attach to. These get
# the scheme prepended rather than substituted in, so the result reads as a
# question instead of a sentence fragment.
_PREFIXABLE = re.compile(
    r"^\s*(?:and\s+|also\s+|then\s+|so\s+)?"
    r"(?:what|how)\s*(?:about|is|are|was|were|much|many|often|is the)?\b",
    re.I,
)


@dataclass(frozen=True)
class Message:
    """One turn of the conversation."""

    role: str  # "user" or "assistant"
    content: str


class History:
    """A rolling window of the most recent messages.

    A deque with a max length, rather than a list that gets trimmed at a call
    site, so the bound is enforced by construction and cannot be forgotten when
    a new caller forgets to trim.
    """

    def __init__(self, maxlen: int = HISTORY_MAXLEN) -> None:
        self.maxlen = maxlen
        self._messages: Deque[Message] = deque(maxlen=maxlen)

    @property
    def messages(self) -> Sequence[Message]:
        """Messages oldest to newest."""
        return tuple(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def append(self, role: str, content: str) -> None:
        if content and content.strip():
            self._messages.append(Message(role=role, content=content))

    def clear(self) -> None:
        self._messages.clear()

    def last(self, role: str) -> str | None:
        for msg in reversed(self._messages):
            if msg.role == role:
                return msg.content
        return None

    def referent(self) -> str | None:
        """The in-scope scheme most recently named, newest message first.

        Returns the *canonical* scheme name, so a follow-up resolves to the same
        string the retrieval filter expects ("HDFC Large Cap Fund" even when the
        user said "HDFC Top 100 Fund").
        """
        for msg in reversed(self._messages):
            scheme = named_scheme(msg.content)
            if scheme:
                return scheme
        return None


def _has_reference(text: str) -> bool:
    """Does this question point back at something already discussed?"""
    if any(re.search(rx, text, re.I) for rx in _REF_PHRASES):
        return True
    # Blank out time idioms first so "this week" cannot register as a reference.
    return bool(_PRONOUNS.search(_TIME_IDIOMS.sub(" ", text)))


def _substitute(question: str, scheme: str) -> str:
    """Replace back-references with the scheme name."""
    out = question
    for rx in _REF_PHRASES:
        out = re.sub(rx, scheme, out, flags=re.I)
    # Possessives first, so "its" becomes "<scheme>'s" rather than being
    # consumed by the bare-pronoun rule and leaving a stray apostrophe.
    out = re.sub(r"\b(its|their|theirs)\b", f"{scheme}'s", out, flags=re.I)
    # Protect time idioms from the bare-pronoun pass, then restore them.
    protected: list[str] = []

    def _hide(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"\x00{len(protected) - 1}\x00"

    out = _TIME_IDIOMS.sub(_hide, out)
    out = _PRONOUNS.sub(scheme, out)
    for i, phrase in enumerate(protected):
        out = out.replace(f"\x00{i}\x00", phrase)
    out = out.replace(f"{scheme} 's", f"{scheme}'s")
    out = re.sub(r"\s+", " ", out).strip()
    return re.sub(r"\s+([?.,])", r"\1", out)


def _prepend(question: str, scheme: str) -> str:
    """Turn a pronoun-only follow-up into a self-contained question."""
    stem = question.strip().rstrip("?").strip()
    if _PREFIXABLE.match(stem):
        return f"{stem} for {scheme}?"
    return f"About {scheme}: {stem}?"


def rewrite_question(question: str, history: Sequence[Message] | None = None) -> str:
    """Resolve a follow-up question against the conversation so far.

    Returns the question unchanged when it already names a scheme, when it has
    no back-reference, or when no in-scope scheme appears in history. A rewrite
    that cannot be made confidently is a no-op: retrieval and generation handle
    a vague question better than they handle a confidently wrong one.
    """
    q = (question or "").strip()
    if not q:
        return q

    # A question that names a scheme is already self-contained. Rewriting it
    # would risk replacing what the user actually asked about.
    if named_scheme(q):
        return q

    if not _has_reference(q):
        return q

    referent = None
    for msg in reversed(list(history or ())):
        referent = named_scheme(msg.content)
        if referent:
            break
    if not referent:
        return q

    rewritten = _substitute(q, referent)
    # If the substitution somehow left the question without a resolvable
    # scheme, state the referent outright rather than shipping a half-rewrite.
    if not rewritten or not named_scheme(rewritten):
        rewritten = _prepend(q, referent)
    return rewritten


def resolve_question(question: str, history: Sequence[Message] | None = None) -> tuple[str, str]:
    """Resolve a follow-up, without letting the rewrite defeat a guardrail.

    Returns (question_to_run, note) where note explains any rewrite. This is the
    only function the product should call: `rewrite_question` on its own is
    safe to use in tests but must not be wired straight into the request path,
    because it will happily rewrite a question the guardrails were about to
    refuse.

    The order is deliberate:

    1. Guard the question **as the user typed it**. If that is refused for a
       policy reason - off topic, PII, advice, injection - stop. The user's
       words are what the refusal is about.
    2. Otherwise resolve the referent and guard the rewritten question, which
       is what actually gets searched and answered. This is where UNSURE
       ("what about its fees?") is rescued.
    """
    raw = (question or "").strip()
    if not raw:
        return raw, ""

    verdict = evaluate(raw)
    if verdict.get("intent") in _REFUSAL_INTENTS:
        # Refuse on the user's own words. Rewriting here would be the guardrail
        # bypass this function exists to prevent.
        return raw, ""

    resolved = rewrite_question(raw, history)
    if resolved == raw:
        return raw, ""
    return resolved, f"resolved from context: {raw}"
