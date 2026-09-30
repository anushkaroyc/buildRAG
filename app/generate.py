"""Grounded generation from retrieved chunks, with a validator that can veto it.

Phase 5's generation half. The architecture rule that shapes this file is
architecture.md §9.4: **the LLM never writes the citation.** The URL is read off
the winning chunk's metadata and appended by this module. That single decision
makes two success criteria structurally true instead of aspirational - SC-2 (one
citation, always) and SC-3 (the domain is on the allowlist) - because the model
has no opportunity to invent, duplicate, or launder a link.

Three layers of defence, cheapest first:

1. **The prompt** asks for <=3 sentences, no URLs, no advice, no returns.
2. **The validator** checks what came back and can trigger one repair retry.
3. **The fallback** replaces a still-bad answer with a fixed string.

The validator is not decoration. A 20B open-weight model asked for a hard format
will occasionally return four sentences, a markdown list, or a comparative claim,
and those are exactly the failure modes the product is judged on (SC-7 brevity,
SC-9 no performance claims). Rather than hoping, `validate()` returns a structured
verdict and `answer()` acts on it.

**Grounding is checked, not assumed.** If the model returns the `NOT IN CONTEXT`
sentinel - or an answer containing no figure at all - we answer "I couldn't
confirm that in the official pages I have" rather than passing on a fluent
paragraph with nothing under it (ST-1).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from . import prompts
from .config import INDEX_SCHEME_NAMES, get_settings
from .retrieve import RetrievalResult, RetrievedChunk

logger = logging.getLogger("app.generate")

# Performance/comparison vocabulary. SC-9 requires zero return or comparison
# claims across the whole run, so this is scanned over the *final* answer text
# before it is returned. Deliberately narrow: "return" also occurs in "exit load
# return" and "returns to the investor" in legitimate disclosures, so each pattern
# requires a comparative or a number nearby instead of banning the word.
FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(outperform\w*|underperform\w*|beat\w*|top[\s-]?performer\w*)\b", "performance comparison"),
    (r"\b(best|worst|highest|lowest)\s+(performing|performer)\b", "ranking"),
    (r"\b(guarantee\w*|assured)\b", "guaranteed return"),
    (r"\bwill\s+(return|rise|fall|grow|gain|deliver)\b", "predicted return"),
    (r"\b(is|are)\s+(safe|risky|good|bad|the\s+best|the\s+right)\b", "suitability claim"),
    (r"\b(recommend\w*|advise\w*|you\s+should|worth\s+buying)\b", "advice"),
    (r"\bI\s+(would|recommend|suggest)\b", "advice in first person"),
)

URL_IN_ANSWER = re.compile(r"https?://", re.I)
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
LIST_MARKER = re.compile(r"(^|\n)\s*(?:[-*\u2022]|\d+[.)])\s+", re.M)
HEADING_MARKER = re.compile(r"(^|\n)\s*#{1,6}\s+", re.M)
# A period only ends a sentence when followed by a capital, digit or quote. This
# is what keeps "Direct Growth." and "e.g." from inflating the count and getting a
# legitimate 2-sentence answer rejected for length.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

MAX_SENTENCES = 3
FRESHNESS_PREFIX = "Last updated from sources:"

# Share of an answer's content words that must appear in the retrieved context
# once the answer is at least LONG_ANSWER_TERMS long. 0.6 rather than "all of
# them" because legitimate paraphrase moves words around ("the scheme's ongoing
# charge" for "Total Expense Ratio"); below 0.6 the text is prose about the
# right topic with no fact in it.
MIN_TERM_OVERLAP = 0.6

# At or above this many content words, an answer is held to MIN_TERM_OVERLAP.
# Below it the bar drops with the number of words, because a one-word answer is
# almost entirely covered by the number check above - see `is_grounded`.
LONG_ANSWER_TERMS = 4


@dataclass
class ValidationResult:
    """Verdict on a generated answer. `ok=False` means do not ship this text."""

    ok: bool
    problems: list[str] = field(default_factory=list)
    sentences: int = 0
    said_not_in_context: bool = False

    def __str__(self) -> str:  # pragma: no cover - diagnostic only
        return "ok" if self.ok else "; ".join(self.problems)


def count_sentences(text: str) -> int:
    """Number of sentences, using the conservative split described above."""
    text = (text or "").strip()
    if not text:
        return 0
    return len([p for p in SENTENCE_SPLIT.split(text) if p.strip()])


# Function words only. This is deliberately NOT retrieve's stopword list: that one
# drops "hdfc", "fund" and "nav" because they are useless for *ranking* (they
# appear in every chunk of a scheme), but an answer is supposed to name its
# scheme, so dropping them here would make a well-formed answer look ungrounded.
_ANSWER_STOPWORDS = frozenset(
    """
    a an the is are was were be been being of for in on at to from by with and or
    but if then than so such as it its this that these those there here he she
    they them his her their we our you your i me my not no do does did has have
    had will would can could should may might must shall about into over under
    per any each all both more most other some such only own same too very
    scheme schemes fund funds plan plans option options growth direct
    """.split()
)

# Numbers as they appear in these documents: 1.00, 35,458.50, 0.41%, 1,250.
# Written out rather than `\d[\d,]*\.?\d*%?`, which swallows the sentence-ending
# period in "Rs. 100." and then fails to find that string in the context,
# reporting a number that is really there as fabricated.
_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?%?")

# Figures are not interchangeable across these qualifiers, and the model gets
# this wrong in a way a number-presence check cannot see. Both of these shipped
# wrong answers before this check existed, on a live run:
#
#   "What is the expense ratio of HDFC Nifty 50 Index Fund - Direct Growth?"
#     -> "0.41% per annum", which the source states for **Regular**. The same
#        source says Direct "shall have a lower expense ratio than Regular".
#   "What is the minimum SIP amount for HDFC Mid Cap Fund?"
#     -> "Rs. 100", which the source states for **redemption / switch-out**, not
#        for a systematic instalment.
#
# Both numbers really are in the cited chunk, so `is_grounded`'s number check
# passes them. The error is *misattribution*: a real figure attached to the wrong
# qualifier. So the check is contextual - locate the number in the source, read
# the words around it, and refuse if they name a qualifier the question did not
# ask about.
_QUALIFIER_CONFLICTS: tuple[frozenset[str], ...] = (
    frozenset({"direct", "regular"}),
    frozenset({"sip", "lump"}),
    frozenset({"growth", "dividend", "idcw"}),
)
# The words that select a member of a conflict family.
_QUALIFIER_WORDS = frozenset(
    "direct regular growth dividend idcw sip lump sum".split()
)

# How much text *before* a number counts as "the words that label it". The
# qualifier is written as a label ("Total Expense Ratio Regular - 0.41 %",
# "Direct Growth: expense ratio 0.35 %"), so this looks backwards only. Looking
# both ways produced a false positive: in "Direct Growth: 0.35 % p.a. Regular:
# 1.40 % p.a." a symmetric window around 0.35 reached the word "Regular" and
# rejected a perfectly correct answer.
_QUALIFIER_WINDOW = 60


def qualifiers_in(text: str) -> frozenset[str]:
    """Which conflict-family qualifiers appear in `text`."""
    return frozenset(re.findall(r"[a-z]+", (text or "").lower())) & _QUALIFIER_WORDS


def qualifier_conflict(question: str, answer: str, context: str) -> str | None:
    """Does the answer attach a figure to a qualifier the sources never gave it?

    This check is only enforced for *plan-sensitive* figure types. For
    scheme-level facts (exit load, benchmark, NAV, AUM, riskometer, tracking
    error, lock-in, capital gains, fund manager) the plan qualifier is often
    omitted in the source text and strict enforcement produces false positives.
    For expense ratio and for minimum amount questions that mention SIP/lump, the
    misattribution risk is real and we must enforce it.
    """
    qlower = (question or "").lower()
    alower = (answer or "").lower()
    sensitive = any(
        p in qlower or p in alower
        for p in (
            "expense ratio",
            "ter",
            "total expense ratio",
            "minimum sip",
            "minimum lump",
            "sip amount",
            "lump sum",
            "minimum investment",
        )
    )
    if not sensitive:
        return None

    asked = qualifiers_in(question)
    if not asked:
        return None
    context_lower = (context or "").lower()

    for number in _NUMBER.findall(answer or ""):
        index = context_lower.find(number)
        if index < 0:
            continue
        window = context_lower[max(0, index - _QUALIFIER_WINDOW) : index]
        nearby = qualifiers_in(window)
        for family in _QUALIFIER_CONFLICTS:
            claimed = family & asked
            if claimed and not (claimed & nearby):
                return (
                    f"the source does not state {number} for "
                    f"{'/'.join(sorted(claimed))}"
                )
    return None


def content_terms(text: str) -> list[str]:
    """Lowercased, de-duplicated content words, for the grounding check."""
    seen: dict[str, None] = {}
    for raw in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if len(raw) < 3 or raw in _ANSWER_STOPWORDS:
            continue
        seen.setdefault(raw, None)
    return list(seen)


def is_grounded(answer: str, context: str, question: str = "") -> tuple[bool, str]:
    """Is this answer actually supported by the retrieved context?

    Three independent conditions, all cheap, all aimed at failures this app
    cannot afford:

    1. **Every number in the answer appears in the context.** A fabricated
       figure is the one error a facts-only product cannot recover from, and it
       is directly checkable. The earlier proxy - "the answer must contain a
       digit" - was not a grounding check at all: it rejected the correct answer
       to "who is the fund manager?", because a person's name has no digits.
    2. **No qualifier mismatch.** A figure the sources attribute to Regular is
       not the Direct figure, and a redemption minimum is not a SIP minimum.
       See `qualifier_conflict` - two wrong answers got past check 1 before
       this existed.
    3. **Most content words overlap the context.** Catches an answer that is
       fluent and ungrounded prose about the right topic.

    `question` is optional so that a format-only caller can omit it; check 2 is
    skipped when it is empty, since there is nothing to compare against.

    Returns (grounded, reason) so the caller can log *why* something was dropped.
    """
    context_lower = (context or "").lower()
    if not context_lower:
        return False, "no context to check against"

    numbers = _NUMBER.findall(answer or "")
    invented = [n for n in numbers if n not in context_lower]
    if invented:
        return False, f"figure(s) not in context: {', '.join(invented[:4])}"

    conflict = qualifier_conflict(question, answer, context)
    if conflict:
        return False, f"qualifier mismatch: {conflict}"

    terms = content_terms(answer)
    if not terms:
        # Nothing checkable (e.g. "NOT IN CONTEXT"), handled by the caller.
        return True, "no content terms to check"

    # The required overlap scales with how much there is to be wrong about. A
    # fixed 0.6 rejects short answers that merely paraphrase: "The ceiling is
    # 2.25 %." against a source saying "Maximum Total Expense Ratio under
    # Regulation 52(6): 2.25 % p.a." scores 0% on word overlap, and refusing that
    # would trade a real fact for a false refusal. Long answers get the strict
    # bar, because long fluent prose is exactly the shape of an ungrounded
    # answer - and a one- or two-word answer has almost nothing to hallucinate
    # once its number is verified above.
    if len(terms) >= LONG_ANSWER_TERMS:
        required = MIN_TERM_OVERLAP
    elif len(terms) == 1:
        required = 0.0
    else:
        required = 1.0 / len(terms)

    overlap = sum(1 for t in terms if t in context_lower) / len(terms)
    if overlap < required:
        return False, f"only {overlap:.0%} of content words appear in the context"
    return True, f"{overlap:.0%} of content words appear in the context"


def validate(answer: str) -> ValidationResult:
    """Check a generated answer against the format and vocabulary rules.

    Pure function, no model and no network - which is what lets every failure mode
    be tested in `tests/test_validator.py` without spending a token.

    Grounding is *not* checked here, because it needs the context: see
    `is_grounded`. Keeping the two apart means a format-only test can call this
    with a one-line string, and the number-fabrication check is not accidentally
    skipped just because a test had no context to hand.
    """
    text = (answer or "").strip()
    if not text:
        return ValidationResult(ok=False, problems=["empty answer"])

    said_not_in_context = text.upper().startswith(prompts.NOT_IN_CONTEXT_SENTINEL.upper())
    sentences = count_sentences(text)
    problems: list[str] = []

    if sentences > MAX_SENTENCES:
        problems.append(f"too long: {sentences} sentences (max {MAX_SENTENCES})")

    # SC-2: the citation is appended by us, so the model must not have written one.
    # Any URL or markdown link means it invented or duplicated a source.
    if URL_IN_ANSWER.search(text):
        problems.append("contains a URL; the citation must come from chunk metadata")
    if MARKDOWN_LINK.search(text):
        problems.append("contains a markdown link")

    if LIST_MARKER.search(text):
        problems.append("contains a list; prose only")
    if HEADING_MARKER.search(text):
        problems.append("contains a heading")

    for pattern, label in FORBIDDEN_PATTERNS:
        if re.search(pattern, text, re.I):
            problems.append(f"{label}: matches /{pattern}/")

    return ValidationResult(
        ok=not problems,
        problems=problems,
        sentences=sentences,
        said_not_in_context=said_not_in_context,
    )


def _scheme_label(scheme: str) -> str:
    """`scheme=<index name>`, plus the name the user is likely to have typed.

    HDFC renamed HDFC Top 100 Fund to HDFC Large Cap Fund (SEBI, March 2026), and
    the ingested documents use the new name while users type the old one. The
    model is strict about this in a way that is easy to miss: asked about "HDFC
    Top 100 Fund" with context labelled "HDFC Large Cap Fund", it returns NOT IN
    CONTEXT rather than guessing that they are the same scheme - measured, and
    the same context answered correctly when the question used the new name.
    Retrieval already knows the mapping (retrieve.INDEX_SCHEME_NAMES, which exists
    so the Chroma filter can translate it); without it in the prompt that
    knowledge stops at the filter and the answer is lost.
    """
    if not scheme:
        return "scheme=unknown"
    canonical = [k for k, v in INDEX_SCHEME_NAMES.items() if v == scheme]
    if not canonical or canonical[0] == scheme:
        return f"scheme={scheme}"
    return f"scheme={scheme} (also called {canonical[0]}; same scheme)"


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Render the retrieved chunks for the prompt.

    Each chunk is labelled with its scheme, document type and as-of date so the
    model attributes a figure to a specific page instead of blending two together,
    which is the mechanism behind cross-scheme bleed (gate 5.13).
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = [
            f"source {i}",
            _scheme_label(chunk.scheme),
            f"document={chunk.page_type or 'unknown'}",
        ]
        if chunk.as_of:
            meta.append(f"as of {chunk.as_of}")
        blocks.append(f"[{' | '.join(meta)}]\n{chunk.document.strip()}")
    return "\n\n---\n\n".join(blocks)


def citation_for(chunk: RetrievedChunk) -> dict:
    """The citation payload, assembled from metadata only (architecture 9.4)."""
    from urllib.parse import urlparse

    host = (urlparse(chunk.url).hostname or "").lower()
    return {
        "url": chunk.url,
        "title": chunk.title or chunk.scheme or host,
        "scheme": chunk.scheme,
        "as_of": chunk.as_of,
        "page_type": chunk.page_type,
        "host": host,
        # Checked rather than assumed, so a re-crawl that adds an off-allowlist
        # source is visible in the eval output instead of silently cited (SC-3).
        "domain_ok": prompts.is_allowlisted(chunk.url),
    }


def freshness_stamp(chunk: RetrievedChunk) -> str:
    """`Last updated from sources: <date>` (SC-8).

    Takes the date of the chunk being cited, not the most recent date across all
    retrieved chunks. With exactly one citation those two numbers have to agree:
    dating an answer with a newer page than the one linked invites the reader to
    believe the cited page is more current than it is, which is the optimistic
    direction of error. When no chunk carries an `as_of` the stamp is omitted
    rather than guessed at.
    """
    return f"{FRESHNESS_PREFIX} {chunk.as_of}" if chunk.as_of else ""


def degraded(reason: str) -> dict:
    """ST-2: the LLM was unavailable. Never substitute an ungrounded answer."""
    logger.warning("generation degraded: %s", reason)
    payload = prompts.unknown_answer()
    payload["answer"] = prompts.DEGRADED_REPLY
    payload["intent"] = "degraded"
    payload["citations"] = []
    payload["degraded_reason"] = reason
    return payload


def call_groq(messages: list[dict], max_tokens: int) -> tuple[str, dict]:
    """One Groq chat call. Returns (text, usage). Raises on any failure."""
    from groq import Groq

    settings = get_settings()
    client = Groq(api_key=settings.groq_api_key, timeout=settings.guard_timeout_s * 6)
    kwargs = {
        "model": settings.groq_model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    # gpt-oss is a reasoning model: without this its reasoning trace can consume
    # the whole token budget and return an empty string (measured 98 vs 46
    # completion tokens on a one-sentence answer, architecture 6.1). The floor
    # leaves room for reasoning plus a 3-sentence answer.
    if "gpt-oss" in settings.groq_model:
        kwargs["reasoning_effort"] = "low"
        kwargs["max_tokens"] = max(max_tokens, 240)

    resp = client.chat.completions.create(**kwargs)
    usage = {
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
        "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
        "total_tokens": getattr(resp.usage, "total_tokens", 0),
        "finish_reason": resp.choices[0].finish_reason,
    }
    # An empty body with finish_reason="length" means the reasoning trace ate the
    # budget, not that the model had nothing to say. Treat it as a failed call so
    # the caller degrades to ST-2 rather than returning an empty string.
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError(f"empty completion (finish_reason={usage['finish_reason']})")
    return text, usage


def answer(retrieval: RetrievalResult, question: str, *, allow_repair: bool = True) -> dict:
    """Generate a grounded, cited answer from `retrieval`, or explain why not.

    Order of exits, each one a refusal to guess:

    - index missing                 -> degraded
    - no key, or Groq down/429/5xx  -> degraded (ST-2)
    - no chunk >= MIN_SCORE         -> "I don't have that in my sources", 0 tokens
    - model says NOT IN CONTEXT     -> "I couldn't confirm that"
    - validator fails twice         -> the unsupported reply, never the bad text
    """
    if retrieval.index_missing:
        return degraded("index missing")

    if not retrieval.ok:
        # ST-1 / gate 5.8: nothing relevant, so nothing is generated at all.
        payload = prompts.unknown_answer(retrieval.n_candidates)
        payload["citations"] = []
        payload["retrieval"] = retrieval
        return payload

    if not get_settings().groq_configured:
        return degraded("no API key configured")

    chunks = retrieval.chunks
    context = build_context(chunks)
    messages = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"SOURCE CONTEXT:\n{context}\n\nQUESTION: {question.strip()}",
        },
    ]

    usage: dict = {}
    try:
        text, usage = call_groq(messages, max_tokens=180)
    except Exception as exc:  # noqa: BLE001 - any failure degrades, never guesses
        return degraded(f"{type(exc).__name__}: {exc}")

    verdict = validate(text)
    repaired = False
    if not verdict.ok and allow_repair:
        # One retry, naming the specific violations. Naming them matters: a bare
        # "try again" on a 20B model tends to reproduce the same mistake.
        logger.info("validator rejected answer (%s); one repair retry", verdict)
        try:
            retry = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"{prompts.REPAIR_PROMPT}\nProblems: {verdict}."},
            ]
            text, usage = call_groq(retry, max_tokens=180)
            repaired = True
            verdict = validate(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("repair call failed: %s", type(exc).__name__)
            verdict = validate(text)

    citation = citation_for(chunks[0])

    if verdict.said_not_in_context:
        payload = prompts.unsupported_answer()
        payload["citations"] = []
        payload["retrieval"] = retrieval
        payload["usage"] = usage
        return payload

    grounded, reason = is_grounded(text, context, question)
    if not grounded:
        # The model did not decline, but it produced text the sources do not
        # support. Treated exactly like a decline: the user gets the fixed
        # "couldn't confirm" reply, and the reason is logged for diagnosis.
        logger.warning("ungrounded answer dropped (%s): %r", reason, text[:160])
        verdict.problems.append(f"ungrounded: {reason}")

    if not verdict.ok:
        # Still not shippable after the retry. The safe fallback is the unsupported
        # reply rather than the model's text: a validated-bad answer is still an
        # answer, and shipping it would break SC-7/SC-9 silently.
        logger.warning("answer failed validation after retry: %s", verdict)
        payload = prompts.unsupported_answer()
        payload["citations"] = []
        payload["retrieval"] = retrieval
        payload["usage"] = usage
        payload["validator_problems"] = verdict.problems
        return payload

    body = text.strip()
    stamp = freshness_stamp(chunks[0])
    if stamp:
        body = f"{body}\n\n{stamp}"

    return {
        "answer": body,
        "refused": False,
        "intent": "answered",
        "citations": [citation],
        "disclaimer": prompts.SHORT_DISCLAIMER,
        "sentences": verdict.sentences,
        "repaired": repaired,
        "usage": usage,
        "retrieval": retrieval,
    }


__all__ = [
    "FRESHNESS_PREFIX",
    "MAX_SENTENCES",
    "answer",
    "build_context",
    "call_groq",
    "citation_for",
    "content_terms",
    "count_sentences",
    "degraded",
    "freshness_stamp",
    "is_grounded",
    "qualifier_conflict",
    "qualifiers_in",
    "validate",
]
