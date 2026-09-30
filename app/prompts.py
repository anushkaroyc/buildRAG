"""Every user-visible fixed string, in one place.

Nothing in this module is model-generated text. A refusal is chosen from a
constant and returned verbatim; the LLM never writes a refusal, an apology, or
an educational link. That is what makes check 4.5 ("identical template every
time") a structural property rather than something to spot-check.

Three groups of strings:

| group | why it is fixed |
| --- | --- |
| refusal templates | SC-5 requires a polite, opinion-free refusal on 100% of advice probes. A generated refusal varies in tone and sometimes apologises *for the fund*, which reads as an opinion. |
| "I don't know" templates | ST-1. The wording is the product decision: say what is missing and offer the alternative, never a guess. |
| educational links | SC-3 keeps citations on allowlisted domains. The same list gates refusal links, so a refusal cannot quietly send a reviewer off-allowlist. |

Phase 4 stubs `SYSTEM_PROMPT`; Phase 5 completes it with the retrieval
contract (≤3 sentences, one citation, `Last updated from sources:`).
"""

from __future__ import annotations

from .config import ALLOWED_CITATION_DOMAINS

# --- canonical disclaimer (PRD §8, verbatim) --------------------------------

DISCLAIMER = (
    "Facts-only. No investment advice. This assistant shares information from "
    "official public pages of HDFC Mutual Fund, SEBI, and AMFI. It does not "
    "recommend buying, selling, or holding any scheme, and does not compute or "
    "compare returns. Figures reflect the source pages as of the date shown. "
    "Please read the official scheme documents and consult a SEBI-registered "
    "investment adviser before investing."
)

SHORT_DISCLAIMER = "Facts-only. No investment advice."

# --- refusal templates -------------------------------------------------------
# All of them: no opinion, no ranking, no hedge that implies a judgement, and a
# pointer to what the assistant *can* do. None mentions a scheme favourably.

PII_REFUSAL = (
    "I can't help with that. Please don't share personal identifiers such as "
    "PAN, Aadhaar, account or folio numbers, OTPs, email addresses or phone "
    "numbers here \u2014 I don't store them and I can't act on an account."
)

ADVICE_REFUSAL = (
    "I only share facts from official HDFC Mutual Fund, SEBI and AMFI pages, so "
    "I can't recommend, compare or forecast a scheme, and I won't suggest a buy, "
    "sell or allocation decision. Ask me about a scheme's documented terms \u2014 "
    "expense ratio, exit load, minimum SIP or lump-sum amount, lock-in, NAV, "
    "benchmark, riskometer or how to download a statement \u2014 and I'll answer "
    "with a source link."
)

OFF_TOPIC_REFUSAL = (
    "I only answer mutual fund questions about HDFC Mutual Fund's five schemes "
    "covered here, using official HDFC, SEBI and AMFI pages, so I can't help with "
    "that one. Ask me about a scheme's expense ratio, exit load, minimum SIP, "
    "lock-in, NAV, benchmark or riskometer and I'll answer with a source link."
)

# OUT_OF_SCOPE reads as ST-1 ("not in my corpus") rather than a scolding: the
# question is a perfectly good one, just outside this build's five schemes.
OUT_OF_SCOPE_REFUSAL = (
    "I don't have that in my sources. I only cover HDFC Mutual Fund's five "
    "equity schemes \u2014 HDFC Top 100 (Large Cap), Flexi Cap, ELSS Tax Saver, Mid "
    "Cap and Nifty 50 Index Fund \u2014 using official HDFC, SEBI and AMFI pages, so "
    "I won't answer from memory about any other scheme."
)

# Emitted only when the guard is unreachable *and* the question could not be
# placed. Wording avoids implying the assistant understood the topic: it did not.
UNSURE_REFUSAL = (
    "I'm not able to tell whether that is a factual question about these schemes, "
    "so I won't risk answering it. I can look up documented scheme terms \u2014 "
    "expense ratio, exit load, minimum SIP, lock-in, NAV, benchmark, riskometer, "
    "or how to download a statement \u2014 and answer with a source link."
)

REFUSAL_TEMPLATES = {
    "pii": PII_REFUSAL,
    "advice": ADVICE_REFUSAL,
    "off_topic": OFF_TOPIC_REFUSAL,
    "out_of_scope": OUT_OF_SCOPE_REFUSAL,
    "unsure": UNSURE_REFUSAL,
}

# --- "I don't know" templates (ST-1) -----------------------------------------
# Two distinct failure modes, deliberately worded differently: the first says the
# corpus lacks it, the second says the retrieved pages do not state it. Collapsing
# them into one string would hide retrieval regressions during the demo.

NO_CONTEXT_REPLY = (
    "I don't have that in my sources, so I won't guess. I answer from a fixed "
    "set of official HDFC Mutual Fund, SEBI and AMFI pages for five equity "
    "schemes; ask me about a scheme's expense ratio, exit load, minimum SIP or "
    "lump-sum amount, lock-in, NAV, benchmark, riskometer, or how to download a "
    "statement."
)

UNSUPPORTED_REPLY = (
    "I couldn't confirm that in the official pages I have for these schemes, so "
    "I won't fill in the gap from memory. If you tell me which scheme and which "
    "term \u2014 expense ratio, exit load, minimum amount, lock-in, NAV, benchmark "
    "or riskometer \u2014 I'll look it up in the source document and cite it."
)

DEGRADED_REPLY = (
    "Sources are temporarily unavailable, so I can't check that right now. Please "
    "try again shortly, or read the official scheme document on HDFC Mutual "
    "Fund's site."
)

# --- educational links -------------------------------------------------------
# Verified HTTP 200 on 2026-09-30, all on ALLOWED_CITATION_DOMAINS. Deep HDFC MF
# links are deliberately avoided: that site answers every path with an SPA shell
# (HTTP 200 for a URL that does not exist), so a deep link cannot be verified and
# a 404 in a demo is worse than a generic landing page. `tests/test_guardrails.py`
# asserts the domains and gate 4.4 asserts one link per refusal.

EDUCATIONAL_LINKS = {
    "learn_basics": {
        "label": "SEBI \u2014 Investor education",
        "url": "https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doInvestorEducation=yes",
    },
    "mutual_fund_basics": {
        "label": "SEBI \u2014 Mutual funds",
        "url": "https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doMutualFund=yes",
    },
    "risk_and_suitability": {
        "label": "SEBI \u2014 Investor education on risk",
        "url": "https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doInvestorEducation=yes",
    },
    "tax_and_statements": {
        "label": "HDFC Mutual Fund \u2014 FAQs on statements and tax",
        "url": "https://www.hdfcmf.com/faqs",
    },
    "complaints": {
        "label": "SEBI \u2014 Investor grievances (SCORES)",
        "url": "https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doInvestorGrievance=yes",
    },
    "what_i_can_answer": {
        "label": "AMFI \u2014 Association of Mutual Funds of India",
        "url": "https://www.amfiindia.com/",
    },
    "scheme_pages": {
        "label": "HDFC Mutual Fund \u2014 scheme pages",
        "url": "https://www.hdfcmf.com/",
    },
}

# Which link a refusal of each intent gets. Keyed by intent, so a new refusal path
# cannot forget its link (gate 4.4 checks every advice probe carries one).
REFUSAL_LINKS = {
    "pii": None,  # nothing to redirect to; never send a PII refuser to a page
    "advice": EDUCATIONAL_LINKS["risk_and_suitability"],
    "off_topic": EDUCATIONAL_LINKS["mutual_fund_basics"],
    "out_of_scope": EDUCATIONAL_LINKS["scheme_pages"],
    "unsure": EDUCATIONAL_LINKS["learn_basics"],
}

# Advice refusals can point at a *more specific* page depending on what was asked
# (a timing question wants tax guidance; a risk question wants risk education).
# Matched against the question, longest pattern first, and always falls back to
# the generic `advice` link above.
ADVICE_TOPIC_HINTS: tuple[tuple[str, str], ...] = (
    # Profit-taking, switching and tax questions want statement/tax guidance
    # (PRD §4.2 #7: "link to official tax/statement guidance"). Matched before
    # the risk bucket so "when should I sell" does not land on risk education.
    (
        # Deliberately no bare `\btax\b`: it would match the in-scope scheme name
        # "HDFC ELSS Tax Saver Fund" and send a plain "should I buy it?" to the
        # tax FAQ instead of general investor education.
        r"\b(sell|selling|redeem|switch|exit|book\s+profits?|profits?|80\s?c|gst|"
        r"capital\s+gains?|statement|income\s+tax|taxation|"
        r"tax\s+(?:implication|benefit|advantage|slab|proof|saving))\b",
        "tax_and_statements",
    ),
    (r"\b(risk|risky|safe|safety|retire|retirement|volatile|volatility)\b", "risk_and_suitability"),
    (r"\b(complain|complaint|grievance|scam|fraud|wrong)\b", "complaints"),
    (r"\b(what is|what's)\b.*\b(mutual fund|sip|nav|elss)\b", "learn_basics"),
    (r"\b(compare|comparison|better|best|rank|ranking)\b", "mutual_fund_basics"),
)


def advice_link_for(question: str):
    """Pick the most specific educational link for an advice refusal.

    Cosmetic, and deliberately the only per-question variation in a refusal: the
    *prose* stays byte-identical (gate 4.5) while the link is pointed at the
    subject the user raised (PRD §4.2 asks for "a relevant educational link").
    """
    import re

    for pattern, key in ADVICE_TOPIC_HINTS:
        if re.search(pattern, question, re.IGNORECASE):
            return EDUCATIONAL_LINKS[key]
    return REFUSAL_LINKS["advice"]


def refusal_for(intent: str, question: str = "") -> dict:
    """Build the full refusal payload for an intent: fixed text + one link.

    Returns a dict so `POST /ask` can pass it straight to the JSON response
    without re-deriving anything. `question` is used only to choose the link and
    is never stored or logged.
    """
    text = REFUSAL_TEMPLATES.get(intent, UNSURE_REFUSAL)
    if intent == "advice":
        link = advice_link_for(question)
    else:
        link = REFUSAL_LINKS.get(intent)
    return {
        "answer": text,
        "citation": link,
        "refused": True,
        "intent": intent,
        "disclaimer": SHORT_DISCLAIMER,
    }


def unknown_answer(hits_below_threshold: int = 0) -> dict:
    """The ST-1 payload: "I don't know", plus the reason it happened.

    `hits_below_threshold` is reported to the caller for the health/eval logs but
    is not surfaced in the answer text \u2014 telling a user how many near-misses
    were retrieved is noise, and it leaks retrieval internals for no benefit.
    """
    return {
        "answer": NO_CONTEXT_REPLY,
        "citation": None,
        "refused": True,
        "intent": "no_context",
        "disclaimer": SHORT_DISCLAIMER,
        "candidates_below_threshold": hits_below_threshold,
    }


def unsupported_answer() -> dict:
    """Context was retrieved but did not state the fact. Still "I don't know"."""
    return {
        "answer": UNSUPPORTED_REPLY,
        "citation": None,
        "refused": True,
        "intent": "unsupported",
        "disclaimer": SHORT_DISCLAIMER,
    }


# --- link hygiene ------------------------------------------------------------


def is_allowlisted(url: str) -> bool:
    """True when `url`'s host is on the SC-3 domain allowlist."""
    from urllib.parse import urlparse

    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in ALLOWED_CITATION_DOMAINS)


def all_links() -> dict[str, dict]:
    """Every link in this module, for the hygiene test."""
    links = dict(EDUCATIONAL_LINKS)
    for value in REFUSAL_LINKS.values():
        if value:
            links[f"_refusal:{value['url']}"] = value
    return links


# --- system prompt (completed in Phase 5) -----------------------------------

SYSTEM_PROMPT = (
    "You are a facts-only assistant for HDFC Mutual Fund equity schemes. You report "
    "what the official source documents say, and nothing else.\n\n"
    "Rules, in priority order:\n"
    "1. Answer only from the SOURCE CONTEXT. A source label reading 'also called X; "
    "same scheme' means the document's name and the question's name are one scheme - "
    "answer from it. If the context genuinely does not contain the answer, reply with "
    "exactly: NOT IN CONTEXT. Otherwise answer.\n"
    "2. Distinguish a scheme's own documented figure from a regulatory ceiling or a "
    "worked example. 'Maximum Total Expense Ratio under Regulation 52(6)' tiers are "
    "ceilings, not the scheme's expense ratio. An 'illustration of the impact of "
    "expense ratio' is an example, not a scheme figure. Keep a 'no exit load' "
    "statement attached to its holding-period condition.\n"
    "3. At most 3 sentences. Be direct and factual.\n"
    "4. Never give investment advice. Do not recommend, rank, compare or time a "
    "purchase, and never call a scheme good, bad, safe or suitable for anyone.\n"
    "5. Never state, estimate or imply a return, growth rate or ranking. Report "
    "figures as documented, without characterising them.\n"
    "6. Do not write URLs, links or citation markers; the citation is attached "
    "automatically. No preamble, headings, bullets or markdown.\n"
    "7. If the question asks for a recommendation, a comparison between schemes, or a "
    "future return, reply with exactly: NOT IN CONTEXT.\n\n"
    "Write only the answer text."
)

# Sent back by the model when the retrieved context does not support an answer.
# `generate.py` maps this to `UNSUPPORTED_REPLY` rather than showing it to the
# user, so the user-facing wording stays a fixed string (gate 4.5) and the model
# never gets to phrase a refusal.
NOT_IN_CONTEXT_SENTINEL = "NOT IN CONTEXT"

REPAIR_PROMPT = (
    "Your previous answer broke the rules. Rewrite it following every rule again, "
    "using only the SOURCE CONTEXT. Reply with the corrected answer text only, or "
    "with exactly: NOT IN CONTEXT."
)

SYSTEM_PROMPT_NOTE = (
    "Rule 1 and the NOT IN CONTEXT sentinel are load-bearing: the model is asked to "
    "decline in a fixed token rather than to judge whether it knows something, "
    "because a model asked to 'be careful' produces a hedged guess instead of a "
    "clean refusal. `generate.is_grounded` independently checks that every number in "
    "an answer appears in the retrieved context, so a leaked hedge still cannot ship.\n"
    "\n"
    "Rule 2 was rewritten rather than stacked. An earlier version phrased every trap "
    "as a second 'reply NOT IN CONTEXT' trigger, and measured over six golden "
    "questions it answered 3: the extra triggers made the model decline questions it "
    "could have answered (the ELSS exit load, which the sources state plainly) while "
    "adding nothing on the one case it was written for. Stating the distinction "
    "positively - 'tiers are ceilings, not the ratio' - and leaving a single decline "
    "trigger answered 4: the same, keeping the expense-ratio trap closed. Rule 2 is "
    "therefore about *disambiguation*, not about refusing.\n"
    "\n"
    "The honest limit of this corpus is worth knowing when reading the eval: for "
    "several golden questions the sources genuinely do not state the figure (a "
    "scheme's own expense ratio, a riskometer category, a minimum SIP amount). The "
    "model declining those is correct behaviour, and the eval reports them as corpus "
    "gaps rather than counting them as model errors."
)
