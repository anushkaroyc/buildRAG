"""Streamlit chat UI for the facts-only mutual-fund assistant.

    .venv/bin/python -m streamlit run app/ui.py

Reuses the same pipeline as `app.ask` and `app.api`:

    resolve_question -> guardrails.evaluate -> retrieve -> generate.answer

The UI adds no logic of its own. That is deliberate: a second code path for
answering questions would be a second set of bugs, and the guardrails in
particular must not be bypassable by "just calling the model from the UI".

Three things this surface owes the user, from Phase 6:

- **The disclaimer is always on screen** (gate 6.1, SC-10). In the sidebar on
  the welcome screen, and repeated under every answer, because a screenshot of
  one answer is a screenshot of the product.
- **A sources expander under each answer** (gate 6.2). The citation is the
  product's core claim - that the answer came from an official document - so
  hiding it behind a hover would leave the claim unauditable.
- **A clear-chat button** (gate 6.2, PRD 3.2 item 7). Also clears the
  `History` window, so a follow-up cannot resolve against a scheme the user has
  already moved on from.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

# `streamlit run app/ui.py` does not import this file as part of the `app`
# package, so `from . import guardrails` raises
# "attempted relative import with no known parent package" and the page renders
# blank. Two traps make that failure easy to miss:
#
# 1. Streamlit's own `/_stcore/health` still answers "ok", so a smoke test that
#    only checks the port reports a working app that is not.
# 2. An earlier version of this file branched on `if __package__ in (None, "")`.
#    That is not reliable: Streamlit's script runner can populate `__package__`
#    with a truthy value, which sends execution to the relative-import branch and
#    reproduces the same ImportError.
#
# So there is no branch here. The package root goes on sys.path and every import
# is absolute. That is correct whether the file is run by Streamlit, imported as
# `app.ui`, or executed by runpy, because all three resolve `app` the same way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import generate, guardrails as g  # noqa: E402
from app.conversation import History, resolve_question  # noqa: E402
from app.meminfo import current_rss_mb  # noqa: E402
from app.retrieve import retrieve  # noqa: E402

logging.getLogger("app.ui").setLevel(logging.INFO)

st.set_page_config(
    page_title="HDFC mutual fund facts assistant",
    page_icon="📄",
    layout="centered",
)

DISCLAIMER = "Facts-only. No investment advice."

WELCOME = (
    "Ask a factual question about the HDFC mutual fund schemes below. Answers come "
    "only from official HDFC and SEBI documents, and every answer links the one "
    "document it came from."
)

# The three examples required by gate 6.2 / PRD 3.2 item 7. Chosen to show the
# three outcomes a reviewer needs to see: a plain fact, a follow-up that uses
# conversation memory, and a refusal.
EXAMPLES = [
    "What is the exit load on HDFC Top 100 Fund - Direct Growth?",
    "What is the benchmark of HDFC Flexi Cap Fund - Direct Growth?",
    "Should I buy HDFC ELSS Tax Saver Fund to save tax under section 80C?",
]


# ── session state ────────────────────────────────────────────────────────────


def _init_state() -> None:
    """Set up the per-browser-session state.

    `@st.cache_resource` is deliberately *not* used for the index or the
    embedding model. Those are process-wide singletons inside `app.store` and
    `app.embeddings` already, and wrapping them again would be a second place to
    get lifecycle wrong. Caching them per session instead would load a copy of
    the model per browser tab, which is exactly the memory blow-up gate 6.6
    exists to prevent.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = History()


def clear_chat() -> None:
    """Reset the transcript *and* the memory window.

    Clearing only the transcript would leave the referent intact, so the first
    question after a "clear" would silently resolve against the previous
    conversation. The two have to go together or the button lies.
    """
    st.session_state.messages = []
    st.session_state.history.clear()


# ── rendering ────────────────────────────────────────────────────────────────


def _sidebar() -> None:
    with st.sidebar:
        st.markdown(f"### {DISCLAIMER}")
        st.caption(WELCOME)
        st.divider()
        st.markdown("**Example questions**")
        for i, question in enumerate(EXAMPLES):
            # A key per button so Streamlit does not reuse one widget's state.
            if st.button(question, key=f"example_{i}", use_container_width=True):
                st.session_state.pending = question
        st.divider()
        if st.button("Clear chat", use_container_width=True, type="primary"):
            clear_chat()
            st.rerun()
        rss = current_rss_mb()
        st.caption(f"Memory: {rss:.0f} MB" if rss else "Memory: n/a")


def _sources_expander(payload: dict) -> None:
    """The sources panel under one answer.

    Shows the citation, and the retrieval diagnostics that explain a bad
    answer. The chunk scores are here rather than behind a second toggle
    because "the assistant said something odd" is only debuggable by someone
    who can see which document it was reading.
    """
    citations = payload.get("citations") or []
    with st.expander(f"Sources ({len(citations)})", expanded=False):
        if not citations:
            st.caption(
                "No source. This is either a refusal (no document was read, so "
                "no tokens were spent) or a case where the documents did not "
                "state the fact."
            )
            return
        for c in citations:
            if not c.get("domain_ok", True):
                st.error("This link is not on the official-source allowlist.")
            st.markdown(f"**[{c.get('title') or c.get('scheme') or 'source'}]({c['url']})**")
            st.caption(
                f"{c.get('scheme') or '-'} · {c.get('page_type') or '-'} · "
                f"as of {c.get('as_of') or '-'}"
            )
        retrieval = payload.get("retrieval") or {}
        chunks = retrieval.get("chunks") or []
        if chunks:
            st.divider()
            st.caption(
                f"Read {len(chunks)} of {retrieval.get('n_candidates', '?')} candidate "
                f"chunks (score threshold {retrieval.get('min_score', '?')})."
            )
            st.dataframe(
                [
                    {
                        "score": round(ch.get("score", 0), 3),
                        "rank": round(ch.get("rank_score", 0), 3),
                        "lexical": round(ch.get("lexical", 0), 2),
                        "page_type": ch.get("page_type"),
                        "as_of": ch.get("as_of"),
                    }
                    for ch in chunks
                ],
                use_container_width=True,
                hide_index=True,
            )


def _render_message(message: dict) -> None:
    """Render one stored message, with its answer and sources."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.caption(DISCLAIMER)
            if message.get("as_asked"):
                st.caption(f"Asked as: {message['as_asked']}")
            _sources_expander(message.get("payload") or {})


def _ask(question: str) -> dict:
    """Run the pipeline for one question and return the payload.

    Identical to the CLI's order, including the guardrail gate on the rewrite.
    `resolve_question` refuses to rewrite a question the guard is about to turn
    away, which is what stops "and what's the weather?" mid-conversation from
    being answered out of a mutual-fund document.
    """
    asked, _note = resolve_question(question, st.session_state.history.messages)
    st.session_state.history.append("user", question)

    verdict = g.evaluate(asked)
    if verdict.get("refused"):
        st.session_state.history.append("assistant", verdict.get("answer") or "")
        return {"as_asked": asked if asked != question else "", "payload": verdict}

    try:
        found = retrieve(asked, scheme=verdict.get("scheme"))
    except Exception:  # noqa: BLE001 - the index failing is our fault
        logging.exception("retrieval failed")
        return {
            "as_asked": asked if asked != question else "",
            "payload": {
                "answer": "I could not search the source documents just now. Please try again.",
                "refused": True,
                "intent": "retrieval_error",
                "citations": [],
            },
        }

    result = generate.answer(found, asked)
    # Diagnostics for the sources panel. Not the raw text: the excerpts are
    # source text and this is a surface a screenshot can travel from.
    result["retrieval"] = {
        "n_chunks": len(found.chunks),
        "n_candidates": found.n_candidates,
        "min_score": found.min_score,
        "chunks": [
            {
                "score": round(c.score, 4),
                "rank_score": round(c.rank_score, 4),
                "lexical": round(c.lexical, 4),
                "page_type": c.page_type,
                "as_of": c.as_of,
            }
            for c in found.chunks
        ],
    }
    st.session_state.history.append("assistant", result.get("answer") or "")
    return {"as_asked": asked if asked != question else "", "payload": result}


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    _init_state()
    _sidebar()

    st.title("HDFC mutual fund facts assistant")
    st.caption(f"_{DISCLAIMER}_")

    if not st.session_state.messages:
        st.info(WELCOME)

    for message in st.session_state.messages:
        _render_message(message)

    typed = st.chat_input("Ask a factual question")
    question = st.session_state.pop("pending", None) or typed
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    _render_message(st.session_state.messages[-1])

    with st.chat_message("assistant"):
        with st.spinner("Reading the source documents…"):
            turn = _ask(question)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": turn["payload"].get("answer") or "",
            "as_asked": turn["as_asked"],
            "payload": turn["payload"],
        }
    )
    _render_message(st.session_state.messages[-1])


# `streamlit run app/ui.py` does not execute the file as __main__, so the call
# is unconditional rather than guarded.
main()
