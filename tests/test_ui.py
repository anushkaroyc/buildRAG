"""Tests for both UI surfaces.

There are two, and they are not interchangeable:

- `app/ui.py` - the local Streamlit surface, run with `streamlit run`.
- `GET /` - the browser page the Render web service actually serves, from
  `app/templates/index.html` + `static/`. This is the one a reviewer opens.

The UI is deliberately thin, so there is not much logic to test here. What
*is* worth testing is the things that fail silently or only in a browser:

1. `app/ui.py` must be executable as a bare script. `streamlit run` imports it
   as a top-level file, so the relative imports raise ImportError, the app shows
   a blank page, and Streamlit's own `/_stcore/health` still answers "ok" - so
   a smoke test that only checks the port is up reports a working app that is
   not. This is a real regression, caught once already.
2. The three example questions must pass the guard and name an in-scope scheme,
   since gate 6.2 requires them to be clickable and return cited answers.
3. Clearing the chat must clear the memory window too, or a follow-up after a
   "clear" silently resolves against the previous conversation.
4. `GET /` must return 200 and its static assets must resolve. Not theoretical:
   Starlette 1.7 removed the old `TemplateResponse(name, context)` signature
   that the pinned fastapi resolves to, and the failure is a 500 whose traceback
   ends inside Jinja's template cache with `TypeError: unhashable type: 'dict'`
   - naming neither Starlette nor the call site. `/healthz` stayed green
   throughout, because the page and the health check are separate routes.
5. The served page must not build DOM from model or document text via
   `innerHTML`. `/ask` returns LLM output and crawled-page metadata; a page
   that interpolates either as HTML is an XSS sink whose purpose is to display
   retrieved third-party content.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
UI_PATH = REPO / "app" / "ui.py"
TEMPLATE_PATH = REPO / "app" / "templates" / "index.html"
WEB_JS_PATH = REPO / "static" / "app.js"
WEB_CSS_PATH = REPO / "static" / "style.css"


@pytest.fixture(scope="module")
def client():
    """A TestClient for the served app.

    Module-scoped because importing `app.main` and building the Jinja
    environment is the expensive part, and none of these tests mutate app state.
    The index and the embedding model are *not* loaded: both are lazy, which is
    why these tests need neither the network nor an API key.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


def _exec_ui(package: str | None):
    """Execute app/ui.py the way Streamlit does, and return the resulting module.

    Streamlit `exec()`s the compiled source with its own globals rather than
    using importlib, so `import app.ui` does not reproduce the failure. This
    mirrors it, including the `package` value, because a relative import's
    behaviour depends entirely on it.
    """
    source = UI_PATH.read_text(encoding="utf-8")
    glb = {
        "__name__": "__main__",
        "__file__": str(UI_PATH),
        "__doc__": None,
        "__package__": package,
        "__loader__": None,
        "__spec__": None,
        "__builtins__": __builtins__,
    }
    exec(compile(source, str(UI_PATH), "exec"), glb)  # noqa: S102 - see docstring
    return glb


def test_ui_module_runs_as_streamlit_executes_it() -> None:
    """`streamlit run app/ui.py` must not hit an ImportError.

    Streamlit runs the file as a top-level script, outside the `app` package,
    so `from . import guardrails` fails with "attempted relative import with no
    known parent package", the page renders blank, and `/_stcore/health` still
    answers "ok" - so a port check alone reports a working app that is not.
    """
    assert UI_PATH.exists(), "app/ui.py is missing"
    for package in (None, "", "app"):
        try:
            _exec_ui(package)
        except ImportError as exc:  # pragma: no cover - the regression
            pytest.fail(f"app/ui.py fails with __package__={package!r}: {exc}")


def test_ui_uses_absolute_imports_only() -> None:
    """No relative import may survive at module scope."""
    source = UI_PATH.read_text(encoding="utf-8")
    code_lines = [ln for ln in source.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    for ln in code_lines:
        assert not ln.startswith("from ."), f"relative import in code: {ln}"
    assert "sys.path.insert" in source
    # The sys.path insert must come before the app imports.
    assert source.index("sys.path.insert") < source.index("from app import generate")


def test_ui_uses_the_shared_pipeline_not_its_own() -> None:
    """A second code path for answering questions is a second set of bugs."""
    source = UI_PATH.read_text(encoding="utf-8")
    for needed in ("resolve_question", "g.evaluate", "retrieve", "generate.answer"):
        assert needed in source, f"app/ui.py does not use {needed}"
    # It must not call the model directly, or it bypasses the validator.
    assert "call_groq" not in source
    assert "Groq(" not in source


def test_example_questions_pass_the_guard_and_name_a_scheme() -> None:
    """Gate 6.2: the three examples must be clickable and return cited answers."""
    from app import guardrails as g

    EXAMPLES = [
        "What is the exit load on HDFC Top 100 Fund - Direct Growth?",
        "What is the benchmark of HDFC Flexi Cap Fund - Direct Growth?",
        "Should I buy HDFC ELSS Tax Saver Fund to save tax under section 80C?",
    ]
    assert len(EXAMPLES) == 3
    # The first two must reach retrieval. The third is deliberately a refusal:
    # a reviewer needs to see the guardrail fire, so it is allowed to refuse but
    # must refuse for the right reason.
    for question in EXAMPLES[:2]:
        verdict = g.evaluate(question)
        assert verdict.get("intent") == "factual", f"{question!r} was not allowed"
        assert verdict.get("scheme"), f"{question!r} named no scheme"
        assert not verdict.get("refused")

    refusal = g.evaluate(EXAMPLES[2])
    assert refusal.get("intent") == "advice"
    assert refusal.get("refused")


def test_clear_chat_clears_the_memory_window() -> None:
    """Clearing only the transcript leaves the referent, so "clear" would lie."""
    from app.conversation import History, resolve_question

    history = History()
    history.append("user", "What is the benchmark of HDFC Flexi Cap Fund?")
    assert rewrite_would_resolve(history)

    history.clear()
    assert history.referent() is None
    assert not rewrite_would_resolve(history)
    # And the rewrite is a no-op, so a fresh question cannot inherit the scheme.
    assert (
        resolve_question("What about its exit load?", history.messages)[0]
        == "What about its exit load?"
    )


def rewrite_would_resolve(history: History) -> bool:
    from app.conversation import rewrite_question

    return rewrite_question("What about its exit load?", history.messages) != "What about its exit load?"


def test_disclaimer_text_is_present_in_the_ui_source() -> None:
    """Gate 6.1 / SC-10: the disclaimer is a product requirement, not decoration."""
    source = UI_PATH.read_text(encoding="utf-8")
    assert "Facts-only. No investment advice." in source


def test_sources_expander_is_under_each_answer() -> None:
    """The citation is the product's core claim; it has to be reachable."""
    source = UI_PATH.read_text(encoding="utf-8")
    assert "_sources_expander" in source
    assert "expander" in source
    # Called from the assistant-message renderer, not only from main().
    assert source.count("_sources_expander(") >= 2


# ── the served page (GET /) ─────────────────────────────────────────────────


def test_index_route_renders(client) -> None:
    """`GET /` must be a real page, not a 500 and not a placeholder.

    This is the regression test for the Starlette 1.7 signature change: the
    route raised `TypeError: unhashable type: 'dict'` from inside Jinja's
    template cache while `/healthz` kept answering 200, so nothing else in the
    suite would have caught it.
    """
    response = client.get("/")
    assert response.status_code == 200, (
        f"GET / returned {response.status_code}; the deployed page would be "
        f"broken while /healthz stayed green"
    )
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "<title>" in body
    # The placeholder this replaced advertised Phase 6 as pending. If that text
    # comes back, the template was never wired up.
    assert "the browser interface arrives in Phase 6" not in body


def test_static_assets_resolve(client) -> None:
    """A page that 200s while its CSS or JS 404s looks broken and reads fine.

    Checked individually rather than by asserting the HTML mentions the paths:
    the template referencing `/static/app.js` proves nothing about the file
    existing, which is the half that actually breaks.
    """
    body = client.get("/").text
    for asset in ("/static/style.css", "/static/app.js"):
        assert asset in body, f"the page does not reference {asset}"
        response = client.get(asset)
        assert response.status_code == 200, f"{asset} returned {response.status_code}"
    assert len(client.get("/static/style.css").content) > 0
    assert len(client.get("/static/app.js").content) > 0


def test_served_page_shows_the_disclaimer(client) -> None:
    """Gate 6.1 / SC-10, for the surface a reviewer actually opens."""
    body = client.get("/").text
    assert "Facts-only. No investment advice." in body


def test_served_page_posts_to_the_shared_ask_endpoint() -> None:
    """The page must not grow its own answering path.

    The guardrails are the product; a UI that called the model directly would
    bypass the validator and the refusal templates without any test failing.
    """
    source = WEB_JS_PATH.read_text(encoding="utf-8")
    assert '"/ask"' in source or "'/ask'" in source
    for forbidden in ("call_groq", "Groq(", "generate.answer", "guardrails"):
        assert forbidden not in source, (
            f"static/app.js references {forbidden}; the browser client must go "
            f"through POST /ask, not re-implement the pipeline"
        )


def _js_code_only(source: str) -> str:
    """Strip JS comments so a doc comment naming a forbidden API is not a hit.

    The rule being enforced is "never *call* this", and the file documents the
    rule by naming the function. A naive substring test would then fail on the
    comment explaining the very attack it prevents, so the obvious response would
    be to delete the explanation - which is the wrong fix. `//` preceded by a
    colon is left alone so `https://` inside a string is not cut in half.
    """
    without_blocks = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"(?<!:)//[^\n]*", "", without_blocks)


def test_served_page_never_uses_innerhtml() -> None:
    """Answer text is LLM output; citation fields come from crawled pages.

    Both are untrusted. `textContent` cannot execute markup, so the whole
    rendering path uses it and this test fails the moment a future edit reaches
    for the more convenient call.
    """
    source = WEB_JS_PATH.read_text(encoding="utf-8")
    code = _js_code_only(source)
    for forbidden in ("innerHTML", "insertAdjacentHTML", "document.write", "outerHTML"):
        assert forbidden not in code, (
            f"static/app.js builds DOM from model/citation text via {forbidden}; "
            f"use textContent so retrieved content cannot inject markup"
        )
    # textContent has to actually be used, or the page renders nothing.
    assert "textContent" in code


def test_served_page_restricts_link_protocols() -> None:
    """A `javascript:` URL in a document field must not become a live link.

    The only attribute written from data is a citation `href`, so the guard is a
    protocol allowlist: `http`/`https` only. Asserted structurally - the
    rejection of the other two protocols must be present - because asserting the
    *absence* of the string "javascript:" would also fail a comment that
    documents the very attack it prevents.
    """
    source = WEB_JS_PATH.read_text(encoding="utf-8")
    code = _js_code_only(source)
    assert "function safeUrl" in code, "citations are not passed through safeUrl"
    assert 'protocol !== "http:"' in code
    assert 'protocol !== "https:"' in code
    # Every href written from data must go via the guard. `link.href =` appears
    # exactly once, inside addSource, and only ever with a safeUrl() result.
    href_writes = [ln.strip() for ln in code.splitlines() if ".href =" in ln]
    assert len(href_writes) == 1, f"unexpected number of href assignments: {href_writes}"
    assert "safeUrl(" in href_writes[0], (
        f"href is assigned without going through safeUrl: {href_writes[0]}"
    )


def test_served_page_renders_both_citation_shapes(client) -> None:
    """Refusals and answers carry different citation fields; both are shown.

    A refusal returns a singular `citation` (an educational link) with no
    `citations` array, so a UI that reads only `citations` renders "no source"
    on exactly the answers where the guardrail deliberately spent no tokens.
    """
    source = WEB_JS_PATH.read_text(encoding="utf-8")
    assert "payload.citations" in source
    assert "payload.citation" in source


def test_refusal_link_is_an_object_not_a_string() -> None:
    """A refusal `citation` is `{label, url}`, and the page must read it as such.

    Caught by calling the real guard path: an earlier version of the page did
    `url: payload.citation`, which looks right until `safeUrl` is handed an
    object, rejects it as a non-string, and the advice link renders as plain
    unclickable text - a broken citation on exactly the answer whose whole point
    is the link (SC-5). Nothing else failed, because the refusal still returned
    correct text.
    """
    from app import guardrails as g
    from app.prompts import REFUSAL_LINKS

    # The shape the page depends on, taken from the source of truth.
    for intent, link in REFUSAL_LINKS.items():
        if link is None:
            continue
        assert isinstance(link, dict), f"{intent} link is {type(link).__name__}, not a dict"
        assert set(link) >= {"label", "url"}, f"{intent} link keys are {sorted(link)}"

    verdict = g.evaluate("Should I buy HDFC ELSS Tax Saver Fund to save tax under 80C?")
    assert verdict.get("refused") and verdict.get("intent") == "advice"
    citation = verdict.get("citation")
    assert isinstance(citation, dict) and citation.get("url"), (
        f"expected a {{label, url}} refusal link, got {citation!r}"
    )

    # So the page must read .url and .label off it, not treat it as a string.
    code = _js_code_only(WEB_JS_PATH.read_text(encoding="utf-8"))
    assert "payload.citation.url" in code, (
        "the page does not read .url off the refusal link object"
    )
    assert "payload.citation.label" in code


def test_pii_refusal_has_no_link_and_the_page_tolerates_that() -> None:
    """A PII refusal deliberately carries `citation: null`.

    Nothing is rendered for it, and the page must treat that as a normal
    no-source refusal rather than throwing on a null link.
    """
    from app.prompts import REFUSAL_LINKS

    assert REFUSAL_LINKS["pii"] is None, (
        "the PII refusal link is expected to stay null; a PII refuser must "
        "never be redirected somewhere"
    )
    code = _js_code_only(WEB_JS_PATH.read_text(encoding="utf-8"))
    # Guarded by truthiness, so null falls through to the no-source branch.
    assert "else if (payload.citation)" in code


def test_served_layout_does_not_pin_a_fixed_width() -> None:
    """Gate 6.4 (mobile at 375 px), asserted structurally.

    There is no browser in CI, so this cannot be a visual check and does not
    claim to be one. It pins the property that actually breaks a narrow
    viewport: a fixed pixel width on a block element, or a row of chips/buttons
    that cannot wrap. Both are checkable in the stylesheet, and both fail at
    375 px while looking correct on a desktop screenshot.
    """
    css = WEB_CSS_PATH.read_text(encoding="utf-8")
    # The screen-reader-only utility is 1px by definition - that is what hides it
    # visually - so it is excluded before scanning for layout widths.
    scan = re.sub(r"\.visually-hidden\s*\{[^}]*\}", "", css)
    fixed_widths = re.findall(r"(?<!max-)width:\s*(\d+)px", scan)
    assert not fixed_widths, (
        f"static/style.css sets a fixed pixel width ({fixed_widths}); the layout "
        f"must be fluid to stay readable at 375 px"
    )
    # The chip row and the composer actions are the two places a narrow screen
    # would overflow, so both must be allowed to wrap.
    for selector in (".examples", ".composer-actions", ".meta"):
        block = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
        assert block, f"{selector} is missing from the stylesheet"
        assert "flex-wrap: wrap" in block.group(1), (
            f"{selector} cannot wrap, so it overflows a 375 px viewport"
        )
    # The page shell is capped with max-width, which is what keeps prose readable
    # on a desktop without breaking mobile.
    assert "max-width" in css


def test_served_example_questions_pass_the_guard() -> None:
    """Gate 6.2: the page's own example buttons must be clickable and answerable.

    Parsed out of the template rather than duplicated, so editing the button
    text cannot leave a stale copy here claiming a different question is safe.
    """
    from app import guardrails as g

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    questions = re.findall(r'data-question="([^"]+)"', template)
    assert len(questions) == 3, f"expected 3 example questions, found {len(questions)}"

    for question in questions[:2]:
        verdict = g.evaluate(question)
        assert not verdict.get("refused"), f"{question!r} would be refused: {verdict}"
        assert verdict.get("scheme"), f"{question!r} named no in-scope scheme"

    # The third is a deliberate refusal so a reviewer can watch the guard fire.
    assert g.evaluate(questions[2]).get("intent") == "advice"
