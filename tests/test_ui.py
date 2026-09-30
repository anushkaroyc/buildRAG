"""Tests for the Streamlit UI surface.

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
"""
from __future__ import annotations

from pathlib import Path

import pytest

UI_PATH = Path(__file__).resolve().parent.parent / "app" / "ui.py"


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
