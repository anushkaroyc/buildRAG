"""Configuration and dependency-hygiene tests.

These run with no network and no API key (implementation.md Phase 1, check 1.6).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import BRIEF_EMBED_MODEL, EMBED_DIM, Settings

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- defaults ---------------------------------------------------------------


def test_defaults_match_architecture_doc():
    s = Settings(_env_file=None)
    assert s.top_k == 5
    assert s.min_score == 0.35
    assert s.llm_model == "openai/gpt-oss-20b"
    assert s.guard_model == "meta-llama/llama-prompt-guard-2-22m"
    assert s.chroma_path == Path("data/chroma")
    assert s.port == 10000


def test_embed_model_is_the_brief_mandated_one():
    """The brief fixes the model. Changing it silently invalidates the index."""
    s = Settings(_env_file=None)
    assert s.embed_model == BRIEF_EMBED_MODEL
    assert s.embed_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert EMBED_DIM == 384


def test_app_boots_without_api_key():
    """A missing key must not stop the process (PRD ST-2)."""
    s = Settings(_env_file=None, groq_api_key=None)
    assert s.groq_configured is False


# --- env-driven config (check 1.4) ------------------------------------------


def test_env_overrides_are_applied(monkeypatch):
    monkeypatch.setenv("TOP_K", "9")
    monkeypatch.setenv("MIN_SCORE", "0.6")
    monkeypatch.setenv("EMBED_MODEL", "some/other-model")
    s = Settings(_env_file=None)
    assert s.top_k == 9
    assert s.min_score == 0.6
    assert s.embed_model == "some/other-model"


def test_blank_key_counts_as_unconfigured(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "   ")
    s = Settings(_env_file=None)
    assert s.groq_configured is False


# --- secret hygiene (check 1.5 / 1.4) --------------------------------------


def test_public_dict_never_leaks_the_key(monkeypatch):
    secret = "gsk_test_do_not_leak_me"
    monkeypatch.setenv("GROQ_API_KEY", secret)
    s = Settings(_env_file=None)

    public = s.public_dict()
    assert secret not in repr(public)
    assert "groq_api_key" not in public
    assert public["groq_configured"] is True


# --- dependency hygiene (check 1.3) ----------------------------------------


def test_requirements_do_not_include_torch():
    """torch would blow the 512 MB deployment budget. Guard against regression."""
    text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    offending = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
        and any(bad in line for bad in ("torch", "sentence-transformers", "sentence_transformers"))
    ]
    assert not offending, f"forbidden heavy deps in requirements.txt: {offending}"


def test_env_file_is_not_committed():
    """.env must be ignored, and only the example template is tracked."""
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    patterns = {line.strip() for line in gitignore if line.strip() and not line.lstrip().startswith("#")}
    assert ".env" in patterns
    assert "!.env.example" in patterns
    assert (REPO_ROOT / ".env.example").exists()


def test_vcs_actually_ignores_env_file():
    """Belt-and-braces: if git is available, prove the rule works."""
    import subprocess

    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git repository")
    result = subprocess.run(
        ["git", "check-ignore", "-q", ".env"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    assert result.returncode == 0, ".env is not actually ignored by git"
