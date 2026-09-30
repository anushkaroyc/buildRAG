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
    assert s.groq_model == "openai/gpt-oss-20b"
    assert s.guard_model == "meta-llama/llama-prompt-guard-2-22m"
    assert s.chroma_path == Path("data/chroma")
    assert s.port == 10000


def test_groq_model_env_name_is_the_current_one():
    choices = Settings.model_fields["groq_model"].validation_alias.choices
    assert "GROQ_MODEL" in {c.upper() for c in choices}


def test_groq_model_falls_back_to_the_legacy_llm_model_name(monkeypatch):
    """An older .env must not silently revert to the default model."""
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b")
    s = Settings(_env_file=None)
    assert s.groq_model == "openai/gpt-oss-120b"


def test_groq_model_overrides_the_legacy_name(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "new-model")
    monkeypatch.setenv("LLM_MODEL", "old-model")
    assert Settings(_env_file=None).groq_model == "new-model"


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


def test_env_file_is_not_tracked_by_git():
    """Being ignored is not enough; it must not already be in the index."""
    import subprocess

    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git repository")
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ".env"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    assert result.returncode != 0, ".env is tracked in git - the key would be committed"


def test_env_example_is_tracked_and_carries_no_secret():
    """The template ships with the repo, so it must never contain a key."""
    import subprocess

    example = REPO_ROOT / ".env.example"
    assert example.exists()
    if (REPO_ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", ".env.example"],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        assert result.returncode == 0, ".env.example should be tracked"

    text = example.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("GROQ_API_KEY="):
            assert line.strip() == "GROQ_API_KEY=", "GROQ_API_KEY must ship empty"
    assert "GROQ_MODEL=openai/gpt-oss-20b" in text


def test_settings_load_from_a_dotenv_file(tmp_path):
    """Prove the python-dotenv path works, including an empty key degrading."""
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=\nGROQ_MODEL=openai/gpt-oss-120b\nTOP_K=7\n", encoding="utf-8")

    s = Settings(_env_file=env)
    assert s.groq_model == "openai/gpt-oss-120b"
    assert s.top_k == 7
    # An empty value must read as "unconfigured", not as a usable key (ST-2).
    assert s.groq_configured is False


def test_real_environment_beats_the_dotenv_file(monkeypatch, tmp_path):
    """A Render env var must override a committed .env without extra code."""
    env = tmp_path / ".env"
    env.write_text("GROQ_MODEL=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("GROQ_MODEL", "from-environment")
    assert Settings(_env_file=env).groq_model == "from-environment"
