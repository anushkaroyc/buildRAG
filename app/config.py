"""Typed application configuration.

Single source of truth for every tunable. Modules must import from here rather than
hard-coding values, so behaviour stays env-driven and testable (implementation.md
Phase 1, check 1.4).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# The embedding model is mandated by the brief (docs/ProblemStatement.txt, line 46).
# This is the *model*; the runtime is ONNX via fastembed. Installing the
# `sentence-transformers` PyTorch library to load it would pull in torch and OOM
# Render's 512 MB free tier (docs/architecture.md 9.1).
BRIEF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Dimensions produced by the model above. Fixed, asserted in Phase 3.
EMBED_DIM = 384

# --- Scope (PRD 3.1) ---------------------------------------------------------
# One AMC, five schemes. The guardrails use these names to tell an in-scope
# question from an off-topic one, and Phase 5's retriever uses them to scope
# retrieval. Kept here, beside EMBED_DIM, because they are a domain constant
# rather than a runtime tunable.
AMC_NAME = "HDFC Mutual Fund"

IN_SCOPE_SCHEMES = (
    "HDFC Top 100 Fund",
    "HDFC Flexi Cap Fund",
    "HDFC ELSS Tax Saver Fund",
    "HDFC Mid Cap Fund",
    "HDFC Nifty 50 Index Fund",
)

# Names the corpus still carries for the in-scope schemes, so a question phrased
# with a former name is recognised as in-scope rather than refused as off-topic
# (SEBI's Mar-2026 categorisation renamed two of these schemes; sources.csv notes
# which). Also the short forms users actually type.
SCHEME_ALIASES = {
    "HDFC Top 100 Fund": (
        "hdfc top 100",
        "hdfc large cap",
        "hdfc large cap fund",
        "hdfc top 100",
        "top 100 fund",
        "large cap fund",
    ),
    "HDFC Flexi Cap Fund": (
        "hdfc flexi cap",
        "flexi cap fund",
        "flexi cap",
    ),
    "HDFC ELSS Tax Saver Fund": (
        "hdfc elss",
        "hdfc elss tax saver",
        "hdfc tax saver",
        "hdfc taxsaver",
        "elss tax saver",
        "elss",
        "tax saver fund",
    ),
    "HDFC Mid Cap Fund": (
        "hdfc mid cap",
        "hdfc midcap",
        "hdfc mid-cap",
        "hdfc mid cap opportunities",
        "mid cap fund",
        "midcap fund",
    ),
    "HDFC Nifty 50 Index Fund": (
        "hdfc nifty 50",
        "hdfc nifty fifty",
        "hdfc nifty 50 index",
        "nifty 50 index fund",
        "nifty 50",
        "nifty fifty",
    ),
}

# The `scheme` string actually stored in each chunk's metadata, keyed by the
# canonical name above.
#
# These disagree for two of the five schemes, and the disagreement is silent:
# Phase 3's parser took the name off the source document, so the index holds
# "HDFC Large Cap Fund" (post-rename) and "HDFC ELSS - Tax Saver Fund" (the
# hyphenation in HDFC's own title), while `IN_SCOPE_SCHEMES` uses the names the
# PRD and the user actually type. Filtering Chroma on the wrong string returns
# **zero** rows with no error - the failure looks exactly like "this fund isn't
# in the corpus", which is the one diagnosis that sends you in the wrong
# direction (implementation.md 5.13, no cross-scheme bleed).
#
# Kept explicit rather than normalised at query time: a lookup table is auditable
# against `data/parsed/`, whereas a fuzzy matcher would hide a future rename
# behind a silent wrong answer. `tests/test_retrieve.py` asserts these values
# exist in the built index, so a re-parse that renames a scheme fails a test
# instead of quietly emptying a filter.
INDEX_SCHEME_NAMES: dict[str, str] = {
    "HDFC Top 100 Fund": "HDFC Large Cap Fund",
    "HDFC Flexi Cap Fund": "HDFC Flexi Cap Fund",
    "HDFC ELSS Tax Saver Fund": "HDFC ELSS - Tax Saver Fund",
    "HDFC Mid Cap Fund": "HDFC Mid Cap Fund",
    "HDFC Nifty 50 Index Fund": "HDFC Nifty 50 Index Fund",
}

# Domains permitted to be cited as a source of record (SC-3). Answer citations are
# read off chunk metadata, so this is enforced at index time; the guardrails'
# refusal links are checked against the same list by tests/test_guardrails.py so a
# refusal cannot quietly point a reviewer off-allowlist.
ALLOWED_CITATION_DOMAINS = (
    "hdfcfund.com",
    "hdfcmf.com",
    "sebi.gov.in",
    "amfiindia.com",
)


class Settings(BaseSettings):
    """Configuration read from the process environment and `.env`."""

    model_config = SettingsConfigDict(
        # `.env` is read through python-dotenv, which pydantic-settings uses as
        # its dotenv backend. Real environment variables win over `.env`, so a
        # Render env var overrides a committed file without an extra code path.
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- LLM (Groq) -------------------------------------------------------
    # Optional at boot: a missing key must not stop the app from starting
    # (PRD ST-2). The app reports itself unconfigured instead of guessing.
    groq_api_key: str | None = None
    # GROQ_MODEL is the current name. LLM_MODEL is kept as a fallback alias so
    # an existing `.env` or deploy config using the old name keeps working
    # instead of silently reverting to the default model.
    groq_model: str = Field(
        default="openai/gpt-oss-20b",
        validation_alias=AliasChoices("GROQ_MODEL", "LLM_MODEL"),
    )
    guard_model: str = "meta-llama/llama-prompt-guard-2-22m"

    # --- embeddings -------------------------------------------------------
    embed_model: str = BRIEF_EMBED_MODEL

    # --- retrieval --------------------------------------------------------
    top_k: int = 5
    min_score: float = 0.35

    # --- guardrails (Phase 4) ---------------------------------------------
    # `guard_mode` selects how the advice/injection guard reaches its decision.
    # The implementation doc (gate 4.6) makes the false-positive rate on the 20
    # golden factual questions the real constraint, so the default is `auto`:
    #
    #   auto     - deterministic patterns decide first; the 22M guard model is
    #              consulted only for questions the patterns cannot place.
    #              A refusal usually costs 0 API calls (gate 4.8).
    #   model    - consult the guard model first, patterns as the backstop.
    #   keywords - never touch the network (offline demos, zero-token budget).
    #
    # Trade-off, stated plainly: the 22M guard is a jailbreak/harm classifier,
    # not a financial-advice classifier. Asked directly, it calls "Should I buy
    # HDFC ELSS?" *safe*, because nothing about it is harmful. It is therefore
    # reliable for prompt-injection and for catching borderline phrasings, and
    # unreliable as the sole advice detector - which is exactly the scenario the
    # implementation doc anticipates when it says "a simpler guard that works
    # beats a smarter one". Refusals still *sound* identical either way (SC-5).
    guard_mode: str = "auto"
    # The 22M guard classifies, it never answers: one output token.
    guard_timeout_s: float = 5.0
    # If the guard is unreachable and the patterns are inconclusive, refuse
    # rather than guess (architecture §10, gate 4.7).
    guard_fail_closed: bool = True
    # Require at least one content term from the question to appear in the
    # retrieved context before Phase 5 may answer. Tuned in Phase 5 (gate 5.8).
    require_context_support: bool = True

    # --- paths ------------------------------------------------------------
    chroma_path: Path = Path("data/chroma")
    sources_csv: Path = Path("data/sources.csv")
    raw_dir: Path = Path("data/raw")
    parsed_dir: Path = Path("data/parsed")

    # --- server -----------------------------------------------------------
    port: int = 10000

    @property
    def groq_configured(self) -> bool:
        """True when a usable API key is present. Never returns the key itself."""
        return bool(self.groq_api_key and self.groq_api_key.strip())

    @property
    def index_present(self) -> bool:
        """True when a built Chroma index is on disk (Phase 3 onwards)."""
        return self.chroma_path.exists()

    def public_dict(self) -> dict:
        """Config that is safe to serve over HTTP.

        Deliberately excludes the API key. Anything added here becomes public.
        """
        return {
            "groq_model": self.groq_model,
            "guard_model": self.guard_model,
            "embed_model": self.embed_model,
            "embed_dim": EMBED_DIM,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "guard_mode": self.guard_mode,
            "chroma_path": str(self.chroma_path),
            "groq_configured": self.groq_configured,
            "index_present": self.index_present,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance. Restart the process to pick up `.env` changes."""
    return Settings()
