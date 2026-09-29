"""Typed application configuration.

Single source of truth for every tunable. Modules must import from here rather than
hard-coding values, so behaviour stays env-driven and testable (implementation.md
Phase 1, check 1.4).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The embedding model is mandated by the brief (docs/ProblemStatement.txt, line 46).
# This is the *model*; the runtime is ONNX via fastembed. Installing the
# `sentence-transformers` PyTorch library to load it would pull in torch and OOM
# Render's 512 MB free tier (docs/architecture.md 9.1).
BRIEF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Dimensions produced by the model above. Fixed, asserted in Phase 3.
EMBED_DIM = 384


class Settings(BaseSettings):
    """Configuration read from the process environment and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (Groq) -------------------------------------------------------
    # Optional at boot: a missing key must not stop the app from starting
    # (PRD ST-2). The app reports itself unconfigured instead of guessing.
    groq_api_key: str | None = None
    llm_model: str = "openai/gpt-oss-20b"
    guard_model: str = "meta-llama/llama-prompt-guard-2-22m"

    # --- embeddings -------------------------------------------------------
    embed_model: str = BRIEF_EMBED_MODEL

    # --- retrieval --------------------------------------------------------
    top_k: int = 5
    min_score: float = 0.35

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
            "llm_model": self.llm_model,
            "guard_model": self.guard_model,
            "embed_model": self.embed_model,
            "embed_dim": EMBED_DIM,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "chroma_path": str(self.chroma_path),
            "groq_configured": self.groq_configured,
            "index_present": self.index_present,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance. Restart the process to pick up `.env` changes."""
    return Settings()
