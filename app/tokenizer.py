"""The exact word-piece tokenizer used by the Phase 3 embedder.

Why this module exists
----------------------
The 220-token chunk ceiling (architecture 3.4) is only meaningful if "token"
means the same thing to the chunker and to the embedder. Estimating with
`len(text) // 4` is a guess that drifts on financial text - "₹1,01,821.82 Cr."
and ISINs tokenize badly. So the chunker counts with the *same* tokenizer
`fastembed` will use, and the number it reports is the number that gets embedded.

`tokenizer.json` (455 KB) is pulled once from the model repo and committed under
`data/tokenizer/`, so chunking works offline and in the Render container
(gate 2.10) without re-downloading model weights. Only the tokenizer is stored -
no ONNX weights, so nothing here costs meaningful memory.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .config import BRIEF_EMBED_MODEL

TOKENIZER_DIR = Path("data/tokenizer")
TOKENIZER_FILE = TOKENIZER_DIR / "tokenizer.json"

# Matches the sentence-transformers config for this model. fastembed applies the
# same truncation, so a chunk at or under this survives embedding intact.
MODEL_MAX_SEQUENCE_LENGTH = 256

# Architecture 3.4: headroom for special tokens ([CLS]/[SEP]) below the 256 cap.
CHUNK_TOKEN_CEILING = 220


def _download_tokenizer() -> Path:
    """Fetch tokenizer.json from the model repo. Network required, cached after."""
    from huggingface_hub import hf_hub_download

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    if TOKENIZER_FILE.exists():
        return TOKENIZER_FILE
    cached = hf_hub_download(
        BRIEF_EMBED_MODEL, "tokenizer.json", cache_dir=str(TOKENIZER_DIR / "_hf")
    )
    TOKENIZER_FILE.write_bytes(Path(cached).read_bytes())
    return TOKENIZER_FILE


@lru_cache(maxsize=1)
def get_tokenizer():
    """Return a `tokenizers.Tokenizer` for the brief-mandated embedding model.

    Both truncation *and* padding must be switched off.

    The shipped `tokenizer.json` sets `padding = {length: 128, ...}` alongside
    `truncation = {max_length: 128}`. `no_truncation()` alone leaves the padding
    in place, so every string -- including a 7-token one -- comes back as exactly
    128 ids. `count_tokens` would then report 128 for everything, the 220-token
    ceiling would be measured against padded lengths, and a chunk of real content
    could be far longer than 220 while appearing to be 128.
    """
    from tokenizers import Tokenizer

    path = _download_tokenizer()
    tok = Tokenizer.from_file(str(path))
    # The chunker must be the only thing enforcing the 220 ceiling. A silent
    # truncation or a fixed-length pad inside the tokenizer defeats that.
    tok.no_truncation()
    tok.no_padding()
    return tok


def count_tokens(text: str) -> int:
    """Number of word-piece tokens in `text` (no special tokens)."""
    if not text:
        return 0
    return len(get_tokenizer().encode(text, add_special_tokens=False).ids)


def encode_ids(text: str) -> list[int]:
    return get_tokenizer().encode(text, add_special_tokens=False).ids


def decode_ids(ids: list[int]) -> str:
    return get_tokenizer().decode(ids)


def tokenizer_fingerprint() -> dict:
    """Provenance of the tokenizer, recorded in the manifest (gate 3.2)."""
    path = _download_tokenizer()
    meta: dict = {"model": BRIEF_EMBED_MODEL, "tokenizer_file": str(path)}
    try:
        cfg = json.loads(
            (path.parent / "config_sentence_transformers.json").read_text(encoding="utf-8")
        )
        meta["max_seq_length"] = cfg.get("max_seq_length")
    except Exception:
        meta["max_seq_length"] = MODEL_MAX_SEQUENCE_LENGTH
    return meta
