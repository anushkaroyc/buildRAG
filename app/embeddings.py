"""Local ONNX embedder for the brief-mandated model.

The brief fixes the *model* as `sentence-transformers/all-MiniLM-L6-v2`; it does
not fix the backend. We load it through `fastembed` (ONNX Runtime) rather than
`pip install sentence-transformers`, because the latter imports torch at
660-930 MB peak RSS and OOMs Render's 512 MB free tier
(docs/architecture.md 3.5, 9.1). `tests/test_config.py` enforces that.

Two deliberate choices:

- **Lazy singleton.** ONNX session + weights load once, on first use, and are
  shared for the process lifetime. Constructing per call would re-read ~90 MB
  and dominate cold-start time (gate 3.9).
- **Threads capped at 2.** ONNX Runtime sizes its thread arenas from
  `OMP_NUM_THREADS`. Left uncapped it grabs one arena per core, which on a
  many-core build machine inflates RSS well past what the 512 MB budget
  tolerates. This must be set *before* onnxruntime is imported, hence the
  lazy import below.

fastembed 0.8.1 ships only an fp32 `model.onnx` (90 MB) for this model; there is
no `-Q` int8 entry in its registry. int8 is therefore the documented fallback if
gate 3.8 comes out amber, not the default (architecture 9.1, gate 3.8).
"""

from __future__ import annotations

import os

# Must precede `import onnxruntime`, which happens inside get_embedder().
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from .config import BRIEF_EMBED_MODEL  # noqa: E402
from .tokenizer import MODEL_MAX_SEQUENCE_LENGTH  # noqa: E402

EMBED_DIM = 384
EMBED_BATCH_SIZE = 32
ONNX_THREADS = 2

_model = None
_model_meta: dict = {}


def get_model():
    """Return the shared `TextEmbedding`, loading it on first call."""
    global _model, _model_meta
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(
            model_name=BRIEF_EMBED_MODEL,
            threads=ONNX_THREADS,
            lazy_load=False,
        )
        _model_meta = {
            "model": BRIEF_EMBED_MODEL,
            "runtime": "fastembed/onnxruntime (no torch)",
            "onnx_threads": ONNX_THREADS,
            "dim": EMBED_DIM,
            "model_max_seq_length": MODEL_MAX_SEQUENCE_LENGTH,
            "quantized": False,
        }
    return _model


def model_meta() -> dict:
    """Provenance for manifest.json. Loads the model if not already loaded."""
    get_model()
    return dict(_model_meta)


def embed_texts(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    """Embed a list of chunks. Batched so peak memory stays flat."""
    if not texts:
        return []
    model = get_model()
    return [list(map(float, vec)) for vec in model.embed(texts, batch_size=batch_size)]


def embed_query(text: str) -> list[float]:
    """Embed a single search query with the same model and settings."""
    vectors = embed_texts([text])
    return vectors[0] if vectors else []
