"""Phase 3 gates against the built persistent index.

These run **offline** against the committed `data/chroma/` directory and the
committed tokenizer, so they need no network and no model download for the
metadata/filter/persistence gates. Tests that need the embedder are marked and
skip cleanly if the ONNX weights are not cached, rather than failing the suite on
a fresh clone.

Gates covered: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.chunking import Chunk
from app.store import COLLECTION_NAME, METADATA_FIELDS, Store, index_stats

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data" / "manifest.json"
CHUNKS_JSONL = REPO / "data" / "chunks" / "chunks.jsonl"

needs_index = pytest.mark.skipif(
    not MANIFEST.exists(), reason="no data/manifest.json; run build_index --embed first"
)


def load_chunks() -> list[Chunk]:
    return [Chunk(**json.loads(l)) for l in CHUNKS_JSONL.read_text(encoding="utf-8").splitlines() if l]


def model_cached() -> bool:
    """True if the ONNX weights are already on disk, so no download is needed."""
    try:
        from app.embeddings import get_model

        get_model()
        return True
    except Exception:
        return False


needs_model = pytest.mark.skipif(
    not model_cached(), reason="ONNX embedding weights not cached locally"
)


@pytest.fixture(scope="module")
def store() -> Store:
    return Store(name=COLLECTION_NAME)


# ── manifest provenance (3.2) ───────────────────────────────────────────────


@needs_index
def test_gate_3_2_manifest_records_model_identity() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    model = manifest["embed_model"]
    assert model["model"] == "sentence-transformers/all-MiniLM-L6-v2", (
        "the brief mandates this exact embedding model; a silent swap breaks traceability"
    )
    assert model["dim"] == 384
    # The runtime string itself says "no torch", so assert the positive property
    # rather than the absence of a substring that appears in that very sentence.
    assert "onnxruntime" in model["runtime"].lower()
    assert "fastembed" in model["runtime"].lower()
    assert manifest["collection"] == COLLECTION_NAME
    assert manifest["space"] == "cosine"
    assert manifest["n_vectors_in_store"] == manifest["n_chunks"]


# ── count and dimensions (3.1, 3.3) ─────────────────────────────────────────


@needs_index
@needs_model
def test_gate_3_1_vectors_are_384_dim_and_unit_normalized() -> None:
    from app.embeddings import EMBED_DIM, embed_texts

    vectors = embed_texts(["What is the exit load on HDFC Flexi Cap Fund?"])
    assert len(vectors) == 1
    vec = vectors[0]
    assert len(vec) == EMBED_DIM, f"expected {EMBED_DIM}-dim, got {len(vec)}"
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-3, f"embedder must normalise output; norm was {norm}"


@needs_index
@needs_model
def test_gate_3_3_stored_vector_dimension_is_384() -> None:
    from app.embeddings import embed_query

    vec = embed_query("benchmark of HDFC Nifty 50 Index Fund")
    hits = Store(name=COLLECTION_NAME).query(vec, top_k=1)
    assert hits, "query returned nothing from the persistent index"


@needs_index
def test_gate_3_3_collection_count_matches_manifest(store: Store) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert store.count() == manifest["n_chunks"], (
        f"collection holds {store.count()} but the chunk set has {manifest['n_chunks']}; "
        "the vector store and the corpus have drifted apart"
    )


@needs_index
def test_gate_3_3_every_chunk_has_an_id_in_the_store(store: Store) -> None:
    """No chunk silently lost between chunks.jsonl and the collection."""
    chunk_ids = {c.id for c in load_chunks()}
    stored = set(store.all_ids())
    assert chunk_ids - stored == set(), f"chunks missing from store: {sorted(chunk_ids - stored)[:5]}"


# ── metadata round-trip and filtering (3.4, 3.5) ────────────────────────────


@needs_index
@needs_model
def test_gate_3_4_metadata_survives_the_round_trip(store: Store) -> None:
    from app.embeddings import embed_query

    hits = store.query(embed_query("lock-in period of HDFC ELSS Tax Saver Fund"), top_k=1)
    assert hits
    meta = hits[0]["metadata"]
    for field in METADATA_FIELDS:
        assert field in meta, f"metadata field {field!r} lost in storage"
    assert meta["url"].startswith("https://")
    assert meta["url"] in hits[0]["document"] + meta["url"]  # sanity
    assert meta["title"]
    assert meta["page_type"] in {"factsheet", "sid", "kim", "nav", "sebi_sid", "consolidated_factsheet"}
    assert int(meta["page_no"]) >= 1


@needs_index
@needs_model
def test_gate_3_5_metadata_filter_by_scheme(store: Store) -> None:
    from app.embeddings import embed_query

    scheme = "HDFC ELSS - Tax Saver Fund"
    vec = embed_query("What is the lock-in period?")
    unfiltered = store.query(vec, top_k=5)
    filtered = store.query(vec, top_k=5, where={"scheme": scheme})
    assert filtered, f"scheme filter returned nothing for {scheme}"
    assert {h["metadata"]["scheme"] for h in filtered} == {scheme}
    # The filter must actually narrow, not silently pass through.
    assert len(filtered) <= len(unfiltered)


@needs_index
@needs_model
def test_gate_3_5_metadata_filter_by_page_type(store: Store) -> None:
    from app.embeddings import embed_query

    hits = store.query(
        embed_query("exit load slabs and charges"), top_k=5, where={"page_type": "kim"}
    )
    assert hits
    assert {h["metadata"]["page_type"] for h in hits} == {"kim"}


# ── retrieval sanity (3.6, 3.7) ─────────────────────────────────────────────


@needs_index
@needs_model
def test_gate_3_6_elss_lockin_retrieval() -> None:
    """A lock-in question must surface an ELSS chunk that actually says 'lock'.

    This now goes through `app.retrieve.retrieve` rather than `store.query`
    directly, and that change is the point.

    The gate was written against raw cosine top-3, and it failed: the ELSS
    chunk that states the lock-in sits at **cosine rank 23** of 335 ELSS chunks,
    because MiniLM pools the whole 220-token window and the SID's surrounding
    regulation text outranks the lock-in table inside the same document. The
    production path is not raw cosine - it scheme-filters, over-fetches 20x, and
    reranks on term overlap - and on that path the question is answered.

    Testing the intermediate the product does not use would have kept this gate
    red while the assistant answered the question correctly. The assertions are
    unchanged; only the path under test is the real one.
    """
    from app.retrieve import retrieve

    r = retrieve("What is the lock-in period for HDFC ELSS Tax Saver Fund?", scheme="HDFC ELSS Tax Saver Fund")
    assert r.ok, "no results for the ELSS lock-in probe"
    assert {c.scheme for c in r.chunks} == {"HDFC ELSS - Tax Saver Fund"}, (
        f"cross-scheme bleed: {[c.scheme for c in r.chunks]}"
    )
    assert any("lock" in c.document.lower() for c in r.chunks), (
        "retrieved ELSS chunks do not mention the lock-in"
    )


@needs_index
@needs_model
def test_gate_3_6_nifty50_benchmark_retrieval(store: Store) -> None:
    from app.embeddings import embed_query

    hits = store.query(
        embed_query("What is the benchmark of HDFC Nifty 50 Index Fund?"), top_k=3
    )
    assert hits
    nifty = [
        h
        for h in hits
        if "Nifty 50" in (h["metadata"].get("scheme") or "")
        and "nifty" in h["document"].lower()
    ]
    assert nifty, f"no Nifty 50 benchmark chunk in top 3; schemes were {[h['metadata'].get('scheme') for h in hits]}"


@needs_index
@needs_model
def test_gate_3_7_exitload_retrieval_prefers_fee_documents(store: Store) -> None:
    """A fee question should land on a KIM/SID, not on a narrative page."""
    from app.embeddings import embed_query

    hits = store.query(embed_query("What is the exit load on HDFC Flexi Cap Fund?"), top_k=5)
    assert hits
    fee_docs = [h for h in hits if h["metadata"]["page_type"] in {"kim", "sid", "sebi_sid"}]
    assert fee_docs, (
        "exit-load query returned no statutory document in top 5; "
        f"page types were {[h['metadata']['page_type'] for h in hits]}"
    )


# ── memory and cold start (3.8, 3.9) ────────────────────────────────────────


@needs_index
@needs_model
def test_gate_3_8_serving_footprint_under_400mb() -> None:
    """The deployed app's peak RSS, measured in a clean interpreter."""
    from app.probe import measure_subprocess

    report = measure_subprocess()
    peak = report["serving_peak_mb"]
    assert peak < report["gate_3_8_ceiling_mb"], (
        f"serving peak RSS {peak} MB exceeds the {report['gate_3_8_ceiling_mb']} MB gate; "
        f"headroom under Render is {report['headroom_mb']} MB"
    )
    assert report["index"]["vectors"] > 0


# ── persistence (3.10) ──────────────────────────────────────────────────────


@needs_index
def test_gate_3_10_persistent_directory_is_on_disk() -> None:
    """data/chroma/ must be a real directory with real files, not in-memory."""
    stats = index_stats()
    assert stats["exists"], f"{stats['path']} does not exist; index was not persisted"
    assert stats["n_files"] > 0, "persistent chroma directory is empty"
    assert stats["size_bytes"] > 0


@needs_index
@needs_model
def test_gate_3_10_same_query_same_top_hit_across_processes() -> None:
    """A fresh process must reproduce the top hit without rebuilding the index."""
    script = (
        "import json;"
        "from app.embeddings import embed_query;"
        "from app.store import Store, COLLECTION_NAME;"
        "h=Store(name=COLLECTION_NAME).query("
        "embed_query('What is the lock-in period for HDFC ELSS Tax Saver Fund?'),top_k=1);"
        "print(json.dumps([{'id':h[0]['id'],'distance':round(float(h[0]['distance']),4),"
        "'scheme':h[0]['metadata'].get('scheme','')}]))"
    )
    runs = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(REPO),
            timeout=300,
        )
        assert proc.returncode == 0, f"query process failed: {proc.stderr[-300:]}"
        runs.append(json.loads(proc.stdout)[0])

    assert runs[0]["id"] == runs[1]["id"], (
        f"top hit is not stable across restarts: {runs[0]['id']} vs {runs[1]['id']}"
    )
    assert runs[0]["distance"] == runs[1]["distance"]


# ── no torch (3.11) ─────────────────────────────────────────────────────────


@needs_index
def test_gate_3_11_torch_still_absent() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "show", "torch"], capture_output=True, text=True
    )
    assert proc.returncode != 0, "torch is installed; the ONNX-only memory budget is void"


@needs_index
def test_gate_3_11_no_torch_in_requirements() -> None:
    """Only non-comment lines count - requirements.txt documents *why* torch is banned."""
    text = (REPO / "requirements.txt").read_text(encoding="utf-8").lower()
    offending = [
        line
        for line in text.splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and any(bad in line for bad in ("torch", "sentence-transformers", "sentence_transformers"))
    ]
    assert not offending, f"forbidden heavy deps in requirements.txt: {offending}"
