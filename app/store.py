"""ChromaDB persistence for chunk vectors.

`data/chroma/` is a **persistent** directory, not an in-memory collection, and it
is committed to git: the Render container has neither the RAM nor the boot time
to build an index (architecture 4.1, 7).

One collection, cosine space, metadata passed straight through from the chunk
records. Nothing here generates or rewrites citation metadata - the retriever in
Phase 5 reads these fields, which is what makes "exactly one official citation"
structural rather than a prompt instruction (SC-2, SC-3).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import get_settings

COLLECTION_NAME = "mf_faq_chunks"

# Only fields the retriever and the citation builder actually need. Keeping the
# set small keeps each Chroma record small; the full text stays in
# data/parsed/ and data/chunks/chunks.jsonl.
METADATA_FIELDS = (
    "url",
    "title",
    "scheme",
    "page_type",
    "content_type",
    "as_of",
    "doc_date",
    "page_no",
    "n_tokens",
    "is_table",
)


class Store:
    """Thin wrapper over a Chroma `PersistentClient` collection."""

    def __init__(self, path: Path | None = None, name: str = COLLECTION_NAME):
        settings = get_settings()
        self.path = Path(path or settings.chroma_path)
        self.name = name
        self._client = None
        self._collection = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self.path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.path),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            try:
                self._collection = self.client.get_or_create_collection(
                    name=self.name,
                    configuration={"hnsw": {"space": "cosine"}},
                    metadata={"hnsw:space": "cosine"},
                )
            except TypeError:  # older chroma without `configuration`
                self._collection = self.client.get_or_create_collection(
                    name=self.name, metadata={"hnsw:space": "cosine"}
                )
        return self._collection

    def reset(self) -> None:
        """Delete the collection. Used by a full rebuild, never implicitly."""
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass
        self._collection = None

    def destroy_dir(self) -> None:
        """Remove the whole persistent directory."""
        if self.path.exists():
            shutil.rmtree(self.path)
        self._client = None
        self._collection = None

    # ── writes ───────────────────────────────────────────────────────────────

    def upsert(self, chunks, vectors, batch_size: int = 500) -> int:
        """Insert or replace chunks. Returns the number of records written."""
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
            )
        if not chunks:
            return 0

        total = 0
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]
            self.collection.upsert(
                ids=[c.id for c in batch_chunks],
                embeddings=batch_vectors,
                documents=[c.text for c in batch_chunks],
                metadatas=[_metadata(c) for c in batch_chunks],
            )
            total += len(batch_chunks)
        return total

    # ── reads ────────────────────────────────────────────────────────────────

    def count(self) -> int:
        return int(self.collection.count())

    def query(
        self,
        vector: list[float],
        *,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Cosine nearest neighbours, with documents and metadata attached."""
        res = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return [
            {
                "id": ids[i],
                "document": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else None,
            }
            for i in range(len(ids))
        ]

    def get(self, chunk_id: str) -> dict | None:
        res = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        ids = (res.get("ids") or [])
        if not ids:
            return None
        return {
            "id": ids[0],
            "document": (res.get("documents") or [None])[0],
            "metadata": (res.get("metadatas") or [{}])[0],
        }

    def all_ids(self) -> list[str]:
        return list(self.collection.get(include=[])["ids"])


def _metadata(chunk) -> dict:
    """Project a Chunk onto the stored metadata fields, coercing to Chroma's types.

    Chroma rejects None and non-scalar values, so booleans become ints and empty
    strings are kept (an empty `scheme` is meaningful for the all-schemes digest;
    inventing a placeholder would be worse than the truth).
    """
    out: dict = {}
    for field in METADATA_FIELDS:
        value = getattr(chunk, field, None)
        if field == "is_table":
            out[field] = int(bool(value))
        elif field in ("page_no", "n_tokens"):
            out[field] = int(value or 0)
        else:
            out[field] = "" if value is None else str(value)
    return out


def index_stats(path: Path | None = None) -> dict:
    """On-disk facts about the persistent index, for /healthz and the manifest."""
    store = Store(path=path)
    exists = store.path.exists()
    files = list(store.path.rglob("*")) if exists else []
    return {
        "path": str(store.path),
        "exists": exists,
        "n_files": sum(1 for f in files if f.is_file()),
        "size_bytes": sum(f.stat().st_size for f in files if f.is_file()),
    }
