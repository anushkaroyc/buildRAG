"""Token-aware, block-aware chunking under a hard 220-token ceiling.

The brief asks for the chunking strategy to be "decided based on the data"
(architecture 3.5). This module implements the strategy chosen from inspecting
the real parsed corpus in Phase 2; the reasoning is recorded in README.md
(gate 2.9) and the constants below are the ones that decision produced.

The ceiling is not advisory. `sentence-transformers/all-MiniLM-L6-v2` truncates
at 256 word-piece tokens **silently** - no exception, no warning - so a chunk
over 220 tokens gets embedded with its tail dropped. Since exit-load slabs,
lock-in terms and minimum-amount conditions live at the *end* of fee sections,
that failure mode is a confidently wrong answer rather than a crash
(architecture 3.4). `validate_chunks` is the assertion that catches it, and
`tests/test_chunking.py` is what asserts the assertion.

Two things here are measured rather than assumed, and both were wrong the first
time they were written down: the median unit is 170 tokens, not ~35, and the
overlap was inert on 90% of boundaries until `_seed_units` was reworked. The
numbers and the reasoning are at `DEFAULT_CEILING` below.

Token counts come from the model's own tokenizer (see `app/tokenizer.py`), so
"220 tokens" means the same thing here and at embedding time.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .tokenizer import CHUNK_TOKEN_CEILING, count_tokens

# Chosen from the Phase 2 corpus inspection (gate 2.9), and revised after
# measuring the corpus rather than trusting the first guess:
#
#   - 220 keeps a whole factsheet section plus its heading in one chunk without
#     pushing into the 256 truncation zone. Measured mean 168.5, max exactly 220.
#   - Units are whole blocks - a paragraph or a table - and the median unit on
#     this corpus measures **170 tokens**, not the ~35 first assumed. Only blocks
#     over the ceiling get split, so packing on block boundaries is cheap but the
#     granularity is coarse.
#   - That granularity is why the overlap had to be reworked. A 40-token budget
#     cannot buy even one median-sized unit, so seeding with whole units produced
#     an empty seed at ~90% of boundaries - see `_seed_units`, and the measured
#     5.0% -> 51.3% seam coverage in `tests/test_chunking.py`.
#
# The remaining ~49% is arithmetic, not a bug: a ~33-token seed leaves 187 tokens
# of the 220 ceiling, and units above that cannot share it, because
# `_split_long_unit` packs a multi-sentence block up to exactly the ceiling. Getting
# the rest would mean re-tuning the unit granularity, which is a fresh gate 2.9
# decision that invalidates the measured Phase 3/5 recall numbers - so it is left as
# a documented limit rather than changed silently.
DEFAULT_CEILING = CHUNK_TOKEN_CEILING  # 220
DEFAULT_OVERLAP = 40

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
_WHITESPACE = re.compile(r"[ \t ]+")


@dataclass
class Chunk:
    """One embeddable unit plus everything needed to cite it."""

    id: str
    text: str
    n_tokens: int
    n_chars: int
    page_no: int
    sid: str = ""
    is_table: bool = False
    oversized: bool = False
    oversized_reason: str = ""
    # Inherited from the source. SC-2/SC-3 depend on these never being empty.
    url: str = ""
    title: str = ""
    scheme: str = ""
    page_type: str = ""
    content_type: str = ""
    as_of: str = ""
    doc_date: str = ""
    fetched_at: str = ""

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class _Unit:
    """A block already guaranteed to fit the ceiling (or flagged as not fitting)."""

    text: str
    page_no: int
    is_table: bool
    n_tokens: int
    oversized: bool = False
    oversized_reason: str = ""


def _split_words(text: str, ceiling: int) -> list[str]:
    """Last-resort split for a single sentence longer than the ceiling."""
    out: list[str] = []
    buf: list[str] = []
    n = 0
    for word in text.split():
        cost = count_tokens(word + " ")
        if buf and n + cost > ceiling:
            out.append(" ".join(buf))
            buf, n = [], 0
        buf.append(word)
        n += cost
    if buf:
        out.append(" ".join(buf))
    return out


def _split_long_unit(unit: _Unit, ceiling: int) -> list[_Unit]:
    """Break a too-large block into fitting pieces, preserving row/sentence integrity.

    A single line that still will not fit (a dense table row) is emitted as an
    `oversized` unit rather than being truncated, per architecture 3.4. Phase 3
    must then embed it as a two-pass mean of its sub-embeddings, not a truncation.
    """
    if unit.n_tokens <= ceiling:
        return [unit]

    pieces: list[_Unit] = []
    if unit.is_table:
        # Group whole rows. Never cut a row across two chunks.
        rows = [r for r in unit.text.splitlines() if r.strip()]

        def _group(group_rows: list[str]) -> _Unit:
            """One candidate chunk from whole rows, flagged if it still will not fit.

            A group can only exceed the ceiling when it is a single row that is
            itself over budget, since the caller flushes before overflow. Such a row
            is carried whole and flagged rather than truncated (architecture 3.4).

            The flag has to be applied on *every* flush, not just the last one. An
            earlier version only checked the trailing group, so an over-ceiling row
            anywhere but last position was emitted as an ordinary chunk and landed
            in `validate_chunks` as a gate 2.4 violation rather than as the flagged
            unit it is - a silent violation of the one check this plan calls
            non-negotiable. It never fired on the real corpus (no row that long),
            which is exactly why it survived.
            """
            body = "\n".join(group_rows)
            nt = count_tokens(body)
            if nt > ceiling:
                return _Unit(
                    body,
                    unit.page_no,
                    True,
                    nt,
                    True,
                    "table row exceeds the ceiling and cannot be split without cutting a row",
                )
            return _Unit(body, unit.page_no, True, nt)

        buf: list[str] = []
        n = 0
        for row in rows:
            cost = count_tokens(row)
            if buf and n + cost > ceiling:
                pieces.append(_group(buf))
                buf, n = [], 0
            buf.append(row)
            n += cost
        if buf:
            pieces.append(_group(buf))
        return pieces

    # Prose: sentence boundary first, words only if a sentence alone is too big.
    sentences = [s for s in _SENTENCE_SPLIT.split(unit.text) if s.strip()]
    buf, n = [], 0
    for sent in sentences:
        cost = count_tokens(sent)
        if cost > ceiling:
            if buf:
                pieces.append(_Unit(" ".join(buf), unit.page_no, False, n))
                buf, n = [], 0
            for frag in _split_words(sent, ceiling):
                pieces.append(_Unit(frag, unit.page_no, False, count_tokens(frag)))
            continue
        if buf and n + cost > ceiling:
            pieces.append(_Unit(" ".join(buf), unit.page_no, False, n))
            buf, n = [], 0
        buf.append(sent)
        n += cost
    if buf:
        pieces.append(_Unit(" ".join(buf), unit.page_no, False, n))
    return pieces


def _seed_units(prev_chunk: list[_Unit], overlap: int) -> list[_Unit]:
    """Trailing context from the chunk that just closed, for the next chunk's seam.

    A fact that lands on a boundary is invisible to retrieval - the half that
    answers the question is not in either chunk. This returns up to `overlap`
    tokens of the previous chunk's tail.

    **Prose is trimmed to a sentence boundary; a table row never is.** Gate 2.5 is
    about rows: a row cut across the seam is precisely the failure that produces a
    confident wrong answer, so table units are skipped entirely rather than split.
    A sentence boundary inside a paragraph costs nothing comparable.

    This exists because seeding with whole units does not work on this corpus.
    Units here are whole blocks - a paragraph or a table - and after Phase 2's
    corpus inspection the median unit measures **170 tokens** against a 40-token
    overlap budget, with 90% of units larger than the whole budget. Taking whole
    units therefore produced an *empty* seed at ~90% of boundaries: measured on the
    1,888-chunk build, only 5.0% of adjacent chunks shared any text, i.e. the
    overlap existed only on paper. Splitting prose down to the budget is what makes
    the parameter do what `DEFAULT_OVERLAP` says it does: 5.0% -> 51.3%.
    """
    if overlap <= 0:
        return []
    seed: list[_Unit] = []
    n = 0
    for unit in reversed(prev_chunk):
        if unit.is_table or unit.oversized:
            continue
        prose = _Unit(unit.text, unit.page_no, False, unit.n_tokens)
        # Reversed, then inserted at 0: the seed is the *tail* of the previous
        # chunk, which is the part that abuts the seam. Taking the head of a unit
        # instead would prepend context the reader already has and drop the
        # sentence the new chunk actually continues.
        for frag in reversed(_split_long_unit(prose, overlap)):
            if n + frag.n_tokens > overlap:
                break
            seed.insert(0, frag)
            n += frag.n_tokens
        if n >= overlap:
            break
    return seed


def _pack(units: list[_Unit], ceiling: int, overlap: int) -> list[list[_Unit]]:
    """Greedily pack units up to the ceiling, seeding each new chunk with overlap."""
    chunks: list[list[_Unit]] = []
    cur: list[_Unit] = []
    cur_n = 0

    for unit in units:
        if unit.oversized or unit.n_tokens > ceiling:
            if cur:
                chunks.append(cur)
                cur, cur_n = [], 0
            chunks.append([unit])
            continue

        if cur and cur_n + unit.n_tokens > ceiling:
            chunks.append(cur)
            seed = _seed_units(chunks[-1], overlap)
            seed_n = sum(u.n_tokens for u in seed)
            # If the seed leaves no room for the incoming unit, drop the seed
            # rather than emit a chunk that overflows or loop forever.
            cur, cur_n = (seed, seed_n) if seed_n + unit.n_tokens <= ceiling else ([], 0)

        cur.append(unit)
        cur_n += unit.n_tokens

    if cur:
        chunks.append(cur)
    return chunks


def chunk_document(
    doc,
    *,
    ceiling: int = DEFAULT_CEILING,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk a `ParsedDoc`, propagating source metadata onto every chunk.

    Metadata is attached from the source document, never generated by the model.
    That is what makes "exactly one official citation" structurally true rather
    than a prompt instruction the LLM might ignore (architecture 3.2, SC-2/SC-3).
    """
    raw_units: list[_Unit] = []
    for page in doc.pages:
        for block in page.blocks:
            text = _WHITESPACE.sub(" ", block.text).strip()
            if not text:
                continue
            raw_units.append(
                _Unit(
                    text=text,
                    page_no=page.page_no,
                    is_table=block.is_table,
                    n_tokens=count_tokens(text),
                )
            )

    units: list[_Unit] = []
    for unit in raw_units:
        units.extend(_split_long_unit(unit, ceiling))

    packed = _pack(units, ceiling, overlap)

    meta = {
        "url": doc.url,
        "title": doc.title,
        "scheme": doc.scheme,
        "page_type": doc.page_type,
        "content_type": doc.content_type,
        "as_of": doc.as_of or "",
        "doc_date": doc.doc_date or "",
    }

    chunks: list[Chunk] = []
    for i, group in enumerate(packed):
        text = "\n\n".join(u.text for u in group).strip()
        if not text:
            continue
        oversized = any(u.oversized for u in group)
        reason = next((u.oversized_reason for u in group if u.oversized), "")
        chunks.append(
            Chunk(
                id=f"{doc.sid}-{i:04d}",
                text=text,
                n_tokens=count_tokens(text),
                n_chars=len(text),
                page_no=group[0].page_no,
                is_table=all(u.is_table for u in group),
                oversized=oversized,
                oversized_reason=reason,
                **meta,
            )
        )
    return chunks


def validate_chunks(chunks: list[Chunk], ceiling: int = DEFAULT_CEILING) -> dict:
    """Gate 2.4. Returns a report; `violations` must be empty to proceed."""
    violations = [
        {"id": c.id, "n_tokens": c.n_tokens, "page_no": c.page_no, "text": c.text[:160]}
        for c in chunks
        if c.n_tokens > ceiling and not c.oversized
    ]
    flagged = [
        {"id": c.id, "n_tokens": c.n_tokens, "reason": c.oversized_reason}
        for c in chunks
        if c.oversized
    ]
    missing_meta = [
        c.id
        for c in chunks
        if not (c.url and c.page_type and c.content_type and c.title)
    ]
    over_256 = [c.id for c in chunks if c.n_tokens > 256]
    return {
        "n_chunks": len(chunks),
        "ceiling": ceiling,
        "max_tokens": max((c.n_tokens for c in chunks), default=0),
        "mean_tokens": round(sum(c.n_tokens for c in chunks) / len(chunks), 1) if chunks else 0,
        "n_tables": sum(1 for c in chunks if c.is_table),
        "violations": violations,
        "flagged_oversized": flagged,
        "missing_metadata": missing_meta,
        "over_model_limit": over_256,
    }
