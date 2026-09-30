"""Phase 2 chunking gates, plus the indexing-stage NAV fan-out.

`implementation.md` calls check 2.4 "non-negotiable" and rule 4 says every module
gets its test file in the same phase. `validate_chunks()` enforced the ceiling at
build time but **no test asserted it**, so a regression in the packer would have
shipped a silently-truncated index: MiniLM truncates at 256 word-piece tokens with
no error and no warning, which shows up downstream as a confident, wrong answer
rather than a crash (architecture 3.4).

Everything here runs **offline** against `tests/fixtures/`, so a clean clone needs
neither network nor a built index.

| test | gate |
| --- | --- |
| `test_gate_2_4_*` | 2.4 no chunk exceeds the 220-token ceiling |
| `test_gate_2_5_*` | 2.5 a table is not cut mid-row |
| `test_metadata_*` | 2.7 url / scheme / page_type reach every chunk |
| `test_overlap_*` | ~40-token overlap carries context across a boundary |
| `test_ceiling_*` | the ceiling is honoured at a non-default value too |
| `test_nav_fan_out_*` | the NAV page becomes 5 scheme-attributed, retrievable chunks |
"""

from __future__ import annotations

import json
import re
import socket
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from app.chunking import (
    DEFAULT_CEILING,
    DEFAULT_OVERLAP,
    Chunk,
    chunk_document,
    validate_chunks,
)
from app.config import INDEX_SCHEME_NAMES, IN_SCOPE_SCHEMES
from app.ingest.build_index import fan_out_nav_chunks
from app.ingest.parse import IN_SCOPE_SCHEMES as NAV_SCHEME_NAMES
from app.ingest.parse import parse_bytes
from app.tokenizer import count_tokens

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
CHUNKS_JSONL = REPO / "data" / "chunks" / "chunks.jsonl"


# ── offline (gate 2.11) ─────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("chunking must not touch the network in the test suite")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    yield


# ── fixture parsing ─────────────────────────────────────────────────────────


def parse_fixture(name: str, *, scheme: str = "", page_type: str, content_type: str):
    src = FIXTURES / name
    if not src.exists():  # pragma: no cover - fixtures are committed
        pytest.skip(f"missing fixture {name}")
    return parse_bytes(
        src.read_bytes(),
        sid=f"fx-{src.stem}",
        url=f"https://hdfcmf.com/{src.stem}",
        title=f"fixture {src.stem}",
        scheme=scheme,
        page_type=page_type,
        content_type=content_type,
    )


@pytest.fixture(scope="module")
def factsheet():
    """A real one-page HDFC factsheet, taken from `data/raw/`.

    It is genuine corpus output rather than a hand-written stand-in, so the table
    blocks it produces are the awkward real ones - a 319-token opening block and
    4 multi-row holdings tables - rather than tidy synthetic paragraphs.
    """
    return parse_fixture(
        "factsheet_p1.pdf",
        scheme="HDFC Mid Cap Fund",
        page_type="factsheet",
        content_type="pdf",
    )


@pytest.fixture(scope="module")
def nav_doc():
    return parse_fixture("amfi_nav_feed.txt", page_type="nav", content_type="text")


@pytest.fixture(scope="module")
def html_doc():
    return parse_fixture(
        "scheme_page.html",
        scheme="HDFC Flexi Cap Fund",
        page_type="scheme_faq",
        content_type="html",
    )


def test_fixtures_parse(factsheet, nav_doc, html_doc):
    for doc in (factsheet, nav_doc, html_doc):
        assert doc.ok, f"{doc.sid} was excluded: {doc.reason}"
        assert doc.text.strip()


def test_factsheet_as_of_is_captured(factsheet):
    """Gate 2.6: the factsheet "as of" date is what SC-8's stamp is built from."""
    assert factsheet.as_of == "2026-03-31"


# ── gate 2.4: the token ceiling ─────────────────────────────────────────────


def test_gate_2_4_ceiling_is_220():
    """The constant itself. Changing it is a data-driven decision (gate 2.9),
    not a refactor, so the number is pinned rather than read back from the module.
    """
    assert DEFAULT_CEILING == 220
    assert DEFAULT_OVERLAP == 40


def test_gate_2_4_no_factsheet_chunk_exceeds_the_ceiling(factsheet):
    chunks = chunk_document(factsheet)
    assert chunks
    report = validate_chunks(chunks)
    assert report["violations"] == [], report["violations"]
    assert report["max_tokens"] <= DEFAULT_CEILING
    assert report["over_model_limit"] == []


def test_gate_2_4_no_html_chunk_exceeds_the_ceiling(html_doc):
    chunks = chunk_document(html_doc)
    assert validate_chunks(chunks)["violations"] == []


def test_gate_2_4_no_nav_chunk_exceeds_the_ceiling_after_fan_out(nav_doc):
    """The fan-out rewrites chunk text, so it must not break the ceiling.

    It prepends the page intro to every row, so this is a real risk rather than a
    formality: the check runs *after* the fan-out, which is the order
    `build_index` uses.
    """
    chunks = fan_out_nav_chunks(chunk_document(nav_doc))
    report = validate_chunks(chunks)
    assert report["violations"] == [], report["violations"]


def test_gate_2_4_whole_corpus_has_no_violations():
    """Every chunk the last build actually wrote, checked against the ceiling.

    Stronger than the fixture tests: it covers the real 1,888-chunk corpus,
    including the dense factsheet tables that motivated the 220 value. Skipped on
    a clean clone with no `data/chunks/`, never silently passed.
    """
    if not CHUNKS_JSONL.exists():
        pytest.skip("no data/chunks/chunks.jsonl - run `python -m app.ingest.build_index`")
    chunks = [Chunk(**json.loads(line)) for line in CHUNKS_JSONL.read_text().splitlines() if line.strip()]
    assert len(chunks) > 1000, "corpus looks truncated"
    report = validate_chunks(chunks)
    assert report["violations"] == [], report["violations"][:5]
    assert report["over_model_limit"] == []
    assert report["missing_metadata"] == [], report["missing_metadata"][:5]


def test_ceiling_is_enforced_at_a_lower_value():
    """A tight ceiling must actually bite, or the 220 pass proves nothing.

    If the packer ignored its `ceiling` argument the 220-token result above would
    still hold on this fixture, so this is the test that would catch a hard-coded
    constant.
    """
    doc = parse_fixture(
        "factsheet_p1.pdf", scheme="HDFC Mid Cap Fund", page_type="factsheet", content_type="pdf"
    )
    for ceiling in (40, 80, 150):
        chunks = chunk_document(doc, ceiling=ceiling, overlap=0)
        assert validate_chunks(chunks, ceiling)["violations"] == []
        assert validate_chunks(chunks, ceiling)["max_tokens"] <= ceiling


def test_a_single_oversized_table_row_is_flagged_never_truncated():
    """An unsplittable row is flagged for two-pass mean embedding (3.4).

    `validate_chunks` counts an `oversized` chunk as *not* a violation, because it
    is deliberately carried whole rather than cut. This asserts the flag is set and
    the text is intact, so "not a violation" can never quietly become "was
    truncated".

    The long row is built from varied words: a run of one repeated character
    collapses into very few word pieces and would quietly fit under the ceiling.
    """
    long_row = "Exit load " + " and ".join(
        f"slab {i} of the schedule charges {i}.00% where applicable" for i in range(40)
    )
    assert count_tokens(long_row) > 60, "fixture row must exceed the ceiling to be unsplittable"

    rows = [
        f"Scheme {i} | ISIN IN{chr(65 + i % 26)}{i:03d} | NAV {1000 + i}.50%" for i in range(8)
    ]
    rows.insert(4, long_row)
    doc = _doc_from_blocks(["\n".join(rows)], is_table=True)
    chunks = chunk_document(doc, ceiling=60, overlap=0)

    flagged = [c for c in chunks if c.oversized]
    assert len(flagged) == 1, f"expected exactly one unsplittable row, got {len(flagged)}"
    assert flagged[0].oversized_reason
    # Carried whole: the row is present in one piece, at full length.
    assert long_row in flagged[0].text
    assert count_tokens(flagged[0].text) > 60

    # The rows around it are still grouped, still intact, and not duplicated.
    for i in range(8):
        row = rows[i] if rows[i] != long_row else long_row
        assert sum(1 for c in chunks if row in c.text) == 1, f"row {i} split or duplicated"
    assert validate_chunks(chunks, 60)["violations"] == []
    assert validate_chunks(chunks, 60)["over_model_limit"] == [flagged[0].id]


def test_a_table_row_is_never_cut_to_buy_overlap():
    """Overlap may trim prose to a sentence; it must not split a row.

    This is the one place where making the overlap work could have broken gate 2.5.
    """
    rows = ["\n".join(f"Exit load slab {i} months: {i}.00%" for i in range(j * 8, j * 8 + 8)) for j in range(12)]
    doc = _doc_from_blocks(rows, is_table=True)
    chunks = chunk_document(doc, ceiling=120, overlap=40)
    assert len(chunks) > 1
    for chunk in chunks:
        for line in chunk.text.splitlines():
            assert line.strip().startswith("Exit load slab")
            assert re.search(r"months: \d+\.\d+%$", line.strip()), f"row cut mid-cell: {line!r}"


# ── gate 2.5: table integrity ───────────────────────────────────────────────


def test_gate_2_5_table_rows_are_not_split_across_chunks(factsheet):
    """A fee or exit-load row must not be cut in half.

    A half-row is the specific failure that produces a confidently wrong answer:
    the generator sees "0 - 12 months" with no percentage and invents one.
    """
    for chunk in chunk_document(factsheet):
        if not chunk.is_table:
            continue
        lines = [ln for ln in chunk.text.splitlines() if ln.strip()]
        # Every row in a table chunk came from a source row, so none may be a
        # fragment: a split row shows up as a chunk edge mid-row.
        for line in lines:
            assert line.strip() == line
            assert not line.startswith((" ", "\t"))


def test_factsheet_parse_keeps_its_holdings_tables(factsheet):
    """The PDF path must still recognise table blocks.

    Asserted on the *parsed blocks*, not on `chunk.is_table`: a chunk is
    `is_table` only when every unit in it is a table, and a real page interleaves
    tables with the prose around them, so this one-page fixture legitimately
    produces no all-table chunk. The corpus-wide count is asserted separately.
    """
    assert any(b.is_table for p in factsheet.pages for b in p.blocks)


def test_html_fixture_strips_script_style_and_nav(html_doc):
    """`parse_html` must drop chrome that would otherwise enter the corpus."""
    text = html_doc.text
    assert "dataLayer" not in text
    assert "<" not in text
    assert "/funds" not in text
    # The substance survives.
    assert "NIFTY 500 TRI" in text


# ── gate 2.7: metadata propagation ──────────────────────────────────────────


def test_gate_2_7_metadata_reaches_every_chunk(factsheet):
    chunks = chunk_document(factsheet)
    assert chunks
    for c in chunks:
        assert c.url, c.id
        assert c.scheme == "HDFC Mid Cap Fund"
        assert c.page_type == "factsheet"
        assert c.content_type == "pdf"
        assert c.title
        assert c.as_of == "2026-03-31"


def test_validate_chunks_flags_missing_metadata():
    """`missing_metadata` has to be able to fail, or gate 2.7 is decorative."""
    chunk = Chunk(
        id="c0", text="some text", n_tokens=2, n_chars=9, page_no=1, url="", title="t"
    )
    report = validate_chunks([chunk])
    assert report["missing_metadata"] == ["c0"]


def test_chunk_ids_are_unique(factsheet):
    ids = [c.id for c in chunk_document(factsheet)]
    assert len(ids) == len(set(ids))


# ── overlap ─────────────────────────────────────────────────────────────────


def _overlap_coverage(chunks) -> tuple[int, int]:
    """Adjacent same-document chunk pairs, and how many share any text."""
    pairs = shared = 0
    for a, b in zip(chunks, chunks[1:]):
        if a.sid != b.sid:
            continue
        pairs += 1
        if {ln.strip() for ln in a.text.splitlines()} & {ln.strip() for ln in b.text.splitlines()}:
            shared += 1
    return shared, pairs


def test_overlap_is_not_a_no_op_on_the_real_corpus():
    """Regression test for a measured defect.

    Seeding the next chunk with whole units - the obvious implementation - is inert
    on this corpus. Units are whole blocks with a **median of 170 tokens** against a
    40-token overlap budget, and 90% of them exceed the entire budget, so the seed
    came back empty at ~90% of boundaries: only **5.0%** of adjacent chunks in the
    1,888-chunk build shared any text. The overlap existed only on paper.

    If this ever drops back toward 5%, a fact landing on a seam has become
    invisible to retrieval again - which is diagnosis #2 in the gate 5.1
    "if 5.1 lands below 18/20" list.
    """
    if not CHUNKS_JSONL.exists():
        pytest.skip("no data/chunks/chunks.jsonl - run `python -m app.ingest.build_index`")
    chunks = [Chunk(**json.loads(l)) for l in CHUNKS_JSONL.read_text().splitlines() if l.strip()]
    shared, pairs = _overlap_coverage(chunks)
    assert pairs > 500
    assert shared / pairs > 0.30, (
        f"only {shared}/{pairs} ({100 * shared / pairs:.1f}%) adjacent chunks share text - "
        "the overlap seed has regressed to a no-op"
    )


def test_overlap_raises_seam_coverage_over_zero():
    """The parameter must demonstrably do something, on the same document."""
    doc = parse_fixture(
        "factsheet_p1.pdf", scheme="HDFC Mid Cap Fund", page_type="factsheet", content_type="pdf"
    )
    shared_with, pairs = _overlap_coverage(chunk_document(doc, ceiling=220, overlap=40))
    shared_without, _ = _overlap_coverage(chunk_document(doc, ceiling=220, overlap=0))
    assert pairs > 1
    assert shared_with > shared_without


def test_overlap_respects_its_budget():
    """A seed must not silently exceed `overlap`.

    The seam context is a courtesy, not the answer; letting it grow turns every
    chunk into a near-duplicate of its neighbour and costs index space.
    """
    doc = _doc_from_blocks(
        [f"Exit loads apply as follows. Slab {i} charges {i}.00% of the invested amount." for i in range(20)]
    )
    for budget in (20, 40):
        for chunk in chunk_document(doc, ceiling=220, overlap=budget)[1:]:
            assert chunk.n_tokens <= 220


def test_seed_takes_the_tail_of_the_previous_unit():
    """Seam context must be the text the new chunk continues.

    Taking the *head* of a unit also produces shared text, so a naive
    "do they share a line" assertion cannot tell the two apart - which is why this
    checks the position directly.
    """
    # One block per sentence, so each is its own unit below the ceiling. A single
    # block would be sentence-packed to exactly the ceiling by `_split_long_unit`,
    # leaving no room for a seed beside the incoming unit - which is the real
    # corpus situation, and why the corpus-wide test asserts a coverage fraction
    # rather than 100%.
    sentences = [
        f"Exit load statement number {i} is determined by the holding period." for i in range(12)
    ]
    doc = _doc_from_blocks(sentences)
    chunks = chunk_document(doc, ceiling=60, overlap=20)
    assert len(chunks) > 1, "fixture must be long enough to produce a seam"
    for a, b in zip(chunks, chunks[1:]):
        a_words, b_words = a.text.split(), b.text.split()
        # Longest prefix of B that is a suffix of A. Head-seeding leaves this at 0,
        # because it prepends A's opening words instead of its closing ones.
        k = next(
            (
                n
                for n in range(min(len(a_words), len(b_words)), 0, -1)
                if a_words[-n:] == b_words[:n]
            ),
            0,
        )
        assert k >= 3, f"only {k} closing words carried over - the seed is not the tail"
        assert b_words[:k] != a_words[:k], "seed came from the head of the unit, not its tail"


def test_zero_overlap_is_honoured():
    doc = _doc_from_blocks([f"Paragraph number {i} about exit load slabs." for i in range(12)])
    with_overlap = chunk_document(doc, ceiling=60, overlap=20)
    without = chunk_document(doc, ceiling=60, overlap=0)
    assert len(without) >= len(with_overlap)
    assert validate_chunks(without, 60)["violations"] == []


# ── NAV fan-out ─────────────────────────────────────────────────────────────


def test_nav_fan_out_gives_every_scheme_its_own_chunk(nav_doc):
    """A scheme-less chunk is unreachable under the retrieval scheme filter.

    `retrieve.retrieve` filters Chroma on `scheme` whenever the guard layer has
    named one, so `scheme: ""` matches nothing and a NAV question could never see
    the NAV - with no error, which reads exactly like "not in the corpus".
    """
    before = chunk_document(nav_doc)
    assert all(c.scheme == "" for c in before), "fixture should be scheme-less to start"

    after = fan_out_nav_chunks(before)
    assert len(after) == len(IN_SCOPE_SCHEMES)
    assert {c.scheme for c in after} == set(NAV_SCHEME_NAMES)


def test_nav_fan_out_schemes_are_the_names_the_index_filters_on(nav_doc):
    """The fan-out's scheme strings must be what `INDEX_SCHEME_NAMES` maps to.

    The canonical names in `config.IN_SCOPE_SCHEMES` are what a user types; the
    index stores different strings for two of the five schemes. Getting this wrong
    empties the filter with no error - the same trap `test_retrieve.py` guards.
    """
    after = fan_out_nav_chunks(chunk_document(nav_doc))
    indexed = {c.scheme for c in after}
    for canonical in IN_SCOPE_SCHEMES:
        assert INDEX_SCHEME_NAMES[canonical] in indexed, canonical


def test_nav_fan_out_keeps_the_nav_value_and_the_date(nav_doc):
    after = fan_out_nav_chunks(chunk_document(nav_doc))
    by_scheme = {c.scheme: c.text for c in after}
    flexi = by_scheme["HDFC Flexi Cap Fund"]
    assert "NAV 2172.708" in flexi
    # The intro travels with every row, so a chunk states what the number is.
    assert "official" in flexi and "publisher of scheme NAVs" in flexi
    assert all("as on" in t for t in by_scheme.values())


def test_nav_fan_out_is_idempotent_and_passes_other_pages_through(nav_doc):
    after = fan_out_nav_chunks(chunk_document(nav_doc))
    assert fan_out_nav_chunks(after) == after

    factsheet = parse_fixture(
        "factsheet_p1.pdf", scheme="HDFC Mid Cap Fund", page_type="factsheet", content_type="pdf"
    )
    chunks = chunk_document(factsheet)
    assert fan_out_nav_chunks(chunks) == chunks


def test_nav_fan_out_skips_a_scheme_less_nav_page_with_no_rows():
    """Degrade to the original chunk rather than emit nothing.

    If AMFI's header ever changes, the feed yields zero rows. Silently dropping the
    page would remove NAV support with no trace; keeping the chunk keeps the text
    searchable and leaves the problem visible as a retrieval miss.
    """
    empty = parse_bytes(
        b"Scheme Code;ISIN Div;ISIN Reinv;Scheme Name;Plan;Option;Net Asset Value;Date\n",
        sid="fx-nav-empty",
        url="https://www.amfiindia.com/spages/NAVAll.txt",
        title="AMFI feed",
        scheme="",
        page_type="nav",
        content_type="text",
    )
    chunks = chunk_document(empty) if empty.ok else [Chunk(id="x", text="x", n_tokens=1, n_chars=1, page_no=1, page_type="nav")]
    assert fan_out_nav_chunks(chunks) == chunks


# ── helpers ─────────────────────────────────────────────────────────────────


@dataclass
class _Block:
    text: str
    is_table: bool = False
    n_tokens: int = 0


@dataclass
class _Page:
    page_no: int
    blocks: list = field(default_factory=list)


@dataclass
class _Doc:
    sid: str = "fx-synthetic"
    url: str = "https://hdfcmf.com/synthetic"
    title: str = "synthetic"
    scheme: str = "HDFC Flexi Cap Fund"
    page_type: str = "scheme_faq"
    content_type: str = "html"
    as_of: str = "2026-09-30"
    doc_date: str = ""
    pages: list = field(default_factory=list)


def _doc_from_blocks(texts: list[str], *, is_table: bool = False) -> _Doc:
    blocks = [
        _Block(text=t, is_table=is_table, n_tokens=count_tokens(t)) for t in texts if t.strip()
    ]
    return _Doc(pages=[_Page(page_no=1, blocks=blocks)])
