"""Bytes -> clean, structured text.

Produces `ParsedDoc`s made of `Block`s. Blocks are the unit the chunker packs, and
each carries an `is_table` flag so a fee table is never cut mid-row
(architecture 3.4 / gate 2.5).

Three source shapes are handled:

| Shape | Path | Notes |
| --- | --- | --- |
| PDF | `pypdf`, tables detected from the linear text | The corpus is PDF-first |
| HTML | `BeautifulSoup` + `lxml`, boilerplate stripped | SEBI scheme records |
| `;`-delimited text | hand-parsed, filtered to the 5 in-scope schemes | AMFI NAV feed |

Two deliberate choices worth stating:

**pdfplumber is a fallback, not the default.** pypdf already linearises these
factsheets legibly ("Company Industry* Max Financial Services Ltd. Insurance
4.50"). Running pdfplumber over every page as well would put the same table in
the corpus twice, so it is used only for pages where pypdf finds almost no text
but the page is not a scan - i.e. a vector-drawn table we would otherwise lose.

**Repeated page furniture is removed by frequency, not by regex.** Factsheets
repeat the risk disclaimer, the mission/vision block and a "1/3" page marker on
every page. Counting which lines appear on >=60% of pages removes them
generically, including next month's new boilerplate.
"""

from __future__ import annotations

import calendar
import io
import logging
import re
import warnings
from dataclasses import asdict, dataclass, field
from datetime import date

from ..tokenizer import count_tokens

# pypdf logs one "fontTools is required to fully parse the encoding of a CFF
# Type1 font" warning per embedded font. HDFC's PDFs embed several, so a 25-source
# build emits ~200 lines of noise that hides real problems. Extraction is
# unaffected: it is a glyph-mapping hint, not a text-extraction failure.
logging.getLogger("pypdf").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*fontTools is required.*")

# A document yielding fewer than this is treated as a scan and dropped
# (architecture 3.6 - never feed OCR garbage to the embedder).
# PDF-only: a `.txt` feed cannot be a scan, and a legitimately tiny HTML page
# would be wrongly discarded by a 200-token floor.
MIN_PDF_TOKENS = 200
MIN_TEXT_TOKENS = 30
# Below this, a page is a candidate for the pdfplumber vector-table fallback.
MIN_PAGE_CHARS_FOR_TABLE_FALLBACK = 40
MAX_PDFPLUMBER_PAGES = 4
MAX_PDFPLUMBER_BYTES = 3_000_000

_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}
_MONTHS.update({m[:3].lower(): i for m, i in list(_MONTHS.items())})

BOILERPLATE_MIN_PAGES = 2
BOILERPLATE_PAGE_FRACTION = 0.6

# Line-level noise that frequency-counting cannot catch (it appears once).
_NOISE_PATTERNS = [
    re.compile(r"^\d+\s*/\s*\d+$"),                                   # "1/3"
    re.compile(r"^Fund Facts\s*-\s*HDFC", re.I),
    re.compile(r"^PAGE NO\.?$", re.I),
    re.compile(r"mutual fund investments are subject to market risks", re.I),
    re.compile(r"^read all scheme related documents carefully", re.I),
    re.compile(r"^for latest riskometer", re.I),
    re.compile(r"^(mission|vision)\s*:\s*to be", re.I),
    re.compile(r"^contact your MFD", re.I),
    re.compile(r"^source\s*:\s*bloomberg", re.I),
    re.compile(r"^page \d+ of \d+$", re.I),
    re.compile(r"^scheme riskometer", re.I),
    re.compile(r"^benchmark riskometer", re.I),
    re.compile(r"^#\s*for latest riskometer", re.I),
]

# NOTE: the all-schemes digest ("HDFC MF Factsheet - March 2026", 140 pages, ~180
# schemes) was evaluated and deliberately EXCLUDED from sources.csv. Its pages
# that mention an in-scope scheme are cross-scheme roster and fund-manager
# biography pages, so a scope filter naming an in-scope scheme cannot exclude
# them, and they embedded as confident matches to specific questions: an "exit
# load on HDFC Flexi Cap Fund" query retrieved a manager biography as its top
# hit. All five schemes already have three dedicated factsheets each, so the
# digest adds no scheme-specific fact they do not already cover.

# A line is treated as a table row when numbers dominate it, not merely when it
# contains numbers. The first version required only two numeric tokens, which any
# prose sentence with a rate and a period satisfies - so the exit-load bullet
# "an Exit Load of 1.00% is payable if Units are redeemed within 1 year" was
# classified as a table row, got flushed into its own block, and the sentence was
# cut in half at "within 1" | "year from the date of allotment". Requiring the
# numeric share of the line to dominate keeps real tables (rows are mostly
# figures) and stops splitting sentences.
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?\s*%?")
_ROWISH = re.compile(r"(?:%\s|\d+\.\d{2,}|INR\s+\d|\d+\s*%)")
_NUMERIC_TOKENS_MIN = 2
_NUMERIC_SHARE_MIN = 0.45

# Scope guard. An all-schemes digest covers ~180 HDFC funds; without a filter it
# would put 1000+ chunks about out-of-scope schemes (HDFC Liquid Fund, debt
# schemes) into a corpus whose whole promise is 5 equity schemes, and retrieval
# could surface them. Pages not naming an in-scope scheme are dropped.
_SCOPE_TOKENS = [
    r"HDFC\s+Large\s+Cap\s+Fund",
    r"HDFC\s+Flexi\s+Cap\s+Fund",
    r"HDFC\s+ELSS\s*[-–]?\s*Tax\s*Saver(?:\s+Fund)?",
    r"HDFC\s+Mid[-\s]Cap\s+Fund",
    r"HDFC\s+Nifty\s*50\s+Index\s+Fund",
]
_SCOPE_RE = re.compile("|".join(_SCOPE_TOKENS), re.I)


def _norm(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _is_noise(line: str) -> bool:
    s = _norm(line)
    if not s or len(s) > 260:
        return False
    return any(p.search(s) for p in _NOISE_PATTERNS)


def _looks_tabular(line: str) -> bool:
    """True when numbers dominate the line - i.e. it is a row, not a sentence."""
    s = _norm(line)
    if not s or len(s) > 300:
        return False
    if not _ROWISH.search(s):
        return False
    cells = [c for c in re.split(r"\s*\|\s*|\s+", s) if c and c != "|"]
    if not cells:
        return False
    numeric = sum(1 for c in cells if _NUM.fullmatch(c.strip()))
    if numeric < _NUMERIC_TOKENS_MIN:
        return False
    return numeric / len(cells) >= _NUMERIC_SHARE_MIN


@dataclass
class Block:
    """A packable unit of text. Tables are atomic (never split mid-row)."""

    text: str
    is_table: bool = False
    n_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.n_tokens:
            self.n_tokens = count_tokens(self.text)


@dataclass
class Page:
    page_no: int
    blocks: list[Block] = field(default_factory=list)


@dataclass
class ParsedDoc:
    sid: str
    url: str
    title: str
    scheme: str
    page_type: str
    content_type: str
    ok: bool = True
    reason: str = ""
    as_of: str | None = None
    doc_date: str | None = None
    pages: list[Page] = field(default_factory=list)
    n_tokens: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for p in self.pages for b in p.blocks)

    def to_json(self) -> dict:
        return asdict(self)


# ── as-of / document-date capture (gate 2.6, feeds SC-8) ────────────────────


def _iso(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _end_of_month(year: int, month: int) -> str | None:
    return _iso(year, month, calendar.monthrange(year, month)[1])


def _month_num(name: str) -> int | None:
    return _MONTHS.get(name.strip().lower())


_RE_AS_ON = re.compile(
    r"as on\s+(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9}),?\s+(\d{4})", re.I
)
_RE_AUM = re.compile(r"\bAUM\b[\s\S]{0,40}?([A-Za-z]{3,9})\s+(\d{4})")
_RE_DATED = re.compile(r"dated\s+([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})", re.I)
_RE_DMY = re.compile(r"\b(\d{1,2})-([A-Za-z]{3})-(\d{4})\b")


def capture_as_of(text: str) -> str | None:
    """Best guess at the date the *facts* are stated as of, not the fetch date.

    Priority is deliberate: an explicit "as on <date>" beats an AUM month label,
    which beats the document's own date. SC-8 stamps this on every answer, so
    an over-eager match here would put a wrong date in front of a user.
    """
    m = _RE_AS_ON.search(text)
    if m:
        mo = _month_num(m.group(2))
        if mo:
            return _iso(int(m.group(3)), mo, int(m.group(1)))

    m = _RE_AUM.search(text)
    if m:
        mo = _month_num(m.group(1))
        if mo:
            return _end_of_month(int(m.group(2)), mo)

    m = _RE_DMY.search(text)
    if m:
        mo = _month_num(m.group(2))
        if mo:
            return _iso(int(m.group(3)), mo, int(m.group(1)))
    return None


def capture_doc_date(text: str) -> str | None:
    m = _RE_DATED.search(text)
    if m:
        mo = _month_num(m.group(1))
        if mo:
            return _iso(int(m.group(3)), mo, int(m.group(2)))
    return None


# ── PDF ─────────────────────────────────────────────────────────────────────


# Column separator. pdfplumber pads columns with runs of spaces; collapsing them
# to a visible pipe keeps label and value adjacent *and* marks them as a row.
_COL_GAP = re.compile(r"[ \t]{2,}")

# A short standalone line: no digits, no column separator, not sentence-terminated.
_LABELISH = re.compile(r"^[A-Za-z][A-Za-z0-9 /&().,'*-]{1,43}$")
_SENTENCE_END = re.compile(r"[.:;%)\]]$")



def _normalise_line(raw: str) -> str:
    s = _COL_GAP.sub(" | ", raw.replace("\n", " ").strip())
    # HDFC's factsheets use "$$" as a highlight marker; it renders as literal
    # text and reads as noise to the model. "$$ Exit Load: (i) No Exit Load..."
    # becomes "Exit Load: (i) No Exit Load...", which is the sentence we want.
    s = re.sub(r"^\$\$+\s*", "", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def _pdf_page_lines(data: bytes) -> tuple[list[list[str]], list[str]]:
    """Per-page visual lines, in reading order, with column gaps preserved.

    Why pdfplumber instead of pypdf's plain mode
    --------------------------------------------
    pypdf sorts text by *column*, so a two-column "Fund Facts" block comes out as
    all the labels, then all the values:

        Category of Scheme        Large-CapFund
        Fund Manager*             Mr. Rahul Baijal (since July 29, 2022)
        Inception Date            October 11, 1996
        $$ Exit Load
                                  In respect of each purchase ... 1.00% ...

    The label and its value are on the *same visual line*, but they end up in
    different blocks, and therefore in different chunks. An "exit load" question
    would then retrieve the label without the 1.00% figure, or vice versa - the
    worst possible failure for a facts-only bot, because the answer is plausible
    and wrong. `extract_text_lines()` groups by visual line instead, so the pair
    stays together in one block. pypdf's `extraction_mode="layout"` also groups by
    line but pads to page width, which wastes the token budget on whitespace.

    pypdf remains the fallback: it is 20x faster and handles pages pdfplumber
    cannot parse. Whichever succeeds per page wins; failures are logged, not
    silent.
    """
    warnings: list[str] = []
    n_pages = _pdf_page_count(data)
    lines_per_page: list[list[str]] = []

    plumber_lines: list[list[str]] | None = None
    try:
        import pdfplumber

        plumber_lines = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                raw = page.extract_text_lines() or []
                plumber_lines.append([_normalise_line(ln.get("text", "")) for ln in raw])
    except Exception as exc:
        warnings.append(f"pdfplumber unavailable ({type(exc).__name__}); using pypdf only")
        plumber_lines = None

    if plumber_lines is not None and len(plumber_lines) != n_pages:
        warnings.append(
            f"pdfplumber returned {len(plumber_lines)} pages, expected {n_pages}"
        )

    for i in range(n_pages):
        chosen: list[str] = []
        if plumber_lines is not None and i < len(plumber_lines) and plumber_lines[i]:
            chosen = plumber_lines[i]
        else:
            chosen = [
                _normalise_line(ln)
                for ln in _pypdf_page_lines(data, i)
                if _normalise_line(ln)
            ]
            if not chosen:
                warnings.append(f"page {i + 1}: no text extracted by either extractor")
        lines_per_page.append(chosen)

    return lines_per_page, warnings


def _pdf_page_count(data: bytes) -> int:
    from pypdf import PdfReader

    return len(PdfReader(io.BytesIO(data)).pages)


def _pypdf_page_lines(data: bytes, index: int) -> list[str]:
    from pypdf import PdfReader

    try:
        return (PdfReader(io.BytesIO(data)).pages[index].extract_text() or "").splitlines()
    except Exception:
        return []


def parse_pdf(data: bytes, scope_re: "re.Pattern[str] | None" = None) -> tuple[list[Page], list[str]]:
    """Extract per-page text, strip repeated furniture, mark table runs.

    `scope_re`, when given, drops pages that do not name an in-scope scheme.
    """
    raw_lines, warnings = _pdf_page_lines(data)
    n_pages = len(raw_lines)

    # Which lines are page furniture? Count normalised lines per page.
    freq: dict[str, set[int]] = {}
    for i, lines in enumerate(raw_lines):
        for s in lines:
            if s and len(s) <= 260:
                freq.setdefault(s, set()).add(i)

    threshold = max(BOILERPLATE_MIN_PAGES, int(n_pages * BOILERPLATE_PAGE_FRACTION))
    boilerplate = {s for s, pages in freq.items() if len(pages) >= threshold and len(s) <= 260}
    if boilerplate:
        warnings.append(f"stripped {len(boilerplate)} repeated boilerplate lines")

    pages: list[Page] = []
    for i, lines in enumerate(raw_lines):
        blocks: list[Block] = []
        buf: list[str] = []

        def flush() -> None:
            if not buf:
                return
            body = "\n".join(buf).strip()
            if body:
                blocks.append(
                    Block(text=body, is_table=all(_looks_tabular(x) for x in buf))
                )
            buf.clear()

        for s in lines:
            if not s or s in boilerplate or _is_noise(s):
                flush()
                continue

            # A right-column label ("$$ Exit Load") is vertically centred in a
            # left-column bullet list, so line-grouping lands it *between* two
            # lines of the same sentence:
            #     "• In respect of each purchase ... redeemed / switched-"
            #     "$$ Exit Load"
            #     "year from the date of allotment."
            # Keeping the label would break "within 1 year" apart, and the two
            # halves land in different chunks - a fee statement the bot could
            # cite without its rate. A real heading never interrupts a sentence,
            # so when the buffered line is still mid-sentence, drop the label
            # and let the sentence close. Where the label really is a heading it
            # flushes as one, as before.
            if (
                _LABELISH.match(s)
                and "|" not in s
                and not _NUM.search(s)
                and buf
                and not _SENTENCE_END.search(buf[-1])
            ):
                continue

            if _looks_tabular(s):
                flush()
                # Consecutive table rows stay in one block so a fee slab or a
                # market-cap table is never split across a chunk boundary.
                if blocks and blocks[-1].is_table:
                    blocks[-1].text += "\n" + s
                else:
                    blocks.append(Block(text=s, is_table=True))
                continue
            buf.append(s)
        flush()

        page_text = "\n".join(lines)
        # Keep the ORIGINAL page number even when pages are dropped, so a
        # citation can still say "see page 4" of the source PDF.
        if scope_re is not None and not scope_re.search(page_text):
            warnings.append(f"page {i + 1}: out of scope, dropped")
            continue

        pages.append(Page(page_no=i + 1, blocks=blocks))

    return pages, warnings


# ── HTML ────────────────────────────────────────────────────────────────────

_HTML_DROP_TAGS = ["script", "style", "noscript", "svg", "iframe", "nav"]

# Site furniture. Matched on class/id, not tag name: SEBI wraps its entire page
# in <form> and <header>, so dropping those tags by name silently deletes the
# content (this cost one source before it was caught).
_HTML_CHROME = re.compile(
    r"(menu|cookie|consent|banner|marquee|breadcrumb|social|newsletter|popup|modal"
    r"|site-?header|top-?bar|mobile-?header|overlap|search)",
    re.I,
)


def _find_content_root(soup):
    """Prefer the page's own content container over <body>."""
    if soup.find("main"):
        return soup.find("main")
    for ident in ("main-content", "maincontent", "content", "main"):
        node = soup.find(id=re.compile(ident, re.I))
        if node:
            return node
    node = soup.find(attrs={"class": re.compile(r"(main-)?content", re.I)})
    return node or soup.body or soup


def parse_html(data: bytes) -> tuple[list[Page], list[str]]:
    from bs4 import BeautifulSoup

    warnings: list[str] = []
    soup = BeautifulSoup(data, "lxml")
    for tag in soup(_HTML_DROP_TAGS):
        tag.decompose()

    root = _find_content_root(soup)
    if root is None:
        warnings.append("no content container found")
        return [Page(page_no=1, blocks=[])], warnings

    # Strip chrome by class/id signature, leaving <form>/<header> content alone.
    for node in list(root.find_all(True)):
        if node.decomposed:
            continue
        ident = " ".join(filter(None, (node.get("id"), " ".join(node.get("class") or []))))
        if ident and _HTML_CHROME.search(ident):
            node.decompose()

    blocks = [Block(text=_norm(s)) for s in root.stripped_strings if _norm(s)]
    if not blocks:
        warnings.append("no extractable text (JS-rendered?)")
        return [Page(page_no=1, blocks=[])], warnings
    return [Page(page_no=1, blocks=blocks)], warnings


# ── AMFI `;`-delimited NAV feed ─────────────────────────────────────────────

# The feed lists every scheme in India. Only the five in scope are kept, or the
# corpus would be ~20,000 rows of schemes we must never answer about.
IN_SCOPE_SCHEMES = [
    "HDFC Large Cap Fund",
    "HDFC Flexi Cap Fund",
    "HDFC ELSS - Tax Saver Fund",
    "HDFC Mid Cap Fund",
    "HDFC Nifty 50 Index Fund",
]

_RE_NAVI = re.compile(r"^HDFC\s+(ELSS\s*-\s*Tax Saver|Large Cap|Flexi Cap|Mid Cap|Nifty\s*50)\s+Index\s+Fund$", re.I)


def parse_nav_feed(text: str) -> tuple[list[Page], list[str], str | None]:
    """Filter the AMFI feed to in-scope Direct Growth rows.

    Returns (pages, warnings, as_of). Columns per the AMFI header are
    `Scheme Code;ISIN Div;ISIN Reinv;Name;Plan;Option;NAV;Date`.
    """
    warnings: list[str] = []
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return [Page(page_no=1, blocks=[])], ["empty NAV feed"], None

    header = [c.strip() for c in lines[0].split(";")]
    try:
        i_name, i_plan, i_opt, i_nav, i_date = (
            header.index("Scheme Name"),
            header.index("Plan"),
            header.index("Option"),
            header.index("Net Asset Value"),
            header.index("Date"),
        )
    except ValueError:
        return [Page(page_no=1, blocks=[])], [f"unexpected NAV header: {header}"], None

    want = {s.lower(): s for s in IN_SCOPE_SCHEMES}
    canonical: dict[str, str] = {}
    rows: list[str] = []
    as_of: str | None = None

    for line in lines[1:]:
        cols = [c.strip() for c in line.split(";")]
        if len(cols) <= i_date:
            continue
        name = cols[i_name]
        if name.lower() not in want:
            continue
        if cols[i_plan].lower() != "direct plan" or cols[i_opt].lower() != "growth option":
            continue
        canonical[name.lower()] = want[name.lower()]
        rows.append(
            f"{want[name.lower()]} | Direct Plan | Growth Option | NAV {cols[i_nav]} | as on {cols[i_date]}"
        )
        if as_of is None:
            as_of = cols[i_date]

    missing = sorted(set(want.values()) - set(canonical.values()))
    if missing:
        warnings.append(f"not present in AMFI feed: {', '.join(missing)}")

    intro = (
        "AMFI daily NAV feed (Association of Mutual Funds of India), the official "
        "publisher of scheme NAVs. Direct Growth option only, as of the date shown "
        "against each scheme."
    )
    blocks = [Block(text=intro)] + [Block(text=r, is_table=True) for r in rows]
    if not rows:
        warnings.append("no in-scope Direct Growth rows matched")
    return [Page(page_no=1, blocks=blocks)], warnings, as_of


# ── Entry point ─────────────────────────────────────────────────────────────


def parse_bytes(
    data: bytes,
    *,
    sid: str,
    url: str,
    title: str,
    scheme: str,
    page_type: str,
    content_type: str,
) -> ParsedDoc:
    """Parse raw bytes into a `ParsedDoc`, or an exclusion record with a reason."""
    doc = ParsedDoc(
        sid=sid,
        url=url,
        title=title,
        scheme=scheme,
        page_type=page_type,
        content_type=content_type,
    )

    try:
        if content_type == "pdf":
            # An all-schemes digest is only useful for the 5 in-scope schemes.
            scope_re = _SCOPE_RE if page_type == "consolidated_factsheet" else None
            doc.pages, doc.warnings = parse_pdf(data, scope_re=scope_re)
        elif content_type == "html":
            doc.pages, doc.warnings = parse_html(data)
        elif content_type == "text":
            pages, warns, nav_as_of = parse_nav_feed(data.decode("utf-8", "replace"))
            doc.pages, doc.warnings = pages, warns
            doc.as_of = _normalise_date(nav_as_of)
        else:
            doc.ok = False
            doc.reason = f"unsupported content_type {content_type!r}"
            return doc
    except Exception as exc:
        doc.ok = False
        doc.reason = f"parse failed: {type(exc).__name__}: {str(exc)[:160]}"
        return doc

    full = doc.text
    doc.n_tokens = count_tokens(full)

    # A PDF below the floor is treated as image-only (architecture 3.6). Text and
    # HTML use a far lower floor: they cannot be scans, and small is legitimate.
    floor = MIN_PDF_TOKENS if content_type == "pdf" else MIN_TEXT_TOKENS
    if doc.n_tokens < floor:
        doc.ok = False
        doc.reason = (
            f"only {doc.n_tokens} tokens extracted (floor {floor}) "
            f"- excluded as image-only/empty; never embedded (content_type={content_type})"
        )
        return doc

    if doc.as_of is None:
        doc.as_of = capture_as_of(full)
    if doc.doc_date is None:
        doc.doc_date = capture_doc_date(full)
    if doc.as_of is None:
        # SID/KIM state "This Scheme Information Document is dated <date>" and
        # carry no "as on" line. Their terms are as of the document date, and
        # SC-8 needs *some* honest stamp rather than a blank.
        doc.as_of = doc.doc_date

    if doc.page_type == "factsheet" and doc.as_of is None:
        doc.warnings.append("no as-of date found in factsheet (SC-8 stamp will fall back)")
    return doc


def _normalise_date(value: str | None) -> str | None:
    """Accept `29-Sep-2026` or ISO and return ISO."""
    if not value:
        return None
    m = _RE_DMY.match(value.strip())
    if m:
        mo = _month_num(m.group(2))
        if mo:
            return _iso(int(m.group(3)), mo, int(m.group(1)))
    try:
        return date.fromisoformat(value.strip()).isoformat()
    except ValueError:
        return None
