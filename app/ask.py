"""Interactive CLI: ask questions and see exactly which chunks produced the answer.

    .venv/bin/python -m app.ask                        # interactive
    .venv/bin/python -m app.ask "exit load on Flexi Cap"   # one shot
    .venv/bin/python -m app.ask --json "NAV of Mid Cap"     # machine readable
    .venv/bin/python -m app.ask --batch questions.txt        # file, one per line

The point of this tool is that it shows the *whole* pipeline, not just the answer.
Every refusal in the product is correct only if you can see why: a question that
returns "I couldn't confirm that" is either the model declining unsupported text or
retrieval failing to fetch the fact, and those need completely different fixes. So
each turn prints the guard verdict, the candidate and surviving chunk counts with
their scores, the chunk text the model actually read, the token usage, and the
validator's verdict.

The two lines to read first when an answer looks wrong:

    [guard]     was the question even allowed to be answered?
    [retrieve]  did the right chunk make it into the 5 the model saw?

If `[guard]` refuses, nothing downstream ran and no tokens were spent. If
`[retrieve]` shows 0 survivors, the threshold is the problem. If retrieval looks
right and the answer is still wrong, it is generation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from . import generate, guardrails as g
from .config import get_settings
from .conversation import History, resolve_question
from .meminfo import current_rss_mb
from .retrieve import retrieve

# ANSI, disabled when stdout is not a TTY so piped output stays clean.
_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


DIM, BOLD, RED, GREEN, YELLOW, CYAN = "2", "1", "31", "32", "33", "36"

BANNER = _c(
    DIM,
    "facts-only assistant - type a question, or /help. answers come from the "
    "retrieved chunks shown below.",
)

HELP = """
  /help              this text
  /chunks            toggle the retrieved-chunk display
  /raw               toggle the full chunk text (long)
  /json              toggle raw JSON output
  /new               forget the conversation and start fresh
  /history           show the remembered turns
  /quit              exit (Ctrl-D also works)

Follow-ups work: "what about its fees?" is resolved against the last
scheme you mentioned, and the resolved question is printed as [as asked].
""".strip()


class Session:
    """Display state for one CLI run. Kept small and explicit, not global."""

    def __init__(self, show_chunks: bool = True, show_raw: bool = False, as_json: bool = False):
        self.show_chunks = show_chunks
        self.show_raw = show_raw
        self.as_json = as_json
        self.turns = 0
        self.spent = 0
        # The last 10 messages, so a follow-up like "what about its fees?" can
        # be resolved to the scheme under discussion before it is guarded,
        # searched and answered.
        self.history = History()

    # -- individual sections ------------------------------------------------

    def print_guard(self, verdict: dict) -> None:
        intent = verdict.get("intent", "?")
        colour = RED if verdict.get("refused") else GREEN
        scheme = verdict.get("scheme")
        extra = f" scheme={scheme}" if scheme else ""
        print(_c(DIM, "  [guard]    "), _c(colour, intent), _c(DIM, f"{extra}"))

    def print_retrieval(self, r, question: str) -> None:
        if r.index_missing:
            print(_c("  [retrieve]"), _c(RED, "no index on disk - nothing to search"))
            return
        best = r.best
        if best:
            print(
                "  [retrieve]",
                f"{len(r.chunks)}/{r.n_candidates} chunks, "
                f"top cos={best.score:.3f} rank={best.rank_score:.3f} "
                f"(lex={best.lexical:.2f}) threshold={r.min_score:.2f}",
            )
        else:
            reason = (
                f"best candidate scored {r.best_score:.3f}, below {r.min_score:.2f}"
                if r.n_candidates
                else "no candidates at all"
            )
            print("  [retrieve]", _c(YELLOW, f"0/{r.n_candidates} chunks - {reason}"))

        if not self.show_chunks:
            return
        for i, c in enumerate(r.chunks, 1):
            print(
                f"    {_c(DIM, f'[{i}]')} cos={c.score:.3f} rank={c.rank_score:.3f} "
                f"{_c(CYAN, c.page_type or '?'):<9} {(c.scheme or '?')[:28]:<28} "
                f"as_of={c.as_of or '-'}"
            )
            print(f"        {_c(DIM, c.url)}")
            if self.show_raw:
                for line in c.document.splitlines()[:14]:
                    print(f"        {_c(DIM, line)}")
            else:
                head = " ".join(c.document.split())[:150]
                print(f"        {_c(DIM, head)}...")

    def print_answer(self, res: dict, elapsed: float, guard_refused: bool) -> None:
        intent = res.get("intent", "?")
        colour = {"answered": GREEN, "unsupported": YELLOW, "degraded": RED}.get(intent, DIM)
        if guard_refused:
            return
        usage = res.get("usage") or {}
        if usage:
            print(
                _c(DIM, "  [generate] "),
                f"{elapsed:.1f}s, {usage.get('completion_tokens')} completion "
                f"tokens, finish={usage.get('finish_reason')}"
                + (", repaired after validator rejection" if res.get("repaired") else ""),
            )
        if res.get("validator_problems"):
            print(_c(DIM, "  [validator]"), _c(YELLOW, "; ".join(res["validator_problems"])))
        print()
        for line in (res.get("answer") or "").splitlines():
            print(f"  {line}")
        print()
        for c in res.get("citations") or []:
            flag = "" if c.get("domain_ok") else _c(RED, "  [DOMAIN NOT ALLOWLISTED]")
            print(_c(DIM, "  source:  "), f"{c['url']}{flag}")
            print(
                _c(DIM, "           "),
                f"{c.get('title', '')} | {c.get('scheme', '')} | as of {c.get('as_of', '-')}",
            )
        if res.get("disclaimer"):
            print(_c(DIM, f"  {res['disclaimer']}"))
        if self.as_json:
            print()
            print(json.dumps(json_safe(res), indent=2, ensure_ascii=False))
        print()
        print(_c(DIM, "-" * 78))

    # -- one question -------------------------------------------------------

    def ask(self, question: str) -> dict:
        self.turns += 1
        print(_c(BOLD, f"\nQ: {question}"))

        # Resolve "its"/"that fund" against what was said earlier. The scheme
        # filter needs the name and the model needs a self-contained question,
        # so the resolved form is what gets searched and answered - but the
        # rewrite is gated so it can never rescue a question the guardrails were
        # about to refuse. The user's original words are still what we display,
        # with the resolution shown so a bad one is visible rather than silent.
        asked, note = resolve_question(question, self.history.messages)
        if note:
            print(_c(DIM, f"  [as asked] {asked}"))
        self.history.append("user", question)

        verdict = g.evaluate(asked)
        self.print_guard(verdict)

        if verdict.get("refused"):
            # No embedding, no search, no tokens. Saying so explicitly is the
            # point: a refusal here is a deliberate policy outcome, not a failure.
            print(_c(DIM, "  [retrieve] not run - the guard refused before search"))
            res = dict(verdict)
            self.print_answer(res, 0.0, guard_refused=True)
            if self.as_json:
                print(json.dumps(json_safe(res), indent=2, ensure_ascii=False))
                print()
            self.history.append("assistant", res.get("answer") or "")
            return res

        started = time.perf_counter()
        result = retrieve(asked, scheme=verdict.get("scheme"))
        self.print_retrieval(result, asked)
        res = generate.answer(result, asked)
        elapsed = time.perf_counter() - started
        self.spent += (res.get("usage") or {}).get("total_tokens", 0)
        self.print_answer(res, elapsed, guard_refused=False)
        # Remember what was said, not just what was asked: the assistant turn is
        # what names the scheme back, and a later "its exit load?" is resolved
        # against this.
        self.history.append("assistant", res.get("answer") or "")
        return res

    def footer(self) -> None:
        settings = get_settings()
        try:
            rss = current_rss_mb()
            rss_str = f"{rss:.0f} MB" if rss is not None else "n/a"
        except Exception:
            rss_str = "n/a"
        try:
            spent_str = f"{self.spent:,}"
        except Exception:
            spent_str = "n/a"
        print(
            _c(
                DIM,
                f"\n{self.turns} questions | {spent_str} tokens | "
                f"RSS {rss_str} | model {settings.groq_model} | "
                f"key {'configured' if settings.groq_configured else 'MISSING'}",
            )
        )
        if not settings.groq_configured:
            print(_c(YELLOW, "  No GROQ_API_KEY in .env - answers will degrade safely."))


def json_safe(res: dict) -> dict:
    """Strip the non-serialisable dataclasses so --json actually works."""
    out = dict(res)
    r = out.pop("retrieval", None)
    if r is not None:
        out["retrieval"] = {
            "chunks": [c.as_dict() for c in r.chunks],
            "n_candidates": r.n_candidates,
            "min_score": r.min_score,
            "best_score": r.best_score,
            "index_missing": r.index_missing,
        }
    return out


def _run_repl(session: Session) -> int:
    print(BANNER)
    while True:
        try:
            line = input(_c(BOLD, "\nYou: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        low = line.lower()
        if low in ("/quit", "/exit", "/q"):
            break
        if low == "/help":
            print(HELP)
            continue
        if low == "/chunks":
            session.show_chunks = not session.show_chunks
            print(_c(DIM, f"  chunk display {'on' if session.show_chunks else 'off'}"))
            continue
        if low == "/raw":
            session.show_raw = not session.show_raw
            print(_c(DIM, f"  full chunk text {'on' if session.show_raw else 'off'}"))
            continue
        if low == "/json":
            session.as_json = not session.as_json
            print(_c(DIM, f"  json output {'on' if session.as_json else 'off'}"))
            continue
        if low == "/new":
            session.history.clear()
            print(_c(DIM, "  conversation cleared - follow-ups will not resolve"))
            continue
        if low == "/history":
            msgs = session.history.messages
            if not msgs:
                print(_c(DIM, "  nothing remembered yet"))
            for m in msgs:
                who = "you" if m.role == "user" else "bot"
                print(_c(DIM, f"  {who}: ") + " ".join(m.content.split())[:110])
            continue
        if low.startswith("/"):
            print(_c(YELLOW, f"  unknown command {line!r} - try /help"))
            continue
        try:
            session.ask(line)
        except Exception as exc:  # noqa: BLE001 - a CLI must not die on one question
            print(_c(RED, f"  error: {type(exc).__name__}: {exc}"))
    session.footer()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.ask",
        description="Ask the facts-only assistant and inspect the retrieved chunks.",
    )
    parser.add_argument("question", nargs="*", help="ask one question and exit")
    parser.add_argument("--batch", metavar="FILE", help="ask each line of FILE, then exit")
    parser.add_argument("--json", action="store_true", help="print the raw response JSON")
    parser.add_argument("--no-chunks", action="store_true", help="hide the chunk list")
    parser.add_argument("--raw", action="store_true", help="print full chunk text")
    args = parser.parse_args(argv)

    session = Session(
        show_chunks=not args.no_chunks,
        show_raw=args.raw,
        as_json=args.json,
    )

    if args.batch:
        try:
            with open(args.batch, encoding="utf-8") as handle:
                questions = [ln.strip() for ln in handle if ln.strip() and not ln.startswith("#")]
        except OSError as exc:
            print(f"cannot read {args.batch}: {exc}", file=sys.stderr)
            return 2
        for question in questions:
            session.ask(question)
            # A batch file is a list of independent questions, not a dialogue.
            # Carrying the referent across lines would silently change what each
            # one is evaluated against - and would make the eval's numbers
            # depend on the order of the file.
            session.history.clear()
        session.footer()
        return 0

    if args.question:
        question = " ".join(args.question).strip()
        if args.json:
            print(json.dumps(json_safe(session.ask(question)), indent=2, ensure_ascii=False))
        else:
            session.ask(question)
            session.footer()
        return 0

    return _run_repl(session)


if __name__ == "__main__":
    raise SystemExit(main())
