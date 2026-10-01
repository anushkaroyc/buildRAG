"""Measure the **serving-path** memory footprint, isolated from the build tool.

Why this is a separate process
------------------------------
Gate 3.8 is defined as "RSS after loading model + Chroma + a query", i.e. the
footprint of the deployed web app. The offline index builder runs in a much
heavier process: it additionally imports pypdf, pdfplumber and BeautifulSoup to
parse PDFs, which is ~300 MB of parser state that Render never loads. Measuring
the gate inside the builder therefore charges the deployment budget for code
that does not ship, and reports a failure that does not exist.

Conversely, measuring only a toy probe would miss real growth. So both numbers
are reported, each labelled for what it is:

| figure | meaning | gate |
| --- | --- | --- |
| `serving_peak_mb` | model + Chroma + queries, runtime imports only | **3.8** |
| `build_peak_mb` | the same plus the PDF/HTML parser stack | informational |

Run directly for a human-readable report:

    python -m app.probe

It prints JSON when called with `--json`, which is how `app.ingest.build_index`
consumes it.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys

from .meminfo import current_rss_mb, memory_mb, peak_rss_mb

PROBE_QUERIES = [
    "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
    "What is the exit load on HDFC Flexi Cap Fund?",
    "What is the benchmark of HDFC Nifty 50 Index Fund?",
    "What is the expense ratio of HDFC Mid Cap Fund?",
]

GATE_3_8_CEILING_MB = 400
RENDER_BUDGET_MB = 512


def measure(n_queries: int = 4) -> dict:
    """Load model, open the persistent index, run queries, report peak RSS."""
    from .embeddings import embed_query, get_model
    from .store import Store, index_stats

    steps: list[dict] = [{"step": "runtime imports", "rss_mb": memory_mb()}]

    get_model()
    steps.append({"step": "model loaded", "rss_mb": memory_mb()})

    store = Store()
    on_disk = index_stats()
    count = store.count()
    steps.append({"step": "chroma open", "rss_mb": memory_mb()})

    top_hits: list[dict] = []
    for q in PROBE_QUERIES[:n_queries]:
        hits = store.query(embed_query(q), top_k=3)
        if hits:
            m = hits[0]["metadata"]
            top_hits.append(
                {
                    "query": q,
                    "id": hits[0]["id"],
                    "distance": round(float(hits[0]["distance"]), 4),
                    "page_type": m.get("page_type", ""),
                    "scheme": m.get("scheme", ""),
                    "page_no": m.get("page_no", 0),
                }
            )
    steps.append({"step": f"{len(PROBE_QUERIES[:n_queries])} queries", "rss_mb": memory_mb()})

    peak = peak_rss_mb()
    return {
        "serving_peak_mb": peak,
        "serving_current_mb": current_rss_mb() or memory_mb(),
        "steps": steps,
        "index": {
            "path": str(store.path),
            "vectors": count,
            "files": on_disk["n_files"],
            "size_bytes": on_disk["size_bytes"],
        },
        "top_hits": top_hits,
        "gate_3_8_ceiling_mb": GATE_3_8_CEILING_MB,
        "render_budget_mb": RENDER_BUDGET_MB,
        "headroom_mb": round(RENDER_BUDGET_MB - peak, 1),
        "gate_3_8_pass": peak < GATE_3_8_CEILING_MB,
    }


def _parse_report(stdout: str) -> dict | None:
    """Pull the report out of the child's stdout, or None if it never emitted one.

    Scans backwards and skips anything that is not a JSON object carrying the peak,
    so a stray print from a native library (`onnxruntime`, `tokenizers`) landing on
    stdout cannot hide an otherwise valid report.
    """
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "serving_peak_mb" in parsed:
            return parsed
    return None


def _crash_detail(result) -> str:
    """Explain a probe subprocess that produced no report, in the terms that matter."""
    code = result.returncode
    if code < 0:
        how = f"was killed by signal {signal.Signals(-code).name} ({-code})"
    else:
        how = f"exited {code}"

    parts = [f"probe produced no report; child {how}"]
    for label, stream in (("stdout", result.stdout), ("stderr", result.stderr)):
        tail = stream.strip()[-400:]
        if tail:
            parts.append(f"{label}: {tail}")
    if code == -signal.SIGKILL and not result.stderr.strip():
        parts.append(
            "SIGKILL with an empty stderr is the OOM killer, not a Python error: "
            "the build container ran out of cgroup memory while this probe ran "
            "alongside the builder. Rebuild with --skip-probe, or give the build "
            "container more memory."
        )
    return " | ".join(parts)


def measure_subprocess() -> dict:
    """Run `measure()` in a fresh interpreter, so no build state is inherited.

    A nonzero exit code is **not** by itself a failure here. `main()` returns 1 when
    gate 3.8 is not met, and that is a measurement rather than a crash - the report
    has already been written to stdout by then. Conflating the two raised
    `RuntimeError("probe failed (1): ")` with an empty stderr tail, because the
    number lives in `result.stdout`, which was discarded: the one figure this
    function exists to produce was the only thing lost, and an over-budget build
    died with an empty message and no diagnosis.

    So the report is parsed first and returned whatever the exit code; the verdict
    travels with it in `gate_3_8_pass` for the caller to act on. Only a child that
    produced no report at all is an error, and that message says whether it was a
    signal (an OOM kill) or an exit status, and carries both output streams.
    """
    import subprocess
    from pathlib import Path

    result = subprocess.run(
        [sys.executable, "-m", "app.probe", "--json"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=600,
    )

    report = _parse_report(result.stdout)
    if report is not None:
        # `gate_3_8_pass` is authoritative, but carry the code so a reader can see
        # that a measured-and-failed probe returned 1 by design, not by accident.
        report["exit_code"] = result.returncode
        return report

    raise RuntimeError(_crash_detail(result))


def _render(report: dict) -> str:
    rule = "=" * 66
    lines = [rule, "GATE 3.8  serving-path memory (model + Chroma + queries)", rule]
    for s in report["steps"]:
        lines.append(f"  {s['step']:<22} {s['rss_mb']:7.1f} MB")
    peak = report["serving_peak_mb"]
    lines += [
        "",
        f"  PEAK              {peak:7.1f} MB   ceiling {report['gate_3_8_ceiling_mb']} MB   "
        f"{'PASS' if report['gate_3_8_pass'] else 'FAIL'}",
        f"  headroom under Render {report['render_budget_mb']} MB: {report['headroom_mb']:.0f} MB",
        rule,
    ]
    idx = report["index"]
    lines += [
        "",
        f"  persistent index: {idx['vectors']} vectors, {idx['files']} files, "
        f"{idx['size_bytes'] / 1e6:.1f} MB at {idx['path']}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measure serving-path memory and retrieval")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args(argv)

    report = measure()
    if args.json:
        print(json.dumps(report))
    else:
        print(_render(report))
    return 0 if report["gate_3_8_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
