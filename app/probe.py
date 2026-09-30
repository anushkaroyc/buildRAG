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


def measure_subprocess() -> dict:
    """Run `measure()` in a fresh interpreter, so no build state is inherited."""
    import subprocess
    from pathlib import Path

    result = subprocess.run(
        [sys.executable, "-m", "app.probe", "--json"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"probe failed ({result.returncode}): {result.stderr[-400:]}")
    return json.loads(result.stdout)


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
