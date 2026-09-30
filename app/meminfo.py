"""Resident-memory measurement, shared by the server and the build CLI.

Memory is the project's binding deployment constraint (Render free tier = 512 MB,
architecture 9.1), and it is measured from two very different places: the running
app (`/healthz`) and the offline index build (gate 3.8). Keeping one implementation
means a number in a build log and a number on the health endpoint are comparable.

Platform note: Linux (including the Render container) exposes current RSS via
`/proc/self/status`. macOS does not, so there we fall back to `ru_maxrss`, which
is *peak* rather than current. The distinction is preserved in the field names
rather than papered over.
"""

from __future__ import annotations

import platform
import resource
from pathlib import Path


def current_rss_mb() -> float | None:
    """Current resident set size in MB, or None where /proc is unavailable."""
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    try:
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        return None
    return None


def peak_rss_mb() -> float:
    """Peak resident set size in MB.

    `ru_maxrss` is reported in bytes on macOS and kilobytes on Linux.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    return round(peak / divisor, 1)


def memory_mb() -> float:
    """Best available figure, preferring current RSS over peak."""
    current = current_rss_mb()
    return current if current is not None else peak_rss_mb()
