"""Capture a script's console output to its report file.

Six analysis scripts printed their results to stdout and were captured to
``results/reports/*.txt`` by shell redirection at some point in August 2026.
Nothing re-ran that redirection, so those files froze while the analysis moved
on: on 2026-09-20 `analysis_report.txt` still described 311 scored events when
the pipeline produced 307. Worse, the pinned-number tests were reading those
frozen files, so they could not fail.

`tee` removes the possibility. A script wrapped in it writes its report every
time it runs, as a side effect of printing.

    from pfdf_seasonality.report import tee

    if __name__ == "__main__":
        with tee("analysis_report.txt"):
            main()
"""
from __future__ import annotations

import io
import sys
from contextlib import contextmanager
from pathlib import Path

from .paths import REPORTS, ensure_dirs

__all__ = ["tee"]


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s: str) -> int:
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            st.flush()


@contextmanager
def tee(name: str, echo: bool = True):
    """Mirror everything printed inside the block into ``REPORTS/<name>``.

    The file is written even if the block raises, so a partial report is
    available to diagnose the failure rather than a stale one being left in
    place to be mistaken for current.
    """
    ensure_dirs()          # a clean data tree has no results/ folders yet
    dest = Path(REPORTS) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    real = sys.stdout
    sys.stdout = _Tee(real, buf) if echo else buf
    try:
        yield buf
    finally:
        sys.stdout = real
        dest.write_text(buf.getvalue())
        print(f"wrote {dest}")
