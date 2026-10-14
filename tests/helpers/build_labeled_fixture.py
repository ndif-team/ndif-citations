"""One-off: rebuild tests/fixtures/labeled_evidence.json from the local PDF cache.

Run from the ndif-citations root (needs output/ + output/pdfs, i.e. a maintainer laptop):
    .venv/bin/python -m tests.helpers.build_labeled_fixture

Writes human-verified papers' evidence windows, re-extracted with the CURRENT
extractor, so the auto-publish regression test runs anywhere (spec §4.3, §4.7).
Prints a window-count report: every DECREASE must be explained by duplicate matches
of one mention collapsing (inspect the old windows); none may come from a lost
real mention.
"""
from __future__ import annotations

import json
from pathlib import Path

from ndif_citations import config
from ndif_citations.models import Bucket
from ndif_citations.output import load_existing_papers
from ndif_citations.pdf_cache import cached_pdf_path
from ndif_citations.process import _apply_prefilters
from ndif_citations.utils import extract_ndif_context, redact_emails

OUT = Path("output")
FIXTURE = Path("tests/fixtures/labeled_evidence.json")


def main() -> None:
    rows, report = [], []
    for p in load_existing_papers(OUT):
        if not (p.manual_override and p.bucket == Bucket.VERIFIED and p.context_source == "pdf"):
            continue
        pdf = cached_pdf_path(p, OUT)
        if pdf is None:
            continue
        ctx = extract_ndif_context(pdf, window=config.CONTEXT_WINDOW)
        if ctx.startswith("No "):
            continue
        surviving, _ = _apply_prefilters(ctx.split("\n---\n"), p)
        windows = list(dict.fromkeys(surviving))  # compute_context's byte-identical dedupe
        # Stored windows hold byte-duplicates (the bug Task 4 fixes): count distinct ones.
        old_n, new_n = len(dict.fromkeys(p.ndif_context_windows or [])), len(windows)
        if old_n != new_n:
            band_flip = (old_n >= 2) != (new_n >= 2)
            report.append((new_n - old_n, old_n, new_n, band_flip, p.title[:70]))
        rows.append({"title": p.title, "category": p.category.value,
                     "windows": [redact_emails(w) for w in windows]})
    FIXTURE.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} papers → {FIXTURE}")
    for delta, old_n, new_n, flip, title in sorted(report):
        tag = "DECREASE" if delta < 0 else "increase"
        print(f"{tag:8} {old_n}→{new_n}{'  BAND HIGH↔MEDIUM' if flip else ''}  {title}")


if __name__ == "__main__":
    main()
