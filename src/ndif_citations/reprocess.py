"""Targeted force-reprocess of specific fields on specific papers (Task 4.3).

This module powers the "re-run the LLM summary / classification / Surya
thumbnail / affiliation extraction for *this* paper" action in the curator web
app. It is deliberately free of any server imports so it can run on the
``JobRunner`` worker thread (off the request thread, serialized, Surya-safe,
cancelable) without dragging FastAPI into the worker.

The §3.3 recipe (mutate-in-place, write-the-whole-list, NO merge)
-----------------------------------------------------------------
A naive "re-run the field" approach is defeated by the FILL_GAPS protective
hydration/guard block in ``process.process_papers``: that block only fires when
``is_protected_manual`` is true (FILL_GAPS bucket AND the existing paper has
``manual_override=True``), and when it fires it *keeps the curated value* for
any non-empty field. To force a genuine re-run on a curated paper we must defeat
that guard:

1. CLEAR the requested field(s) and their ``has_*`` flags.
2. Temporarily set ``manual_override=False`` so ``is_protected_manual`` is
   false and none of the hydration/guards fire — the field re-runs purely from
   ``processing_needed``.
3. Process the paper in place.
4. Restore ``manual_override=True`` (a reprocess implies the curator wants the
   paper locked afterwards), re-derive ``has_*`` from the fresh values, and
   re-decide the bucket.
5. Write the whole list back (no merge — merge would re-protect the curated
   values we just overwrote).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from ndif_citations.models import Category, DiscoveredPaper, PipelineRun
from ndif_citations.output import load_existing_papers, write_outputs
from ndif_citations.process import _decide_bucket, process_papers
from ndif_citations.router import ProcessingBucket, RoutingDecision

logger = logging.getLogger(__name__)

# The only fields a targeted reprocess may re-run.
ALLOWED_FIELDS: frozenset[str] = frozenset(
    {"summary", "classify", "thumbnail", "affiliations"}
)

# The full processing-flag key set (order kept stable for readability).
_FLAG_KEYS = ("summary", "classify", "thumbnail", "affiliations")


def _clear_field(paper: DiscoveredPaper, field: str) -> None:
    """Clear a single field + its has-flag so process_papers re-runs it."""
    if field == "summary":
        paper.description = ""
        paper.has_summary = False
    elif field == "classify":
        paper.category = Category.UNCLASSIFIED
        paper.has_classification = False
    elif field == "thumbnail":
        paper.image = None
        paper.has_thumbnail = False
    elif field == "affiliations":
        paper.affiliations = ""
        paper.has_affiliations = False
    else:  # pragma: no cover — guarded by ALLOWED_FIELDS check upstream
        raise ValueError(f"unknown reprocess field: {field!r}")


def reprocess_papers(
    out: Path,
    paper_ids: list[str],
    fields: list[str],
    *,
    cancel_check: "Callable[[], bool] | None" = None,
) -> dict:
    """Force a targeted re-run of *fields* on the papers identified by *paper_ids*.

    Parameters
    ----------
    out:
        Output directory (passed to ``load_existing_papers`` / ``write_outputs``).
    paper_ids:
        ``DiscoveredPaper.merge_key()`` values to reprocess.
    fields:
        Subset of ``{"summary", "classify", "thumbnail", "affiliations"}``.
    cancel_check:
        Optional callable forwarded to ``process_papers``; when it returns True
        the in-flight processing raises ``RunCancelled``.

    Returns ``{"reprocessed": [...merge_keys...], "fields": [...]}``.

    Raises
    ------
    ValueError:
        If *fields* contains anything outside the allowed set, or if any
        requested id matches no existing paper.
    """
    bad_fields = [f for f in fields if f not in ALLOWED_FIELDS]
    if bad_fields:
        raise ValueError(
            f"invalid reprocess field(s): {bad_fields!r}; "
            f"allowed: {sorted(ALLOWED_FIELDS)}"
        )
    if not fields:
        raise ValueError("no fields requested for reprocess")

    papers = load_existing_papers(out)
    wanted = set(paper_ids)
    targets = [p for p in papers if p.merge_key() in wanted]

    # Collect-and-raise on any id that matched nothing.
    found_ids = {p.merge_key() for p in targets}
    missing = [pid for pid in paper_ids if pid not in found_ids]
    if missing:
        raise ValueError(f"no paper found for id(s): {missing!r}")

    # Step 2: clear the requested fields + drop manual_override so the
    # protective FILL_GAPS hydration/guards are skipped. Remember each target's
    # original manual_override so we can restore it afterwards.
    original_overrides: dict[int, bool] = {}
    for p in targets:
        original_overrides[id(p)] = p.manual_override
        for field in fields:
            _clear_field(p, field)
        p.manual_override = False

    # Step 3: build FILL_GAPS decisions that re-run ONLY the requested fields.
    decisions = [
        RoutingDecision(
            paper=p,
            bucket=ProcessingBucket.FILL_GAPS,
            existing_paper=p,  # same object — process_papers mutates in place
            processing_needed={k: (k in fields) for k in _FLAG_KEYS},
        )
        for p in targets
    ]

    # Step 4: process in place (LLM summary/classify, Surya thumbnail, affs).
    process_papers(decisions, out, cancel_check=cancel_check)

    # Step 5: restore curation lock, re-derive has_* + bucket from fresh values.
    for p in targets:
        p.manual_override = True
        p.has_summary = bool(p.description)
        p.has_classification = p.category != Category.UNCLASSIFIED
        p.has_thumbnail = bool(p.image)
        p.has_affiliations = bool(p.affiliations)
        p.bucket, p.reason = _decide_bucket(p)

    # Step 6: write the whole list back (NO merge — would re-protect overwrites).
    write_outputs(papers, out, PipelineRun())

    return {"reprocessed": [p.merge_key() for p in targets], "fields": list(fields)}
