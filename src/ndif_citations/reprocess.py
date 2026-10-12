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
4. Restore thumbnails that weren't re-extracted (old PNGs are parked in a temp
   dir, never in ``images/``, and restored in a ``finally``); re-derive
   ``has_*``; re-decide the bucket only for re-classified PENDING papers (hold
   active, so the auto-publish rule gates VERIFIED); then lock
   (``manual_override=True``).
5. Write the whole list back (no merge — merge would re-protect the curated
   values we just overwrote).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional

from ndif_citations.models import Bucket, Category, DiscoveredPaper, PaperReason, PipelineRun
from ndif_citations.output import load_existing_papers, write_outputs
from ndif_citations.process import _decide_bucket, process_papers
from ndif_citations.router import ProcessingBucket, RoutingDecision
from ndif_citations.utils import slugify

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

    # Step 2: remember each paper's decision, clear the requested fields, and drop
    # manual_override so the protective FILL_GAPS hydration/guards are skipped.
    prior = {}
    parked: list[tuple[Path, Path]] = []  # (PNG in images/, parked copy)
    park_dir = Path(tempfile.mkdtemp(prefix="reprocess-thumbs-")) if "thumbnail" in fields else None
    try:
        for p in targets:
            prior[id(p)] = (p.bucket, p.reason, p.reason_detail, p.category, p.image)
            for field in fields:
                _clear_field(p, field)
            if park_dir is not None:
                # Park the old PNG outside images/ (a git-tracked dir): process_papers skips
                # extraction while the file exists, and the finally below puts it back if
                # no new thumbnail replaced it.
                png = out / "images" / f"{slugify(p.title)}.png"
                if png.exists():
                    shutil.move(png, park_dir / f"{len(parked)}.png")
                    parked.append((png, park_dir / f"{len(parked)}.png"))
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
    finally:
        # Success, error, or RunCancelled: restore every parked PNG that wasn't
        # replaced. shutil.move, not Path.replace: the temp dir may be on another
        # filesystem.
        for png, parked_png in parked:
            if not png.exists():
                shutil.move(parked_png, png)
        if park_dir is not None:
            shutil.rmtree(park_dir, ignore_errors=True)

    # Step 5: re-link restored thumbnails, re-derive has_*, decide the bucket, then
    # lock. Pending papers are decided with the hold active (lock state irrelevant),
    # so they reach VERIFIED only via the auto-publish rule. A verified paper whose
    # category changed goes back to review (a published category — above all the
    # uses_ndif badge — never changes without a human). Discards stand. Without
    # "classify", the bucket never changes.
    for p in targets:
        prior_bucket, prior_reason, prior_detail, prior_category, prior_image = prior[id(p)]
        if "thumbnail" in fields and not p.image:
            png = out / "images" / f"{slugify(p.title)}.png"
            if prior_image and prior_image.startswith("/images/") and (out / prior_image[1:]).exists():
                p.image = prior_image  # not the title slug (e.g. after a title edit); never moved
            elif png.exists():  # the old thumbnail, restored above
                p.image = f"/images/{png.name}"
        p.has_summary = bool(p.description)
        p.has_classification = p.category != Category.UNCLASSIFIED
        p.has_thumbnail = bool(p.image)
        p.has_affiliations = bool(p.affiliations)
        reclassified = "classify" in fields
        auto_discarded = p.bucket == Bucket.DISCARDED and p.reason == PaperReason.ZERO_PDF_HITS
        if reclassified and prior_bucket == Bucket.PENDING:
            if not auto_discarded:  # keep process_papers' zero-hit discard
                p.manual_override = False
                p.bucket, p.reason = _decide_bucket(p)
        elif reclassified and prior_bucket == Bucket.VERIFIED and p.category != prior_category:
            p.bucket, p.reason = Bucket.PENDING, PaperReason.NEEDS_REVIEW
            p.reason_detail = f"re-classified: {prior_category.value} → {p.category.value}"
        else:
            p.bucket, p.reason, p.reason_detail = prior_bucket, prior_reason, prior_detail
        p.manual_override = True

    # Step 6: write the whole list back (NO merge — would re-protect overwrites).
    write_outputs(papers, out, PipelineRun())

    return {"reprocessed": [p.merge_key() for p in targets], "fields": list(fields)}
