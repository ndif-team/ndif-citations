"""Read-only service helpers for the papers API.

All functions are pure (no side-effects) and operate on a given output_dir
Path so they are trivially testable via dependency_overrides.
"""
from __future__ import annotations

from pathlib import Path

from ndif_citations.models import DiscoveredPaper
from ndif_citations.output import load_existing_papers


def resolve(out: Path, paper_id: str) -> DiscoveredPaper | None:
    """Find the paper whose ``merge_key() == paper_id``.

    Shared lookup used by read endpoints and later mutation tasks.
    Returns ``None`` if the paper is not found.
    """
    papers = load_existing_papers(out)
    for paper in papers:
        if paper.merge_key() == paper_id:
            return paper
    return None


def _paper_to_row(paper: DiscoveredPaper) -> dict:
    """Convert a DiscoveredPaper to the list-row dict shape."""
    return {
        "id": paper.merge_key(),
        "title": paper.title,
        "authors": paper.authors,
        "venue": paper.venue,
        "year": paper.year,
        "category": paper.category.value,
        "bucket": paper.bucket.value,
        "confidence_band": paper.category_confidence_band.value,
        "reason": paper.reason.value if paper.reason else None,
        "source": paper.source.value,
        "has_image": bool(paper.image),
        "manual_override": paper.manual_override,
        "url": paper.url,
    }


def list_rows(
    out: Path,
    *,
    bucket: str | None = None,
    q: str | None = None,
    sort: str = "year_desc",
) -> list[dict]:
    """Return a filtered, sorted list of paper row dicts.

    Parameters
    ----------
    out:
        Output directory (passed to ``load_existing_papers``).
    bucket:
        Optional filter: ``"pending"``, ``"verified"``, or ``"discarded"``.
        If ``None`` all buckets are included.
    q:
        Optional case-insensitive substring search over title + authors + venue.
    sort:
        ``"year_desc"`` (default), ``"year_asc"``, or ``"title"``.
    """
    papers = load_existing_papers(out)

    # Filter by bucket
    if bucket is not None:
        papers = [p for p in papers if p.bucket.value == bucket]

    # Substring search
    if q is not None:
        q_lower = q.lower()
        papers = [
            p for p in papers
            if q_lower in p.title.lower()
            or q_lower in p.authors.lower()
            or q_lower in p.venue.lower()
        ]

    # Sort
    if sort == "year_asc":
        papers.sort(key=lambda p: (p.year, p.title.lower()))
    elif sort == "title":
        papers.sort(key=lambda p: p.title.lower())
    else:  # year_desc (default)
        papers.sort(key=lambda p: (-p.year, p.title.lower()))

    return [_paper_to_row(p) for p in papers]


def get_paper(out: Path, paper_id: str) -> dict | None:
    """Return ``to_full_dict()`` for the paper with the given merge_key, or None."""
    paper = resolve(out, paper_id)
    if paper is None:
        return None
    return paper.to_full_dict()
