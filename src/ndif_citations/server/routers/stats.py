"""REST router for dashboard stats — ``/api/stats``.

Endpoints
---------
GET  /api/stats   Aggregate counts for the curation dashboard.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from ndif_citations.output import load_existing_papers, load_existing_repos
from ndif_citations.server import deps

router = APIRouter(prefix="/api", tags=["stats"])


@router.get("/stats")
def get_stats(out: Path = Depends(deps.get_output_dir)) -> dict:
    """Return aggregate counts for the curation dashboard.

    Response shape::

        {
            "papers": {"verified": int, "pending": int, "discarded": int, "total": int},
            "repos":  {"research": int, "course": int, "experiment": int, "total": int},
            "categories": {"uses_ndif": int, "uses_nnsight": int, "referencing": int,
                           "unclassified": int},
        }

    Papers counts are split by bucket; repo counts are split by repo_type;
    category counts tally all papers (across all buckets) by category value.
    """
    papers = load_existing_papers(out)
    repos = load_existing_repos(out)

    # --- paper bucket counts ---
    paper_counts: dict[str, int] = {"verified": 0, "pending": 0, "discarded": 0}
    for p in papers:
        key = p.bucket.value
        if key in paper_counts:
            paper_counts[key] += 1

    # --- repo type counts ---
    repo_counts: dict[str, int] = {"research": 0, "course": 0, "experiment": 0}
    for r in repos:
        key = r.repo_type
        if key in repo_counts:
            repo_counts[key] += 1

    # --- category counts (all papers, all buckets) ---
    cat_counts: dict[str, int] = {
        "uses_ndif": 0,
        "uses_nnsight": 0,
        "referencing": 0,
        "unclassified": 0,
    }
    for p in papers:
        key = p.category.value
        if key in cat_counts:
            cat_counts[key] += 1

    return {
        "papers": {
            **paper_counts,
            "total": sum(paper_counts.values()),
        },
        "repos": {
            **repo_counts,
            "total": sum(repo_counts.values()),
        },
        "categories": cat_counts,
    }
