"""REST router for papers — ``/api/papers``.

Endpoints
---------
GET  /api/papers                  List all papers (optional bucket/q/sort filters).
GET  /api/papers/{paper_id:path}  Fetch a single paper by merge_key.

The ``{paper_id:path}`` converter allows merge_keys that contain colons or
slashes (e.g. ``arxiv:2407.14561``, ``doi:10.1234/foo``, ``title:some title``)
to be passed as a single URL segment without URL-encoding.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ndif_citations.server import deps
from ndif_citations.server.services import papers_svc

router = APIRouter(prefix="/api", tags=["papers"])


@router.get("/papers")
def list_papers(
    bucket: Optional[str] = None,
    q: Optional[str] = None,
    sort: str = "year_desc",
    out: Path = Depends(deps.get_output_dir),
) -> list[dict]:
    """Return all papers matching the given filters.

    Query parameters
    ----------------
    bucket:
        Filter to a single bucket: ``pending``, ``verified``, or ``discarded``.
    q:
        Case-insensitive substring search over title, authors, and venue.
    sort:
        Sort order: ``year_desc`` (default), ``year_asc``, or ``title``.
    """
    return papers_svc.list_rows(out, bucket=bucket, q=q, sort=sort)


@router.get("/papers/{paper_id:path}")
def get_paper(
    paper_id: str,
    out: Path = Depends(deps.get_output_dir),
) -> dict:
    """Return full metadata for the paper whose merge_key equals *paper_id*.

    The ``:path`` converter ensures that merge_keys containing colons, slashes,
    or other special characters (e.g. ``arxiv:2407.14561``) are captured as a
    single path parameter rather than being split by FastAPI's routing.

    Returns ``404`` if no paper with the given merge_key exists.
    """
    result = papers_svc.get_paper(out, paper_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id!r} not found")
    return result
