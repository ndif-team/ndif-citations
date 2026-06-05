"""REST router for papers — ``/api/papers``.

Endpoints
---------
GET   /api/papers                       List all papers (optional bucket/q/sort filters).
GET   /api/papers/{paper_id:path}       Fetch a single paper by merge_key.
PATCH /api/papers/{paper_id:path}       Edit one or more fields on a paper.
POST  /api/papers/{paper_id:path}/bucket  Move a paper to a different bucket.

The ``{paper_id:path}`` converter allows merge_keys that contain colons or
slashes (e.g. ``arxiv:2407.14561``, ``doi:10.1234/foo``, ``title:some title``)
to be passed as a single URL segment without URL-encoding.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ndif_citations.server import deps
from ndif_citations.server.services import papers_svc

router = APIRouter(prefix="/api", tags=["papers"])


# ---------------------------------------------------------------------------
# Request body schemas (mutation endpoints)
# ---------------------------------------------------------------------------

class EditRequest(BaseModel):
    """Body for PATCH /api/papers/{paper_id}."""
    fields: dict[str, str]


class BucketRequest(BaseModel):
    """Body for POST /api/papers/{paper_id}/bucket."""
    bucket: Literal["pending", "verified", "discarded"]
    reason: Optional[str] = None
    detail: Optional[str] = None


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


@router.patch("/papers/{paper_id:path}")
def edit_paper(
    paper_id: str,
    body: EditRequest,
    out: Path = Depends(deps.get_output_dir),
    _guard: None = Depends(deps.require_no_active_run),
) -> dict:
    """Edit one or more fields on a paper identified by *paper_id*.

    All values in ``fields`` must be raw strings; they are parsed via the same
    ``EditableField.parse`` logic used by the CLI ``edit`` command.

    Returns ``404`` if no paper with the given merge_key exists.
    Returns ``422`` if a field name is unknown/non-editable or a value fails parsing.
    Returns ``409`` if a pipeline run is currently active.
    """
    try:
        return papers_svc.edit_paper(out, paper_id, body.fields)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id!r} not found")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/papers/{paper_id:path}/bucket")
def set_bucket(
    paper_id: str,
    body: BucketRequest,
    out: Path = Depends(deps.get_output_dir),
    _guard: None = Depends(deps.require_no_active_run),
) -> dict:
    """Move a paper to a different bucket (promote / demote / discard).

    Unifies the ``promote`` / ``demote`` / ``discard`` CLI commands into a
    single endpoint: the caller supplies ``bucket`` (required) and optionally
    ``reason`` + ``detail``.

    Returns ``404`` if no paper with the given merge_key exists.
    Returns ``422`` if *reason* is not a valid ``PaperReason`` value.
    Returns ``409`` if a pipeline run is currently active.
    """
    try:
        return papers_svc.set_bucket(out, paper_id, body.bucket, body.reason, body.detail)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id!r} not found")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
