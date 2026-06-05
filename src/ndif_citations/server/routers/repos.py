"""REST router for repos — ``/api/repos``.

Endpoints
---------
GET   /api/repos                         List all repos (optional repo_type/q/sort filters).
GET   /api/repos/{owner}/{repo}          Fetch a single repo by owner/repo.
PATCH /api/repos/{owner}/{repo}          Edit fields on a repo (curator override).
POST  /api/repos/{owner}/{repo}/exclude  Exclude a repo from the DB and future runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ndif_citations.server import deps
from ndif_citations.server.services import repos_svc

router = APIRouter(prefix="/api", tags=["repos"])


# ---------------------------------------------------------------------------
# Request body schemas
# ---------------------------------------------------------------------------

class RepoEditRequest(BaseModel):
    """Body for PATCH /api/repos/{owner}/{repo}."""

    model_config = {"extra": "forbid"}

    repo_type: Optional[str] = None
    linked_paper_url: Optional[str] = None
    description: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/repos")
def list_repos(
    repo_type: Optional[str] = None,
    q: Optional[str] = None,
    sort: str = "stars_desc",
    out: Path = Depends(deps.get_output_dir),
) -> list[dict]:
    """Return all repos matching the given filters.

    Query parameters
    ----------------
    repo_type:
        Filter to a single type: ``research``, ``course``, or ``experiment``.
    q:
        Case-insensitive substring search over ``owner/repo`` and description.
    sort:
        Sort order: ``stars_desc`` (default), ``recent`` (last_commit desc, None last),
        or ``name`` (owner/repo alphabetical).
    """
    return repos_svc.list_rows(out, repo_type=repo_type, q=q, sort=sort)


@router.get("/repos/{owner}/{repo}")
def get_repo(
    owner: str,
    repo: str,
    out: Path = Depends(deps.get_output_dir),
) -> dict:
    """Return full metadata for the repo identified by ``owner/repo``.

    Returns ``404`` if no matching repo exists.
    """
    repo_id = f"{owner}/{repo}"
    result = repos_svc.get_repo(out, repo_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Repo {repo_id!r} not found")
    return result


@router.patch("/repos/{owner}/{repo}")
def edit_repo(
    owner: str,
    repo: str,
    body: RepoEditRequest,
    out: Path = Depends(deps.get_output_dir),
    _guard: None = Depends(deps.require_no_active_run),
) -> dict:
    """Edit curator-controlled fields on a repo.

    Allowed fields: ``repo_type``, ``linked_paper_url``, ``description``.
    Sets ``manual_override=True`` so the pipeline won't re-tag this repo.

    Returns ``404`` if no matching repo exists.
    Returns ``422`` if a field value is invalid or an unknown field is supplied.
    Returns ``409`` if a pipeline run is currently active.
    """
    repo_id = f"{owner}/{repo}"

    # Build the fields dict from explicitly-set values in the request body.
    # We only include fields that were explicitly provided (not None by default).
    # For optional string fields, None means "clear the value" when the key is
    # included — but we only include a field if the caller sent it.
    # Pydantic v2: use model_fields_set to detect which fields were sent.
    fields: dict = {}
    for field_name in body.model_fields_set:
        fields[field_name] = getattr(body, field_name)

    if not fields:
        # Nothing to do — return current state
        result = repos_svc.get_repo(out, repo_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Repo {repo_id!r} not found")
        return result

    try:
        return repos_svc.edit_repo(out, repo_id, fields)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Repo {repo_id!r} not found")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/repos/{owner}/{repo}/exclude")
def exclude_repo(
    owner: str,
    repo: str,
    out: Path = Depends(deps.get_output_dir),
    _guard: None = Depends(deps.require_no_active_run),
) -> dict:
    """Exclude a repo: add to ``excluded_github_repos`` settings and remove from DB.

    The exclusion is persisted to ``settings.json`` so future pipeline runs
    will skip the repo during discovery. The repo is also removed from the
    current on-disk JSON outputs.

    Returns ``{"excluded": "<owner>/<repo>", "remaining": <count>, "was_present": <bool>}``.
    Returns ``409`` if a pipeline run is currently active.
    """
    repo_id = f"{owner}/{repo}"
    return repos_svc.exclude_repo(out, repo_id)
