"""REST router for publishing — ``/api/publish`` (spec §5.7).

GET  /api/publish/preview   What "Push catalog" would publish (local vs last commit).
POST /api/publish           Commit + push the catalog to ndif-citations main; ndif.us
                            pulls it on its next daily build. 409 on a conflict (the
                            curator's commit is kept on a catalog-conflict-* branch).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from ndif_citations import catalog_git, publish
from ndif_citations.server import deps

router = APIRouter(prefix="/api", tags=["publish"])


@router.get("/publish/preview")
def get_preview(out: Path = Depends(deps.get_output_dir)) -> dict:
    return publish.preview(out, out.parent)


@router.post("/publish")
def push(out: Path = Depends(deps.get_output_dir),
         _guard: None = Depends(deps.require_no_active_run)) -> dict:
    try:
        return catalog_git.push_catalog(out.parent, f"data: curator update {datetime.now():%Y-%m-%d %H:%M}")
    except catalog_git.CatalogPushError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
