"""REST router for pipeline runs — ``/api/runs``.

Endpoints
---------
POST  /api/runs           Start a new pipeline run.
GET   /api/runs           List all persisted run records (history).
GET   /api/runs/{run_id}  Fetch a single run record by ID.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ndif_citations.jobs import JobRunner, RunActiveError
from ndif_citations.server.deps import get_output_dir, get_runner

router = APIRouter(prefix="/api", tags=["runs"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class StartRunRequest(BaseModel):
    mode: Literal["fresh", "incremental"]
    skip_papers: bool = False
    skip_github: bool = False


class StartRunResponse(BaseModel):
    run_id: str
    state: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/runs", response_model=StartRunResponse)
def start_run(
    body: StartRunRequest,
    runner: JobRunner = Depends(get_runner),
    out: Path = Depends(get_output_dir),
) -> StartRunResponse:
    """Start a new pipeline run.

    Returns 409 if a run is already active (``RunActiveError``).
    Returns 422 if *mode* is not ``"fresh"`` or ``"incremental"`` (Pydantic
    validation via ``Literal``).
    """
    try:
        run_id = runner.start(
            out,
            mode=body.mode,
            skip_papers=body.skip_papers,
            skip_github=body.skip_github,
        )
    except RunActiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return StartRunResponse(run_id=run_id, state="running")


@router.get("/runs/{run_id}")
def get_run(
    run_id: str,
    runner: JobRunner = Depends(get_runner),
    out: Path = Depends(get_output_dir),
) -> dict:
    """Fetch a single run record by ID.

    Resolution order:
    1. ``runner.status(run_id)`` — in-memory record (current/most-recent run).
    2. Persisted file ``out/runs/{run_id}.json`` — lets clients fetch a
       finished run's record after a newer run has started (the runner only
       tracks the most-recent run in memory).
    3. ``HTTPException(404)`` if neither source has the record.
    """
    # 1. Try in-memory runner.
    try:
        return runner.status(run_id).to_dict()
    except KeyError:
        pass

    # 2. Fall back to the persisted file.
    run_file = Path(out) / "runs" / f"{run_id}.json"
    if run_file.exists():
        try:
            return json.loads(run_file.read_text())
        except (OSError, ValueError):
            pass

    # 3. Not found.
    raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")


@router.get("/runs")
def list_runs(
    runner: JobRunner = Depends(get_runner),
    out: Path = Depends(get_output_dir),
) -> list[dict]:
    """Return all persisted run records, most-recent first."""
    return runner.history(out)
