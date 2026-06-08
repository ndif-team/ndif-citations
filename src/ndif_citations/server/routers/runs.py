"""REST router for pipeline runs — ``/api/runs``.

Endpoints
---------
POST  /api/runs                  Start a new pipeline run.
GET   /api/runs                  List all persisted run records (history).
GET   /api/runs/active           Return the currently active run (if any).
GET   /api/runs/preflight        Credential preflight check (no side-effects).
GET   /api/runs/{run_id}         Fetch a single run record by ID.
GET   /api/runs/{run_id}/events  Server-Sent Events stream of progress events.
POST  /api/runs/{run_id}/cancel  Request cancellation of a run.
POST  /api/runs/{run_id}/gate    Submit the curator gate selection for a run.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ndif_citations.jobs import GateError, JobRunner, RunActiveError
from ndif_citations.preflight import preflight as run_preflight
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


class GateRequest(BaseModel):
    process_ids: list[str] = []
    discard_ids: list[str] = []
    edits: dict[str, dict[str, str]] = {}


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
    pf = run_preflight(skip_papers=body.skip_papers, skip_github=body.skip_github)
    if not pf["ok"]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "missing required keys",
                "blocking": pf["blocking"],
                "warnings": pf["warnings"],
            },
        )
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


@router.get("/runs/active")
def get_active_run(
    runner: JobRunner = Depends(get_runner),
) -> dict:
    """Return the currently active run record, or ``{"active": null}`` if idle.

    An active run is one in state ``"running"`` or ``"awaiting_review"``.  The
    frontend can call this endpoint after a browser refresh to resume the gate
    UI without losing track of the in-flight run.

    When awaiting review the returned record includes ``paper_candidates`` and
    ``repo_candidates`` so the gate UI can render them immediately.

    **Route ordering:** this route is defined BEFORE ``GET /runs/{run_id}`` so
    FastAPI does not treat the literal path segment ``"active"`` as a run_id.
    """
    if runner.active:
        try:
            record = runner.status()
        except KeyError:
            return {"active": None}
        return {"active": record.to_dict()}
    return {"active": None}


@router.get("/runs/preflight")
def get_preflight(skip_papers: bool = False, skip_github: bool = False) -> dict:
    """Return preflight credential check for the requested entity set.

    Query params mirror ``StartRunRequest.skip_papers`` / ``skip_github``.
    Response: ``{ok: bool, blocking: list[str], warnings: list[str]}``.

    **Route ordering:** declared BEFORE ``GET /runs/{run_id}`` so the literal
    path segment ``"preflight"`` is never captured as a run_id.
    """
    return run_preflight(skip_papers=skip_papers, skip_github=skip_github)


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


# ---------------------------------------------------------------------------
# SSE live events + cancel
# ---------------------------------------------------------------------------


def _sse_data(payload: dict) -> str:
    """Format a payload as an SSE ``data:`` frame."""
    return f"data: {json.dumps(payload)}\n\n"


@router.get("/runs/{run_id}/events")
def stream_run_events(
    run_id: str,
    runner: JobRunner = Depends(get_runner),
    out: Path = Depends(get_output_dir),
) -> StreamingResponse:
    """Stream a run's progress events as Server-Sent Events.

    Resolution order:
    1. ``runner.subscribe(run_id)`` — the in-memory run (live tail for a running
       run, or a buffer replay for a terminal one).
    2. If the runner doesn't know the run (``KeyError``) but a persisted file
       ``out/runs/{run_id}.json`` exists, replay its ``events`` from disk.
    3. Otherwise ``HTTPException(404)``.

    Every stream ends with an ``event: end`` frame so clients can close cleanly.
    The ``KeyError``/404 decision is made *before* building the StreamingResponse
    so an unknown run id surfaces as a real 404 (not a half-open stream).
    """
    # Decide the event source eagerly so a 404 is a real HTTP error, not a
    # stream that opens then immediately closes. ``subscribe`` is a generator
    # whose KeyError would only fire on first iteration (inside the response
    # body), so probe ``runner.status`` here — it raises KeyError synchronously.
    known_to_runner = False
    try:
        runner.status(run_id)
        known_to_runner = True
    except KeyError:
        known_to_runner = False

    file_events: list[dict] | None = None
    if not known_to_runner:
        run_file = Path(out) / "runs" / f"{run_id}.json"
        if run_file.exists():
            try:
                data = json.loads(run_file.read_text())
                file_events = data.get("events", [])
            except (OSError, ValueError):
                file_events = None
        if file_events is None:
            raise HTTPException(
                status_code=404, detail=f"Run {run_id!r} not found"
            )

    def _generate() -> Iterator[str]:
        if known_to_runner:
            # subscribe() snapshots the buffer + registers a live queue under the
            # runner lock; iterating it yields the buffered prefix then live
            # events until the run's sentinel.
            for ev in runner.subscribe(run_id):
                yield _sse_data(ev.to_dict())
        else:
            for ev in file_events or []:
                yield _sse_data(ev)
        # Final marker so clients know the stream is complete and can close.
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/runs/{run_id}/cancel")
def cancel_run(
    run_id: str,
    runner: JobRunner = Depends(get_runner),
    out: Path = Depends(get_output_dir),
) -> dict:
    """Request cancellation of a run.

    * If the runner knows the run, ``runner.cancel(run_id)`` sets its cancel
      token (a no-op if the run is already terminal) and we return 200
      ``{"status": "cancelling"}``.
    * If the runner does NOT know the run but a persisted file exists, the run
      already finished — cancel is a safe no-op, still 200.
    * If neither the runner nor a persisted file knows the run → 404.
    """
    known_to_runner = False
    try:
        runner.status(run_id)
        known_to_runner = True
    except KeyError:
        known_to_runner = False

    if not known_to_runner:
        run_file = Path(out) / "runs" / f"{run_id}.json"
        if not run_file.exists():
            raise HTTPException(
                status_code=404, detail=f"Run {run_id!r} not found"
            )

    # Safe no-op if the run is not (or no longer) running.
    runner.cancel(run_id)
    return {"status": "cancelling"}


@router.post("/runs/{run_id}/gate")
def submit_gate(
    run_id: str,
    body: GateRequest,
    runner: JobRunner = Depends(get_runner),
) -> dict:
    """Submit the curator's gate selection and resume the parked worker.

    The run must be in state ``"awaiting_review"`` — i.e. it is an incremental
    run that has completed the discover→enrich→route stages and is waiting for
    a human decision on which candidates to process.

    * ``process_ids`` — candidate paper IDs to send through the LLM process stage.
    * ``discard_ids`` — candidate paper IDs to mark as MANUAL_DISCARD and merge.
    * ``edits`` — pre-processing field edits keyed by paper ID, each a
      ``{field_name: raw_value}`` dict (validated against the edit schema).

    Status codes:

    * 200 ``{"status": "processing", "run_id": run_id}`` — selection accepted
      and the worker has been unblocked to run process + finalize.
    * 404 — no run with ``run_id`` is known to the runner.
    * 409 — the run exists but is NOT in state ``"awaiting_review"`` (already
      done, running, cancelled, etc.).
    * 422 — the body failed Pydantic validation, OR an ``edits`` entry
      references an unknown / non-editable field.
    """
    # Resolve the run; raise 404 for unknown ids.
    try:
        record = runner.status(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    # The run must be awaiting review; anything else is a 409.
    if record.state != "awaiting_review":
        raise HTTPException(
            status_code=409,
            detail="run is not awaiting review",
        )

    # Delegate to the runner — raises GateError if edits are invalid.
    try:
        runner.submit_gate(
            run_id,
            process_ids=body.process_ids,
            discard_ids=body.discard_ids,
            edits=body.edits,
        )
    except GateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"status": "processing", "run_id": run_id}
