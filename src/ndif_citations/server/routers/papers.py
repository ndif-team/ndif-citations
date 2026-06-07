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

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ndif_citations.jobs import JobRunner, RunActiveError
from ndif_citations.server import deps
from ndif_citations.server.services import papers_svc
from ndif_citations.utils import extract_arxiv_id_from_url

router = APIRouter(prefix="/api", tags=["papers"])

# Allowed targeted-reprocess fields (mirrors reprocess.ALLOWED_FIELDS).
_REPROCESS_FIELDS = {"summary", "classify", "thumbnail", "affiliations"}


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


class ReprocessRequest(BaseModel):
    """Body for POST /api/papers/{paper_id}/reprocess."""
    fields: list[str]


class BatchReprocessRequest(BaseModel):
    """Body for POST /api/papers/reprocess (batch)."""
    ids: list[str]
    fields: list[str]


class ReprocessResponse(BaseModel):
    """Response for the reprocess / reextract-thumbnail endpoints."""
    run_id: str
    state: str


class AddPaperRequest(BaseModel):
    """Body for POST /api/papers/add."""
    url: str


class AddPaperResponse(BaseModel):
    """Response for POST /api/papers/add."""
    run_id: str
    state: str


# ---------------------------------------------------------------------------
# Add-by-URL endpoint (Task 4.5)
# ---------------------------------------------------------------------------


@router.post("/papers/add", response_model=AddPaperResponse)
def add_paper(
    body: AddPaperRequest,
    out: Path = Depends(deps.get_output_dir),
    runner: JobRunner = Depends(deps.get_runner),
) -> AddPaperResponse:
    """Add a single paper by URL via the JobRunner.

    The URL must be non-empty and either:
    * yield a parseable arXiv ID (e.g. ``https://arxiv.org/abs/2407.14561``), or
    * look like a generic http/https URL.

    Heavy work (S2 lookup, enrichment, LLM classification) runs on the
    JobRunner worker — the client watches progress via
    ``GET /api/runs/{run_id}`` or the SSE events stream.

    Returns ``{"run_id": ..., "state": "running"}``.

    * 422 — ``url`` is empty or clearly invalid (not http/https and not an
      arXiv URL).
    * 409 — a run/job is already active.
    """
    url = (body.url or "").strip()
    if not url:
        raise HTTPException(status_code=422, detail="url must not be empty")

    # Accept if: parseable arXiv ID OR starts with http:// or https://
    arxiv_id = extract_arxiv_id_from_url(url)
    if arxiv_id is None and not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=422,
            detail="url must be an http/https URL or a recognisable arXiv URL",
        )

    from ndif_citations import manual_add

    try:
        run_id = runner.start_job(
            out,
            lambda cc: manual_add.add_paper_by_url(out, url, cancel_check=cc),
            kind="add",
        )
    except RunActiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return AddPaperResponse(run_id=run_id, state="running")


@router.post("/papers/reprocess", response_model=ReprocessResponse)
def batch_reprocess_papers(
    body: BatchReprocessRequest,
    out: Path = Depends(deps.get_output_dir),
    runner: JobRunner = Depends(deps.get_runner),
) -> ReprocessResponse:
    """Force a targeted re-run of *fields* on a batch of papers.

    Validates that:
    * ``ids`` is non-empty and every id resolves to an existing paper (404 if any unknown).
    * ``fields`` is a non-empty subset of ``{summary, classify, thumbnail, affiliations}`` (422).

    The heavy work runs on the ``JobRunner`` worker — the client polls
    ``GET /api/runs/{run_id}`` or the SSE events stream.

    Returns ``{"run_id": ..., "state": "running"}``.

    * 404 — one or more ids unknown.
    * 422 — ids empty or fields empty/invalid.
    * 409 — a run/job is already active.

    Note: this route is defined BEFORE ``GET /api/papers/{paper_id:path}`` and
    ``POST /api/papers/{paper_id:path}/reprocess`` so the literal segment
    ``/papers/reprocess`` is matched first and not captured by the ``:path``
    converter.
    """
    from ndif_citations import reprocess

    if not body.ids:
        raise HTTPException(status_code=422, detail="ids must not be empty")

    if not body.fields or not set(body.fields).issubset(_REPROCESS_FIELDS):
        raise HTTPException(
            status_code=422,
            detail=f"fields must be a non-empty subset of {sorted(_REPROCESS_FIELDS)}",
        )

    # Validate that every requested id resolves to an existing paper.
    missing = [pid for pid in body.ids if papers_svc.resolve(out, pid) is None]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"unknown paper id(s): {missing!r}",
        )

    ids = list(body.ids)
    fields = list(body.fields)
    try:
        run_id = runner.start_job(
            out,
            lambda cc: reprocess.reprocess_papers(out, ids, fields, cancel_check=cc),
            kind="reprocess",
        )
    except RunActiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return ReprocessResponse(run_id=run_id, state="running")


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


@router.get("/papers/{paper_id:path}/pdf")
def get_paper_pdf(
    paper_id: str,
    out: Path = Depends(deps.get_output_dir),
) -> FileResponse:
    """Serve the cached PDF for a paper identified by *paper_id*.

    Returns the PDF file with ``Content-Type: application/pdf``.

    * 404 — no paper with the given merge_key.
    * 404 — paper exists but no PDF has been cached.
    * 404 — internal path-traversal guard triggered (should not happen in practice).
    """
    from ndif_citations.pdf_cache import cached_pdf_path
    from ndif_citations.server.services import papers_svc

    paper = papers_svc.resolve(out, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    path = cached_pdf_path(paper, out)
    if path is None:
        raise HTTPException(status_code=404, detail="No cached PDF")

    # Path-traversal guard: resolved file must live directly inside out/pdfs/
    resolved = path.resolve()
    pdfs_dir = (out / "pdfs").resolve()
    if resolved.parent != pdfs_dir:
        raise HTTPException(status_code=404, detail="Invalid PDF path")

    return FileResponse(str(resolved), media_type="application/pdf")


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


# ---------------------------------------------------------------------------
# Targeted force-reprocess + image upload (Task 4.3)
# ---------------------------------------------------------------------------


def _start_reprocess_job(
    out: Path,
    runner: JobRunner,
    paper_id: str,
    fields: list[str],
) -> ReprocessResponse:
    """Validate + queue a targeted reprocess job on the JobRunner.

    Shared by ``/reprocess`` and ``/reextract-thumbnail``. Heavy work (LLM /
    Surya) runs on the worker thread, serialized behind the single-run gate.

    * 404 — the paper does not exist.
    * 422 — *fields* is empty or contains a non-allowed field.
    * 409 — a run/job is already active (``RunActiveError``).
    """
    from ndif_citations import reprocess

    if papers_svc.resolve(out, paper_id) is None:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id!r} not found")

    if not fields or not set(fields).issubset(_REPROCESS_FIELDS):
        raise HTTPException(
            status_code=422,
            detail=(
                f"fields must be a non-empty subset of {sorted(_REPROCESS_FIELDS)}"
            ),
        )

    try:
        run_id = runner.start_job(
            out,
            lambda cc: reprocess.reprocess_papers(
                out, [paper_id], list(fields), cancel_check=cc
            ),
            kind="reprocess",
        )
    except RunActiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return ReprocessResponse(run_id=run_id, state="running")


@router.post("/papers/{paper_id:path}/reprocess", response_model=ReprocessResponse)
def reprocess_paper(
    paper_id: str,
    body: ReprocessRequest,
    out: Path = Depends(deps.get_output_dir),
    runner: JobRunner = Depends(deps.get_runner),
) -> ReprocessResponse:
    """Force a targeted re-run of the requested *fields* on a single paper.

    The heavy work (LLM summary/classification, Surya thumbnail, affiliation
    extraction) runs on the JobRunner worker — the client watches progress via
    ``GET /api/runs/{run_id}`` + the SSE events stream.

    Returns ``{"run_id": ..., "state": "running"}``.

    * 404 — no paper with the given merge_key.
    * 422 — ``fields`` empty or contains a non-allowed value.
    * 409 — a run/job is already active.
    """
    return _start_reprocess_job(out, runner, paper_id, body.fields)


@router.post(
    "/papers/{paper_id:path}/reextract-thumbnail", response_model=ReprocessResponse
)
def reextract_thumbnail(
    paper_id: str,
    out: Path = Depends(deps.get_output_dir),
    runner: JobRunner = Depends(deps.get_runner),
) -> ReprocessResponse:
    """Re-run ONLY the Surya thumbnail extraction for a single paper.

    Convenience wrapper over ``/reprocess`` with ``fields=["thumbnail"]``. Same
    status-code mapping (404 / 409).
    """
    return _start_reprocess_job(out, runner, paper_id, ["thumbnail"])


@router.post("/papers/{paper_id:path}/image")
def upload_image(
    paper_id: str,
    file: UploadFile = File(...),
    out: Path = Depends(deps.get_output_dir),
    _guard: None = Depends(deps.require_no_active_run),
) -> dict:
    """Upload a curated PNG thumbnail for a paper (fast synchronous file write).

    Unlike reprocess, this is not heavy work, so it runs on the request thread —
    but it still requires no active run (``require_no_active_run`` → 409) so it
    doesn't race a pipeline write to ``research-papers-full.json``.

    The PNG is saved to ``out/images/{slugify(title)}.png`` (the same path
    convention ``extract_thumbnail`` / ``write_outputs`` use), and the paper's
    ``image`` is set to ``/images/{filename}`` with ``has_thumbnail=True`` and
    ``manual_override=True``.

    Returns the updated paper's ``to_full_dict()``.

    * 404 — no paper with the given merge_key.
    * 422 — the upload is not a PNG (content-type or magic bytes).
    * 409 — a run/job is already active.
    """
    try:
        return papers_svc.upload_image(out, paper_id, file)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id!r} not found")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
