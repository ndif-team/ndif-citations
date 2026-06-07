"""REST router for publishing curated outputs to the site — ``/api/publish``.

Endpoints
---------
GET  /api/publish/target   Report the detected + configured publish target and
                           whether the resolved one is valid.
PUT  /api/publish/target   Validate a path and persist it as ``publish_target``.
POST /api/publish          Dry-run (return ``diff``) or apply (copy files +
                           force-overwrite changed images), guarded against an
                           active run.

The heavy lifting lives in :mod:`ndif_citations.publish` (no server imports);
this router only resolves the target, persists the setting, and maps results to
HTTP status codes.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ndif_citations import config, publish, settings_store
from ndif_citations.server import deps

router = APIRouter(prefix="/api", tags=["publish"])


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class TargetRequest(BaseModel):
    """Body for PUT /api/publish/target."""
    path: str


class PublishRequest(BaseModel):
    """Body for POST /api/publish."""
    dry_run: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _configured_target() -> str | None:
    """Return the persisted ``publish_target`` setting (or None)."""
    overrides = settings_store.load_overrides(config._SETTINGS_FILE)
    return overrides.get("publish_target")


def _resolve_target() -> Path:
    """Resolve the effective publish target: configured setting, else detect.

    Raises HTTPException(400) if neither yields a valid target.
    """
    configured = _configured_target()
    if configured:
        path = Path(configured)
        if publish.validate_target(path):
            return path
        raise HTTPException(
            status_code=400,
            detail=f"Configured publish_target {configured!r} is not a valid target.",
        )

    detected = publish.detect_target()
    if detected is not None:
        return detected

    raise HTTPException(
        status_code=400,
        detail=(
            "No publish target configured or detected. Set one with "
            "PUT /api/publish/target."
        ),
    )


# ---------------------------------------------------------------------------
# GET /api/publish/target
# ---------------------------------------------------------------------------

@router.get("/publish/target")
def get_target() -> dict:
    """Report the detected + configured target and whether the resolved one is valid.

    ``valid`` reflects the resolved target (configured if set, else detected).
    """
    detected = publish.detect_target()
    configured = _configured_target()

    if configured:
        valid = publish.validate_target(Path(configured))
    else:
        valid = detected is not None and publish.validate_target(detected)

    return {
        "detected": str(detected) if detected is not None else None,
        "configured": configured,
        "valid": valid,
    }


# ---------------------------------------------------------------------------
# PUT /api/publish/target
# ---------------------------------------------------------------------------

@router.put("/publish/target")
def put_target(body: TargetRequest) -> dict:
    """Validate *body.path* and persist it as ``publish_target``.

    422 if the path is not a valid target (missing ``public/data`` +
    ``public/images``, or it is the refused ``ndif-website`` / a build ``out``
    dir).
    """
    path = Path(body.path)
    if not publish.validate_target(path):
        raise HTTPException(
            status_code=422,
            detail=(
                f"{body.path!r} is not a valid publish target. It must be an "
                "ndif.us dir with public/data/ and public/images/, and must not "
                "be the ndif-website project or a Next build 'out' dir."
            ),
        )

    settings_store.save(config._SETTINGS_FILE, {"publish_target": str(path)})
    config.reload_settings()
    return {"publish_target": str(path)}


# ---------------------------------------------------------------------------
# POST /api/publish
# ---------------------------------------------------------------------------

_BUILD_HINT = (
    "Run `bun run build` in ndif-web-beta to see changes on the site "
    "(data is imported at build time)."
)


@router.post("/publish")
def do_publish(
    body: PublishRequest,
    out: Path = Depends(deps.get_output_dir),
    _guard: None = Depends(deps.require_no_active_run),
) -> dict:
    """Dry-run (return ``diff``) or apply (copy + force-overwrite) the outputs.

    * ``dry_run=true`` (default) → return the ``diff(out, target)`` dict.
    * ``dry_run=false`` → ``apply(out, target)``; return
      ``{summary, diff, build_hint}``.

    Guarded by ``require_no_active_run`` (409) — publishing while a pipeline run
    is in progress would race the slim-output writers.
    400 if no target is configured or detected; the publish module raises a
    clear FileNotFoundError (→ 400) if the pipeline has not been run.
    """
    target = _resolve_target()

    try:
        if body.dry_run:
            return publish.diff(out, target)
        summary = publish.apply(out, target)
        # Recompute diff post-apply for the response (should be empty buckets,
        # but informative — reflects what was just published).
        return {
            "summary": summary,
            "diff": publish.diff(out, target),
            "build_hint": _BUILD_HINT,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
