"""REST router for settings + venues — ``/api/settings`` and ``/api/venues``.

Endpoints
---------
GET   /api/settings        Return effective settings (defaults + overrides).
PUT   /api/settings        Save a partial settings dict; reload config globals.
GET   /api/venues          Return raw known_venues.json content.
PUT   /api/venues          Validate + write a full venues dict; reload config.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from ndif_citations import config, settings_store
from ndif_citations.server.deps import require_no_active_run

router = APIRouter(prefix="/api", tags=["settings"])

# Valid venue type values — mirrored from test_known_venues_schema.py schema rules.
_VALID_VENUE_TYPES = {"conference", "workshop", "journal", "preprint"}


# ---------------------------------------------------------------------------
# Settings endpoints
# ---------------------------------------------------------------------------


@router.get("/settings")
def get_settings() -> dict:
    """Return the effective settings (DEFAULTS merged with on-disk overrides).

    Safe to return: ``settings.json`` / DEFAULTS contain no secrets (API keys
    live in ``.env``).
    """
    return settings_store.load(config._SETTINGS_FILE)


@router.put("/settings")
def put_settings(
    body: dict,
    _guard: None = Depends(require_no_active_run),
) -> dict:
    """Save a partial settings dict and reload config globals.

    The body may contain any subset of known settings keys.  All values are
    type-validated by ``settings_store.save``; invalid keys/types → **422**.
    A run in progress → **409** (evaluated before any writes).

    Returns the new effective settings after applying the override.
    """
    try:
        settings_store.save(config._SETTINGS_FILE, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    config.reload_settings()
    return settings_store.load(config._SETTINGS_FILE)


# ---------------------------------------------------------------------------
# Venues endpoints
# ---------------------------------------------------------------------------


@router.get("/venues")
def get_venues() -> dict:
    """Return the raw ``known_venues.json`` content.

    Returns ``{"venues": {}}`` if the file does not exist.
    """
    if not config._VENUES_FILE.exists():
        return {"venues": {}}
    with open(config._VENUES_FILE) as f:
        return json.load(f)


def _validate_venues(venues_dict: dict) -> None:
    """Raise HTTPException(422) if *venues_dict* fails the canonical schema.

    Rules (mirrored from ``test_known_venues_schema.py``):
    - Each venue entry must have ``type`` ∈ {conference, workshop, journal, preprint}.
    - ``aliases`` if present must be a list of strings.
    - ``parent`` if present must be a string.
    """
    for canonical, entry in venues_dict.items():
        if not isinstance(entry, dict):
            raise HTTPException(
                status_code=422,
                detail=f"Venue {canonical!r}: entry must be a JSON object, got {type(entry).__name__}",
            )
        venue_type = entry.get("type")
        if venue_type not in _VALID_VENUE_TYPES:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Venue {canonical!r}: 'type' must be one of "
                    f"{sorted(_VALID_VENUE_TYPES)}, got {venue_type!r}"
                ),
            )
        aliases = entry.get("aliases")
        if aliases is not None:
            if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
                raise HTTPException(
                    status_code=422,
                    detail=f"Venue {canonical!r}: 'aliases' must be a list of strings",
                )
        parent = entry.get("parent")
        if parent is not None and not isinstance(parent, str):
            raise HTTPException(
                status_code=422,
                detail=f"Venue {canonical!r}: 'parent' must be a string",
            )


@router.put("/venues")
def put_venues(
    body: dict,
    _guard: None = Depends(require_no_active_run),
) -> dict:
    """Validate + write a full venues dict and reload config.

    Expected body: ``{"venues": {canonical: {"type": ..., "aliases"?: [...], "parent"?: ...}}}``.

    Schema errors → **422**.  A run in progress → **409**.
    Returns the saved content.
    """
    venues_dict = body.get("venues")
    if venues_dict is None or not isinstance(venues_dict, dict):
        raise HTTPException(
            status_code=422,
            detail="Body must be {\"venues\": {...}}",
        )

    # Validate each entry before touching disk.
    _validate_venues(venues_dict)

    # Write pretty JSON to the venues file.
    config._VENUES_FILE.parent.mkdir(parents=True, exist_ok=True)
    config._VENUES_FILE.write_text(json.dumps(body, indent=2))

    # Reload the in-memory KNOWN_VENUES derived dict.
    config.reload_venues()

    return body
