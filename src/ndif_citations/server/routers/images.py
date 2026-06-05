"""REST router for paper thumbnail images — ``/api/images/{slug}``.

Endpoints
---------
GET  /api/images/{slug}   Serve a single PNG thumbnail from ``out/images/``.

Security
--------
Path traversal is blocked before any filesystem access:
  - Slugs containing ``/`` or ``..`` are rejected with 404.
  - The resolved file path must be a direct child of the images directory
    (``parent == images_dir``) — this catches any edge cases that slip
    through the string check.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ndif_citations.server import deps

router = APIRouter(prefix="/api", tags=["images"])


@router.get("/images/{slug}")
def serve_image(
    slug: str,
    out: Path = Depends(deps.get_output_dir),
) -> FileResponse:
    """Serve ``out/images/{slug}`` as a PNG.

    Returns 404 if the file does not exist.
    Returns 404 if the slug contains path traversal sequences (``/`` or ``..``).
    """
    # Guard 1: reject any slug that looks like a path traversal attempt.
    # This covers URL-encoded variants like ``..%2F`` that FastAPI decodes
    # before handing them to us, as well as bare ``/`` or ``..`` segments.
    if "/" in slug or ".." in slug:
        raise HTTPException(status_code=404, detail="Invalid image slug")

    images_dir = out / "images"
    image_path = images_dir / slug

    # Guard 2: verify the resolved path is a direct child of images_dir —
    # catches any remaining edge cases (e.g. symlinks).
    try:
        resolved = image_path.resolve()
        images_resolved = images_dir.resolve()
        if resolved.parent != images_resolved:
            raise HTTPException(status_code=404, detail="Invalid image slug")
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail="Invalid image slug")

    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail=f"Image {slug!r} not found")

    return FileResponse(str(image_path), media_type="image/png")
