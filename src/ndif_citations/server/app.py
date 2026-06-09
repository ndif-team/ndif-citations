"""FastAPI application factory for ndif-citations.

Usage (uvicorn)
---------------
    uvicorn ndif_citations.server.app:app --host 127.0.0.1 --port 8000

The ``serve`` CLI command (Task 2.6) will call ``uvicorn.run`` with the
appropriate host/port; this module only builds the app.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ndif_citations.server.routers import images, keys, papers, publish, repos, runs, settings, stats

# SPA dist directory — populated by the frontend build (Task 3+).
_WEB_DIST = Path(__file__).resolve().parent.parent.parent.parent / "web" / "dist"


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application.

    * Mounts the ``/runs`` REST router under ``/api``.
    * If ``web/dist/`` exists, serves it as a static SPA at ``/`` with
      ``html=True`` so the frontend's client-side routing works.  The guard
      prevents a startup error when the frontend hasn't been built yet.
    """
    app = FastAPI(
        title="ndif-citations",
        description="Pipeline API for the NDIF citations tracker.",
        version="0.1.0",
    )

    # JSON API — always mounted.
    app.include_router(runs.router)
    app.include_router(papers.router)
    app.include_router(repos.router)
    app.include_router(stats.router)
    app.include_router(images.router)
    app.include_router(publish.router)
    app.include_router(settings.router)
    app.include_router(keys.router)

    # SPA — only if the dist directory exists (optional; absent before the
    # frontend is built). Hashed assets are served from /assets; every other
    # non-/api path falls back to index.html so client-side routes (e.g.
    # /papers) work on direct navigation and refresh.
    if _WEB_DIST.exists():
        assets_dir = _WEB_DIST / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        _index = _WEB_DIST / "index.html"
        _dist_root = _WEB_DIST.resolve()
        # The SPA shell (and the unhashed root statics like favicon) must never be
        # cached: Vite content-hashes /assets, but index.html keeps a stable URL, so
        # a cached shell would point at deleted asset hashes after a rebuild (F-009).
        # Hashed /assets are served by the StaticFiles mount above and stay cacheable.
        _no_store = {"Cache-Control": "no-store"}

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str) -> FileResponse:
            # /api/* is handled by the routers above; an unknown API path must
            # 404 as JSON, not silently return the SPA shell.
            if full_path == "api" or full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not Found")
            # Serve a real static file when one exists (favicon, robots, etc.),
            # otherwise the SPA shell for client-side routing.
            if full_path:
                candidate = (_WEB_DIST / full_path).resolve()
                if candidate.is_file() and candidate.is_relative_to(_dist_root):
                    return FileResponse(candidate, headers=_no_store)
            return FileResponse(_index, headers=_no_store)

    return app


# Module-level instance consumed by uvicorn / the serve command.
app = create_app()
