"""FastAPI application factory for ndif-citations.

Usage (uvicorn)
---------------
    uvicorn ndif_citations.server.app:app --host 127.0.0.1 --port 8000

The ``serve`` CLI command (Task 2.6) will call ``uvicorn.run`` with the
appropriate host/port; this module only builds the app.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ndif_citations.server.routers import images, papers, repos, runs, stats

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

    # SPA static files — only if the dist directory exists (optional).
    if _WEB_DIST.exists():
        app.mount(
            "/",
            StaticFiles(directory=str(_WEB_DIST), html=True),
            name="spa",
        )

    return app


# Module-level instance consumed by uvicorn / the serve command.
app = create_app()
