"""FastAPI dependencies for ndif-citations server.

Module-level singletons (``_runner``) are overridable via
``app.dependency_overrides`` in tests — see ``get_runner`` and ``get_output_dir``.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import Depends, HTTPException

from ndif_citations import config
from ndif_citations.jobs import JobRunner

# Module-level singleton — the app's single runner instance.
_runner = JobRunner()


def get_runner() -> JobRunner:
    """Return the module-level JobRunner instance.

    Override via ``app.dependency_overrides[get_runner]`` in tests to inject a
    fresh, isolated runner per test.
    """
    return _runner


def get_output_dir() -> Path:
    """Return the configured output directory (default: ``output/``).

    Override via ``app.dependency_overrides[get_output_dir]`` in tests to
    redirect all output to a temporary directory.
    """
    return config.get_output_dir(None)


def require_no_active_run(
    runner: JobRunner = Depends(get_runner),
) -> None:
    """Dependency guard: raise 409 if a run is already in progress.

    Intended for mutating endpoints (e.g. POST /runs) that must not start a
    second concurrent run. Define here once; use as a FastAPI dependency where
    needed.
    """
    if runner.active:
        raise HTTPException(status_code=409, detail="A run is already in progress")
