"""API tests for POST /api/papers/add (Task 1B.3 — gated manual-add run).

Scenarios
---------
1. Happy path: valid arXiv URL → 200 {run_id, state="running"}; run parks at
   review gate (gated run, does NOT auto-complete to "done").
2. Generic http URL (no arXiv ID) accepted → 200.
3. Invalid URL (empty string) → 422.
4. Invalid URL (non-http, non-arXiv string) → 422.
5. Conflict: a run is already active → 409.

What is faked / why
-------------------
* ``JobRunner.start_manual_add`` — monkeypatched on the class so the route
  contract can be asserted without any real worker thread.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from ndif_citations.jobs import RunActiveError
from ndif_citations.server import deps
from ndif_citations.server.app import create_app

_NEW_ARXIV_ID = "9999.00003"
_NEW_URL = f"https://arxiv.org/abs/{_NEW_ARXIV_ID}"
_GENERIC_HTTP_URL = "https://example.com/papers/some-paper"


# ---------------------------------------------------------------------------
# Test 1 — happy path: valid arXiv URL → starts a gated run
# ---------------------------------------------------------------------------


def test_add_paper_happy_path(monkeypatch, fixture_state):
    seen = {}

    def fake(self, out, seed, *, pdf_bytes=None):
        seen["seed"] = seed
        seen["pdf"] = pdf_bytes
        return "run-123"

    monkeypatch.setattr("ndif_citations.jobs.JobRunner.start_manual_add", fake)
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    client = TestClient(app, raise_server_exceptions=True)

    resp = client.post("/api/papers/add", json={"url": _NEW_URL})
    assert resp.status_code == 200, resp.text
    assert resp.json()["run_id"] == "run-123"
    assert resp.json()["state"] == "running"
    assert seen["seed"].arxiv_id == _NEW_ARXIV_ID
    assert seen["pdf"] is None


# ---------------------------------------------------------------------------
# Test 2 — generic http URL (no arXiv ID) is accepted
# ---------------------------------------------------------------------------


def test_add_paper_generic_http_url_accepted(monkeypatch, fixture_state):
    """A plain https URL with no arXiv ID should be accepted (200) not 422."""
    seen = {}

    def fake(self, out, seed, *, pdf_bytes=None):
        seen["seed"] = seed
        seen["pdf"] = pdf_bytes
        return "run-abc"

    monkeypatch.setattr("ndif_citations.jobs.JobRunner.start_manual_add", fake)
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    client = TestClient(app, raise_server_exceptions=True)

    resp = client.post("/api/papers/add", json={"url": _GENERIC_HTTP_URL})
    assert resp.status_code == 200, resp.text
    assert "run_id" in resp.json()


# ---------------------------------------------------------------------------
# Test 3 — empty URL → 422
# ---------------------------------------------------------------------------


def test_add_paper_empty_url_422(monkeypatch, fixture_state):
    def fake(self, out, seed, *, pdf_bytes=None):
        return "run-x"

    monkeypatch.setattr("ndif_citations.jobs.JobRunner.start_manual_add", fake)
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    client = TestClient(app, raise_server_exceptions=True)

    resp = client.post("/api/papers/add", json={"url": ""})
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# Test 4 — clearly invalid URL (not http, not arXiv) → 422
# ---------------------------------------------------------------------------


def test_add_paper_invalid_url_422(monkeypatch, fixture_state):
    def fake(self, out, seed, *, pdf_bytes=None):
        return "run-x"

    monkeypatch.setattr("ndif_citations.jobs.JobRunner.start_manual_add", fake)
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    client = TestClient(app, raise_server_exceptions=True)

    resp = client.post("/api/papers/add", json={"url": "not-a-url-at-all"})
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# Test 5 — run already active → 409
# ---------------------------------------------------------------------------


def test_add_paper_conflict_409(fixture_state, monkeypatch):
    """With an active run, POST /api/papers/add → 409."""

    def fake(self, out, seed, *, pdf_bytes=None):
        raise RunActiveError("a pipeline run is already active")

    monkeypatch.setattr("ndif_citations.jobs.JobRunner.start_manual_add", fake)
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state

    client = TestClient(app, raise_server_exceptions=True)
    resp = client.post("/api/papers/add", json={"url": _NEW_URL})
    assert resp.status_code == 409, resp.text
    assert "run_id" not in resp.json()
