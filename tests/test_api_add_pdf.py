"""API tests for POST /api/papers/add-pdf (Task 1B.3).

Scenarios
---------
1. Happy path: PDF upload + title → 200 {run_id, state="running"}.
2. Missing title → 422.
3. Non-PDF bytes → 422.
4. Conflict: a run is already active → 409.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ndif_citations.jobs import RunActiveError
from ndif_citations.server import deps
from ndif_citations.server.app import create_app


@pytest.fixture()
def client(fixture_state: Path) -> TestClient:
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    return TestClient(app, raise_server_exceptions=True)


def test_add_pdf_starts_gated_run(client, monkeypatch):
    seen = {}

    def fake(self, out, seed, *, pdf_bytes=None):
        seen["seed"] = seed
        seen["pdf"] = pdf_bytes
        return "run-9"

    monkeypatch.setattr("ndif_citations.jobs.JobRunner.start_manual_add", fake)
    r = client.post(
        "/api/papers/add-pdf",
        data={"title": "Paywalled Paper", "doi": "10.1/x"},
        files={"file": ("p.pdf", b"%PDF-1.4\nx\n", "application/pdf")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["run_id"] == "run-9"
    assert seen["seed"].title == "Paywalled Paper"
    assert seen["pdf"] == b"%PDF-1.4\nx\n"


def test_add_pdf_requires_title(client):
    r = client.post(
        "/api/papers/add-pdf",
        data={},
        files={"file": ("p.pdf", b"%PDF-1.4\nx\n", "application/pdf")},
    )
    assert r.status_code == 422


def test_add_pdf_rejects_non_pdf(client):
    r = client.post(
        "/api/papers/add-pdf",
        data={"title": "X"},
        files={"file": ("p.txt", b"nope", "text/plain")},
    )
    assert r.status_code == 422


def test_add_pdf_conflict_409(fixture_state, monkeypatch):
    def fake(self, out, seed, *, pdf_bytes=None):
        raise RunActiveError("a pipeline run is already active")

    monkeypatch.setattr("ndif_citations.jobs.JobRunner.start_manual_add", fake)
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    c = TestClient(app, raise_server_exceptions=True)
    r = c.post(
        "/api/papers/add-pdf",
        data={"title": "X"},
        files={"file": ("p.pdf", b"%PDF-1.4\nx\n", "application/pdf")},
    )
    assert r.status_code == 409
    assert "run_id" not in r.json()


def test_add_pdf_rejects_oversize(client, monkeypatch):
    monkeypatch.setattr("ndif_citations.pdf_cache._MAX_PDF_BYTES", 8)
    r = client.post(
        "/api/papers/add-pdf",
        data={"title": "X"},
        files={"file": ("p.pdf", b"%PDF-1.4\ntoo big", "application/pdf")},
    )
    assert r.status_code == 422
