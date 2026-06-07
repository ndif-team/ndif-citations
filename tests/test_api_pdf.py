"""API tests for GET /api/papers/{id}/pdf — Task 4 (cached-PDF serving).

Fixture state (from mini-research-papers-full.json):
  arxiv:2602.16080 — "Activation Steering via Generative Causal Mediation" (verified)
  arxiv:2604.08058 — "Machine Learning..." (discarded)

Tests:
  - 200 + application/pdf when cached PDF exists for a paper.
  - 404 when paper exists but has no cached PDF.
  - 404 when paper_id does not exist.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ndif_citations.server import deps
from ndif_citations.server.app import create_app


# Known IDs from the mini fixture
ARXIV_ID_WITH_PDF = "arxiv:2602.16080"   # verified paper — we'll write a fake PDF
ARXIV_ID_NO_PDF = "arxiv:2604.08058"     # discarded — no PDF written in fixture
UNKNOWN_ID = "arxiv:9999.00000"


# ---------------------------------------------------------------------------
# Fixture: TestClient with get_output_dir → fixture_state
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(fixture_state: Path) -> TestClient:
    """TestClient with get_output_dir overridden to fixture_state."""
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def client_with_pdf(fixture_state: Path) -> TestClient:
    """TestClient + a pre-written PDF for arxiv:2602.16080."""
    pdfs_dir = fixture_state / "pdfs"
    pdfs_dir.mkdir(exist_ok=True)
    (pdfs_dir / "arxiv-2602.16080.pdf").write_bytes(b"%PDF-1.4 test content")

    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_get_paper_pdf_200(client_with_pdf: TestClient):
    """GET /api/papers/{id}/pdf returns 200 + application/pdf for a cached PDF."""
    resp = client_with_pdf.get(f"/api/papers/{ARXIV_ID_WITH_PDF}/pdf")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/pdf")


def test_get_paper_pdf_content(client_with_pdf: TestClient):
    """The response body matches the file content exactly."""
    resp = client_with_pdf.get(f"/api/papers/{ARXIV_ID_WITH_PDF}/pdf")
    assert resp.status_code == 200
    assert resp.content == b"%PDF-1.4 test content"


def test_get_paper_pdf_no_cached_file_returns_404(client: TestClient):
    """404 when the paper exists but no PDF has been cached."""
    resp = client.get(f"/api/papers/{ARXIV_ID_NO_PDF}/pdf")
    assert resp.status_code == 404


def test_get_paper_pdf_unknown_paper_returns_404(client: TestClient):
    """404 when the paper_id doesn't exist in the catalog."""
    resp = client.get(f"/api/papers/{UNKNOWN_ID}/pdf")
    assert resp.status_code == 404


def test_get_paper_detail_still_works(client: TestClient):
    """Confirm the existing GET /api/papers/{id} detail route is unaffected."""
    resp = client.get(f"/api/papers/{ARXIV_ID_WITH_PDF}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["arxiv_id"] == "2602.16080"
