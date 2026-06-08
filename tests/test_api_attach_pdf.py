from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from ndif_citations.server import deps
from ndif_citations.server.app import create_app

ID_OK = "arxiv:2602.16080"   # exists in the mini fixture
UNKNOWN = "arxiv:9999.00000"
_PDF = b"%PDF-1.4\nfake\n"

@pytest.fixture()
def client(fixture_state: Path) -> TestClient:
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    return TestClient(app, raise_server_exceptions=True)

def test_attach_pdf_ok(client, fixture_state):
    r = client.post(f"/api/papers/{ID_OK}/pdf", files={"file": ("p.pdf", _PDF, "application/pdf")})
    assert r.status_code == 200, r.text
    assert r.json()["has_pdf"] is True
    assert (fixture_state / "pdfs" / "arxiv-2602.16080.pdf").exists()

def test_attach_pdf_non_pdf_422(client):
    r = client.post(f"/api/papers/{ID_OK}/pdf", files={"file": ("p.txt", b"nope", "text/plain")})
    assert r.status_code == 422

def test_attach_pdf_unknown_404(client):
    r = client.post(f"/api/papers/{UNKNOWN}/pdf", files={"file": ("p.pdf", _PDF, "application/pdf")})
    assert r.status_code == 404
