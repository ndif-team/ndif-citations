from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from ndif_citations.server import deps
from ndif_citations.server.app import create_app

ID_OK = "arxiv:2602.16080"
UNKNOWN = "arxiv:9999.00000"

@pytest.fixture()
def client(fixture_state: Path, monkeypatch) -> TestClient:
    monkeypatch.setattr("ndif_citations.process.compute_context",
                        lambda paper, out: (["uses nnsight to trace activations"], "abstract", None))
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    return TestClient(app, raise_server_exceptions=True)

def test_evidence_backfill_ok(client):
    r = client.post(f"/api/papers/{ID_OK}/evidence")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ndif_context_windows"] == ["uses nnsight to trace activations"]
    assert body["context_source"] == "abstract"

def test_evidence_unknown_404(client):
    r = client.post(f"/api/papers/{UNKNOWN}/evidence")
    assert r.status_code == 404

def test_evidence_409_during_active_run(fixture_state, monkeypatch):
    monkeypatch.setattr("ndif_citations.process.compute_context",
                        lambda paper, out: ([], "none", None))
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    stub = MagicMock(); stub.active = True
    app.dependency_overrides[deps.get_runner] = lambda: stub
    c = TestClient(app, raise_server_exceptions=True)
    r = c.post(f"/api/papers/{ID_OK}/evidence")
    assert r.status_code == 409
