from __future__ import annotations
import os
import pytest
from fastapi.testclient import TestClient
from ndif_citations.server import deps
from ndif_citations.server.app import create_app

@pytest.fixture(autouse=True)
def _restore_env():
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear(); os.environ.update(saved)
        from ndif_citations import config
        config.reload_settings()

@pytest.fixture()
def client(fixture_state):
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    return TestClient(app, raise_server_exceptions=True)

def test_get_preflight_blocks_without_llm(client, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    r = client.get("/api/runs/preflight", params={"skip_papers": False, "skip_github": True})
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert any("LLM_API_KEY" in b for b in r.json()["blocking"])

def test_get_preflight_ok_when_keys_present(client, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "x")
    r = client.get("/api/runs/preflight", params={"skip_papers": False, "skip_github": True})
    assert r.status_code == 200
    assert r.json()["ok"] is True

def test_post_runs_blocked_without_required_key(client, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    r = client.post("/api/runs", json={"mode": "incremental", "skip_papers": False, "skip_github": True})
    assert r.status_code == 422
    assert "LLM_API_KEY" in r.text
