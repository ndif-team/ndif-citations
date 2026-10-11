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

def test_get_preflight_validate_blocks_dead_github_token(client, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_dead")
    monkeypatch.setattr("ndif_citations.key_validation.test_github",
                        lambda t: {"ok": False, "detail": "HTTP 401"})
    r = client.get("/api/runs/preflight",
                   params={"skip_papers": True, "skip_github": False, "validate": True})
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert any("rejected by GitHub" in b for b in r.json()["blocking"])

def test_post_runs_blocked_without_required_key(client, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    r = client.post("/api/runs", json={"mode": "incremental", "skip_papers": False, "skip_github": True})
    assert r.status_code == 422
    assert "LLM_API_KEY" in r.text


def test_post_runs_blocks_when_llm_check_fails(monkeypatch, fixture_state):
    from fastapi.testclient import TestClient
    from ndif_citations.server import deps
    from ndif_citations.server.app import create_app
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setattr("ndif_citations.key_validation.test_llm",
                        lambda *a, **k: {"ok": False, "detail": "HTTP 410"})
    class _NoRun:  # before Step 4 the preflight passes: never launch a real (networked) run
        def start(self, *a, **k):
            raise AssertionError("run must not start")

    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    app.dependency_overrides[deps.get_runner] = lambda: _NoRun()
    resp = TestClient(app).post("/api/runs", json={"mode": "incremental", "skip_github": True})
    assert resp.status_code == 422
    assert any("LLM check failed" in b for b in resp.json()["detail"]["blocking"])
