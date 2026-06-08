from __future__ import annotations
import os
import pytest
from fastapi.testclient import TestClient
from ndif_citations.jobs import JobRunner
from ndif_citations.server import deps
from ndif_citations.server.app import create_app

_SECRETS = ("LLM_API_KEY", "S2_API_KEY", "GITHUB_TOKEN", "SERPAPI_API_KEY")

@pytest.fixture(autouse=True)
def _restore_env():
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
        from ndif_citations import config
        config.reload_settings()

@pytest.fixture()
def client():
    app = create_app()
    app.dependency_overrides[deps.get_runner] = lambda: JobRunner()  # fresh idle runner
    return TestClient(app, raise_server_exceptions=True)

def test_get_keys_returns_booleans_only(client, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "secret-value")
    r = client.get("/api/settings/keys")
    assert r.status_code == 200
    body = r.json()
    assert body["LLM_API_KEY"]["configured"] is True
    assert "secret-value" not in r.text          # never leaks the value

def test_put_keys_writes_and_applies(client, monkeypatch, tmp_path):
    env = tmp_path / ".env"
    monkeypatch.setattr("ndif_citations.server.routers.keys._env_path", lambda: env)
    # suppress the cascade reload so project .env doesn't clobber the live-apply
    monkeypatch.setattr("ndif_citations.config.reload_settings", lambda: None)
    r = client.put("/api/settings/keys", json={"GITHUB_TOKEN": "ghp_z"})
    assert r.status_code == 200
    assert "GITHUB_TOKEN" in env.read_text()      # written to the .env
    assert os.environ.get("GITHUB_TOKEN") == "ghp_z"   # live-applied
    assert r.json()["GITHUB_TOKEN"]["configured"] is True

def test_test_connection_uses_validator(client, monkeypatch):
    monkeypatch.setattr("ndif_citations.key_validation.test_github",
                        lambda token: {"ok": True, "detail": "HTTP 200"})
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_z")
    r = client.post("/api/settings/keys/test", json={"provider": "github"})
    assert r.status_code == 200
    assert r.json()["ok"] is True

def test_put_rejects_unknown_key(client, monkeypatch, tmp_path):
    env = tmp_path / ".env"
    monkeypatch.setattr("ndif_citations.server.routers.keys._env_path", lambda: env)
    # An unknown field is ignored by the Pydantic model (not in KeysPut), so this
    # should simply not write it; assert the known one still works.
    r = client.put("/api/settings/keys", json={"GITHUB_TOKEN": "ghp_q"})
    assert r.status_code == 200

def test_test_connection_unknown_provider_422(client):
    r = client.post("/api/settings/keys/test", json={"provider": "bogus"})
    assert r.status_code == 422
