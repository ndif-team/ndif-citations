from __future__ import annotations
import os
import pytest
from fastapi.testclient import TestClient
from ndif_citations import config
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
    # Point BOTH _env_path() and reload_settings() at the same tmp .env (mirrors prod,
    # where _PROJECT_ROOT/.env is the single file written and reloaded).
    monkeypatch.setattr(config, "_PROJECT_ROOT", tmp_path)
    r = client.put("/api/settings/keys", json={"GITHUB_TOKEN": "ghp_z"})
    assert r.status_code == 200
    assert "GITHUB_TOKEN" in (tmp_path / ".env").read_text()      # written to .env
    assert os.environ.get("GITHUB_TOKEN") == "ghp_z"              # live-applied to env
    assert config.GITHUB_TOKEN == "ghp_z"                         # reload synced the config global
    assert r.json()["GITHUB_TOKEN"]["configured"] is True

def test_test_connection_uses_validator(client, monkeypatch):
    monkeypatch.setattr("ndif_citations.key_validation.test_github",
                        lambda token: {"ok": True, "detail": "HTTP 200"})
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_z")
    r = client.post("/api/settings/keys/test", json={"provider": "github"})
    assert r.status_code == 200
    assert r.json()["ok"] is True

def test_put_extra_fields_stripped_by_pydantic(client, monkeypatch, tmp_path):
    """Unknown fields are ignored by Pydantic — only known secrets are processed."""
    monkeypatch.setattr(config, "_PROJECT_ROOT", tmp_path)
    r = client.put("/api/settings/keys", json={"UNKNOWN_KEY": "val", "GITHUB_TOKEN": "ghp_q"})
    assert r.status_code == 200
    assert "UNKNOWN_KEY" not in (tmp_path / ".env").read_text()

def test_put_value_error_from_store_returns_422(client, monkeypatch):
    """If secrets_store.set_keys raises ValueError, the router maps it to 422."""
    def _raise(path, changes):
        raise ValueError("unknown secret key: 'X'")
    monkeypatch.setattr("ndif_citations.secrets_store.set_keys", _raise)
    r = client.put("/api/settings/keys", json={"GITHUB_TOKEN": "ghp_q"})
    assert r.status_code == 422

def test_test_connection_unknown_provider_422(client):
    r = client.post("/api/settings/keys/test", json={"provider": "bogus"})
    assert r.status_code == 422
