from __future__ import annotations
import os
import pytest
from ndif_citations import secrets_store

_SECRETS = ("LLM_API_KEY", "S2_API_KEY", "GITHUB_TOKEN", "SERPAPI_API_KEY")

@pytest.fixture(autouse=True)
def _restore_env():
    """Snapshot the secret env vars; restore + reload after each test so the
    live-apply in set_keys never leaks into the rest of the suite."""
    saved = {k: os.environ.get(k) for k in _SECRETS}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        from ndif_citations import config
        config.reload_settings()

def test_set_keys_upserts_only_provided(tmp_path):
    env = tmp_path / ".env"
    env.write_text("LLM_API_KEY=old\nLLM_MODEL=keep-me\n")
    secrets_store.set_keys(env, {"LLM_API_KEY": "new", "GITHUB_TOKEN": "ghp_x"})
    text = env.read_text()
    assert "LLM_API_KEY=new" in text
    assert "GITHUB_TOKEN=ghp_x" in text
    assert "LLM_MODEL=keep-me" in text          # untouched non-secret line preserved

def test_set_keys_blank_is_skipped(tmp_path):
    env = tmp_path / ".env"
    env.write_text("S2_API_KEY=keepme\n")
    secrets_store.set_keys(env, {"S2_API_KEY": "", "GITHUB_TOKEN": "ghp_y"})
    text = env.read_text()
    assert "S2_API_KEY=keepme" in text          # blank = keep existing
    assert "GITHUB_TOKEN=ghp_y" in text

def test_configured_status_booleans_only(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.delenv("S2_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    status = secrets_store.configured_status()
    assert status == {"LLM_API_KEY": True, "S2_API_KEY": False,
                      "GITHUB_TOKEN": False, "SERPAPI_API_KEY": False}
    assert all(isinstance(v, bool) for v in status.values())
