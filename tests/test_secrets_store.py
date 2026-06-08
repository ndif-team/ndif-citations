from __future__ import annotations
import os
import pytest
from dotenv import dotenv_values
from ndif_citations import secrets_store

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

def test_set_keys_upserts_only_provided(tmp_path):
    env = tmp_path / ".env"
    env.write_text("LLM_API_KEY=old\nLLM_MODEL=keep-me\n")
    secrets_store.set_keys(env, {"LLM_API_KEY": "new", "GITHUB_TOKEN": "ghp_x"})
    vals = dotenv_values(env)
    assert vals["LLM_API_KEY"] == "new"
    assert vals["GITHUB_TOKEN"] == "ghp_x"
    assert vals["LLM_MODEL"] == "keep-me"      # untouched non-secret line preserved

def test_set_keys_blank_is_skipped(tmp_path):
    env = tmp_path / ".env"
    env.write_text("S2_API_KEY=keepme\n")
    secrets_store.set_keys(env, {"S2_API_KEY": "", "GITHUB_TOKEN": "ghp_y"})
    vals = dotenv_values(env)
    assert vals["S2_API_KEY"] == "keepme"
    assert vals["GITHUB_TOKEN"] == "ghp_y"

def test_configured_status_booleans_only(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.delenv("S2_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    status = secrets_store.configured_status()
    assert status == {"LLM_API_KEY": True, "S2_API_KEY": False,
                      "GITHUB_TOKEN": False, "SERPAPI_API_KEY": False}
    assert all(isinstance(v, bool) for v in status.values())
