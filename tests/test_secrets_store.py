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

def test_set_keys_strips_pasted_surrounding_quotes(tmp_path):
    # A user who pastes a token wrapped in quotes shouldn't store/auth with the quotes.
    env = tmp_path / ".env"
    env.write_text("")
    secrets_store.set_keys(env, {"GITHUB_TOKEN": "'ghp_quoted'", "S2_API_KEY": '"s2quoted"'})
    vals = dotenv_values(env)
    assert vals["GITHUB_TOKEN"] == "ghp_quoted"
    assert vals["S2_API_KEY"] == "s2quoted"


def test_set_keys_writes_tokens_without_quotes(tmp_path):
    # Tokens with '_' / '-' aren't str.isalnum(), so dotenv's quote_mode='auto'
    # used to write KEY='value' (cosmetic, but alarming on inspection). We use
    # quote_mode='never' → the raw .env line has NO surrounding quotes, and the
    # value still round-trips cleanly.
    env = tmp_path / ".env"
    env.write_text("")
    token = "github_pat_11ABC_xyz-456"          # underscores + hyphen → not alnum
    secrets_store.set_keys(env, {"GITHUB_TOKEN": token, "LLM_API_KEY": "sk-ant-api03-x_y"})
    raw = env.read_text()
    assert f"GITHUB_TOKEN={token}\n" in raw, f"expected unquoted line, got: {raw!r}"
    assert "GITHUB_TOKEN='" not in raw and 'GITHUB_TOKEN="' not in raw
    assert "LLM_API_KEY='" not in raw and 'LLM_API_KEY="' not in raw
    vals = dotenv_values(env)
    assert vals["GITHUB_TOKEN"] == token        # round-trips clean
    assert vals["LLM_API_KEY"] == "sk-ant-api03-x_y"

def test_set_keys_quote_only_value_is_skipped(tmp_path):
    # '' (just quotes -> empty after stripping) is treated as blank = keep existing.
    env = tmp_path / ".env"
    env.write_text("GITHUB_TOKEN=keepme\n")
    secrets_store.set_keys(env, {"GITHUB_TOKEN": "''"})
    assert dotenv_values(env)["GITHUB_TOKEN"] == "keepme"

def test_refresh_secrets_from_file_reflects_edits_and_removals(tmp_path, monkeypatch):
    # F-002: out-of-band .env edits should be reflected. Keys present in the file
    # are synced into os.environ; keys absent from the file are popped (removal).
    env = tmp_path / ".env"
    env.write_text("GITHUB_TOKEN=fromfile\nLLM_API_KEY=llmfromfile\n")
    for k in _SECRETS:
        monkeypatch.setenv(k, "stale")            # os.environ drifted away from the file
    status = secrets_store.refresh_secrets_from_file(env)
    assert os.environ["GITHUB_TOKEN"] == "fromfile"     # edit reflected
    assert os.environ["LLM_API_KEY"] == "llmfromfile"
    assert "S2_API_KEY" not in os.environ               # not in file -> popped (removal reflected)
    assert "SERPAPI_API_KEY" not in os.environ
    assert status == {"LLM_API_KEY": True, "S2_API_KEY": False,
                      "GITHUB_TOKEN": True, "SERPAPI_API_KEY": False}

def test_clear_key_unsets_file_and_env(tmp_path, monkeypatch):
    from ndif_citations import config
    monkeypatch.setattr(config, "_PROJECT_ROOT", tmp_path)   # reload_settings reads tmp .env
    env = tmp_path / ".env"
    env.write_text("GITHUB_TOKEN=abc\nLLM_MODEL=keep-me\n")
    monkeypatch.setenv("GITHUB_TOKEN", "abc")
    status = secrets_store.clear_key(env, "GITHUB_TOKEN")
    assert "GITHUB_TOKEN" not in dotenv_values(env)          # removed from file
    assert not os.environ.get("GITHUB_TOKEN")               # removed from env
    assert status["GITHUB_TOKEN"] is False
    assert dotenv_values(env)["LLM_MODEL"] == "keep-me"      # other lines preserved

def test_clear_key_rejects_unknown(tmp_path):
    with pytest.raises(ValueError):
        secrets_store.clear_key(tmp_path / ".env", "NOT_A_SECRET")

def test_configured_status_booleans_only(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.delenv("S2_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    status = secrets_store.configured_status()
    assert status == {"LLM_API_KEY": True, "S2_API_KEY": False,
                      "GITHUB_TOKEN": False, "SERPAPI_API_KEY": False}
    assert all(isinstance(v, bool) for v in status.values())
