import requests
from ndif_citations import key_validation as kv

class _Resp:
    def __init__(self, status): self.status_code = status

def test_llm_ok(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **k: _Resp(200))
    r = kv.test_llm("https://x/v1", "key")
    assert r["ok"] is True

def test_llm_bad_key(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **k: _Resp(401))
    r = kv.test_llm("https://x/v1", "key")
    assert r["ok"] is False

def test_github_ok(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **k: _Resp(200))
    assert kv.test_github("ghp_x")["ok"] is True

def test_s2_ok(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **k: _Resp(200))
    assert kv.test_s2("s2key")["ok"] is True

def test_email_format():
    assert kv.validate_email("a@b.org")["ok"] is True
    assert kv.validate_email("not-an-email")["ok"] is False

def test_github_bad_token(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **k: _Resp(401))
    assert kv.test_github("bad")["ok"] is False

def test_s2_bad_key(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **k: _Resp(403))
    assert kv.test_s2("bad")["ok"] is False

def test_llm_network_error_does_not_leak_secret(monkeypatch):
    def _raise(*a, **k):
        raise requests.exceptions.ConnectionError("boom at https://x/v1")
    monkeypatch.setattr(requests, "get", _raise)
    r = kv.test_llm("https://x/v1", "supersecret")
    assert r["ok"] is False
    assert "supersecret" not in r["detail"]
    assert r["detail"] == "ConnectionError"
