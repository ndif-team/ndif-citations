import requests
from ndif_citations import key_validation as kv

class _Resp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


_OK_CHAT = {"choices": [{"message": {"content": "ok"}}]}

def test_llm_ok(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda url, **k: _Resp(200, _OK_CHAT))
    monkeypatch.setattr(requests, "get", lambda url, **k: _Resp(401))  # /models must NOT be the path
    r = kv.test_llm("https://x/v1", "key")
    assert r["ok"] is True

def test_llm_bad_key(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda url, **k: _Resp(401))
    monkeypatch.setattr(requests, "get", lambda url, **k: _Resp(200))  # /models 200 must NOT mask a bad key
    r = kv.test_llm("https://x/v1", "key")
    assert r["ok"] is False

def test_llm_validates_via_chat_completion(monkeypatch):
    # The key bug (F-006): /models on some providers (NVIDIA) returns 200 without auth,
    # so a garbage key reads "ok". A 1-token chat completion actually exercises auth.
    calls = {}
    def _post(url, **k):
        calls["url"] = url
        calls["json"] = k.get("json")
        return _Resp(200, _OK_CHAT)
    monkeypatch.setattr(requests, "post", _post)
    monkeypatch.setattr(requests, "get", lambda url, **k: _Resp(200))
    kv.test_llm("https://x/v1", "key", "my-model")
    assert calls["url"].endswith("/chat/completions")
    assert calls["json"]["model"] == "my-model"
    assert calls["json"]["max_tokens"] == 5

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
    monkeypatch.setattr(requests, "post", _raise)
    r = kv.test_llm("https://x/v1", "supersecret")
    assert r["ok"] is False
    assert "supersecret" not in r["detail"]
    assert r["detail"] == "ConnectionError"

def test_serpapi_ok(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **k: _Resp(200))
    assert kv.test_serpapi("serpkey")["ok"] is True

def test_serpapi_bad_key(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **k: _Resp(401))
    assert kv.test_serpapi("bad")["ok"] is False

def test_serpapi_empty_requires_key():
    r = kv.test_serpapi("")
    assert r["ok"] is False
    assert "required" in r["detail"]


def test_llm_empty_reply_is_not_ok(monkeypatch):
    monkeypatch.setattr(requests, "post",
                        lambda url, **k: _Resp(200, {"choices": [{"message": {"content": None}}]}))
    r = kv.test_llm("https://x/v1", "key", "nvidia/nemotron-3-super-120b-a12b")
    assert r["ok"] is False
    assert "empty reply" in r["detail"]


def test_llm_merges_extra_body(monkeypatch):
    seen = {}
    def _post(url, **k):
        seen.update(k["json"])
        return _Resp(200, _OK_CHAT)
    monkeypatch.setattr(requests, "post", _post)
    kv.test_llm("https://x/v1", "key", "m", {"chat_template_kwargs": {"enable_thinking": False}})
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


def test_github_uses_rate_limit_endpoint(monkeypatch):
    seen = {}
    def _get(url, **k):
        seen["url"] = url
        return _Resp(200, {"resources": {"core": {"remaining": 987}}})
    monkeypatch.setattr(requests, "get", _get)
    r = kv.test_github("ghs_installation_token")
    assert seen["url"] == "https://api.github.com/rate_limit"
    assert r["ok"] is True and "987" in r["detail"]
