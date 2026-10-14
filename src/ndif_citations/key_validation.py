"""Cheap live validation ("Test connection") for provider credentials.

Each function returns {"ok": bool, "detail": str} and never echoes the secret.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse
import requests

_TIMEOUT = 10
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _result(ok: bool, detail: str) -> dict:
    return {"ok": ok, "detail": detail}


def test_llm(base_url: str, api_key: str, model: str = "", extra_body: dict | None = None) -> dict:
    # A tiny chat completion, NOT GET /models: some OpenAI-compatible providers
    # (e.g. NVIDIA integrate) serve /models without enforcing auth. We also require a
    # non-empty reply: a reasoning model with its budget spent "thinking" returns
    # content=None, which the pipeline cannot use.
    if not (base_url and api_key):
        return _result(False, "base_url and api_key required")
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return _result(False, "base_url must be a http(s) URL")
    payload = {"model": model or "gpt-3.5-turbo",
               "messages": [{"role": "user", "content": "Reply with the word ok."}],
               "max_tokens": 5}
    if extra_body:
        payload.update(extra_body)
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=30,
        )
        if r.status_code >= 400:
            return _result(False, f"HTTP {r.status_code}")
        try:
            content = r.json()["choices"][0]["message"].get("content")
        except Exception:                        # noqa: BLE001
            content = None
        if not (content and content.strip()):
            return _result(False, f"HTTP {r.status_code} but empty reply — reasoning model? set llm_extra_body")
        return _result(True, f"HTTP {r.status_code}")
    except Exception as e:                       # noqa: BLE001
        return _result(False, f"{type(e).__name__}")


def test_github(token: str) -> dict:
    # /rate_limit works for PATs AND Actions installation tokens (GET /user 403s for
    # the latter) and does not count against the limit.
    if not token:
        return _result(False, "token required")
    try:
        r = requests.get("https://api.github.com/rate_limit",
                         headers={"Authorization": f"Bearer {token}"}, timeout=_TIMEOUT)
        if r.status_code >= 400:
            return _result(False, f"HTTP {r.status_code}")
        try:
            remaining = r.json()["resources"]["core"]["remaining"]
            return _result(True, f"HTTP {r.status_code} · {remaining} API calls left this hour")
        except Exception:                        # noqa: BLE001
            return _result(True, f"HTTP {r.status_code}")
    except Exception as e:                       # noqa: BLE001
        return _result(False, f"{type(e).__name__}")


def test_s2(api_key: str) -> dict:
    if not api_key:
        return _result(False, "api_key required")
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2407.14561?fields=title",
            headers={"x-api-key": api_key}, timeout=_TIMEOUT)
        return _result(r.status_code < 400, f"HTTP {r.status_code}")
    except Exception as e:                       # noqa: BLE001
        return _result(False, f"{type(e).__name__}")


def test_serpapi(api_key: str) -> dict:
    # GET serpapi.com/account validates the key (200) vs rejects it (401) WITHOUT
    # consuming a search credit (it returns plan/usage metadata, not a search).
    if not api_key:
        return _result(False, "api_key required")
    try:
        r = requests.get("https://serpapi.com/account",
                         params={"api_key": api_key}, timeout=_TIMEOUT)
        return _result(r.status_code < 400, f"HTTP {r.status_code}")
    except Exception as e:                       # noqa: BLE001
        return _result(False, f"{type(e).__name__}")


def validate_email(addr: str) -> dict:
    ok = bool(_EMAIL_RE.match(addr or ""))
    return _result(ok, "valid format" if ok else "invalid format")
