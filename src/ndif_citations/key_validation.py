"""Cheap live validation ("Test connection") for provider credentials.

Each function returns {"ok": bool, "detail": str} and never echoes the secret.
"""
from __future__ import annotations

import re
import requests

_TIMEOUT = 10
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _result(ok: bool, detail: str) -> dict:
    return {"ok": ok, "detail": detail}


def test_llm(base_url: str, api_key: str) -> dict:
    if not (base_url and api_key):
        return _result(False, "base_url and api_key required")
    try:
        r = requests.get(f"{base_url.rstrip('/')}/models",
                         headers={"Authorization": f"Bearer {api_key}"}, timeout=_TIMEOUT)
        return _result(r.status_code < 400, f"HTTP {r.status_code}")
    except Exception as e:                       # noqa: BLE001
        return _result(False, f"{type(e).__name__}")


def test_github(token: str) -> dict:
    if not token:
        return _result(False, "token required")
    try:
        r = requests.get("https://api.github.com/user",
                         headers={"Authorization": f"Bearer {token}"}, timeout=_TIMEOUT)
        return _result(r.status_code < 400, f"HTTP {r.status_code}")
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


def validate_email(addr: str) -> dict:
    return _result(bool(_EMAIL_RE.match(addr or "")), "format check")
