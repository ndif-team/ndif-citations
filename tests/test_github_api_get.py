"""Tests for _github_api_get 403 semantics (rate limit vs org-policy permission).

Regression for 2026-06-10: a single org-policy 403 (fine-grained PAT rejected
by e.g. ndif-team) tripped the global rate-limit kill switch and silently
skipped enrichment for every remaining repo in the run.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from ndif_citations import utils


class FakeResponse:
    def __init__(self, status_code: int, json_body: dict | None = None,
                 headers: dict | None = None):
        self.status_code = status_code
        self._json = json_body or {}
        self.headers = headers or {}

    def json(self):
        return self._json


@pytest.fixture(autouse=True)
def reset_flags():
    utils._github_rate_limited = False
    utils._github_anon_rate_limited = False
    yield
    utils._github_rate_limited = False
    utils._github_anon_rate_limited = False


def _policy_403():
    return FakeResponse(
        403,
        {"message": "The 'ndif-team' organization forbids access via a fine-grained "
                    "personal access tokens if the token's lifetime is greater than 366 days."},
        {"x-ratelimit-remaining": "4500"},
    )


def _quota_403():
    return FakeResponse(
        403,
        {"message": "API rate limit exceeded for user ID 1."},
        {"x-ratelimit-remaining": "0"},
    )


def test_permission_403_does_not_trip_global_kill_switch(monkeypatch):
    monkeypatch.setattr("ndif_citations.config.GITHUB_TOKEN", "tok")
    # Authed call → policy 403; anonymous retry → 200.
    responses = [_policy_403(), FakeResponse(200, {"full_name": "o/r"})]
    with patch.object(utils.requests, "get", side_effect=responses):
        data, status = utils._github_api_get("/repos/o/r")
    assert status == 200
    assert data == {"full_name": "o/r"}
    assert utils._github_rate_limited is False


def test_quota_403_trips_global_kill_switch(monkeypatch):
    monkeypatch.setattr("ndif_citations.config.GITHUB_TOKEN", "tok")
    with patch.object(utils.requests, "get", return_value=_quota_403()):
        data, status = utils._github_api_get("/repos/o/r")
    assert status == 403
    assert data is None
    assert utils._github_rate_limited is True
    # Subsequent calls short-circuit without any HTTP request.
    with patch.object(utils.requests, "get", side_effect=AssertionError("must not be called")):
        data, status = utils._github_api_get("/repos/o/other")
    assert (data, status) == (None, 0)


def test_429_trips_global_kill_switch(monkeypatch):
    monkeypatch.setattr("ndif_citations.config.GITHUB_TOKEN", "tok")
    with patch.object(utils.requests, "get",
                      return_value=FakeResponse(429, {}, {"x-ratelimit-remaining": "10"})):
        _, status = utils._github_api_get("/repos/o/r")
    assert status == 429
    assert utils._github_rate_limited is True


def test_permission_403_with_failed_anon_retry_returns_403(monkeypatch):
    monkeypatch.setattr("ndif_citations.config.GITHUB_TOKEN", "tok")
    # Authed → policy 403; anon retry → anonymous quota exhausted.
    anon_quota = FakeResponse(403, {"message": "API rate limit exceeded"},
                              {"x-ratelimit-remaining": "0"})
    with patch.object(utils.requests, "get", side_effect=[_policy_403(), anon_quota]):
        data, status = utils._github_api_get("/repos/o/r")
    assert (data, status) == (None, 403)
    assert utils._github_rate_limited is False, "authed quota is still fine"
    assert utils._github_anon_rate_limited is True
    # Next permission-403 must NOT retry anonymously (only one authed call made).
    calls = []

    def record(url, **kw):
        calls.append(kw.get("headers", {}))
        return _policy_403()

    with patch.object(utils.requests, "get", side_effect=record):
        data, status = utils._github_api_get("/repos/o/r2")
    assert (data, status) == (None, 403)
    assert len(calls) == 1


def test_anonymous_mode_403_trips_kill_switch_via_message(monkeypatch):
    # No token at all: a 403 with a rate-limit message must stop the run
    # (anonymous quota is the only quota).
    monkeypatch.setattr("ndif_citations.config.GITHUB_TOKEN", None)
    resp = FakeResponse(403, {"message": "API rate limit exceeded"}, {})
    with patch.object(utils.requests, "get", return_value=resp):
        _, status = utils._github_api_get("/repos/o/r")
    assert status == 403
    assert utils._github_rate_limited is True
