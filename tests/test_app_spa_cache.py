"""F-009 — the SPA shell (index.html) must not be cached; hashed /assets stay cacheable.

Symptom: after a rebuild, a returning browser reused a stale ``index.html`` that
pointed at now-deleted asset hashes (the original GitHub-key "Test -> 404"). The
shell is served by ``spa_fallback`` with no ``Cache-Control``, so it gets cached.
Fix: serve the shell (and the root-static fallback that can also yield index.html)
with ``Cache-Control: no-store`` while the content-hashed ``/assets`` mount keeps
its long cache.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ndif_citations.server import app as app_module
from ndif_citations.server.app import create_app


@pytest.fixture()
def spa_client(tmp_path: Path, monkeypatch) -> TestClient:
    """TestClient whose SPA dist is an isolated tmp tree (index.html + one hashed asset)."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>app</title>")
    (dist / "assets" / "app-DEADBEEF.js").write_text("console.log('hi')")
    (dist / "favicon.svg").write_text("<svg/>")
    # create_app reads the module global fresh each call, so patching before the
    # call makes it register the SPA fallback against our tmp dist.
    monkeypatch.setattr(app_module, "_WEB_DIST", dist)
    return TestClient(create_app(), raise_server_exceptions=True)


def test_spa_root_is_not_cached(spa_client: TestClient):
    resp = spa_client.get("/")
    assert resp.status_code == 200
    assert "no-store" in resp.headers.get("cache-control", "")


def test_spa_client_route_is_not_cached(spa_client: TestClient):
    # A deep client-side route falls back to the SPA shell; it must also be no-store.
    resp = spa_client.get("/papers")
    assert resp.status_code == 200
    assert "no-store" in resp.headers.get("cache-control", "")


def test_index_html_direct_is_not_cached(spa_client: TestClient):
    # /index.html is served via the root-static branch; it must not be cached either.
    resp = spa_client.get("/index.html")
    assert resp.status_code == 200
    assert "no-store" in resp.headers.get("cache-control", "")


def test_hashed_assets_stay_cacheable(spa_client: TestClient):
    resp = spa_client.get("/assets/app-DEADBEEF.js")
    assert resp.status_code == 200
    # The content-addressed bundle must NOT be marked no-store, or we lose asset caching.
    assert "no-store" not in resp.headers.get("cache-control", "")
