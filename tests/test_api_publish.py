"""TDD tests for the publish REST endpoints — Task 5.1.

GET  /api/publish/target  → {detected, configured, valid}
PUT  /api/publish/target  → validate + persist publish_target setting
POST /api/publish         → dry_run diff | apply + build_hint (guarded 409)

All tests use a FAKE site target under tmp_path and a monkeypatched
``config._SETTINGS_FILE`` so the real settings.json / site repo are never touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ndif_citations import config, publish, settings_store
from ndif_citations.jobs import JobRunner
from ndif_citations.server import deps
from ndif_citations.server.app import create_app


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Point config._SETTINGS_FILE at a throwaway file so PUT never writes the
    real settings.json, and restore publish_target after each test."""
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(config, "_SETTINGS_FILE", settings_file)
    yield settings_file
    # reload to clear any in-process state derived from the temp file
    config.reload_settings()


def _make_target(root: Path) -> Path:
    site = root / "ndif-website"
    (site / "public" / "data").mkdir(parents=True)
    (site / "public" / "images").mkdir(parents=True)
    (site / "public" / "data" / "research-papers.json").write_text("[]\n")
    (site / "public" / "data" / "github-repos.json").write_text("[]\n")
    return site


def _slim_paper(title: str, url: str, image: str | None = None) -> dict:
    p = {
        "title": title, "authors": ["A"], "venue": "v", "year": 2024,
        "url": url, "description": "d", "category": "uses_nnsight",
    }
    if image:
        p["image"] = image
    return p


def _seed_out(out: Path, papers, repos, images=None) -> None:
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "research-papers.json").write_text(json.dumps(papers, indent=2) + "\n")
    (out / "github-repos.json").write_text(json.dumps(repos, indent=2) + "\n")
    for name, payload in (images or {}).items():
        (out / "images" / name).write_bytes(payload)


def _make_client(fixture_state, runner: JobRunner | None = None):
    _runner = runner if runner is not None else JobRunner()
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    app.dependency_overrides[deps.get_runner] = lambda: _runner
    return TestClient(app, raise_server_exceptions=True), _runner


# ---------------------------------------------------------------------------
# GET /api/publish/target
# ---------------------------------------------------------------------------

def test_get_target_detected(monkeypatch, fixture_state, tmp_path):
    """With nothing configured, detected comes from detect_target()."""
    ndif_us = _make_target(tmp_path)
    monkeypatch.setattr(publish, "detect_target", lambda start=None: ndif_us)

    client, _ = _make_client(fixture_state)
    resp = client.get("/api/publish/target")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["detected"] == str(ndif_us)
    assert body["configured"] is None
    assert body["valid"] is True  # detected target is valid


def test_get_target_uses_configured(monkeypatch, fixture_state, tmp_path, isolated_settings):
    """A configured publish_target is reported and validated."""
    ndif_us = _make_target(tmp_path)
    settings_store.save(config._SETTINGS_FILE, {"publish_target": str(ndif_us)})
    monkeypatch.setattr(publish, "detect_target", lambda start=None: None)

    client, _ = _make_client(fixture_state)
    resp = client.get("/api/publish/target")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["configured"] == str(ndif_us)
    assert body["valid"] is True


# ---------------------------------------------------------------------------
# PUT /api/publish/target
# ---------------------------------------------------------------------------

def test_put_target_valid(fixture_state, tmp_path, isolated_settings):
    ndif_us = _make_target(tmp_path)
    client, _ = _make_client(fixture_state)

    resp = client.put("/api/publish/target", json={"path": str(ndif_us)})
    assert resp.status_code == 200, resp.text
    assert resp.json()["publish_target"] == str(ndif_us)

    # Persisted to the (temp) settings file.
    saved = settings_store.load_overrides(config._SETTINGS_FILE)
    assert saved["publish_target"] == str(ndif_us)


def test_put_target_accepts_ndif_website(fixture_state, tmp_path, isolated_settings):
    # ndif-website is the production publish target — a valid layout is accepted.
    website = tmp_path / "ndif-website"
    (website / "public" / "data").mkdir(parents=True)
    (website / "public" / "images").mkdir(parents=True)

    client, _ = _make_client(fixture_state)
    resp = client.put("/api/publish/target", json={"path": str(website)})
    assert resp.status_code == 200, resp.text

    # Persisted as the publish target.
    saved = settings_store.load_overrides(config._SETTINGS_FILE)
    assert saved.get("publish_target") == str(website)


def test_put_target_refuses_build_out_dir(fixture_state, tmp_path, isolated_settings):
    # A Next build-output dir (out/data, no public/) is still not a valid target.
    site = tmp_path / "ndif-website"
    (site / "out" / "data").mkdir(parents=True)
    (site / "out" / "images").mkdir(parents=True)

    client, _ = _make_client(fixture_state)
    resp = client.put("/api/publish/target", json={"path": str(site)})
    assert resp.status_code == 422, resp.text

    saved = settings_store.load_overrides(config._SETTINGS_FILE)
    assert saved.get("publish_target") is None


def test_put_target_invalid_path(fixture_state, tmp_path, isolated_settings):
    client, _ = _make_client(fixture_state)
    resp = client.put("/api/publish/target", json={"path": str(tmp_path / "does-not-exist")})
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# POST /api/publish — dry run
# ---------------------------------------------------------------------------

def test_post_publish_dry_run(monkeypatch, tmp_path):
    """dry_run=true returns the diff and writes nothing."""
    ndif_us = _make_target(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _seed_out(
        out,
        papers=[_slim_paper("New", "https://x/new")],
        repos=[],
    )
    monkeypatch.setattr(publish, "detect_target", lambda start=None: ndif_us)

    client, _ = _make_client(out)
    resp = client.post("/api/publish", json={"dry_run": True})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert {p["url"] for p in body["papers"]["added"]} == {"https://x/new"}
    # Nothing written: target papers JSON still empty.
    assert json.loads((ndif_us / "public" / "data" / "research-papers.json").read_text()) == []


def test_post_publish_dry_run_default(monkeypatch, tmp_path):
    """Omitting dry_run defaults to a dry run (read-only)."""
    ndif_us = _make_target(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _seed_out(out, papers=[_slim_paper("A", "https://x/a")], repos=[])
    monkeypatch.setattr(publish, "detect_target", lambda start=None: ndif_us)

    client, _ = _make_client(out)
    resp = client.post("/api/publish", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "papers" in body and "images" in body
    # default is dry run → nothing written
    assert json.loads((ndif_us / "public" / "data" / "research-papers.json").read_text()) == []


# ---------------------------------------------------------------------------
# POST /api/publish — apply
# ---------------------------------------------------------------------------

def test_post_publish_apply(monkeypatch, tmp_path):
    """dry_run=false applies and returns summary + diff + build_hint."""
    ndif_us = _make_target(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _seed_out(
        out,
        papers=[_slim_paper("New", "https://x/new", image="/images/t.png")],
        repos=[],
        images={"t.png": b"FRESH"},
    )
    monkeypatch.setattr(publish, "detect_target", lambda start=None: ndif_us)

    client, _ = _make_client(out)
    resp = client.post("/api/publish", json={"dry_run": False})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "summary" in body
    assert "diff" in body
    assert "bun run build" in body["build_hint"]

    # Target JSON actually updated + image copied.
    dest = json.loads((ndif_us / "public" / "data" / "research-papers.json").read_text())
    assert [p["url"] for p in dest] == ["https://x/new"]
    assert (ndif_us / "public" / "images" / "t.png").read_bytes() == b"FRESH"


def test_post_publish_apply_blocked_by_active_run(monkeypatch, tmp_path):
    """apply during an active run → 409 (require_no_active_run guard)."""
    ndif_us = _make_target(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _seed_out(out, papers=[_slim_paper("A", "https://x/a")], repos=[])
    monkeypatch.setattr(publish, "detect_target", lambda start=None: ndif_us)

    runner = JobRunner()
    # Simulate an active run.
    monkeypatch.setattr(type(runner), "active", property(lambda self: True))

    client, _ = _make_client(out, runner=runner)
    resp = client.post("/api/publish", json={"dry_run": False})
    assert resp.status_code == 409, resp.text


def test_post_publish_no_target(monkeypatch, tmp_path, isolated_settings):
    """No configured + no detected target → 400."""
    out = tmp_path / "out"
    out.mkdir()
    _seed_out(out, papers=[], repos=[])
    monkeypatch.setattr(publish, "detect_target", lambda start=None: None)

    client, _ = _make_client(out)
    resp = client.post("/api/publish", json={"dry_run": True})
    assert resp.status_code == 400, resp.text
