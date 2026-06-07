"""TDD tests for GET/PUT /api/settings and GET/PUT /api/venues.

Isolation contract
------------------
All tests monkeypatch ``config._SETTINGS_FILE`` and ``config._VENUES_FILE``
to tmp paths so the real ``settings.json`` and ``data/known_venues.json`` are
NEVER written.  The ``restore_config`` autouse fixture snapshots and restores
every config global that the endpoints can mutate (including KNOWN_VENUES) plus
the two ``_*_FILE`` paths themselves.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ndif_citations import config, settings_store
from ndif_citations.jobs import JobRunner
from ndif_citations.server import deps
from ndif_citations.server.app import create_app


# ---------------------------------------------------------------------------
# Config snapshot/restore — MUST run for every test in this module.
# ---------------------------------------------------------------------------

_TRACKED_ATTRS = [
    # Settings-mapped config globals
    "MIN_PAPER_YEAR",
    "SHARED_PAPER_THRESHOLD",
    "EXCLUDED_GITHUB_REPOS",
    "KNOWN_COURSE_SOURCES",
    "COURSE_NAME_PATTERNS",
    "NDIF_KEYWORDS",
    "NDIF_README_KEYWORDS_REGEX",
    "NDIF_README_KEYWORDS_SUBSTR",
    "NDIF_README_NEGATIVE_PATTERNS",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "LLM_RATE_LIMIT_SLEEP",
    "S2_RATE_LIMIT_SLEEP",
    "GITHUB_RATE_LIMIT_SLEEP",
    # Secrets (reload_settings re-reads env)
    "LLM_API_KEY",
    "S2_API_KEY",
    "GITHUB_TOKEN",
    "SERPAPI_API_KEY",
    "OPENALEX_EMAIL",
    "UNPAYWALL_EMAIL",
    # Venues-derived dict
    "KNOWN_VENUES",
]


@pytest.fixture(autouse=True)
def restore_config():
    """Snapshot all mutable config globals + file-path attributes; restore after test."""
    snapshot = {attr: copy.deepcopy(getattr(config, attr, None)) for attr in _TRACKED_ATTRS}
    orig_settings_file = config._SETTINGS_FILE
    orig_venues_file = config._VENUES_FILE
    yield
    # Restore every tracked attribute
    for attr, val in snapshot.items():
        setattr(config, attr, val)
    # Restore the two path attributes
    config._SETTINGS_FILE = orig_settings_file
    config._VENUES_FILE = orig_venues_file


# ---------------------------------------------------------------------------
# Test client factory
# ---------------------------------------------------------------------------

def _make_client(
    tmp_settings: Path,
    tmp_venues: Path,
    runner: JobRunner | None = None,
) -> TestClient:
    """Build a TestClient with config paths redirected to tmp files."""
    # Redirect config module path attributes BEFORE the app resolves them.
    config._SETTINGS_FILE = tmp_settings
    config._VENUES_FILE = tmp_venues

    _runner = runner if runner is not None else JobRunner()
    app = create_app()
    app.dependency_overrides[deps.get_runner] = lambda: _runner
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# 1. GET /api/settings → 200 with defaults
# ---------------------------------------------------------------------------

def test_get_settings_returns_defaults(tmp_path):
    """GET /api/settings returns 200 with default keys when no overrides file exists."""
    tmp_settings = tmp_path / "settings.json"   # does NOT exist yet
    tmp_venues = tmp_path / "venues.json"

    client = _make_client(tmp_settings, tmp_venues)
    resp = client.get("/api/settings")

    assert resp.status_code == 200
    data = resp.json()
    assert "min_paper_year" in data
    assert data["min_paper_year"] == settings_store.DEFAULTS["min_paper_year"]
    assert "llm_rate_limit_sleep" in data
    assert "s2_rate_limit_sleep" in data


# ---------------------------------------------------------------------------
# 2. PUT /api/settings with valid partial → 200; config reloaded
# ---------------------------------------------------------------------------

def test_put_settings_valid_partial(tmp_path):
    """PUT /api/settings with valid partial updates the returned value and config global."""
    tmp_settings = tmp_path / "settings.json"
    tmp_venues = tmp_path / "venues.json"

    client = _make_client(tmp_settings, tmp_venues)
    resp = client.put("/api/settings", json={"min_paper_year": 2025})

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["min_paper_year"] == 2025

    # config global reloaded
    assert config.MIN_PAPER_YEAR == 2025

    # On-disk file must contain ONLY the override (no bloat)
    raw = json.loads(tmp_settings.read_text())
    assert raw == {"min_paper_year": 2025}


# ---------------------------------------------------------------------------
# 3. PUT /api/settings with bad type → 422
# ---------------------------------------------------------------------------

def test_put_settings_invalid_type_422(tmp_path):
    """PUT /api/settings with wrong type raises 422 (not 500)."""
    tmp_settings = tmp_path / "settings.json"
    tmp_venues = tmp_path / "venues.json"

    client = _make_client(tmp_settings, tmp_venues)
    resp = client.put("/api/settings", json={"min_paper_year": "NaN"})

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 4. PUT /api/settings during active run → 409
# ---------------------------------------------------------------------------

def test_put_settings_during_active_run_409(tmp_path):
    """PUT /api/settings with an active run returns 409."""
    tmp_settings = tmp_path / "settings.json"
    tmp_venues = tmp_path / "venues.json"

    # Stub runner with active=True
    class _ActiveRunner:
        active = True

    client = _make_client(tmp_settings, tmp_venues, runner=_ActiveRunner())
    resp = client.put("/api/settings", json={"min_paper_year": 2025})

    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# 5. GET /api/venues → 200 with {"venues": {...}}
# ---------------------------------------------------------------------------

def test_get_venues_returns_venues_dict(tmp_path):
    """GET /api/venues returns 200 with {"venues": {...}} structure."""
    tmp_settings = tmp_path / "settings.json"
    # Write a minimal real venues file to tmp
    tmp_venues = tmp_path / "venues.json"
    tmp_venues.write_text(json.dumps({"venues": {"ICML": {"type": "conference"}}}))

    client = _make_client(tmp_settings, tmp_venues)
    resp = client.get("/api/venues")

    assert resp.status_code == 200
    data = resp.json()
    assert "venues" in data
    assert isinstance(data["venues"], dict)


def test_get_venues_missing_file_returns_empty(tmp_path):
    """GET /api/venues returns {"venues": {}} when the file does not exist."""
    tmp_settings = tmp_path / "settings.json"
    tmp_venues = tmp_path / "venues_missing.json"  # does not exist

    client = _make_client(tmp_settings, tmp_venues)
    resp = client.get("/api/venues")

    assert resp.status_code == 200
    assert resp.json() == {"venues": {}}


# ---------------------------------------------------------------------------
# 6. PUT /api/venues valid → 200; file updated; config.KNOWN_VENUES reloaded
# ---------------------------------------------------------------------------

def test_put_venues_valid(tmp_path):
    """PUT /api/venues with valid body updates the file and reloads KNOWN_VENUES."""
    tmp_settings = tmp_path / "settings.json"
    tmp_venues = tmp_path / "venues.json"

    payload = {
        "venues": {
            "ICML": {
                "type": "conference",
                "aliases": ["International Conference on Machine Learning"],
            }
        }
    }

    client = _make_client(tmp_settings, tmp_venues)
    resp = client.put("/api/venues", json=payload)

    assert resp.status_code == 200, resp.text
    assert resp.json() == payload

    # File on disk updated
    saved = json.loads(tmp_venues.read_text())
    assert saved == payload

    # KNOWN_VENUES derived dict rebuilt
    assert "ICML" in config.KNOWN_VENUES.get("conferences", [])


# ---------------------------------------------------------------------------
# 7. PUT /api/venues with invalid type → 422
# ---------------------------------------------------------------------------

def test_put_venues_invalid_type_422(tmp_path):
    """PUT /api/venues with unknown venue type returns 422."""
    tmp_settings = tmp_path / "settings.json"
    tmp_venues = tmp_path / "venues.json"

    payload = {"venues": {"X": {"type": "bogus"}}}

    client = _make_client(tmp_settings, tmp_venues)
    resp = client.put("/api/venues", json=payload)

    assert resp.status_code == 422
    # File must NOT have been written
    assert not tmp_venues.exists()


# ---------------------------------------------------------------------------
# 8. PUT /api/venues during active run → 409
# ---------------------------------------------------------------------------

def test_put_venues_during_active_run_409(tmp_path):
    """PUT /api/venues with an active run returns 409."""
    tmp_settings = tmp_path / "settings.json"
    tmp_venues = tmp_path / "venues.json"

    class _ActiveRunner:
        active = True

    payload = {"venues": {"ICML": {"type": "conference"}}}

    client = _make_client(tmp_settings, tmp_venues, runner=_ActiveRunner())
    resp = client.put("/api/venues", json=payload)

    assert resp.status_code == 409
