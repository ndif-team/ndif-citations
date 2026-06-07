"""Tests for the repos API — Task 4.4.

Covers:
  GET  /api/repos                      — list + filter + sort
  GET  /api/repos/{owner}/{repo}       — single repo by owner/repo
  PATCH /api/repos/{owner}/{repo}      — curator edit
  POST /api/repos/{owner}/{repo}/exclude — exclude repo

Fixture state (mini-github-repos-full.json):
  callummcdougall/ARENA_3.0   (repo_type=course,    stars=1060)
  hijohnnylin/neuronpedia     (repo_type=research,  stars=829)
  saprmarks/dictionary_learning (repo_type=research, stars=414, linked_paper_url set)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ndif_citations import config
from ndif_citations.server import deps
from ndif_citations.server.app import create_app

# ---------------------------------------------------------------------------
# Known repo IDs from the fixture
# ---------------------------------------------------------------------------

ARENA_OWNER = "callummcdougall"
ARENA_REPO = "ARENA_3.0"
ARENA_ID = f"{ARENA_OWNER}/{ARENA_REPO}"

NEURO_OWNER = "hijohnnylin"
NEURO_REPO = "neuronpedia"
NEURO_ID = f"{NEURO_OWNER}/{NEURO_REPO}"

DICT_OWNER = "saprmarks"
DICT_REPO = "dictionary_learning"
DICT_ID = f"{DICT_OWNER}/{DICT_REPO}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(fixture_state: Path) -> TestClient:
    """Return a TestClient with get_output_dir overridden to fixture_state."""
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def client_with_tmp_settings(fixture_state: Path, tmp_path: Path, monkeypatch):
    """TestClient + monkeypatched _SETTINGS_FILE pointing at a fresh tmp file.

    Returns (client, settings_file_path) so tests can inspect the written JSON.
    """
    tmp_settings = tmp_path / "test_settings.json"
    monkeypatch.setattr(config, "_SETTINGS_FILE", tmp_settings)
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    return TestClient(app, raise_server_exceptions=True), tmp_settings


# ---------------------------------------------------------------------------
# 1. GET /api/repos — list all repos
# ---------------------------------------------------------------------------

def test_list_repos_returns_all(client: TestClient):
    resp = client.get("/api/repos")
    assert resp.status_code == 200
    rows = resp.json()

    assert len(rows) == 3

    expected_keys = {
        "id", "owner", "repo", "url", "description", "stars", "forks",
        "language", "repo_type", "category", "linked_paper_url",
        "last_commit", "manual_override",
    }
    for row in rows:
        assert expected_keys == set(row.keys()), f"Row keys mismatch: {set(row.keys())}"

    # ARENA_3.0 must be present
    ids = {r["id"] for r in rows}
    assert ARENA_ID in ids


def test_list_repos_default_sort_stars_desc(client: TestClient):
    resp = client.get("/api/repos")
    assert resp.status_code == 200
    rows = resp.json()
    stars = [r["stars"] for r in rows if r["stars"] is not None]
    assert stars == sorted(stars, reverse=True), "Default sort should be stars descending"


# ---------------------------------------------------------------------------
# 2. GET /api/repos?repo_type=course — filter by repo_type
# ---------------------------------------------------------------------------

def test_list_repos_filter_course(client: TestClient):
    resp = client.get("/api/repos", params={"repo_type": "course"})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["id"] == ARENA_ID
    assert rows[0]["repo_type"] == "course"


def test_list_repos_filter_research(client: TestClient):
    resp = client.get("/api/repos", params={"repo_type": "research"})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2
    for row in rows:
        assert row["repo_type"] == "research"


def test_list_repos_filter_experiment_empty(client: TestClient):
    resp = client.get("/api/repos", params={"repo_type": "experiment"})
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# 3. GET /api/repos?q=<substring> — text search
# ---------------------------------------------------------------------------

def test_list_repos_query_filter(client: TestClient):
    # "neuronpedia" appears only in owner/repo
    resp = client.get("/api/repos", params={"q": "neuronpedia"})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["id"] == NEURO_ID


def test_list_repos_query_no_match(client: TestClient):
    resp = client.get("/api/repos", params={"q": "zzz_no_match_zzz"})
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# 4. GET /api/repos?sort=<sort> — ordering
# ---------------------------------------------------------------------------

def test_list_repos_sort_name(client: TestClient):
    resp = client.get("/api/repos", params={"sort": "name"})
    assert resp.status_code == 200
    rows = resp.json()
    ids_lower = [r["id"].lower() for r in rows]
    assert ids_lower == sorted(ids_lower), "name sort: not alphabetical"


def test_list_repos_sort_recent(client: TestClient):
    resp = client.get("/api/repos", params={"sort": "recent"})
    assert resp.status_code == 200
    rows = resp.json()
    # All three fixture repos have last_commit set; verify descending order
    dates = [r["last_commit"] for r in rows if r["last_commit"] is not None]
    assert dates == sorted(dates, reverse=True), "recent sort: not descending by last_commit"


# ---------------------------------------------------------------------------
# 5. GET /api/repos/{owner}/{repo} — single repo
# ---------------------------------------------------------------------------

def test_get_repo_known(client: TestClient):
    resp = client.get(f"/api/repos/{NEURO_OWNER}/{NEURO_REPO}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["owner"] == NEURO_OWNER
    assert data["repo"] == NEURO_REPO
    # to_full_dict() has more fields than the list row
    assert "readme_arxiv_ids" in data
    assert "category" in data


def test_get_repo_unknown_returns_404(client: TestClient):
    resp = client.get("/api/repos/unknown-owner/unknown-repo")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 6. PATCH /api/repos/{owner}/{repo} — edit repo_type
# ---------------------------------------------------------------------------

def test_patch_repo_type_updates_and_sets_manual_override(client: TestClient, fixture_state: Path):
    # ARENA_3.0 is currently "course" — promote to "research" as a curator override
    resp = client.patch(
        f"/api/repos/{ARENA_OWNER}/{ARENA_REPO}",
        json={"repo_type": "research"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["repo_type"] == "research"
    assert data["manual_override"] is True

    # Verify persisted to disk
    full_json = fixture_state / "github-repos-full.json"
    raw = json.loads(full_json.read_text())
    arena = next(r for r in raw if r["owner"] == ARENA_OWNER and r["repo"] == ARENA_REPO)
    assert arena["repo_type"] == "research"
    assert arena["manual_override"] is True


def test_patch_repo_type_invalid_returns_422(client: TestClient):
    resp = client.patch(
        f"/api/repos/{ARENA_OWNER}/{ARENA_REPO}",
        json={"repo_type": "invalid_type"},
    )
    assert resp.status_code == 422


def test_patch_unknown_field_returns_422(client: TestClient):
    resp = client.patch(
        f"/api/repos/{ARENA_OWNER}/{ARENA_REPO}",
        json={"stars": 9999},
    )
    assert resp.status_code == 422


def test_patch_unknown_repo_returns_404(client: TestClient):
    resp = client.patch(
        "/api/repos/no-such-owner/no-such-repo",
        json={"repo_type": "research"},
    )
    assert resp.status_code == 404


def test_patch_linked_paper_url(client: TestClient):
    resp = client.patch(
        f"/api/repos/{ARENA_OWNER}/{ARENA_REPO}",
        json={"linked_paper_url": "https://arxiv.org/abs/2407.14561"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["linked_paper_url"] == "https://arxiv.org/abs/2407.14561"
    assert data["manual_override"] is True


def test_patch_description(client: TestClient):
    resp = client.patch(
        f"/api/repos/{NEURO_OWNER}/{NEURO_REPO}",
        json={"description": "Updated description"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["description"] == "Updated description"
    assert data["manual_override"] is True


# ---------------------------------------------------------------------------
# 7. POST /api/repos/{owner}/{repo}/exclude
# ---------------------------------------------------------------------------

def test_exclude_repo_removes_from_db_and_adds_to_settings(
    fixture_state: Path, tmp_path: Path, monkeypatch
):
    """Exclude a repo: verify DB removal and settings update."""
    tmp_settings = tmp_path / "test_settings.json"
    monkeypatch.setattr(config, "_SETTINGS_FILE", tmp_settings)

    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    client = TestClient(app, raise_server_exceptions=True)

    resp = client.post(f"/api/repos/{DICT_OWNER}/{DICT_REPO}/exclude")
    assert resp.status_code == 200
    result = resp.json()
    assert result["excluded"] == DICT_ID
    assert result["remaining"] == 2  # 3 repos minus the excluded one
    assert result["was_present"] is True

    # Repo must be gone from github-repos-full.json
    full_json = fixture_state / "github-repos-full.json"
    raw = json.loads(full_json.read_text())
    ids = [f"{r['owner']}/{r['repo']}" for r in raw]
    assert DICT_ID not in ids

    # owner/repo must be in the tmp settings file
    settings_data = json.loads(tmp_settings.read_text())
    assert DICT_ID in settings_data["excluded_github_repos"]


def test_exclude_repo_idempotent_not_in_db(
    fixture_state: Path, tmp_path: Path, monkeypatch
):
    """Excluding a repo not in the DB still adds it to excluded list, returns was_present=False."""
    tmp_settings = tmp_path / "test_settings.json"
    monkeypatch.setattr(config, "_SETTINGS_FILE", tmp_settings)

    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    client = TestClient(app, raise_server_exceptions=True)

    resp = client.post("/api/repos/ghost-owner/ghost-repo/exclude")
    assert resp.status_code == 200
    result = resp.json()
    assert result["excluded"] == "ghost-owner/ghost-repo"
    assert result["remaining"] == 3  # all 3 still present (it wasn't there)
    assert result["was_present"] is False

    # Still added to settings
    settings_data = json.loads(tmp_settings.read_text())
    assert "ghost-owner/ghost-repo" in settings_data["excluded_github_repos"]


# ---------------------------------------------------------------------------
# 8. Mutation during active run → 409
# ---------------------------------------------------------------------------

class _FakeActiveRunner:
    """Stub runner that always reports active=True."""
    active = True


def test_patch_during_active_run_returns_409(fixture_state: Path):
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    app.dependency_overrides[deps.get_runner] = lambda: _FakeActiveRunner()
    client = TestClient(app, raise_server_exceptions=True)

    resp = client.patch(
        f"/api/repos/{NEURO_OWNER}/{NEURO_REPO}",
        json={"repo_type": "experiment"},
    )
    assert resp.status_code == 409


def test_exclude_during_active_run_returns_409(fixture_state: Path):
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    app.dependency_overrides[deps.get_runner] = lambda: _FakeActiveRunner()
    client = TestClient(app, raise_server_exceptions=True)

    resp = client.post(f"/api/repos/{ARENA_OWNER}/{ARENA_REPO}/exclude")
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# 9. _tag_repo_type honors manual_override — confirmed by code inspection
# ---------------------------------------------------------------------------
# The guard is present at discover.py line 1124-1125:
#   if repo.manual_override:
#       return repo.repo_type
# This test verifies the behavior via the model layer directly.

def test_tag_repo_type_skips_manual_override_repos():
    """_tag_repo_type must return the existing repo_type when manual_override=True."""
    from ndif_citations.discover import _tag_repo_type
    from ndif_citations.models import DiscoveredRepo

    # Repo that WOULD be tagged "course" by name pattern, but manual_override=True
    repo = DiscoveredRepo(
        owner="test",
        repo="ARENA_homework_assignment",
        url="https://github.com/test/ARENA_homework_assignment",
        repo_type="research",   # manually set to research
        manual_override=True,
    )
    result = _tag_repo_type(repo, unlinked_set=set())
    assert result == "research", (
        "_tag_repo_type should return the existing repo_type for manual_override repos, "
        f"got {result!r}"
    )
