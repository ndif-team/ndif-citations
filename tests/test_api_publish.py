"""Publish router = preview + push (spec §5.7)."""
from fastapi.testclient import TestClient

from ndif_citations.server import deps
from ndif_citations.server.app import create_app


def _client(fixture_state):
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    return TestClient(app)


def test_preview_endpoint_shape(fixture_state, monkeypatch):
    monkeypatch.setattr("ndif_citations.publish.preview",
                        lambda out, root: {"papers": {"added": [], "changed": [], "removed": []},
                                           "repos": {"added": [], "changed": [], "removed": []},
                                           "images": {"new": [], "changed": []}})
    r = _client(fixture_state).get("/api/publish/preview")
    assert r.status_code == 200 and set(r.json()) == {"papers", "repos", "images"}


def test_push_endpoint_returns_push_result(fixture_state, monkeypatch):
    monkeypatch.setattr("ndif_citations.catalog_git.push_catalog",
                        lambda repo, msg, **k: {"pushed": True, "commit": "abc123", "conflict_branch": None,
                                                "message": "pushed abc123"})
    r = _client(fixture_state).post("/api/publish")
    assert r.status_code == 200 and r.json()["commit"] == "abc123"


def test_push_conflict_is_409_with_message(fixture_state, monkeypatch):
    from ndif_citations import catalog_git
    def _boom(repo, msg, **k):
        raise catalog_git.CatalogPushError("conflict; saved on branch 'catalog-conflict-x'")
    monkeypatch.setattr("ndif_citations.catalog_git.push_catalog", _boom)
    r = _client(fixture_state).post("/api/publish")
    assert r.status_code == 409 and "catalog-conflict-x" in r.json()["detail"]


def test_push_blocked_by_active_run(fixture_state, monkeypatch):
    from types import SimpleNamespace
    pushes = []
    monkeypatch.setattr("ndif_citations.catalog_git.push_catalog", lambda repo, msg, **k: pushes.append(msg))
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    app.dependency_overrides[deps.get_runner] = lambda: SimpleNamespace(active=True)
    r = TestClient(app).post("/api/publish")
    assert r.status_code == 409 and pushes == []


def test_publish_target_endpoints_are_gone(fixture_state):
    assert _client(fixture_state).get("/api/publish/target").status_code == 404
