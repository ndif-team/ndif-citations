"""Cross-origin writes to /api/* are refused (CSRF guard on the unauthenticated local app)."""
import pytest
from fastapi.testclient import TestClient

from ndif_citations.server import deps
from ndif_citations.server.app import create_app


@pytest.fixture()
def client_and_pushes(fixture_state, monkeypatch):
    pushes = []
    monkeypatch.setattr("ndif_citations.catalog_git.push_catalog",
                        lambda repo, msg, **k: pushes.append(msg) or {"pushed": True, "commit": "abc",
                                                                      "conflict_branch": None, "message": "ok"})
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    return TestClient(app), pushes


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://127.0.0.1.evil.example"])
def test_foreign_origin_write_is_refused(client_and_pushes, origin):
    client, pushes = client_and_pushes
    r = client.post("/api/publish", headers={"Origin": origin})
    assert r.status_code == 403 and r.json() == {"detail": "cross-origin request refused"}
    assert pushes == []


@pytest.mark.parametrize("headers", [{"Origin": "http://127.0.0.1:8723"}, {"Origin": "http://[::1]:8723"},
                                     {"Origin": "http://localhost:5173"}, {}])
def test_loopback_or_absent_origin_write_is_allowed(client_and_pushes, headers):
    client, pushes = client_and_pushes
    r = client.post("/api/publish", headers=headers)
    assert r.status_code == 200 and len(pushes) == 1


def test_read_with_foreign_origin_is_not_blocked(client_and_pushes, monkeypatch):
    monkeypatch.setattr("ndif_citations.publish.preview", lambda out, root: {"papers": {}, "repos": {}, "images": {}})
    client, _ = client_and_pushes
    assert client.get("/api/publish/preview", headers={"Origin": "https://evil.example"}).status_code == 200
