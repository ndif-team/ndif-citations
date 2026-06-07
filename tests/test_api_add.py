"""API tests for POST /api/papers/add (Task 4.5 — Part C).

Scenarios
---------
1. Happy path: valid arXiv URL → 200 {run_id, state="running"}; poll until
   "done"; assert the paper is in the DB.
2. Invalid URL (empty string) → 422.
3. Invalid URL (non-http, non-arXiv string) → 422.
4. Conflict: a run is already active → 409.

What is faked / why
-------------------
* ``semanticscholar.SemanticScholar`` — no network.
* ``ndif_citations.extract.enrich_papers`` — identity pass-through.
* ``ndif_citations.process.*`` / ``pdf_cache.get_cached_pdf`` — no LLM / PDF.
  Installed via ``install_pipeline_fakes`` (shared helper) plus direct patches
  on the process module so the worker thread completes quickly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from ndif_citations import orchestrator
from ndif_citations.jobs import JobRunner
from ndif_citations.models import Category, Confidence
from ndif_citations.server import deps
from ndif_citations.server.app import create_app
from tests.helpers.fakes import install_pipeline_fakes

_NEW_ARXIV_ID = "9999.00003"
_NEW_URL = f"https://arxiv.org/abs/{_NEW_ARXIV_ID}"
_GENERIC_HTTP_URL = "https://example.com/papers/some-paper"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _install_add_fakes(monkeypatch) -> None:
    """Install all fakes needed for add_paper_by_url to complete without I/O."""
    import semanticscholar

    fake_s2_paper = MagicMock()
    fake_s2_paper.title = f"Fake Added Paper {_NEW_ARXIV_ID}"
    fake_s2_paper.authors = []
    fake_s2_paper.abstract = "Test abstract."
    fake_s2_paper.venue = "TestConf 2099"
    fake_s2_paper.publicationDate = "2099-01-01"
    fake_s2_paper.externalIds = {}
    fake_s2_paper.paperId = "fake-s2-id"
    fake_s2_paper.openAccessPdf = None

    fake_sch = MagicMock()
    fake_sch.get_paper.return_value = fake_s2_paper
    monkeypatch.setattr(semanticscholar, "SemanticScholar", lambda **kw: fake_sch)

    import ndif_citations.extract as extract_mod

    monkeypatch.setattr(extract_mod, "enrich_papers", lambda papers, raw_dir=None: papers)

    import ndif_citations.pdf_cache as pdf_cache_mod
    import ndif_citations.process as process_mod

    monkeypatch.setattr(
        process_mod, "generate_summary",
        lambda paper: "Fake summary for API add test.",
    )
    monkeypatch.setattr(
        process_mod, "classify_category",
        lambda paper, output_dir, pdf_path=None: (Category.USES_NNSIGHT, 0.85, Confidence.HIGH),
    )
    monkeypatch.setattr(
        process_mod, "extract_thumbnail",
        lambda paper, output_dir, pdf_path=None: None,
    )
    monkeypatch.setattr(
        pdf_cache_mod, "get_cached_pdf",
        lambda paper, output_dir: None,
    )


def _make_client(monkeypatch, fixture_state: Path, runner: JobRunner | None = None):
    """Build a TestClient with fakes + output-dir + runner overrides."""
    _install_add_fakes(monkeypatch)
    _runner = runner if runner is not None else JobRunner()
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    app.dependency_overrides[deps.get_runner] = lambda: _runner
    return TestClient(app, raise_server_exceptions=True), _runner


# ---------------------------------------------------------------------------
# Test 1 — happy path: valid arXiv URL → run completes, paper in DB
# ---------------------------------------------------------------------------


def test_add_paper_happy_path(monkeypatch, fixture_state):
    client, runner = _make_client(monkeypatch, fixture_state)

    resp = client.post("/api/papers/add", json={"url": _NEW_URL})
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert "run_id" in data, f"missing run_id: {data}"
    assert data["state"] == "running", f"unexpected state: {data}"

    run_id = data["run_id"]

    assert _wait_until(
        lambda: client.get(f"/api/runs/{run_id}").json().get("state") == "done"
    ), f"job did not finish; state={client.get(f'/api/runs/{run_id}').json().get('state')!r}"

    # Paper should be in the output JSON
    full_json = fixture_state / "research-papers-full.json"
    assert full_json.exists(), "research-papers-full.json was not written"

    db_data = json.loads(full_json.read_text())
    all_papers = db_data["pending"] + db_data["verified"] + db_data["discarded"]
    arxiv_ids = [p.get("arxiv_id") for p in all_papers]
    assert _NEW_ARXIV_ID in arxiv_ids, (
        f"arxiv_id={_NEW_ARXIV_ID!r} not in DB after add; got: {arxiv_ids}"
    )


# ---------------------------------------------------------------------------
# Test 2 — generic http URL (no arXiv ID) is accepted
# ---------------------------------------------------------------------------


def test_add_paper_generic_http_url_accepted(monkeypatch, fixture_state):
    """A plain https URL with no arXiv ID should be accepted (200) not 422."""
    client, runner = _make_client(monkeypatch, fixture_state)

    resp = client.post("/api/papers/add", json={"url": _GENERIC_HTTP_URL})
    assert resp.status_code == 200, resp.text
    assert "run_id" in resp.json()


# ---------------------------------------------------------------------------
# Test 3 — empty URL → 422
# ---------------------------------------------------------------------------


def test_add_paper_empty_url_422(monkeypatch, fixture_state):
    client, runner = _make_client(monkeypatch, fixture_state)

    resp = client.post("/api/papers/add", json={"url": ""})
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# Test 4 — clearly invalid URL (not http, not arXiv) → 422
# ---------------------------------------------------------------------------


def test_add_paper_invalid_url_422(monkeypatch, fixture_state):
    client, runner = _make_client(monkeypatch, fixture_state)

    resp = client.post("/api/papers/add", json={"url": "not-a-url-at-all"})
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# Test 5 — run already active → 409
# ---------------------------------------------------------------------------


def test_add_paper_conflict_409(fixture_state):
    """With an active run, POST /api/papers/add → 409."""
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state

    stub = MagicMock()
    stub.active = True
    stub.start_job.side_effect = __import__("ndif_citations.jobs", fromlist=["RunActiveError"]).RunActiveError(
        "a pipeline run is already active"
    )
    app.dependency_overrides[deps.get_runner] = lambda: stub

    client = TestClient(app, raise_server_exceptions=True)
    resp = client.post("/api/papers/add", json={"url": _NEW_URL})
    assert resp.status_code == 409, resp.text
