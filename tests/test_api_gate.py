"""Tests for the gate REST endpoints — Task 3.2.

POST  /api/runs/{run_id}/gate  — submit the curator selection, unblock worker.
GET   /api/runs/active         — return the in-progress run (or null if idle).

Uses ``TestClient`` + ``app.dependency_overrides`` to inject:
  * a fresh ``JobRunner`` per test (per-test runner isolation)
  * a temp-dir ``get_output_dir`` (fixture_state)
  * ``install_pipeline_fakes`` to make pipeline stages fast/deterministic

The fakes give two gate candidates:
  * NEW paper    — arxiv:9999.99999   (genuinely new)
  * REPROCESS    — arxiv:2602.16080   (existing paper whose content hash changed)

Key invariant for non-hanging tests: every test that parks the worker at the
gate MUST call submit_gate (or cancel) before finishing — otherwise the daemon
thread blocks forever and the next test's fresh runner races with the parked one.
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from ndif_citations import orchestrator
from ndif_citations.jobs import JobRunner
from ndif_citations.server import deps
from ndif_citations.server.app import create_app
from tests.helpers.fakes import install_pipeline_fakes

# Candidate paper IDs produced by the fake pipeline.
NEW_PAPER_ID = "arxiv:9999.99999"
EXISTING_PAPER_ID = "arxiv:2602.16080"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    """Poll *predicate* until truthy or *timeout* seconds elapse."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _make_client(monkeypatch, fixture_state, runner: JobRunner | None = None):
    """Return a (TestClient, runner) pair with fakes and output-dir override active."""
    install_pipeline_fakes(monkeypatch, orchestrator)

    _runner = runner if runner is not None else JobRunner()
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    app.dependency_overrides[deps.get_runner] = lambda: _runner

    return TestClient(app, raise_server_exceptions=True), _runner


# ---------------------------------------------------------------------------
# 1. Happy-path: full gate flow → process a candidate → done.
# ---------------------------------------------------------------------------


def test_gate_flow_process(monkeypatch, fixture_state):
    """POST /runs → poll to awaiting_review → GET /active → POST /gate → done."""
    client, runner = _make_client(monkeypatch, fixture_state)

    # Start an incremental run.
    resp = client.post("/api/runs", json={"mode": "incremental"})
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    # Poll until the worker parks at the gate.
    assert _wait_until(
        lambda: client.get(f"/api/runs/{run_id}").json().get("state") == "awaiting_review"
    ), f"run did not reach gate; state={client.get(f'/api/runs/{run_id}').json().get('state')!r}"

    # Verify paper_candidates are non-empty in the run record.
    run_data = client.get(f"/api/runs/{run_id}").json()
    assert run_data["state"] == "awaiting_review"
    candidates = run_data["paper_candidates"]
    assert len(candidates) > 0, "expected at least one paper candidate at the gate"

    # GET /active should return the awaiting_review record.
    active_resp = client.get("/api/runs/active")
    assert active_resp.status_code == 200, active_resp.text
    active_data = active_resp.json()
    assert active_data["active"] is not None, "expected active run, got null"
    assert active_data["active"]["state"] == "awaiting_review"
    assert len(active_data["active"]["paper_candidates"]) > 0

    # Pick one candidate ID to process.
    candidate_ids = [c["id"] for c in candidates]
    assert NEW_PAPER_ID in candidate_ids, (
        f"expected {NEW_PAPER_ID!r} in candidates: {candidate_ids}"
    )

    # Submit the gate selection — process the new paper.
    gate_resp = client.post(
        f"/api/runs/{run_id}/gate",
        json={"process_ids": [NEW_PAPER_ID], "discard_ids": [], "edits": {}},
    )
    assert gate_resp.status_code == 200, gate_resp.text
    gate_data = gate_resp.json()
    assert gate_data["status"] == "processing"
    assert gate_data["run_id"] == run_id

    # Poll until done.
    assert _wait_until(
        lambda: client.get(f"/api/runs/{run_id}").json().get("state") == "done",
        timeout=8.0,
    ), f"run did not reach done; state={client.get(f'/api/runs/{run_id}').json().get('state')!r}"


# ---------------------------------------------------------------------------
# 2. Wrong state → 409.
# ---------------------------------------------------------------------------


def test_gate_wrong_state_409(monkeypatch, fixture_state):
    """POST /gate on a run that is NOT awaiting_review → 409."""
    client, runner = _make_client(monkeypatch, fixture_state)

    # Start a fresh run (no gate) and wait for it to finish.
    resp = client.post("/api/runs", json={"mode": "fresh"})
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    assert _wait_until(
        lambda: client.get(f"/api/runs/{run_id}").json().get("state") == "done",
        timeout=8.0,
    ), f"fresh run did not finish; state={client.get(f'/api/runs/{run_id}').json().get('state')!r}"

    # Now submit gate on the done run — must be 409.
    gate_resp = client.post(
        f"/api/runs/{run_id}/gate",
        json={"process_ids": [], "discard_ids": [], "edits": {}},
    )
    assert gate_resp.status_code == 409, gate_resp.text
    assert "awaiting review" in gate_resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 3. Invalid edit field → 422 (run stays at the gate, not consumed).
# ---------------------------------------------------------------------------


def test_gate_invalid_edit_422(monkeypatch, fixture_state):
    """POST /gate with an unknown edit field → 422; run remains awaiting_review."""
    client, runner = _make_client(monkeypatch, fixture_state)

    resp = client.post("/api/runs", json={"mode": "incremental"})
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    assert _wait_until(
        lambda: client.get(f"/api/runs/{run_id}").json().get("state") == "awaiting_review"
    ), "run did not reach gate"

    # Submit with a bogus field name — should be 422.
    gate_resp = client.post(
        f"/api/runs/{run_id}/gate",
        json={
            "process_ids": [NEW_PAPER_ID],
            "discard_ids": [],
            "edits": {NEW_PAPER_ID: {"bogus": "x"}},
        },
    )
    assert gate_resp.status_code == 422, gate_resp.text

    # Run must still be awaiting_review (not consumed by the bad request).
    state = client.get(f"/api/runs/{run_id}").json().get("state")
    assert state == "awaiting_review", (
        f"run was consumed by invalid edit; state={state!r}"
    )

    # Clean up: submit a valid gate so the worker thread can finish.
    cleanup = client.post(
        f"/api/runs/{run_id}/gate",
        json={"process_ids": [NEW_PAPER_ID], "discard_ids": [], "edits": {}},
    )
    assert cleanup.status_code == 200, cleanup.text
    assert _wait_until(
        lambda: client.get(f"/api/runs/{run_id}").json().get("state") == "done",
        timeout=8.0,
    ), "run did not finish after cleanup gate"


# ---------------------------------------------------------------------------
# 4. Unknown run ID → 404.
# ---------------------------------------------------------------------------


def test_gate_unknown_run_404(monkeypatch, fixture_state):
    """POST /gate for a non-existent run_id → 404."""
    client, _ = _make_client(monkeypatch, fixture_state)

    gate_resp = client.post(
        "/api/runs/does-not-exist/gate",
        json={"process_ids": [], "discard_ids": [], "edits": {}},
    )
    assert gate_resp.status_code == 404, gate_resp.text


# ---------------------------------------------------------------------------
# 5. No active run → GET /active returns {"active": null}.
# ---------------------------------------------------------------------------


def test_active_none_when_idle(monkeypatch, fixture_state):
    """GET /runs/active on a fresh runner (nothing started) → {"active": null}."""
    client, _ = _make_client(monkeypatch, fixture_state)

    resp = client.get("/api/runs/active")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"active": None}


# ---------------------------------------------------------------------------
# 6. Active run parked at the gate → GET /active returns the record.
# ---------------------------------------------------------------------------


def test_active_returns_running_or_awaiting(monkeypatch, fixture_state):
    """GET /active while a run is awaiting_review returns the record + candidates."""
    client, runner = _make_client(monkeypatch, fixture_state)

    resp = client.post("/api/runs", json={"mode": "incremental"})
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    # Poll until parked at the gate.
    assert _wait_until(
        lambda: client.get(f"/api/runs/{run_id}").json().get("state") == "awaiting_review"
    ), "run did not reach gate"

    active_resp = client.get("/api/runs/active")
    assert active_resp.status_code == 200, active_resp.text
    active_data = active_resp.json()

    assert active_data["active"] is not None, "expected active run"
    assert active_data["active"]["state"] == "awaiting_review"
    assert active_data["active"]["run_id"] == run_id
    # paper_candidates must be present and non-empty when awaiting review.
    assert len(active_data["active"]["paper_candidates"]) > 0, (
        "expected paper_candidates in active awaiting_review record"
    )

    # IMPORTANT: unblock the worker so the test does not leave a parked daemon.
    cleanup = client.post(
        f"/api/runs/{run_id}/gate",
        json={"process_ids": [], "discard_ids": [], "edits": {}},
    )
    assert cleanup.status_code == 200, cleanup.text
    assert _wait_until(
        lambda: client.get(f"/api/runs/{run_id}").json().get("state") == "done",
        timeout=8.0,
    ), "run did not finish after cleanup gate"
