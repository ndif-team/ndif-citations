"""Tests for the /api/runs REST router (Tasks 2.3-2.4).

Uses FastAPI TestClient with:
  * ``app.dependency_overrides[deps.get_output_dir]`` → fixture_state tmp dir
  * ``app.dependency_overrides[deps.get_runner]``    → fresh JobRunner per test
  * ``install_pipeline_fakes`` to make pipeline runs fast/deterministic

Per-test runner isolation avoids cross-test bleed from the module-level
singleton in deps.py.
"""
from __future__ import annotations

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from ndif_citations import orchestrator
from ndif_citations.jobs import JobRunner
from ndif_citations.server import deps
from ndif_citations.server.app import create_app
from tests.helpers.fakes import install_pipeline_fakes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_until(predicate, timeout: float = 3.0, interval: float = 0.02) -> bool:
    """Poll *predicate* until truthy or *timeout* seconds elapse."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _make_client(monkeypatch, fixture_state, runner: JobRunner | None = None):
    """Return a (TestClient, runner) pair with fakes + output-dir override active."""
    install_pipeline_fakes(monkeypatch, orchestrator)

    _runner = runner if runner is not None else JobRunner()
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    app.dependency_overrides[deps.get_runner] = lambda: _runner

    return TestClient(app, raise_server_exceptions=True), _runner


# ---------------------------------------------------------------------------
# 1. Start a run and poll status until done.
# ---------------------------------------------------------------------------

def test_start_run_and_poll_status(monkeypatch, fixture_state):
    client, runner = _make_client(monkeypatch, fixture_state)

    resp = client.post("/api/runs", json={"mode": "incremental", "skip_github": True})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "run_id" in data
    assert data["state"] == "running"
    run_id = data["run_id"]

    # Poll GET /api/runs/{run_id} until state == "done".
    assert _wait_until(
        lambda: client.get(f"/api/runs/{run_id}").json().get("state") == "done"
    ), f"run did not finish; last state={client.get(f'/api/runs/{run_id}').json().get('state')!r}"

    status_data = client.get(f"/api/runs/{run_id}").json()
    assert status_data["state"] == "done"
    assert status_data["run_id"] == run_id
    # counts should be present on success
    assert "counts" in status_data
    assert isinstance(status_data["counts"], dict)


# ---------------------------------------------------------------------------
# 2. Second POST while first is running → 409.
# ---------------------------------------------------------------------------

def test_start_run_conflict_409(monkeypatch, fixture_state):
    import ndif_citations.process as process_mod

    install_pipeline_fakes(monkeypatch, orchestrator)

    release = threading.Event()
    entered = threading.Event()

    def _blocking_summary(paper):
        entered.set()
        release.wait(timeout=5.0)
        return f"Fake summary for: {paper.title}"

    monkeypatch.setattr(process_mod, "generate_summary", _blocking_summary)

    runner = JobRunner()
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    app.dependency_overrides[deps.get_runner] = lambda: runner

    client = TestClient(app, raise_server_exceptions=True)

    # First POST — should succeed.
    resp1 = client.post("/api/runs", json={"mode": "incremental"})
    assert resp1.status_code == 200, resp1.text
    run_id = resp1.json()["run_id"]

    # Wait until the worker is blocked inside the pipeline.
    assert entered.wait(timeout=2.0), "worker never reached blocking summary"
    assert runner.active

    # Second POST while first is active — must return 409.
    resp2 = client.post("/api/runs", json={"mode": "incremental"})
    assert resp2.status_code == 409, resp2.text

    # Release the block and let the first run finish.
    release.set()
    assert _wait_until(lambda: runner.status().state == "done"), (
        f"run did not finish; state={runner.status().state}"
    )

    # Verify first run completed successfully.
    status_data = client.get(f"/api/runs/{run_id}").json()
    assert status_data["state"] == "done"


# ---------------------------------------------------------------------------
# 3. GET unknown run_id → 404.
# ---------------------------------------------------------------------------

def test_get_unknown_run_404(monkeypatch, fixture_state):
    client, _ = _make_client(monkeypatch, fixture_state)

    resp = client.get("/api/runs/does-not-exist")
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# 4. GET /runs lists runs after a completed run.
# ---------------------------------------------------------------------------

def test_history_lists_runs(monkeypatch, fixture_state):
    client, runner = _make_client(monkeypatch, fixture_state)

    resp = client.post("/api/runs", json={"mode": "fresh"})
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    assert _wait_until(lambda: runner.status().state == "done"), (
        f"run did not finish; state={runner.status().state}"
    )

    hist_resp = client.get("/api/runs")
    assert hist_resp.status_code == 200, hist_resp.text
    runs_list = hist_resp.json()
    assert isinstance(runs_list, list)
    run_ids = [r["run_id"] for r in runs_list]
    assert run_id in run_ids, f"{run_id!r} not in history: {run_ids}"


# ---------------------------------------------------------------------------
# 5. Invalid mode → 422.
# ---------------------------------------------------------------------------

def test_invalid_mode_422(monkeypatch, fixture_state):
    client, _ = _make_client(monkeypatch, fixture_state)

    resp = client.post("/api/runs", json={"mode": "bogus"})
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# 6. Fallback: GET finished run A after run B starts (file fallback).
# ---------------------------------------------------------------------------

def test_get_finished_run_by_id_after_new_run(monkeypatch, fixture_state):
    """Run A completes; run B starts (blocking); GET /api/runs/{A} still returns A."""
    import ndif_citations.process as process_mod

    install_pipeline_fakes(monkeypatch, orchestrator)

    # ---- Run A: fast, no blocking -------------------------------------------
    runner_a = JobRunner()
    app_a = create_app()
    app_a.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    app_a.dependency_overrides[deps.get_runner] = lambda: runner_a
    client_a = TestClient(app_a, raise_server_exceptions=True)

    resp_a = client_a.post("/api/runs", json={"mode": "incremental"})
    assert resp_a.status_code == 200, resp_a.text
    run_id_a = resp_a.json()["run_id"]

    # Wait for run A to finish.
    assert _wait_until(lambda: runner_a.status().state == "done"), (
        f"run A did not finish; state={runner_a.status().state}"
    )

    # ---- Run B: blocking so runner_b.active == True when we query A ----------
    release_b = threading.Event()
    entered_b = threading.Event()

    def _blocking_summary(paper):
        entered_b.set()
        release_b.wait(timeout=5.0)
        return f"Fake summary for: {paper.title}"

    monkeypatch.setattr(process_mod, "generate_summary", _blocking_summary)

    runner_b = JobRunner()
    app_b = create_app()
    app_b.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    app_b.dependency_overrides[deps.get_runner] = lambda: runner_b
    client_b = TestClient(app_b, raise_server_exceptions=True)

    resp_b = client_b.post("/api/runs", json={"mode": "incremental"})
    assert resp_b.status_code == 200, resp_b.text
    run_id_b = resp_b.json()["run_id"]

    # Wait until B is in the pipeline (blocking).
    assert entered_b.wait(timeout=2.0), "run B never started blocking"
    assert runner_b.active

    # ---- Fetch A via runner_b (runner_b only knows about B) -----------------
    # runner_b.status(run_id_a) will raise KeyError → fallback to file
    resp_get_a = client_b.get(f"/api/runs/{run_id_a}")
    assert resp_get_a.status_code == 200, (
        f"expected 200 for run A via file fallback, got {resp_get_a.status_code}: {resp_get_a.text}"
    )
    a_data = resp_get_a.json()
    assert a_data["run_id"] == run_id_a
    assert a_data["state"] == "done"

    # Release B and let it finish.
    release_b.set()
    assert _wait_until(lambda: runner_b.status().state == "done"), (
        f"run B did not finish; state={runner_b.status().state}"
    )
    # Ensure run_id_b is accessible too.
    assert client_b.get(f"/api/runs/{run_id_b}").json()["state"] == "done"


# ---------------------------------------------------------------------------
# SSE live events + cancel (Task 2.5)
# ---------------------------------------------------------------------------


def _parse_sse_lines(lines) -> tuple[list[dict], bool]:
    """Parse SSE ``data:`` / ``event:`` lines.

    Returns (data_payloads, saw_end). ``data: {}`` that follows an ``event: end``
    line is treated as the end marker, not a real event payload.
    """
    payloads: list[dict] = []
    saw_end = False
    pending_event: str | None = None
    for raw in lines:
        line = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
        if line.startswith("event:"):
            pending_event = line[len("event:"):].strip()
            if pending_event == "end":
                saw_end = True
            continue
        if line.startswith("data:"):
            body = line[len("data:"):].strip()
            if pending_event == "end":
                pending_event = None
                continue
            try:
                payloads.append(json.loads(body))
            except ValueError:
                pass
            pending_event = None
    return payloads, saw_end


# ---------------------------------------------------------------------------
# 7. SSE replay of a completed run: events + end marker, stream closes.
# ---------------------------------------------------------------------------

def test_sse_replays_completed_run(monkeypatch, fixture_state):
    client, runner = _make_client(monkeypatch, fixture_state)

    resp = client.post("/api/runs", json={"mode": "incremental"})
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    assert _wait_until(lambda: runner.status().state == "done"), (
        f"run did not finish; state={runner.status().state}"
    )

    # Collect the stream on a helper thread guarded by a join timeout so a hang
    # can't wedge CI.
    result: dict = {}

    def _collect():
        with client.stream("GET", f"/api/runs/{run_id}/events") as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            payloads, saw_end = _parse_sse_lines(r.iter_lines())
            result["payloads"] = payloads
            result["saw_end"] = saw_end

    t = threading.Thread(target=_collect, daemon=True)
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive(), "SSE replay stream did not close (hung)"

    assert result.get("saw_end") is True, "missing SSE end marker"
    payloads = result.get("payloads", [])
    assert len(payloads) >= 1, "expected at least one event in the replay"
    assert any(p.get("type") == "stage_start" for p in payloads), (
        f"expected a stage_start event; got {[p.get('type') for p in payloads]}"
    )


# ---------------------------------------------------------------------------
# 8. SSE opened against a still-running run: exercises the live subscribe()
#    path (not file replay), delivers events + end marker, and closes.
#
# NOTE on true live interleaving over TestClient
# ----------------------------------------------
# Starlette's TestClient uses httpx's ASGITransport, which BUFFERS the entire
# response body before returning it (ASGIResponseStream.__aiter__ yields
# ``b"".join(self._body)`` — one chunk, after the ASGI app has run to
# completion). So ``iter_lines`` cannot observe an SSE event *before* the run
# finishes — every chunk is delivered in one batch at the end. This is a
# transport limitation, not a bug in subscribe()/the endpoint.
#
# True live interleaving (an event delivered while the worker is still blocked)
# IS verified directly at the JobRunner layer in
# tests/test_jobs.py::test_subscribe_live_then_terminates, which drives
# subscribe() without the HTTP transport. Here we instead prove the HTTP layer
# routes a *running* run through the live subscribe() generator (registering a
# real subscriber queue, NOT replaying a persisted file — the run file does not
# exist yet while the run is still running) and that the stream terminates via
# the sentinel once the run completes.
# ---------------------------------------------------------------------------

def test_sse_live_during_run(monkeypatch, fixture_state):
    import ndif_citations.process as process_mod

    install_pipeline_fakes(monkeypatch, orchestrator)

    entered = threading.Event()
    release = threading.Event()

    def _blocking_summary(paper):
        entered.set()
        release.wait(timeout=5.0)
        return f"Fake summary for: {paper.title}"

    monkeypatch.setattr(process_mod, "generate_summary", _blocking_summary)

    runner = JobRunner()
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    app.dependency_overrides[deps.get_runner] = lambda: runner
    client = TestClient(app, raise_server_exceptions=True)

    resp = client.post("/api/runs", json={"mode": "incremental"})
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    # Worker parked mid-pipeline (discover/enrich/route already emitted events).
    assert entered.wait(timeout=2.0), "worker never reached the blocking summary"
    assert runner.active

    # The run file must NOT exist yet (run still in flight) — proving the stream
    # below resolves through the live subscribe() path (which registers a real
    # subscriber queue), not the persisted-file replay branch.
    assert not (fixture_state / "runs" / f"{run_id}.json").exists()

    # Prove the live queue branch is actually taken: open a direct subscribe()
    # against the still-running run and confirm a queue is registered. (The HTTP
    # stream uses the same code path; TestClient buffers the HTTP response so we
    # can't observe its queue registration timing, hence this direct probe.)
    probe = runner.subscribe(run_id)
    first = next(probe)  # forces the snapshot+register critical section to run
    assert first.type == "stage_start"
    assert len(runner.status(run_id)._subscribers) == 1, (
        "subscribe() did not register a live queue for a running run"
    )

    result: dict = {}

    def _collect():
        with client.stream("GET", f"/api/runs/{run_id}/events") as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            payloads, saw_end = _parse_sse_lines(r.iter_lines())
            result["payloads"] = payloads
            result["saw_end"] = saw_end

    t = threading.Thread(target=_collect, daemon=True)
    t.start()

    # Fully drain the probe subscriber on its own thread so it consumes the
    # sentinel and unregisters its queue (no leak, never blocks _append).
    drained: list = []
    drain = threading.Thread(target=lambda: drained.extend(probe), daemon=True)
    drain.start()

    # Release the worker so the run finishes; the sentinel ends every stream.
    release.set()

    t.join(timeout=5.0)
    drain.join(timeout=5.0)
    assert not t.is_alive(), "SSE live stream did not close after run finished"
    assert not drain.is_alive(), "probe subscribe() did not terminate"

    assert _wait_until(lambda: runner.status().state == "done")
    assert result.get("saw_end") is True, "missing SSE end marker"
    payloads = result.get("payloads", [])
    assert any(p.get("type") == "stage_start" for p in payloads), (
        f"expected stage_start in stream; got {[p.get('type') for p in payloads]}"
    )
    # The probe subscriber queue is unregistered after the stream completes.
    assert _wait_until(
        lambda: len(runner.status(run_id)._subscribers) == 0, timeout=2.0
    ), "subscriber queue leaked after stream completed"


# ---------------------------------------------------------------------------
# 9. Cancel endpoint: 200 {"status":"cancelling"}, run reaches "cancelled".
# ---------------------------------------------------------------------------

def test_cancel_endpoint(monkeypatch, fixture_state):
    import ndif_citations.process as process_mod

    install_pipeline_fakes(monkeypatch, orchestrator)

    entered = threading.Event()
    release = threading.Event()
    call_count = [0]

    def _blocking_summary(paper):
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0:
            entered.set()
            release.wait(timeout=5.0)
        return f"Fake summary for: {paper.title}"

    monkeypatch.setattr(process_mod, "generate_summary", _blocking_summary)

    runner = JobRunner()
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    app.dependency_overrides[deps.get_runner] = lambda: runner
    client = TestClient(app, raise_server_exceptions=True)

    resp = client.post("/api/runs", json={"mode": "incremental"})
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    assert entered.wait(timeout=2.0), "worker never entered blocking summary"
    assert runner.active

    cancel_resp = client.post(f"/api/runs/{run_id}/cancel")
    assert cancel_resp.status_code == 200, cancel_resp.text
    assert cancel_resp.json() == {"status": "cancelling"}

    # Release item 0; the loop hits cancel_check at the top of item 1.
    release.set()

    assert _wait_until(
        lambda: client.get(f"/api/runs/{run_id}").json().get("state") == "cancelled",
        timeout=4.0,
    ), f"run did not reach cancelled; last={client.get(f'/api/runs/{run_id}').json().get('state')!r}"


# ---------------------------------------------------------------------------
# 10. Cancel of an unknown run id → 404.
# ---------------------------------------------------------------------------

def test_cancel_unknown_run_404(monkeypatch, fixture_state):
    client, _ = _make_client(monkeypatch, fixture_state)
    resp = client.post("/api/runs/does-not-exist/cancel")
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# 11. SSE events for an unknown run id → 404.
# ---------------------------------------------------------------------------

def test_sse_unknown_run_404(monkeypatch, fixture_state):
    client, _ = _make_client(monkeypatch, fixture_state)
    resp = client.get("/api/runs/does-not-exist/events")
    assert resp.status_code == 404, resp.text
