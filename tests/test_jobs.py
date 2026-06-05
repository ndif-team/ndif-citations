"""Tests for the in-process JobRunner (Task 2.1).

The JobRunner drives ``orchestrator.run_pipeline`` on a daemon thread, captures
the progress-event buffer, and persists a run record to ``out/runs/<run_id>.json``.

Cancel (Task 2.2) and SSE/HTTP (Tasks 2.3-2.5) are explicitly out of scope here.

All tests use the deterministic fake harness (``install_pipeline_fakes``) so they
never touch the network or an LLM. The "active" test parks the worker thread
*inside* the pipeline by installing a ``generate_summary`` fake that blocks on a
``threading.Event`` the test controls, then releases it.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from ndif_citations import orchestrator
from ndif_citations.jobs import JobRunner, RunActiveError, RunRecord
from tests.helpers.fakes import install_pipeline_fakes


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Poll *predicate* until it returns truthy or *timeout* seconds elapse."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# ---------------------------------------------------------------------------
# 1. Happy path: a run completes, fills counts, captures events, persists JSON.
# ---------------------------------------------------------------------------

def test_run_completes_and_persists(monkeypatch, fixture_state):
    install_pipeline_fakes(monkeypatch, orchestrator)
    out = fixture_state

    runner = JobRunner()
    run_id = runner.start(out, mode="incremental")

    assert _wait_until(lambda: runner.status().state == "done"), (
        f"run did not finish; state={runner.status().state}"
    )

    rec = runner.status()
    assert isinstance(rec, RunRecord)
    assert rec.run_id == run_id
    assert rec.state == "done"
    assert rec.mode == "incremental"
    assert rec.error is None
    assert rec.finished_at is not None

    # Counts populated from FinalizeResult.run_stats.
    assert rec.counts, "counts should be populated on success"
    assert "total_unique" in rec.counts

    # Events captured; the pipeline emits stage_start at every stage.
    assert rec.events, "events buffer should be non-empty"
    assert any(ev.type == "stage_start" for ev in rec.events)

    # Persisted to out/runs/<run_id>.json and valid JSON.
    run_file = out / "runs" / f"{run_id}.json"
    assert run_file.exists()
    data = json.loads(run_file.read_text())
    assert data["run_id"] == run_id
    assert data["state"] == "done"
    assert isinstance(data["events"], list) and data["events"]
    assert data["events"][0]["type"]  # serialized via ProgressEvent.to_dict()

    # history() surfaces the persisted run.
    hist = runner.history(out)
    assert any(h["run_id"] == run_id for h in hist)


# ---------------------------------------------------------------------------
# 2. One-run-at-a-time: a second start() while active raises RunActiveError.
# ---------------------------------------------------------------------------

def test_second_start_while_active_raises(monkeypatch, fixture_state):
    install_pipeline_fakes(monkeypatch, orchestrator)
    out = fixture_state

    # Block the worker thread *inside* the pipeline: generate_summary is called
    # from process_papers, so parking it here keeps the run in state "running".
    import ndif_citations.process as process_mod

    release = threading.Event()
    entered = threading.Event()

    def _blocking_summary(paper):
        entered.set()
        release.wait(timeout=5.0)
        return f"Fake summary for: {paper.title}"

    monkeypatch.setattr(process_mod, "generate_summary", _blocking_summary)

    runner = JobRunner()
    run_id = runner.start(out, mode="incremental")

    # Wait until the worker is parked inside the pipeline.
    assert entered.wait(timeout=2.0), "worker never reached the blocking summary"
    assert runner.active is True

    # A second start() while one is active must raise.
    with pytest.raises(RunActiveError):
        runner.start(out, mode="incremental")

    # Release the block and let the first run finish.
    release.set()
    assert _wait_until(lambda: runner.status().state == "done"), (
        f"run did not finish after release; state={runner.status().state}"
    )
    assert runner.active is False
    assert runner.status().run_id == run_id

    run_file = out / "runs" / f"{run_id}.json"
    assert run_file.exists()


# ---------------------------------------------------------------------------
# 3. Error path: an exception in the pipeline is captured and persisted.
# ---------------------------------------------------------------------------

def test_run_error_is_captured(monkeypatch, fixture_state):
    install_pipeline_fakes(monkeypatch, orchestrator)
    out = fixture_state

    boom = RuntimeError("discovery exploded")

    def _exploding_s2(raw_dir=None):
        raise boom

    monkeypatch.setattr(orchestrator, "discover_s2_citations", _exploding_s2)

    runner = JobRunner()
    run_id = runner.start(out, mode="incremental")

    assert _wait_until(lambda: runner.status().state == "error"), (
        f"run did not reach error state; state={runner.status().state}"
    )

    rec = runner.status()
    assert rec.state == "error"
    assert rec.error is not None
    assert "discovery exploded" in rec.error
    assert rec.finished_at is not None
    assert not runner.active

    # Persisted even on failure.
    run_file = out / "runs" / f"{run_id}.json"
    assert run_file.exists()
    data = json.loads(run_file.read_text())
    assert data["state"] == "error"
    assert "discovery exploded" in data["error"]
