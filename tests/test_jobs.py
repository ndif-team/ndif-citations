"""Tests for the in-process JobRunner (Tasks 2.1-2.2).

The JobRunner drives ``orchestrator.run_pipeline`` on a daemon thread, captures
the progress-event buffer, and persists a run record to ``out/runs/<run_id>.json``.

SSE/HTTP (Tasks 2.3-2.5) are explicitly out of scope here.

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


# ---------------------------------------------------------------------------
# 4. Cancel: a cancel() during processing ends the run in state "cancelled".
# ---------------------------------------------------------------------------

def test_cancel_stops_run(monkeypatch, fixture_state):
    install_pipeline_fakes(monkeypatch, orchestrator)
    out = fixture_state

    # Snapshot the research-papers-full.json bytes BEFORE the run so we can
    # assert they are byte-for-byte unchanged after a cancelled run (nothing
    # was written to disk by the in-flight run).
    papers_file = out / "research-papers-full.json"
    papers_bytes_before = papers_file.read_bytes()

    # Install a blocking generate_summary fake:
    #   * For item 0 (first paper): sets `entered`, then waits for `release`.
    #   * For item 1+: returns normally — but the cancel_check will fire at the
    #     top of the process_papers loop for item 1, raising RunCancelled before
    #     generate_summary is even called for that item.
    import ndif_citations.process as process_mod

    entered = threading.Event()
    release = threading.Event()
    call_count = [0]  # mutable counter shared with fake

    def _blocking_summary(paper):
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0:
            entered.set()
            release.wait(timeout=5.0)
        return f"Fake summary for: {paper.title}"

    monkeypatch.setattr(process_mod, "generate_summary", _blocking_summary)

    runner = JobRunner()
    runner.start(out, mode="incremental")

    # Wait until the worker is parked inside processing of item 0.
    assert entered.wait(timeout=2.0), "worker never entered the blocking summary"
    assert runner.active is True

    # Cancel while the worker is blocked inside item 0's processing.
    runner.cancel()

    # Now release item 0. The loop advances to item 1, hits cancel_check at the
    # top-of-loop, and RunCancelled propagates out of process_stage.
    release.set()

    # Poll until the run reaches a terminal state.
    assert _wait_until(
        lambda: runner.status().state in ("cancelled", "done", "error"),
        timeout=4.0,
    ), f"run did not reach terminal state; state={runner.status().state}"

    rec = runner.status()
    assert rec.state == "cancelled", f"expected cancelled, got {rec.state!r}"
    assert rec.error is None, f"error should be None for a cancelled run, got {rec.error!r}"
    assert not runner.active

    # Persisted JSON must show state="cancelled" and must NOT contain cancel_event.
    run_file = out / "runs" / f"{rec.run_id}.json"
    assert run_file.exists(), "persisted run file missing"
    data = json.loads(run_file.read_text())
    assert data["state"] == "cancelled"
    assert "cancel_event" not in data, "cancel_event must not appear in persisted JSON"

    # Safety assertion: the original research-papers-full.json is byte-unchanged
    # (cancel happened before finalize_stage could write anything).
    papers_bytes_after = papers_file.read_bytes()
    assert papers_bytes_after == papers_bytes_before, (
        "research-papers-full.json was modified by a cancelled run — "
        "finalize_stage should not have run"
    )


# ---------------------------------------------------------------------------
# 5. Cancel no-op: cancel() on a finished (or non-existent) run is safe.
# ---------------------------------------------------------------------------

def test_cancel_noop_when_not_running(monkeypatch, fixture_state):
    install_pipeline_fakes(monkeypatch, orchestrator)
    out = fixture_state

    runner = JobRunner()

    # No run started yet — cancel() must not raise.
    runner.cancel()  # no-op, nothing active

    # Start and let a run finish normally.
    run_id = runner.start(out, mode="incremental")
    assert _wait_until(lambda: runner.status().state == "done"), (
        f"run did not finish; state={runner.status().state}"
    )

    # Cancel after the run is already done — must be a safe no-op.
    runner.cancel()
    runner.cancel(run_id=run_id)

    # State must remain "done" (cancel must not change a terminal state).
    assert runner.status().state == "done"


# ---------------------------------------------------------------------------
# 6. subscribe(): replays the buffered events of a completed run and terminates.
# ---------------------------------------------------------------------------

def test_subscribe_replays_completed_run(monkeypatch, fixture_state):
    """After a run finishes, subscribe(run_id) yields the buffered events and
    terminates (no hang) — exercising the terminal/buffer-replay path."""
    install_pipeline_fakes(monkeypatch, orchestrator)
    out = fixture_state

    runner = JobRunner()
    run_id = runner.start(out, mode="incremental")

    assert _wait_until(lambda: runner.status().state == "done"), (
        f"run did not finish; state={runner.status().state}"
    )

    # Guard against a hang: drain the generator on a helper thread with a join
    # timeout. A terminal-run subscribe must NOT block on a queue.
    collected: list = []

    def _drain():
        collected.extend(runner.subscribe(run_id))

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    t.join(timeout=2.0)
    assert not t.is_alive(), "subscribe() hung on a completed run"

    assert collected, "subscribe() yielded no events for a completed run"
    types = [ev.type for ev in collected]
    assert "stage_start" in types, f"expected a stage_start event, got {types}"

    # The replay must match the buffer exactly (same objects, same order).
    assert collected == runner.status().events


# ---------------------------------------------------------------------------
# 7. subscribe(): unknown run id raises KeyError.
# ---------------------------------------------------------------------------

def test_subscribe_unknown_run_raises(monkeypatch, fixture_state):
    install_pipeline_fakes(monkeypatch, orchestrator)
    runner = JobRunner()
    with pytest.raises(KeyError):
        # The generator body raises lazily on first iteration.
        list(runner.subscribe("nope-does-not-exist"))


# ---------------------------------------------------------------------------
# 8. subscribe(): delivers live events while the worker is still running, then
#    terminates cleanly once the run finishes.
# ---------------------------------------------------------------------------

def test_subscribe_live_then_terminates(monkeypatch, fixture_state):
    """A subscriber opened while the run is blocked mid-pipeline receives the
    early buffered events, then the sentinel-driven loop terminates after the
    worker is released."""
    install_pipeline_fakes(monkeypatch, orchestrator)
    out = fixture_state

    import ndif_citations.process as process_mod

    entered = threading.Event()
    release = threading.Event()

    def _blocking_summary(paper):
        entered.set()
        release.wait(timeout=5.0)
        return f"Fake summary for: {paper.title}"

    monkeypatch.setattr(process_mod, "generate_summary", _blocking_summary)

    runner = JobRunner()
    run_id = runner.start(out, mode="incremental")

    # Worker parked inside the pipeline (early stages already emitted events).
    assert entered.wait(timeout=2.0), "worker never reached the blocking summary"
    assert runner.active is True

    collected: list = []

    def _drain():
        collected.extend(runner.subscribe(run_id))

    t = threading.Thread(target=_drain, daemon=True)
    t.start()

    # While still blocked, the subscriber should have received the buffered
    # prefix (at least one stage_start from discover/enrich/route).
    assert _wait_until(
        lambda: any(ev.type == "stage_start" for ev in collected),
        timeout=2.0,
    ), f"no stage_start delivered live; got {[e.type for e in collected]}"

    # Release the worker; the run finishes and the sentinel ends the stream.
    release.set()
    t.join(timeout=4.0)
    assert not t.is_alive(), "subscribe() did not terminate after the run finished"

    assert _wait_until(lambda: runner.status().state == "done")
    types = [ev.type for ev in collected]
    assert "stage_start" in types
