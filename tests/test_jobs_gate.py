"""Tests for the JobRunner human-in-the-loop gate (Task 3.1).

In ``mode="incremental"`` the JobRunner drives the orchestrator stages
individually (discover -> enrich -> route) and then PAUSES at a gate so a
curator can pick which NEW/REPROCESS candidates get expensive LLM processing.
The worker blocks on a per-run ``threading.Event`` until ``submit_gate`` (or
``cancel``) is called.

``mode="fresh"`` is unchanged — it runs ``run_pipeline`` end-to-end with NO
gate.

All tests use the deterministic fake harness (``install_pipeline_fakes``) so
they never touch the network or an LLM. The fakes give:
  * one genuinely NEW paper (arxiv:9999.99999)
  * one existing-match paper (arxiv:2602.16080) which routes to REPROCESS
    (its abstract differs from the fixture, so the content hash changed).
Both are NEW/REPROCESS → both are gate candidates.

Poll loops use GIL-releasing ``time.sleep`` and are guarded with timeouts so a
gate that never unblocks fails fast instead of hanging the suite.
"""
from __future__ import annotations

import json
import time

import pytest

from ndif_citations import orchestrator
from ndif_citations.jobs import GateError, JobRunner, RunActiveError
from tests.helpers.fakes import EXISTING_ARXIV_ID, install_pipeline_fakes

NEW_PAPER_ID = "arxiv:9999.99999"
EXISTING_PAPER_ID = f"arxiv:{EXISTING_ARXIV_ID}"


def _wait_until(predicate, timeout: float = 3.0, interval: float = 0.01) -> bool:
    """Poll *predicate* until it returns truthy or *timeout* seconds elapse."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _candidate_ids(record) -> list[str]:
    return [c["id"] for c in record.paper_candidates]


def _all_papers(out) -> dict:
    """Load research-papers-full.json as {bucket: [paper dicts]}."""
    return json.loads((out / "research-papers-full.json").read_text())


def _find_paper(out, arxiv_id: str):
    """Return (bucket_name, paper_dict) for the first paper matching arxiv_id."""
    data = _all_papers(out)
    for bucket_name in ("pending", "verified", "discarded"):
        for p in data.get(bucket_name, []):
            if p.get("arxiv_id") == arxiv_id:
                return bucket_name, p
    return None, None


# ---------------------------------------------------------------------------
# 1. Incremental run pauses at the gate (worker blocked, awaiting_review).
# ---------------------------------------------------------------------------

def test_incremental_pauses_at_gate(monkeypatch, fixture_state):
    install_pipeline_fakes(monkeypatch, orchestrator)
    out = fixture_state

    runner = JobRunner()
    runner.start(out, mode="incremental")

    assert _wait_until(lambda: runner.status().state == "awaiting_review"), (
        f"run did not reach the gate; state={runner.status().state}"
    )

    rec = runner.status()
    # The genuinely-new paper is surfaced as a candidate.
    assert NEW_PAPER_ID in _candidate_ids(rec), (
        f"new paper not in candidates: {_candidate_ids(rec)}"
    )
    # Candidate dicts carry the display fields.
    new_cand = next(c for c in rec.paper_candidates if c["id"] == NEW_PAPER_ID)
    assert new_cand["title"] == "Fake New Paper for Test Harness"
    assert new_cand["processing_bucket"] in ("new", "reprocess")

    # Worker is parked at the gate — NOT terminal — and still counts as active.
    assert rec.state == "awaiting_review"
    assert runner.active is True

    # A second start() while awaiting_review must raise.
    with pytest.raises(RunActiveError):
        runner.start(out, mode="incremental")

    # Clean up: unblock the worker so the daemon thread can finish.
    runner.submit_gate(rec.run_id, process_ids=[], discard_ids=[], edits={})
    assert _wait_until(lambda: runner.status().state == "done")


# ---------------------------------------------------------------------------
# 2. Gate: process the selected new paper → it is merged + processed.
# ---------------------------------------------------------------------------

def test_gate_process_selected(monkeypatch, fixture_state):
    install_pipeline_fakes(monkeypatch, orchestrator)
    out = fixture_state

    runner = JobRunner()
    runner.start(out, mode="incremental")
    assert _wait_until(lambda: runner.status().state == "awaiting_review")

    rec = runner.status()
    runner.submit_gate(
        rec.run_id, process_ids=[NEW_PAPER_ID], discard_ids=[], edits={}
    )

    assert _wait_until(lambda: runner.status().state == "done"), (
        f"run did not finish; state={runner.status().state}"
    )

    bucket_name, paper = _find_paper(out, "9999.99999")
    assert paper is not None, "new paper was not merged into the output"
    # Processed via the fakes → USES_NNSIGHT / HIGH → verified.
    assert bucket_name == "verified", f"expected verified, got {bucket_name}"
    assert paper["has_summary"] is True


# ---------------------------------------------------------------------------
# 3. Gate: discard the new paper → discarded bucket + manual_override.
# ---------------------------------------------------------------------------

def test_gate_discard(monkeypatch, fixture_state):
    install_pipeline_fakes(monkeypatch, orchestrator)
    out = fixture_state

    runner = JobRunner()
    runner.start(out, mode="incremental")
    assert _wait_until(lambda: runner.status().state == "awaiting_review")

    rec = runner.status()
    runner.submit_gate(
        rec.run_id, process_ids=[], discard_ids=[NEW_PAPER_ID], edits={}
    )

    assert _wait_until(lambda: runner.status().state == "done"), (
        f"run did not finish; state={runner.status().state}"
    )

    bucket_name, paper = _find_paper(out, "9999.99999")
    assert paper is not None, "discarded paper was not merged into the output"
    assert bucket_name == "discarded", f"expected discarded, got {bucket_name}"
    assert paper["manual_override"] is True
    assert paper["reason"] == "manual_discard"


# ---------------------------------------------------------------------------
# 4. Gate: edit a field then process → edited value survives to the output.
# ---------------------------------------------------------------------------

def test_gate_edit_then_process(monkeypatch, fixture_state):
    install_pipeline_fakes(monkeypatch, orchestrator)
    out = fixture_state

    runner = JobRunner()
    runner.start(out, mode="incremental")
    assert _wait_until(lambda: runner.status().state == "awaiting_review")

    rec = runner.status()
    runner.submit_gate(
        rec.run_id,
        process_ids=[NEW_PAPER_ID],
        discard_ids=[],
        edits={NEW_PAPER_ID: {"venue": "ICML 2025"}},
    )

    assert _wait_until(lambda: runner.status().state == "done"), (
        f"run did not finish; state={runner.status().state}"
    )

    _bucket_name, paper = _find_paper(out, "9999.99999")
    assert paper is not None, "edited paper was not merged into the output"
    assert paper["venue"] == "ICML 2025", f"venue not applied; got {paper['venue']!r}"


# ---------------------------------------------------------------------------
# 5. Gate: an unselected candidate (neither process nor discard) is dropped.
# ---------------------------------------------------------------------------

def test_gate_unselected_dropped(monkeypatch, fixture_state):
    install_pipeline_fakes(monkeypatch, orchestrator)
    out = fixture_state

    runner = JobRunner()
    runner.start(out, mode="incremental")
    assert _wait_until(lambda: runner.status().state == "awaiting_review")

    rec = runner.status()
    cand_ids = _candidate_ids(rec)
    # The fakes route BOTH papers to NEW/REPROCESS, so there are two candidates.
    assert NEW_PAPER_ID in cand_ids and EXISTING_PAPER_ID in cand_ids, (
        f"expected both papers as candidates, got {cand_ids}"
    )

    # Snapshot the EXISTING paper's on-disk state before the run.
    _before_bucket, before = _find_paper(out, EXISTING_ARXIV_ID)
    assert before is not None

    # Process only the new paper; leave the existing-match candidate unselected.
    runner.submit_gate(
        rec.run_id, process_ids=[NEW_PAPER_ID], discard_ids=[], edits={}
    )

    assert _wait_until(lambda: runner.status().state == "done"), (
        f"run did not finish; state={runner.status().state}"
    )

    # New paper was processed and merged.
    _bn, new_paper = _find_paper(out, "9999.99999")
    assert new_paper is not None

    # The dropped (unselected) candidate must NOT have been reprocessed/merged:
    # it should still exist (it was already in the DB) but unchanged by this run
    # — i.e. it stays in its original bucket and is not freshly reclassified.
    after_bucket, after = _find_paper(out, EXISTING_ARXIV_ID)
    assert after is not None, "existing paper vanished — drop should not delete it"
    # The dropped candidate is not processed: its content_hash is unchanged
    # (it was not re-merged from the discovered candidate).
    assert after["content_hash"] == before["content_hash"], (
        "dropped candidate appears to have been reprocessed/merged"
    )


# ---------------------------------------------------------------------------
# 6. Cancel during the gate → "cancelled", nothing written (byte-unchanged).
# ---------------------------------------------------------------------------

def test_cancel_during_gate(monkeypatch, fixture_state):
    install_pipeline_fakes(monkeypatch, orchestrator)
    out = fixture_state

    papers_file = out / "research-papers-full.json"
    repos_file = out / "github-repos-full.json"
    papers_before = papers_file.read_bytes()
    repos_before = repos_file.read_bytes()

    runner = JobRunner()
    runner.start(out, mode="incremental")
    assert _wait_until(lambda: runner.status().state == "awaiting_review")

    rec = runner.status()
    runner.cancel(rec.run_id)

    assert _wait_until(lambda: runner.status().state == "cancelled"), (
        f"run did not reach cancelled; state={runner.status().state}"
    )

    cancelled = runner.status()
    assert cancelled.state == "cancelled"
    assert cancelled.error is None
    assert runner.active is False

    # Nothing was written to the on-disk DB (cancel = abandon, no finalize).
    assert papers_file.read_bytes() == papers_before, (
        "research-papers-full.json was modified by a cancelled gate run"
    )
    assert repos_file.read_bytes() == repos_before, (
        "github-repos-full.json was modified by a cancelled gate run"
    )


# ---------------------------------------------------------------------------
# 7. Fresh mode never enters the gate; runs end-to-end to "done".
# ---------------------------------------------------------------------------

def test_fresh_mode_no_gate(monkeypatch, fixture_state):
    install_pipeline_fakes(monkeypatch, orchestrator)
    out = fixture_state

    seen_states: list[str] = []

    runner = JobRunner()
    runner.start(out, mode="fresh")

    # Sample states while it runs; fresh mode must never pause at the gate.
    def _poll():
        st = runner.status().state
        seen_states.append(st)
        return st == "done"

    assert _wait_until(_poll, timeout=4.0), (
        f"fresh run did not finish; states seen={seen_states}"
    )
    assert "awaiting_review" not in seen_states, (
        f"fresh mode entered the gate: {seen_states}"
    )
    assert runner.status().state == "done"


# ---------------------------------------------------------------------------
# 8. submit_gate on a run that is not awaiting_review raises GateError.
# ---------------------------------------------------------------------------

def test_submit_gate_wrong_state_raises(monkeypatch, fixture_state):
    install_pipeline_fakes(monkeypatch, orchestrator)
    out = fixture_state

    runner = JobRunner()
    run_id = runner.start(out, mode="incremental")
    assert _wait_until(lambda: runner.status().state == "awaiting_review")

    # Finish the run normally.
    runner.submit_gate(run_id, process_ids=[], discard_ids=[], edits={})
    assert _wait_until(lambda: runner.status().state == "done")

    # A second submit_gate on the now-done run must raise.
    with pytest.raises((GateError, ValueError)):
        runner.submit_gate(run_id, process_ids=[], discard_ids=[], edits={})


# ---------------------------------------------------------------------------
# 9. submit_gate with an unknown edit field → GateError before the run advances.
# ---------------------------------------------------------------------------

def test_gate_edit_unknown_field_raises(monkeypatch, fixture_state):
    """A typo'd field name in edits raises GateError synchronously and the run
    stays in 'awaiting_review' (not consumed).  A subsequent valid submit_gate
    must still work and drive the run to 'done'.
    """
    install_pipeline_fakes(monkeypatch, orchestrator)
    out = fixture_state

    runner = JobRunner()
    runner.start(out, mode="incremental")
    assert _wait_until(lambda: runner.status().state == "awaiting_review")

    rec = runner.status()

    # Bad edit: 'bogus_field' is not in the editable schema.
    with pytest.raises(GateError, match="bogus_field"):
        runner.submit_gate(
            rec.run_id,
            process_ids=[NEW_PAPER_ID],
            discard_ids=[],
            edits={NEW_PAPER_ID: {"bogus_field": "x"}},
        )

    # The run must still be awaiting_review — not consumed.
    assert runner.status().state == "awaiting_review", (
        f"run was consumed by a bad-edit submit_gate; state={runner.status().state}"
    )

    # A valid submit_gate must now still work.
    runner.submit_gate(rec.run_id, process_ids=[NEW_PAPER_ID], discard_ids=[], edits={})
    assert _wait_until(lambda: runner.status().state == "done"), (
        f"run did not finish after valid submit_gate; state={runner.status().state}"
    )


# ---------------------------------------------------------------------------
# 10. submit_gate with an unparseable edit value → GateError, run not consumed.
# ---------------------------------------------------------------------------

def test_gate_edit_parse_error_raises(monkeypatch, fixture_state):
    """A value that cannot be parsed for its field type raises GateError
    synchronously and the run stays in 'awaiting_review' (not consumed).
    """
    install_pipeline_fakes(monkeypatch, orchestrator)
    out = fixture_state

    runner = JobRunner()
    runner.start(out, mode="incremental")
    assert _wait_until(lambda: runner.status().state == "awaiting_review")

    rec = runner.status()

    # 'year' expects an integer; 'not-a-number' must fail to parse.
    with pytest.raises(GateError, match="year"):
        runner.submit_gate(
            rec.run_id,
            process_ids=[NEW_PAPER_ID],
            discard_ids=[],
            edits={NEW_PAPER_ID: {"year": "not-a-number"}},
        )

    # The run must still be awaiting_review — not consumed.
    assert runner.status().state == "awaiting_review", (
        f"run was consumed by a parse-error submit_gate; state={runner.status().state}"
    )

    # Clean up: finish the run normally.
    runner.submit_gate(rec.run_id, process_ids=[], discard_ids=[], edits={})
    assert _wait_until(lambda: runner.status().state == "done"), (
        f"run did not finish after cleanup submit_gate; state={runner.status().state}"
    )
