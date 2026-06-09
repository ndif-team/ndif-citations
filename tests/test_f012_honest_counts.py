"""F-012 — the run UI over-represented LLM work.

Approving 1 gate candidate streamed ``Processing 5/128 … 113/128`` because the
process loop iterated EVERY routed decision (incl. SKIP/PROTECTED no-ops) and the
FE headlined off ``item_start`` (one per iteration), not off the real-work items.

This locks in the honest accounting:
  * route_stage records a per-bucket breakdown (on RouteResult + the route_summary
    event) and finalize copies it onto the run record's ``bucket_*`` fields.
  * process_papers emits ``item_skip`` for SKIP/PROTECTED no-ops and ``item_start``
    (carrying ``work_idx``/``work_total``) only for papers it actually works on.
  * the gate exposes the routing breakdown so the curator can see, before clicking,
    that approving also triggers automatic gap-fills on existing papers.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from ndif_citations import events, orchestrator
from ndif_citations.events import ProgressEvent
from ndif_citations.models import Bucket, Category, Confidence, PipelineRun
from ndif_citations.orchestrator import EnrichResult
from ndif_citations.process import process_papers
from ndif_citations.router import ProcessingBucket, RoutingDecision

from tests.conftest import make_paper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decision(bucket: ProcessingBucket, *, needs_summary: bool = True) -> RoutingDecision:
    """A routing decision in *bucket*. SKIP/PROTECTED carry all-false needs."""
    work = bucket not in (ProcessingBucket.SKIP, ProcessingBucket.PROTECTED)
    needs = {
        "summary": work and needs_summary,
        "classify": False,
        "thumbnail": False,
        "affiliations": False,
    }
    existing = make_paper(title="Existing") if bucket != ProcessingBucket.NEW else None
    return RoutingDecision(
        paper=make_paper(title=f"{bucket.value} paper"),
        bucket=bucket,
        existing_paper=existing,
        processing_needed=needs,
    )


def _install_process_fakes(monkeypatch):
    import ndif_citations.pdf_cache as pdf_cache_mod
    import ndif_citations.process as process_mod

    monkeypatch.setattr(process_mod, "generate_summary", lambda paper: f"summary:{paper.title}")
    monkeypatch.setattr(
        process_mod, "classify_category",
        lambda paper, output_dir, pdf_path=None: (Category.USES_NNSIGHT, 0.85, Confidence.HIGH),
    )
    monkeypatch.setattr(process_mod, "extract_thumbnail", lambda paper, output_dir, pdf_path=None: None)
    monkeypatch.setattr(pdf_cache_mod, "get_cached_pdf", lambda paper, output_dir: None)


@pytest.fixture()
def event_sink():
    captured: list[ProgressEvent] = []
    events.set_sink(captured.append)
    try:
        yield captured
    finally:
        events.clear_sink()


# ---------------------------------------------------------------------------
# 1. route_stage records a per-bucket breakdown
# ---------------------------------------------------------------------------

def test_route_stage_records_bucket_breakdown(monkeypatch, tmp_path, event_sink):
    decisions = [
        _decision(ProcessingBucket.NEW),
        _decision(ProcessingBucket.SKIP),
        _decision(ProcessingBucket.SKIP),
        _decision(ProcessingBucket.FILL_GAPS),
        _decision(ProcessingBucket.PROTECTED),
    ]
    monkeypatch.setattr(orchestrator, "route_papers", lambda discovered, existing: decisions)

    e = EnrichResult(papers=[make_paper()], repos=[], removal_counts={}, existing_repos=[])
    result = orchestrator.route_stage(
        tmp_path, e, skip_papers=False, skip_github=True, fresh=True
    )

    # The breakdown rides on RouteResult for finalize to lift onto the record.
    assert result.bucket_counts == {"new": 1, "skip": 2, "fill_gaps": 1, "protected": 1}

    summary = next(ev for ev in event_sink if ev.type == "route_summary" and ev.data.get("kind") == "papers")
    # Real work = NEW + REPROCESS + FILL_GAPS (= 2 here), NOT len(decisions) (= 5).
    assert summary.data["to_process"] == 2
    assert summary.data["skipped"] == 3
    assert summary.data["new"] == 1
    assert summary.data["fill_gaps"] == 1
    assert summary.data["skip"] == 2
    assert summary.data["protected"] == 1


# ---------------------------------------------------------------------------
# 2. process_papers separates real work from no-ops in its events
# ---------------------------------------------------------------------------

def test_process_papers_emits_item_skip_for_noops(monkeypatch, tmp_path, event_sink):
    _install_process_fakes(monkeypatch)
    decisions = [
        _decision(ProcessingBucket.NEW),
        _decision(ProcessingBucket.SKIP),
        _decision(ProcessingBucket.PROTECTED),
        _decision(ProcessingBucket.FILL_GAPS),
    ]

    process_papers(decisions, tmp_path)

    starts = [e for e in event_sink if e.type == "item_start"]
    skips = [e for e in event_sink if e.type == "item_skip"]

    # Only the 2 real-work papers (NEW, FILL_GAPS) get item_start; the 2 no-ops skip.
    assert len(starts) == 2
    assert len(skips) == 2
    # The headline denominator is the real work, not len(decisions).
    assert all(s.data["work_total"] == 2 for s in starts)
    assert [s.data["work_idx"] for s in starts] == [1, 2]
    # item_skip names the skipped paper so the log can show it distinctly.
    assert all("title" in s.data for s in skips)


# ---------------------------------------------------------------------------
# 3. finalize lifts the breakdown onto the run record (end-to-end)
# ---------------------------------------------------------------------------

def test_run_pipeline_populates_run_stats_buckets(monkeypatch, fixture_state: Path):
    from tests.helpers.fakes import install_pipeline_fakes

    install_pipeline_fakes(monkeypatch, orchestrator)
    result = orchestrator.run_pipeline(fixture_state, mode="incremental", skip_github=True)

    rs = result.run_stats
    bucket_total = (
        rs.bucket_new + rs.bucket_reprocess + rs.bucket_fill_gaps
        + rs.bucket_skip + rs.bucket_protected
    )
    # The fixtures route at least one brand-new paper, and the bucket_* fields are
    # populated (previously always 0 — they were never written by the gated path).
    assert rs.bucket_new >= 1
    assert bucket_total >= 1


# ---------------------------------------------------------------------------
# 4. the gate exposes the routing breakdown for the work-preview
# ---------------------------------------------------------------------------

def test_gate_exposes_route_breakdown(monkeypatch, fixture_state: Path):
    from ndif_citations.jobs import JobRunner
    from tests.helpers.fakes import install_pipeline_fakes

    install_pipeline_fakes(monkeypatch, orchestrator)
    runner = JobRunner()
    run_id = runner.start(fixture_state, mode="incremental", skip_github=True)

    deadline = time.time() + 3.0
    while time.time() < deadline and runner.status().state != "awaiting_review":
        time.sleep(0.01)

    rec = runner.status()
    assert rec.state == "awaiting_review"
    # The breakdown lets the gate preview real work before the curator clicks.
    assert isinstance(rec.route_breakdown, dict)
    assert rec.route_breakdown  # non-empty
    # Fixtures yield 1 NEW + 1 REPROCESS gate candidates.
    assert rec.route_breakdown.get("new", 0) + rec.route_breakdown.get("reprocess", 0) >= 2

    runner.submit_gate(run_id, process_ids=[], discard_ids=[], edits={})
