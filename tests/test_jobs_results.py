"""Tests for JobRunner._build_result_papers (Task C1).

Verifies that after a pipeline run, result_papers contains only papers that
are new (merge_key absent before run) or changed (category/bucket pair differs),
excluding silent gap-fills (same category/bucket).
"""
from __future__ import annotations

from ndif_citations import orchestrator
from ndif_citations.jobs import JobRunner
from ndif_citations.models import Bucket, Category, Confidence, PipelineRun
from tests.conftest import make_paper


def _p(**kw):
    """Create a DiscoveredPaper with sensible defaults; kw overrides."""
    title = kw.pop("title", "T")
    arxiv_id = kw.pop("arxiv_id", None)
    return make_paper(title=title, arxiv_id=arxiv_id, **kw)


def test_build_result_papers_new_and_changed_only():
    """Only new and (category,bucket)-changed papers appear in result_papers."""
    # Pre-existing paper: UNCLASSIFIED / PENDING
    existing = _p(
        title="Existing",
        arxiv_id="1.1",
        category=Category.UNCLASSIFIED,
        bucket=Bucket.PENDING,
    )
    pre = {
        existing.merge_key(): (existing.category.value, existing.bucket.value)
    }

    # Same paper, now classified → category changed → should appear
    changed = _p(
        title="Existing",
        arxiv_id="1.1",
        category=Category.USES_NNSIGHT,
        bucket=Bucket.PENDING,
        category_confidence_band=Confidence.HIGH,
    )
    # Brand-new paper → should appear as is_new
    new = _p(
        title="Brand New",
        arxiv_id="2.2",
        category=Category.USES_NDIF,
        bucket=Bucket.PENDING,
        category_confidence_band=Confidence.HIGH,
    )
    # Same category/bucket as before → silent gap-fill, must NOT appear
    untouched = _p(
        title="Untouched",
        arxiv_id="3.3",
        category=Category.UNCLASSIFIED,
        bucket=Bucket.PENDING,
    )
    pre[untouched.merge_key()] = (
        untouched.category.value,
        untouched.bucket.value,
    )

    result = orchestrator.FinalizeResult(
        merged_papers=[changed, new, untouched],
        merged_repos=[],
        run_stats=PipelineRun(),
    )

    rows = JobRunner._build_result_papers(pre, result)
    ids = {r["id"] for r in rows}

    assert changed.merge_key() in ids, "changed paper must appear"
    assert new.merge_key() in ids, "new paper must appear"
    assert untouched.merge_key() not in ids, "untouched paper must NOT appear"

    new_row = next(r for r in rows if r["id"] == new.merge_key())
    assert new_row["is_new"] is True
    assert new_row["category"] == "uses_ndif"       # Category.USES_NDIF.value
    assert new_row["confidence_band"] == "high"     # Confidence.HIGH.value
    assert new_row["bucket"] == "pending"            # Bucket.PENDING.value

    changed_row = next(r for r in rows if r["id"] == changed.merge_key())
    assert changed_row["is_new"] is False
    assert changed_row["category"] == "uses_nnsight"


def test_build_result_papers_empty_pre_snapshot():
    """When pre-snapshot is empty, every merged paper is treated as new."""
    p1 = _p(title="First", arxiv_id="1.1", category=Category.USES_NDIF, bucket=Bucket.PENDING)
    p2 = _p(title="Second", arxiv_id="2.2", category=Category.REFERENCING, bucket=Bucket.VERIFIED)
    result = orchestrator.FinalizeResult(
        merged_papers=[p1, p2], merged_repos=[], run_stats=PipelineRun()
    )
    rows = JobRunner._build_result_papers({}, result)
    assert len(rows) == 2
    assert all(r["is_new"] is True for r in rows)


def test_build_result_papers_empty_merged():
    """Empty merged_papers → empty result_papers."""
    result = orchestrator.FinalizeResult(
        merged_papers=[], merged_repos=[], run_stats=PipelineRun()
    )
    rows = JobRunner._build_result_papers({"arxiv:x": ("uses_ndif", "pending")}, result)
    assert rows == []


def test_build_result_papers_bucket_change_counts_as_changed():
    """A bucket change (same category) must be surfaced as changed."""
    p = _p(title="T", arxiv_id="9.9", category=Category.USES_NDIF, bucket=Bucket.VERIFIED)
    pre = {p.merge_key(): ("uses_ndif", "pending")}  # bucket was pending, now verified
    result = orchestrator.FinalizeResult(
        merged_papers=[p], merged_repos=[], run_stats=PipelineRun()
    )
    rows = JobRunner._build_result_papers(pre, result)
    assert len(rows) == 1
    assert rows[0]["is_new"] is False
    assert rows[0]["bucket"] == "verified"
