"""Tests for per-item progress events and cancel hook in process_papers / process_repos.

Task 1.7: Per-item progress events + cancel hook.
"""
from __future__ import annotations

import pytest

from ndif_citations import events
from ndif_citations.events import RunCancelled
from ndif_citations.models import (
    Bucket,
    Category,
    DiscoveredPaper,
    DiscoveredRepo,
    DiscoverySource,
)
from ndif_citations.router import (
    ProcessingBucket,
    RepoRoutingDecision,
    RoutingDecision,
)
from ndif_citations.process import process_papers, process_repos

from tests.conftest import make_paper, make_repo


def _new_decision(paper: DiscoveredPaper) -> RoutingDecision:
    """Wrap a paper in a NEW routing decision with all processing enabled."""
    return RoutingDecision(
        paper=paper,
        bucket=ProcessingBucket.NEW,
        existing_paper=None,
        processing_needed={"summary": True, "classify": True, "thumbnail": True, "affiliations": True},
    )


def _new_repo_decision(repo: DiscoveredRepo) -> RepoRoutingDecision:
    """Wrap a repo in a NEW routing decision."""
    return RepoRoutingDecision(
        repo=repo,
        bucket=ProcessingBucket.NEW,
        existing_repo=None,
    )


# ---------------------------------------------------------------------------
# Helper: install the standard process-module fakes so no network calls occur
# ---------------------------------------------------------------------------

def _install_process_fakes(monkeypatch):
    """Patch the four expensive callables in process.py and pdf_cache."""
    import ndif_citations.process as process_mod
    import ndif_citations.pdf_cache as pdf_cache_mod
    from ndif_citations.models import Confidence

    monkeypatch.setattr(process_mod, "generate_summary", lambda paper: f"Fake summary: {paper.title}")
    monkeypatch.setattr(
        process_mod,
        "classify_category",
        lambda paper, output_dir, pdf_path=None: (Category.USES_NNSIGHT, 0.85, Confidence.HIGH),
    )
    monkeypatch.setattr(process_mod, "extract_thumbnail", lambda paper, output_dir, pdf_path=None: None)
    monkeypatch.setattr(pdf_cache_mod, "get_cached_pdf", lambda paper, output_dir: None)


# ---------------------------------------------------------------------------
# 1. item_start + item_step events are emitted during process_papers
# ---------------------------------------------------------------------------

class TestProcessPapersEmitsItemEvents:
    def test_item_start_and_summary_step_emitted(self, monkeypatch, tmp_path):
        _install_process_fakes(monkeypatch)

        paper = make_paper(
            title="Test Emit Paper",
            abstract="We use nnsight to probe language models.",
            category=Category.USES_NNSIGHT,
            bucket=Bucket.VERIFIED,
        )
        decisions = [_new_decision(paper)]

        collected: list[events.ProgressEvent] = []
        events.set_sink(collected.append)
        try:
            process_papers(decisions, tmp_path)
        finally:
            events.clear_sink()

        # At least one item_start was emitted
        item_starts = [e for e in collected if e.type == "item_start"]
        assert len(item_starts) >= 1, "Expected at least one item_start event"

        start = item_starts[0]
        assert start.stage == "process"
        assert start.data["title"] == "Test Emit Paper"
        assert start.data["idx"] == 0
        assert start.data["total"] == 1

        # At least one item_step with step=="summary" was emitted
        summary_steps = [
            e for e in collected
            if e.type == "item_step" and e.data.get("step") == "summary"
        ]
        assert len(summary_steps) >= 1, "Expected at least one item_step(step='summary') event"


# ---------------------------------------------------------------------------
# 2. cancel_check fires before item 2 → RunCancelled with completed==1
# ---------------------------------------------------------------------------

class TestProcessPapersCancelRaisesWithCompleted:
    def test_cancel_after_first_item(self, monkeypatch, tmp_path):
        _install_process_fakes(monkeypatch)

        papers = [
            make_paper(
                title=f"Paper {n}",
                arxiv_id=f"2401.0000{n}",
                abstract="Uses nnsight to study transformers.",
                category=Category.USES_NNSIGHT,
                bucket=Bucket.VERIFIED,
            )
            for n in range(3)
        ]
        decisions = [_new_decision(p) for p in papers]

        # cancel_check returns False on first call (item 0), True on second (item 1)
        call_count = {"n": 0}

        def cancel_check() -> bool:
            result = call_count["n"] > 0  # False for idx 0, True for idx 1+
            call_count["n"] += 1
            return result

        with pytest.raises(RunCancelled) as exc_info:
            process_papers(decisions, tmp_path, cancel_check=cancel_check)

        exc = exc_info.value
        assert exc.completed == 1, f"Expected 1 completed item, got {exc.completed}"
        assert len(exc.results) == 1, f"Expected 1 result in partial list, got {len(exc.results)}"


# ---------------------------------------------------------------------------
# 3. cancel_check=None → normal completion (sanity / no regression)
# ---------------------------------------------------------------------------

class TestProcessPapersNoCancelCheckIsUnchanged:
    def test_completes_all_items_without_cancel_check(self, monkeypatch, tmp_path):
        _install_process_fakes(monkeypatch)

        papers = [
            make_paper(
                title=f"Paper {n}",
                arxiv_id=f"2401.9999{n}",
                abstract="Uses nnsight to study transformers.",
                category=Category.USES_NNSIGHT,
                bucket=Bucket.VERIFIED,
            )
            for n in range(3)
        ]
        decisions = [_new_decision(p) for p in papers]

        result = process_papers(decisions, tmp_path, cancel_check=None)
        assert len(result) == 3, f"Expected 3 results, got {len(result)}"


# ---------------------------------------------------------------------------
# 4. process_repos emits item_start and item_step(step="classify")
# ---------------------------------------------------------------------------

class TestProcessReposEmitsItemEvents:
    def test_item_start_and_classify_step_emitted(self):
        repos = [
            make_repo(owner="test-owner", repo="test-repo", has_classification=True),
        ]
        decisions = [_new_repo_decision(r) for r in repos]

        collected: list[events.ProgressEvent] = []
        events.set_sink(collected.append)
        try:
            process_repos(decisions)
        finally:
            events.clear_sink()

        item_starts = [e for e in collected if e.type == "item_start"]
        assert len(item_starts) >= 1, "Expected at least one item_start event"

        start = item_starts[0]
        assert start.data["title"] == "test-owner/test-repo"
        assert start.data["idx"] == 0

        classify_steps = [
            e for e in collected
            if e.type == "item_step" and e.data.get("step") == "classify"
        ]
        assert len(classify_steps) >= 1, "Expected at least one item_step(step='classify')"


# ---------------------------------------------------------------------------
# 5. process_repos cancel_check fires → RunCancelled with completed==1
# ---------------------------------------------------------------------------

class TestProcessReposCancelRaisesWithCompleted:
    def test_cancel_after_first_repo(self):
        repos = [
            make_repo(owner=f"owner-{n}", repo=f"repo-{n}", has_classification=True)
            for n in range(3)
        ]
        decisions = [_new_repo_decision(r) for r in repos]

        call_count = {"n": 0}

        def cancel_check() -> bool:
            result = call_count["n"] > 0
            call_count["n"] += 1
            return result

        with pytest.raises(RunCancelled) as exc_info:
            process_repos(decisions, cancel_check=cancel_check)

        exc = exc_info.value
        assert exc.completed == 1
        assert len(exc.results) == 1
