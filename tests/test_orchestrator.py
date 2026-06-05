"""Tests for the event-emitting pipeline orchestrator (Task 1.6).

Two layers of coverage:

1. A light unit test that ``discover_stage`` emits ``stage_start`` / ``source_count``
   and returns a ``DiscoverResult`` when its discovery callables are monkeypatched
   to return small lists.
2. An end-to-end stage test that drives all five stages via ``run_pipeline`` using
   the Task 1.0 fake harness (``install_pipeline_fakes`` + ``fixture_state``) — no
   network, no LLM. This is the main correctness signal for the extract-not-rewrite
   refactor: it asserts the merged set includes the pre-existing fixture papers plus
   the one genuinely-new fake paper, and that the output files are written.

The strict legacy-vs-orchestrator parity test lives in Task 1.8, not here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ndif_citations import events, orchestrator
from ndif_citations.events import ProgressEvent
from ndif_citations.models import DiscoveredPaper, DiscoverySource
from ndif_citations.orchestrator import DiscoverResult, FinalizeResult
from tests.helpers.fakes import EXISTING_ARXIV_ID, install_pipeline_fakes


@pytest.fixture()
def event_sink():
    """Install an events sink that records every emitted ProgressEvent.

    Yields the list of captured events; the sink is cleared on teardown.
    """
    captured: list[ProgressEvent] = []
    events.set_sink(captured.append)
    try:
        yield captured
    finally:
        events.clear_sink()


# ---------------------------------------------------------------------------
# 1. Unit test: discover_stage emits and returns the right shape
# ---------------------------------------------------------------------------

class TestDiscoverStage:
    def test_emits_stage_start_and_source_count(self, monkeypatch, tmp_path, event_sink):
        out = tmp_path / "output"
        out.mkdir()

        p1 = DiscoveredPaper(title="P1", arxiv_id="1234.00001", source=DiscoverySource.S2_CITATION)
        p2 = DiscoveredPaper(title="P2", arxiv_id="1234.00002", source=DiscoverySource.OPENALEX_FULLTEXT)

        monkeypatch.setattr(orchestrator, "discover_s2_citations", lambda raw_dir: [p1])
        monkeypatch.setattr(orchestrator, "discover_openalex", lambda raw_dir: [p2])
        monkeypatch.setattr(orchestrator, "discover_scholar", lambda raw_dir, force_refresh=False: [])
        monkeypatch.setattr(orchestrator, "discover_github_dependents", lambda raw_dir: [])

        result = orchestrator.discover_stage(
            out, skip_papers=False, skip_github=True, fresh=False
        )

        assert isinstance(result, DiscoverResult)
        # Both unique papers survive (distinct arxiv ids, no year filter — year 0)
        assert len(result.papers) == 2
        assert result.repos == []

        types = [e.type for e in event_sink]
        assert "stage_start" in types
        assert "source_count" in types

        # source_count carries the per-source counts the CLI used to print
        source_ev = next(e for e in event_sink if e.type == "source_count")
        assert source_ev.data["s2"] == 1
        assert source_ev.data["openalex"] == 1
        assert source_ev.data["scholar"] == 0

        # run_stats mirrors the per-source counts
        assert result.run_stats.s2_citations_found == 1
        assert result.run_stats.openalex_found == 1


# ---------------------------------------------------------------------------
# 2. End-to-end: run_pipeline drives all five stages on the fake harness
# ---------------------------------------------------------------------------

class TestRunPipelineEndToEnd:
    def test_incremental_run_merges_existing_plus_new(
        self, monkeypatch, fixture_state: Path, event_sink
    ):
        install_pipeline_fakes(monkeypatch, orchestrator)

        result = orchestrator.run_pipeline(
            fixture_state, mode="incremental", skip_github=True
        )

        # Returns a FinalizeResult
        assert isinstance(result, FinalizeResult)

        # Output files written
        website_json = fixture_state / "research-papers.json"
        full_json = fixture_state / "research-papers-full.json"
        assert website_json.exists()
        assert full_json.exists()

        # The merged set includes the 3 pre-existing verified fixture papers plus
        # the 1 genuinely-new fake paper (arxiv 9999.99999). The fake paper sharing
        # EXISTING_ARXIV_ID merges into the existing record rather than adding a row.
        merged_arxiv_ids = {p.arxiv_id for p in result.merged_papers}
        assert EXISTING_ARXIV_ID in merged_arxiv_ids
        assert "9999.99999" in merged_arxiv_ids

        # run_stats reflects the merge: 1 brand-new paper, existing population > 0
        assert result.run_stats.new_papers == 1
        assert result.run_stats.existing_papers > 0

        # The full JSON keeps the 3-bucket structure and now contains the new paper
        full_data = json.loads(full_json.read_text())
        assert set(full_data.keys()) == {"pending", "verified", "discarded"}
        all_full_ids = {
            p.get("arxiv_id")
            for bucket in ("pending", "verified", "discarded")
            for p in full_data[bucket]
        }
        assert "9999.99999" in all_full_ids
        assert EXISTING_ARXIV_ID in all_full_ids

        # A report event was emitted at the end
        assert any(e.type == "report" for e in event_sink)
