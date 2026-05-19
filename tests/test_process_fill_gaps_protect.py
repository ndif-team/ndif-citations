"""Process.py guard: when manual_override=True and routing is FILL_GAPS,
only write to fields that are currently empty.

Companion to tests/test_router_protected_fill_gaps.py — the router routes
the paper, this guard enforces 'don't overwrite curated values'.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ndif_citations.models import Bucket, Category, Confidence
from ndif_citations.process import process_papers
from ndif_citations.router import ProcessingBucket, RoutingDecision
from tests.conftest import make_paper


class TestFillGapsProtectsManualFields:
    def test_description_is_preserved_when_present(self, tmp_path, monkeypatch):
        existing = make_paper(
            arxiv_id="2407.14561",
            description="Emma's curated summary that must NOT be overwritten.",
            manual_override=True,
            has_summary=True,
            has_classification=True,
            has_thumbnail=False,
            has_affiliations=True,
        )
        decision = RoutingDecision(
            paper=existing,
            bucket=ProcessingBucket.FILL_GAPS,
            existing_paper=existing,
            processing_needed={"summary": False, "classify": False,
                               "thumbnail": True, "affiliations": False},
        )
        monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf",
                            lambda p, o: None)
        monkeypatch.setattr("ndif_citations.process.generate_summary",
                            lambda p: "PIPELINE-GENERATED summary (must not appear)")
        result = process_papers([decision], tmp_path)
        assert len(result) == 1
        assert result[0].description == "Emma's curated summary that must NOT be overwritten."

    def test_guard_prevents_overwrite_when_flag_drift(self, tmp_path, monkeypatch):
        """Defense in depth: even if has_summary is False (router thinks
        summary needs filling), if the description is actually non-empty
        and manual_override is set, don't overwrite.
        """
        existing = make_paper(
            arxiv_id="2407.14561",
            description="Curator's text — has_summary flag drifted False but field is set.",
            manual_override=True,
            has_summary=False,  # drift: field non-empty but flag says missing
            has_classification=True,
            has_thumbnail=True,
            has_affiliations=True,
        )
        decision = RoutingDecision(
            paper=existing,
            bucket=ProcessingBucket.FILL_GAPS,
            existing_paper=existing,
            processing_needed={"summary": True,  # router asked to fill
                               "classify": False, "thumbnail": False,
                               "affiliations": False},
        )
        monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf",
                            lambda p, o: None)
        monkeypatch.setattr("ndif_citations.process.generate_summary",
                            lambda p: "PIPELINE-GENERATED (must not appear)")
        result = process_papers([decision], tmp_path)
        assert result[0].description == (
            "Curator's text — has_summary flag drifted False but field is set."
        )

    def test_summary_is_filled_when_empty_on_manual_override(self, tmp_path, monkeypatch):
        existing = make_paper(
            arxiv_id="2407.14561",
            description="",
            manual_override=True,
            has_summary=False,
            has_classification=True,
            has_thumbnail=True,
            has_affiliations=True,
        )
        decision = RoutingDecision(
            paper=existing,
            bucket=ProcessingBucket.FILL_GAPS,
            existing_paper=existing,
            processing_needed={"summary": True, "classify": False,
                               "thumbnail": False, "affiliations": False},
        )
        monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf",
                            lambda p, o: None)
        monkeypatch.setattr("ndif_citations.process.generate_summary",
                            lambda p: "NEW summary from pipeline")
        result = process_papers([decision], tmp_path)
        assert result[0].description == "NEW summary from pipeline"
