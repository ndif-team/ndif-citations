"""Tests for the manual_override → FILL_GAPS routing path.

When a paper has manual_override=True but some has_* flags are False,
the router should send it to FILL_GAPS (not PROTECTED) so the pipeline
can backfill empty fields without overwriting curated values.

The companion guard in process.py ensures FILL_GAPS writes only land on
fields that are actually empty when manual_override is set.
"""
from __future__ import annotations

from ndif_citations.router import ProcessingBucket, route_papers
from tests.conftest import make_paper


class TestProtectedFillGaps:
    def test_protected_with_no_empty_fields_stays_protected(self):
        existing = make_paper(
            arxiv_id="2407.14561",
            manual_override=True,
            has_summary=True,
            has_classification=True,
            has_thumbnail=True,
            has_affiliations=True,
        )
        discovered = make_paper(arxiv_id="2407.14561")
        decisions = route_papers([discovered], [existing])
        assert decisions[0].bucket == ProcessingBucket.PROTECTED
        assert all(v is False for v in decisions[0].processing_needed.values())

    def test_protected_with_empty_description_routes_to_fill_gaps(self):
        existing = make_paper(
            arxiv_id="2407.14561",
            manual_override=True,
            has_summary=False,
            has_classification=True,
            has_thumbnail=True,
            has_affiliations=True,
        )
        discovered = make_paper(arxiv_id="2407.14561")
        decisions = route_papers([discovered], [existing])
        assert decisions[0].bucket == ProcessingBucket.FILL_GAPS
        assert decisions[0].processing_needed["summary"] is True
        assert decisions[0].processing_needed["classify"] is False
        assert decisions[0].processing_needed["thumbnail"] is False
        assert decisions[0].processing_needed["affiliations"] is False

    def test_protected_with_multiple_empty_fields(self):
        existing = make_paper(
            arxiv_id="2407.14561",
            manual_override=True,
            has_summary=False,
            has_classification=False,
            has_thumbnail=False,
            has_affiliations=False,
        )
        discovered = make_paper(arxiv_id="2407.14561")
        decisions = route_papers([discovered], [existing])
        assert decisions[0].bucket == ProcessingBucket.FILL_GAPS
        assert all(
            decisions[0].processing_needed[f] is True
            for f in ("summary", "classify", "thumbnail", "affiliations")
        )
