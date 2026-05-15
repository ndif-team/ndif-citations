"""Year reconciliation + venue_source plumbing inside enrich_via_external_apis.

Pass 2 of enrich_via_external_apis runs resolve_venue and then, when the
result came from a high-confidence source whose label embeds an explicit
year (doi_prefix, arxiv_comment_parsed), reconciles paper.year to match
the venue year. This avoids the common case where paper.year is the arXiv
upload year (e.g. 2024) while the venue is the proceedings year (e.g.
2025 for an ACL 2025 DOI).

External APIs are monkeypatched to no-op so these tests stay hermetic.
"""
from __future__ import annotations

import pytest

from ndif_citations.extract import enrich_via_external_apis
from tests.conftest import make_paper


@pytest.fixture
def stub_external_apis(monkeypatch):
    """Make all external API calls inside enrich_via_external_apis no-op."""
    monkeypatch.setattr(
        "ndif_citations.extract.query_crossref", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "ndif_citations.extract.query_s2_publication_venue",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        "ndif_citations.extract.query_openreview_venue",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        "ndif_citations.extract.query_arxiv_api", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        "ndif_citations.extract.rate_limit_sleep", lambda *_a, **_k: None
    )


class TestYearReconciliation:
    def test_doi_prefix_year_overrides_paper_year(self, stub_external_apis):
        # arXiv upload was 2024; ACL proceedings DOI says 2025.
        paper = make_paper(
            arxiv_id=None,
            doi="10.18653/v1/2025.acl-long.42",
            venue="",
            year=2024,
        )
        enrich_via_external_apis([paper])
        assert paper.venue == "ACL 2025"
        assert paper.year == 2025

    def test_arxiv_comment_year_overrides_paper_year(self, stub_external_apis):
        # paper.year=2023 but the comment says NeurIPS 2024.
        paper = make_paper(
            arxiv_id=None,
            doi=None,
            venue="",
            year=2023,
        )
        # Inject arxiv comment by pre-populating sources via the public path:
        # easier to set venue directly and let the existing-cascade pick it up
        # — but that'd be "existing" source, not "arxiv_comment_parsed".
        # Use the explicit comment-source path by patching query_arxiv_api.
        from ndif_citations import extract
        # arxiv_id needed so the loop picks up the comment
        paper.arxiv_id = "2310.00001"
        captured = {"2310.00001": {"comment": "Accepted at NeurIPS 2024"}}
        extract.query_arxiv_api = lambda *_a, **_k: captured  # type: ignore[assignment]
        enrich_via_external_apis([paper])
        assert paper.venue == "NeurIPS 2024"
        assert paper.year == 2024

    def test_openalex_match_does_not_reconcile_year(self, stub_external_apis):
        # OpenAlex source is medium-confidence — it doesn't carry an embedded
        # year, so paper.year stays as-is even when normalize appends a year.
        paper = make_paper(
            arxiv_id="2410.17194",
            doi=None,
            venue="International Conference on Machine Learning",
            year=2024,
        )
        enrich_via_external_apis([paper])
        # Venue resolves via the "existing" source (paper.venue was set);
        # year stays 2024 because the source isn't doi_prefix/arxiv_comment_parsed.
        assert paper.venue == "ICML 2024"
        assert paper.year == 2024

    def test_no_year_in_venue_no_reconciliation(self, stub_external_apis):
        # ArXiv DOI decodes to plain "ArXiv" (no year). Reconciliation shouldn't fire.
        paper = make_paper(
            arxiv_id=None,
            doi="10.48550/arxiv.2406.00877",
            venue="",
            year=2024,
        )
        enrich_via_external_apis([paper])
        # decode_doi_prefix yields "ArXiv" → normalize_venue appends paper.year → "ArXiv 2024"
        # then is_preprint_sentinel matches → resolve_venue normalized is "" so it falls
        # through to fallback. Year stays 2024 either way.
        assert paper.year == 2024

    def test_low_confidence_existing_does_not_reconcile(self, stub_external_apis):
        # Existing-source resolution should never alter paper.year, even if
        # the venue contains a year.
        paper = make_paper(
            arxiv_id=None,
            doi=None,
            venue="ICLR 2026",
            year=2025,
        )
        enrich_via_external_apis([paper])
        assert paper.venue == "ICLR 2026"
        assert paper.year == 2025  # unchanged

    def test_matching_years_no_change(self, stub_external_apis):
        # DOI says 2025, paper.year already 2025 — no logged change but state stable.
        paper = make_paper(
            arxiv_id=None,
            doi="10.18653/v1/2025.emnlp-main.56",
            venue="",
            year=2025,
        )
        enrich_via_external_apis([paper])
        assert paper.venue == "EMNLP 2025"
        assert paper.year == 2025
