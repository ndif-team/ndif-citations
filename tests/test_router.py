"""Tests for route_papers, route_repos and helpers (router.py)."""
import pytest

from ndif_citations.router import (
    ProcessingBucket,
    _detect_venue_type,
    _is_venue_upgrade,
    get_bucket_summary,
    route_papers,
    route_repos,
)
from tests.conftest import make_paper, make_repo


# ---------------------------------------------------------------------------
# _detect_venue_type
# ---------------------------------------------------------------------------

class TestDetectVenueType:
    def test_arxiv_is_preprint(self):
        assert _detect_venue_type("arXiv 2025") == "preprint"

    def test_neurips_is_conference(self):
        assert _detect_venue_type("NeurIPS 2024") == "conference"

    def test_workshop_is_conference(self):
        assert _detect_venue_type("Workshop on X") == "conference"

    def test_iclr_is_conference(self):
        assert _detect_venue_type("ICLR 2025") == "conference"

    def test_ieee_transactions_is_journal(self):
        assert _detect_venue_type("IEEE Transactions on Y") == "journal"

    def test_nature_is_journal(self):
        assert _detect_venue_type("Nature") == "journal"

    def test_unknown_returns_unknown(self):
        assert _detect_venue_type("Some Random Title") == "unknown"

    def test_empty_returns_unknown(self):
        assert _detect_venue_type("") == "unknown"

    def test_case_insensitive(self):
        assert _detect_venue_type("ARXIV 2024") == "preprint"


# ---------------------------------------------------------------------------
# _is_venue_upgrade
# ---------------------------------------------------------------------------

class TestIsVenueUpgrade:
    def test_preprint_to_conference_is_upgrade(self):
        assert _is_venue_upgrade("arXiv 2024", "NeurIPS 2024") is True

    def test_preprint_to_journal_is_upgrade(self):
        assert _is_venue_upgrade("arXiv 2024", "Nature 2025") is True

    def test_conference_to_conference_not_upgrade(self):
        assert _is_venue_upgrade("ICLR 2024", "NeurIPS 2024") is False

    def test_journal_to_preprint_not_upgrade(self):
        assert _is_venue_upgrade("Nature", "arXiv 2025") is False

    def test_unknown_to_conference_not_upgrade(self):
        assert _is_venue_upgrade("Some Title", "NeurIPS 2024") is False

    def test_preprint_to_workshop_is_upgrade(self):
        # Workshop is considered "conference" type
        assert _is_venue_upgrade("arXiv 2024", "NeurIPS 2024 Workshop") is True


# ---------------------------------------------------------------------------
# route_papers
# ---------------------------------------------------------------------------

class TestRoutePapers:
    def test_new_paper_bucket(self):
        discovered = [make_paper(arxiv_id="2407.00001")]
        existing = []
        decisions = route_papers(discovered, existing)
        assert len(decisions) == 1
        assert decisions[0].bucket == ProcessingBucket.NEW
        assert all(decisions[0].processing_needed.values())

    def test_protected_paper_ignores_hash_change(self):
        # manual_override + all has_* flags set → PROTECTED even when hash changes.
        # (manual_override with missing flags now routes to FILL_GAPS — see
        # tests/test_router_protected_fill_gaps.py)
        existing_p = make_paper(
            arxiv_id="2407.00001",
            manual_override=True,
            has_summary=True,
            has_classification=True,
            has_thumbnail=True,
            has_affiliations=True,
        )
        discovered_p = make_paper(arxiv_id="2407.00001", abstract="different abstract")
        discovered_p.content_hash = "aabbccddeeff0011"
        decisions = route_papers([discovered_p], [existing_p])
        assert decisions[0].bucket == ProcessingBucket.PROTECTED
        assert not any(decisions[0].processing_needed.values())

    def test_skip_when_complete_and_unchanged(self):
        p = make_paper(
            arxiv_id="2407.00001",
            has_summary=True,
            has_classification=True,
            has_thumbnail=True,
            has_affiliations=True,
        )
        existing = make_paper(
            arxiv_id="2407.00001",
            has_summary=True,
            has_classification=True,
            has_thumbnail=True,
            has_affiliations=True,
        )
        # Give them the same content_hash
        p.content_hash = existing.content_hash
        decisions = route_papers([p], [existing])
        assert decisions[0].bucket == ProcessingBucket.SKIP
        assert not any(decisions[0].processing_needed.values())

    def test_fill_gaps_when_missing_summary(self):
        existing_p = make_paper(
            arxiv_id="2407.00001",
            has_summary=False,
            has_classification=True,
            has_thumbnail=True,
            has_affiliations=True,
        )
        discovered_p = make_paper(arxiv_id="2407.00001")
        discovered_p.content_hash = existing_p.content_hash
        decisions = route_papers([discovered_p], [existing_p])
        assert decisions[0].bucket == ProcessingBucket.FILL_GAPS
        assert decisions[0].processing_needed["summary"] is True
        assert decisions[0].processing_needed["classify"] is False
        assert decisions[0].processing_needed["thumbnail"] is False

    def test_reprocess_on_content_hash_change(self):
        existing_p = make_paper(arxiv_id="2407.00001")
        discovered_p = make_paper(arxiv_id="2407.00001", abstract="changed abstract")
        assert existing_p.content_hash != discovered_p.content_hash
        decisions = route_papers([discovered_p], [existing_p])
        assert decisions[0].bucket == ProcessingBucket.REPROCESS
        assert all(decisions[0].processing_needed.values())

    def test_reprocess_on_venue_upgrade(self):
        existing_p = make_paper(arxiv_id="2407.00001", venue="arXiv 2024")
        discovered_p = make_paper(arxiv_id="2407.00001", venue="ICLR 2025")
        # Same hash (same title + abstract)
        discovered_p.content_hash = existing_p.content_hash
        decisions = route_papers([discovered_p], [existing_p])
        assert decisions[0].bucket == ProcessingBucket.REPROCESS

    def test_match_priority_arxiv_over_doi(self):
        existing_by_arxiv = make_paper(arxiv_id="2407.00001", doi="10.1/A", title="Paper A")
        existing_by_doi = make_paper(arxiv_id=None, doi="10.1/B", title="Paper B")
        discovered = make_paper(arxiv_id="2407.00001", doi="10.1/B")
        decisions = route_papers([discovered], [existing_by_arxiv, existing_by_doi])
        # Should match by arxiv_id (existing_by_arxiv), not by doi
        assert decisions[0].existing_paper is existing_by_arxiv

    def test_doi_match_when_no_arxiv_id(self):
        existing_p = make_paper(arxiv_id=None, doi="10.1234/test")
        discovered_p = make_paper(arxiv_id=None, doi="10.1234/test")
        existing_p.content_hash = discovered_p.content_hash
        existing_p.has_summary = True
        existing_p.has_classification = True
        existing_p.has_thumbnail = True
        existing_p.has_affiliations = True
        decisions = route_papers([discovered_p], [existing_p])
        assert decisions[0].bucket == ProcessingBucket.SKIP

    def test_processing_bucket_set_on_paper(self):
        discovered = [make_paper(arxiv_id="2407.00001")]
        route_papers(discovered, [])
        assert discovered[0].processing_bucket == "new"


# ---------------------------------------------------------------------------
# route_repos
# ---------------------------------------------------------------------------

class TestRouteRepos:
    def test_new_repo(self):
        discovered = [make_repo(owner="u", repo="new-repo")]
        decisions = route_repos(discovered, [])
        assert decisions[0].bucket == ProcessingBucket.NEW

    def test_protected_repo(self):
        existing = make_repo(owner="u", repo="r", manual_override=True)
        discovered = make_repo(owner="u", repo="r")
        decisions = route_repos([discovered], [existing])
        assert decisions[0].bucket == ProcessingBucket.PROTECTED

    def test_reprocess_on_hash_change(self):
        existing = make_repo(owner="u", repo="r", description="old")
        discovered = make_repo(owner="u", repo="r", description="new")
        assert existing.content_hash != discovered.content_hash
        decisions = route_repos([discovered], [existing])
        assert decisions[0].bucket == ProcessingBucket.REPROCESS

    def test_fill_gaps_when_no_classification(self):
        existing = make_repo(owner="u", repo="r", has_classification=False)
        discovered = make_repo(owner="u", repo="r")
        discovered.content_hash = existing.content_hash
        decisions = route_repos([discovered], [existing])
        assert decisions[0].bucket == ProcessingBucket.FILL_GAPS

    def test_skip_when_complete(self):
        existing = make_repo(owner="u", repo="r", has_classification=True)
        discovered = make_repo(owner="u", repo="r")
        discovered.content_hash = existing.content_hash
        decisions = route_repos([discovered], [existing])
        assert decisions[0].bucket == ProcessingBucket.SKIP

    def test_processing_bucket_set_on_repo(self):
        discovered = [make_repo()]
        route_repos(discovered, [])
        assert discovered[0].processing_bucket == "new"


# ---------------------------------------------------------------------------
# get_bucket_summary
# ---------------------------------------------------------------------------

class TestGetBucketSummary:
    def test_counts_correctly(self):
        papers = [
            make_paper(arxiv_id="2407.00001"),
            make_paper(arxiv_id="2407.00002"),
            make_paper(arxiv_id="2407.00003"),
        ]
        decisions = route_papers(papers, [])
        summary = get_bucket_summary(decisions)
        assert summary.get("new") == 3
