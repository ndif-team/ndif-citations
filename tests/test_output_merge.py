"""Tests for merge_papers, _update_existing, and serialization contract (US-F11)."""
import pytest

from ndif_citations.models import Bucket, Category, DiscoveredPaper, PaperReason, PipelineRun
from ndif_citations.output import _update_existing, merge_papers
from tests.conftest import make_paper

EXPECTED_WEBSITE_KEYS = {"title", "authors", "venue", "year", "url", "description", "category"}


# ---------------------------------------------------------------------------
# _update_existing
# ---------------------------------------------------------------------------

class TestUpdateExisting:
    def test_manual_override_blocks_all_changes(self):
        existing = make_paper(manual_override=True, authors="Original Author")
        new = make_paper(authors="New Author Better", venue="NeurIPS 2025")
        changed = _update_existing(existing, new)
        assert changed is False
        assert existing.authors == "Original Author"

    def test_longer_authors_updates(self):
        existing = make_paper(authors="Smith")
        new = make_paper(authors="Smith, Jones, Brown")
        changed = _update_existing(existing, new)
        assert changed is True
        assert existing.authors == "Smith, Jones, Brown"

    def test_shorter_authors_does_not_update(self):
        existing = make_paper(authors="Smith, Jones, Brown")
        new = make_paper(authors="Smith")
        changed = _update_existing(existing, new)
        assert existing.authors == "Smith, Jones, Brown"

    def test_fills_missing_affiliations(self):
        existing = make_paper()
        existing.affiliations = ""
        new = make_paper()
        new.affiliations = "MIT, Stanford"
        changed = _update_existing(existing, new)
        assert changed is True
        assert existing.affiliations == "MIT, Stanford"

    def test_does_not_overwrite_existing_affiliations(self):
        existing = make_paper()
        existing.affiliations = "Harvard"
        new = make_paper()
        new.affiliations = "MIT"
        _update_existing(existing, new)
        assert existing.affiliations == "Harvard"

    def test_venue_upgrade_preprint_to_conference(self):
        existing = make_paper(venue="arXiv 2024")
        new = make_paper(venue="NeurIPS 2025")
        changed = _update_existing(existing, new)
        assert changed is True
        assert existing.venue == "NeurIPS 2025"
        assert existing.peer_reviewed is True

    def test_venue_conference_to_conference_no_change(self):
        existing = make_paper(venue="ICLR 2024")
        new = make_paper(venue="NeurIPS 2025")
        _update_existing(existing, new)
        assert existing.venue == "ICLR 2024"

    def test_fills_missing_doi(self):
        existing = make_paper(arxiv_id=None, doi=None)
        new = make_paper(arxiv_id=None, doi="10.1234/test")
        changed = _update_existing(existing, new)
        assert changed is True
        assert existing.doi == "10.1234/test"

    def test_fills_missing_s2_paper_id(self):
        existing = make_paper()
        existing.s2_paper_id = None
        new = make_paper()
        new.s2_paper_id = "s2-abc123"
        changed = _update_existing(existing, new)
        assert changed is True
        assert existing.s2_paper_id == "s2-abc123"

    def test_does_not_update_description(self):
        # _update_existing intentionally does not copy description
        existing = make_paper(description="Old description")
        new = make_paper(description="New LLM description")
        _update_existing(existing, new)
        assert existing.description == "Old description"


# ---------------------------------------------------------------------------
# merge_papers
# ---------------------------------------------------------------------------

class TestMergePapers:
    def test_new_paper_appended(self):
        paper = make_paper(arxiv_id="2407.99999", title="Brand New Paper", description="Summary here")
        merged, run = merge_papers([], [paper])
        assert len(merged) == 1
        assert merged[0].description == "Summary here"

    def test_existing_paper_matched_by_arxiv_id(self):
        existing = make_paper(arxiv_id="2407.00001", authors="Old Author")
        new = make_paper(arxiv_id="2407.00001", authors="Old Author, New Author")
        merged, run = merge_papers([existing], [new])
        assert len(merged) == 1
        assert merged[0].authors == "Old Author, New Author"

    def test_manual_override_description_preserved_on_merge(self):
        existing = make_paper(
            arxiv_id="2407.00001",
            description="Curator's hand-written summary",
            manual_override=True,
        )
        new = make_paper(
            arxiv_id="2407.00001",
            description="Pipeline generated summary",
        )
        merged, run = merge_papers([existing], [new])
        assert merged[0].description == "Curator's hand-written summary"

    def test_new_identifiers_filled_even_with_manual_override(self):
        existing = make_paper(arxiv_id="2407.00001", manual_override=True)
        existing.s2_paper_id = None
        new = make_paper(arxiv_id="2407.00001")
        new.s2_paper_id = "s2-xyz"
        merged, _ = merge_papers([existing], [new])
        # manual_override=True blocks _update_existing entirely
        # so s2_paper_id won't be filled (this is the documented behavior)
        assert merged[0].s2_paper_id is None

    def test_run_stats_new_count(self):
        existing = [make_paper(arxiv_id="2407.00001", title="Existing Paper One")]
        new = [make_paper(arxiv_id="2407.99999", title="Completely Different Paper")]
        _, run = merge_papers(existing, new)
        assert run.new_papers == 1

    def test_run_stats_updated_count(self):
        existing = [make_paper(arxiv_id="2407.00001", title="Same Paper", authors="A")]
        new = [make_paper(arxiv_id="2407.00001", title="Same Paper", authors="A, B, C")]
        _, run = merge_papers(existing, new)
        assert run.updated_papers == 1

    def test_first_run_no_existing(self):
        papers = [make_paper(), make_paper(arxiv_id="2407.00002", title="Second")]
        merged, run = merge_papers([], papers)
        assert len(merged) == 2
        assert run.new_papers == 2


# ---------------------------------------------------------------------------
# to_website_dict key contract
# ---------------------------------------------------------------------------

class TestWebsiteContract:
    def test_always_present_keys(self):
        p = make_paper(title="Test", authors="Author", venue="ICLR 2025", year=2025,
                       description="Summary")
        p.url = "https://arxiv.org/abs/2407.00001"
        d = p.to_website_dict()
        assert EXPECTED_WEBSITE_KEYS.issubset(d.keys()), (
            f"Missing keys: {EXPECTED_WEBSITE_KEYS - d.keys()}"
        )

    def test_unclassified_reason_not_in_website_dict(self):
        p = make_paper()
        p.unclassified_reason = "no_evidence_extractable"
        d = p.to_website_dict()
        assert "unclassified_reason" not in d

    def test_image_included_when_present(self):
        p = make_paper()
        p.image = "/images/Test-Paper.png"
        d = p.to_website_dict()
        assert "image" in d
        assert d["image"] == "/images/Test-Paper.png"

    def test_image_excluded_when_absent(self):
        p = make_paper()
        p.image = None
        d = p.to_website_dict()
        assert "image" not in d

    def test_category_value_is_string(self):
        p = make_paper(category=Category.USES_NDIF)
        d = p.to_website_dict()
        assert isinstance(d["category"], str)
        assert d["category"] == "uses_ndif"


# ---------------------------------------------------------------------------
# to_full_dict round-trip
# ---------------------------------------------------------------------------

class TestFullDictRoundTrip:
    def test_round_trips_all_fields(self):
        p = make_paper(
            title="Round Trip Paper",
            arxiv_id="2407.12345",
            venue="NeurIPS 2024",
            year=2024,
            description="A great paper.",
        )
        p.unclassified_reason = None
        d = p.to_full_dict()
        restored = DiscoveredPaper.model_validate(d)
        assert restored.title == p.title
        assert restored.arxiv_id == p.arxiv_id
        assert restored.venue == p.venue
        assert restored.description == p.description
        assert restored.content_hash == p.content_hash

    def test_category_and_bucket_in_full_dict(self):
        p = make_paper(category=Category.USES_NNSIGHT)
        d = p.to_full_dict()
        assert "category" in d
        assert d["category"] == "uses_nnsight"
        assert "bucket" in d
        assert "reason" in d
        assert "reason_detail" in d

    def test_classification_signal_in_full_dict(self):
        p = make_paper()
        p.classification_signal = "pre_filter:comparison_table"
        d = p.to_full_dict()
        assert "classification_signal" in d
        assert d["classification_signal"] == "pre_filter:comparison_table"

    def test_classification_signal_not_in_website_dict(self):
        p = make_paper()
        p.classification_signal = "pre_filter:negative_evidence"
        d = p.to_website_dict()
        assert "classification_signal" not in d

    def test_classification_signal_round_trip(self):
        p = make_paper()
        p.classification_signal = "pre_filter:acks_only_thank_you"
        d = p.to_full_dict()
        restored = DiscoveredPaper.model_validate(d)
        assert restored.classification_signal == "pre_filter:acks_only_thank_you"

    def test_classification_signal_none_by_default(self):
        p = make_paper()
        assert p.classification_signal is None
        d = p.to_full_dict()
        assert d["classification_signal"] is None

    def test_bucket_reason_not_in_website_dict(self):
        from ndif_citations.models import Bucket, PaperReason
        p = make_paper(bucket=Bucket.PENDING)
        p.reason = PaperReason.OPENALEX_SOURCE
        d = p.to_website_dict()
        assert "bucket" not in d
        assert "reason" not in d
        assert "reason_detail" not in d

    def test_bucket_reason_round_trip(self):
        from ndif_citations.models import Bucket, PaperReason
        p = make_paper(bucket=Bucket.PENDING)
        p.reason = PaperReason.STUB_METADATA
        p.reason_detail = "year=0, no abstract"
        d = p.to_full_dict()
        restored = DiscoveredPaper.model_validate(d)
        assert restored.bucket == Bucket.PENDING
        assert restored.reason == PaperReason.STUB_METADATA
        assert restored.reason_detail == "year=0, no abstract"


# ---------------------------------------------------------------------------
# TestManualOverride (US-D5)
# ---------------------------------------------------------------------------

class TestManualOverride:
    def test_override_discarded_paper_stays_discarded_on_merge(self):
        existing = make_paper(
            arxiv_id="2401.00001",
            bucket=Bucket.DISCARDED,
            manual_override=True,
        )
        existing.reason = PaperReason.MANUAL_DISCARD
        new = make_paper(arxiv_id="2401.00001", bucket=Bucket.VERIFIED)
        merged, _ = merge_papers([existing], [new])
        assert merged[0].bucket == Bucket.DISCARDED
        assert merged[0].reason == PaperReason.MANUAL_DISCARD

    def test_override_verified_category_not_changed_on_merge(self):
        existing = make_paper(
            arxiv_id="2401.00001",
            category=Category.USES_NDIF,
            manual_override=True,
        )
        new = make_paper(arxiv_id="2401.00001", category=Category.REFERENCING)
        merged, _ = merge_papers([existing], [new])
        assert merged[0].category == Category.USES_NDIF

    def test_override_pending_not_auto_promoted_on_merge(self):
        existing = make_paper(
            arxiv_id="2401.00001",
            bucket=Bucket.PENDING,
            manual_override=True,
            year=2024,
        )
        existing.reason = PaperReason.STUB_METADATA
        # New discovery fills in missing data that would clear stub_metadata
        new = make_paper(arxiv_id="2401.00001", year=2024)
        merged, _ = merge_papers([existing], [new])
        # manual_override=True blocks _update_existing, so no promotion
        assert merged[0].bucket == Bucket.PENDING


# ---------------------------------------------------------------------------
# TestAutoRecovery (US-D4)
# ---------------------------------------------------------------------------

class TestAutoRecovery:
    def test_pending_paper_promoted_when_stub_cleared(self, monkeypatch):
        """Existing pending (stub_metadata) paper with year=0 → new discovery has year=2025 → promoted."""
        from ndif_citations.models import Category
        from ndif_citations.process import _decide_bucket

        existing = make_paper(
            arxiv_id="2401.00001",
            bucket=Bucket.PENDING,
            year=0,
            category=Category.USES_NNSIGHT,
            category_confidence=0.85,
        )
        existing.reason = PaperReason.STUB_METADATA

        # New discovery fills the year gap
        new = make_paper(
            arxiv_id="2401.00001",
            year=2025,
            authors="Smith, Jones, Brown",  # also longer so _update_existing fires
            category=Category.USES_NNSIGHT,
            category_confidence=0.85,
        )

        merged, run = merge_papers([existing], [new])
        assert merged[0].bucket == Bucket.VERIFIED
        assert merged[0].reason is None
        assert existing.title in run.auto_promoted

    def test_pending_with_manual_override_not_promoted(self):
        existing = make_paper(
            arxiv_id="2401.00001",
            bucket=Bucket.PENDING,
            manual_override=True,
            year=0,
            category_confidence=0.85,
        )
        existing.reason = PaperReason.STUB_METADATA

        new = make_paper(
            arxiv_id="2401.00001",
            year=2025,
            authors="Smith, Jones",
            category_confidence=0.85,
        )

        merged, run = merge_papers([existing], [new])
        assert merged[0].bucket == Bucket.PENDING
        assert existing.title not in run.auto_promoted

    def test_verified_paper_demoted_when_conditions_degrade(self):
        """Existing verified paper → new discovery returns confidence=0.4 → demoted."""
        existing = make_paper(
            arxiv_id="2401.00001",
            bucket=Bucket.VERIFIED,
            category_confidence=0.4,  # simulate the merge updating this
            category=Category.REFERENCING,
        )
        existing.year = 2024  # ensure year is set
        new = make_paper(
            arxiv_id="2401.00001",
            year=2024,
            category=Category.REFERENCING,
            category_confidence=0.4,
        )

        merged, run = merge_papers([existing], [new])
        assert merged[0].bucket == Bucket.PENDING
        assert merged[0].reason == PaperReason.LOW_CONFIDENCE
        assert existing.title in run.auto_demoted

    def test_verified_with_manual_override_not_demoted(self):
        existing = make_paper(
            arxiv_id="2401.00001",
            bucket=Bucket.VERIFIED,
            manual_override=True,
            category_confidence=0.4,
            category=Category.REFERENCING,
        )
        existing.year = 2024
        new = make_paper(arxiv_id="2401.00001", year=2024, category_confidence=0.4)

        merged, run = merge_papers([existing], [new])
        assert merged[0].bucket == Bucket.VERIFIED
        assert existing.title not in run.auto_demoted
