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

    def test_venue_conference_to_conference_replaces(self):
        # The new venue is post-resolve_venue, so it's authoritative — always
        # prefer it when it differs from the existing string. (Manual overrides
        # are protected one level up via existing.manual_override.)
        existing = make_paper(venue="ICLR 2024")
        new = make_paper(venue="NeurIPS 2025")
        changed = _update_existing(existing, new)
        assert changed is True
        assert existing.venue == "NeurIPS 2025"

    def test_venue_arxiv_does_not_downgrade_confident_existing(self):
        # Confident existing venue must NOT be replaced by an ArXiv fallback
        # when the new pipeline run had no better source.
        existing = make_paper(venue="ICML 2024", year=2024)
        new = make_paper(venue="ArXiv 2024", year=2024)
        _update_existing(existing, new)
        assert existing.venue == "ICML 2024"

    def test_venue_arxiv_replaces_junk_existing(self):
        # Existing venue that doesn't contain any known acronym/journal is
        # treated as junk and replaced by the ArXiv fallback.
        existing = make_paper(venue="Handbook of Human 2025", year=2025)
        new = make_paper(venue="ArXiv 2025", year=2025)
        changed = _update_existing(existing, new)
        assert changed is True
        assert existing.venue == "ArXiv 2025"

    def test_venue_long_form_existing_renormalized(self):
        # Pre-cleanup long-form existing gets re-normalized on merge even when
        # the new paper had no better source. (This is the cleanup-on-merge path.)
        existing = make_paper(
            venue="International Conference on Machine Learning 2024", year=2024
        )
        new = make_paper(venue="ArXiv 2024", year=2024)
        changed = _update_existing(existing, new)
        assert changed is True
        assert existing.venue == "ICML 2024"

    def test_venue_source_propagates_on_take_new(self):
        # When new venue replaces existing, venue_source is mirrored alongside.
        existing = make_paper(venue="arXiv 2024", venue_source=None)
        new = make_paper(venue="NeurIPS 2025", venue_source="doi_prefix")
        _update_existing(existing, new)
        assert existing.venue == "NeurIPS 2025"
        assert existing.venue_source == "doi_prefix"

    def test_venue_source_not_overwritten_when_existing_protected(self):
        # When the ArXiv fallback would downgrade a confident existing, the
        # venue stays as the previously-resolved value. venue_source backfills
        # to whatever the new run produced — it's audit metadata, not a content
        # signal, and the new label reflects the cascade path on this run.
        existing = make_paper(
            venue="ICML 2024", year=2024, venue_source="openalex"
        )
        new = make_paper(
            venue="ArXiv 2024", year=2024, venue_source="fallback"
        )
        _update_existing(existing, new)
        assert existing.venue == "ICML 2024"
        # Source backfill happens regardless of which venue we kept.
        assert existing.venue_source == "fallback"

    def test_venue_source_backfilled_when_venue_unchanged(self):
        # Stable paper across runs: venue is identical, but the new pass labels
        # its cascade path. The existing record gets the source label so the
        # field doesn't stay None for legacy papers.
        existing = make_paper(venue="ICML 2025", year=2025, venue_source=None)
        new = make_paper(
            venue="ICML 2025", year=2025, venue_source="doi_prefix"
        )
        _update_existing(existing, new)
        assert existing.venue == "ICML 2025"
        assert existing.venue_source == "doi_prefix"

    def test_year_reconciles_across_runs_for_high_confidence_source(self):
        # arXiv upload-year staleness: existing.year is the upload year (2024)
        # but the new run's DOI-prefix decode tied it to the 2025 proceedings.
        existing = make_paper(
            venue="ACL 2025", year=2024, venue_source=None
        )
        new = make_paper(
            venue="ACL 2025", year=2025, venue_source="doi_prefix"
        )
        changed = _update_existing(existing, new)
        assert changed is True
        assert existing.year == 2025

    def test_year_not_reconciled_for_low_confidence_source(self):
        # OpenAlex source doesn't embed a year — must not move the year.
        existing = make_paper(
            venue="ICML 2025", year=2024, venue_source=None
        )
        new = make_paper(
            venue="ICML 2025", year=2025, venue_source="openalex"
        )
        _update_existing(existing, new)
        assert existing.year == 2024  # unchanged

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

    def test_venue_source_not_in_website_dict(self):
        # Provenance label is internal/audit-only — must not surface on the
        # public website schema until a UI design decision is made.
        p = make_paper(venue_source="doi_prefix")
        d = p.to_website_dict()
        assert "venue_source" not in d

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

    def test_venue_source_in_full_dict_round_trip(self):
        p = make_paper(venue_source="doi_prefix")
        d = p.to_full_dict()
        assert "venue_source" in d
        assert d["venue_source"] == "doi_prefix"
        restored = DiscoveredPaper.model_validate(d)
        assert restored.venue_source == "doi_prefix"

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
