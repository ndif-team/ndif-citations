"""Tests for _decide_bucket demotion rules."""
import pytest

from ndif_citations.models import Bucket, Category, DiscoverySource, PaperReason
from ndif_citations.process import _decide_bucket, _has_usable_abstract
from tests.conftest import make_paper


class TestHasUsableAbstract:
    def test_none_abstract_false(self):
        paper = make_paper(abstract=None)
        assert _has_usable_abstract(paper) is False

    def test_comma_only_abstract_false(self):
        paper = make_paper(abstract=",")
        assert _has_usable_abstract(paper) is False

    def test_short_abstract_false(self):
        paper = make_paper(abstract="Short.")
        assert _has_usable_abstract(paper) is False

    def test_whitespace_only_false(self):
        paper = make_paper(abstract="   ")
        assert _has_usable_abstract(paper) is False

    def test_good_abstract_true(self):
        paper = make_paper(abstract="We propose a new method for activation patching in language models.")
        assert _has_usable_abstract(paper) is True

    def test_exactly_20_chars_true(self):
        paper = make_paper(abstract="A" * 20)
        assert _has_usable_abstract(paper) is True

    def test_19_chars_false(self):
        paper = make_paper(abstract="A" * 19)
        assert _has_usable_abstract(paper) is False


class TestDecideBucket:
    # --- Rule 1: stub_metadata ---

    def test_year_zero_demoted(self):
        paper = make_paper(year=0, category=Category.REFERENCING, category_confidence=0.85)
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.PENDING
        assert reason == PaperReason.STUB_METADATA

    def test_no_arxiv_no_doi_demoted(self):
        paper = make_paper(
            arxiv_id=None, doi=None,
            category=Category.REFERENCING, category_confidence=0.85,
            year=2024,
        )
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.PENDING
        assert reason == PaperReason.STUB_METADATA

    def test_missing_abstract_demoted_regardless_of_pdf_url(self):
        paper = make_paper(
            abstract=None,
            category=Category.REFERENCING, category_confidence=0.85,
            year=2024,
        )
        paper.pdf_url = "https://arxiv.org/pdf/2401.00001"  # has PDF, still no abstract
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.PENDING
        assert reason == PaperReason.STUB_METADATA

    def test_malformed_comma_abstract_demoted(self):
        paper = make_paper(
            abstract=",",
            category=Category.USES_NNSIGHT, category_confidence=0.85,
            year=2024,
        )
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.PENDING
        assert reason == PaperReason.STUB_METADATA

    def test_short_abstract_demoted(self):
        paper = make_paper(
            abstract="Too short.",
            category=Category.USES_NNSIGHT, category_confidence=0.85,
            year=2024,
        )
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.PENDING
        assert reason == PaperReason.STUB_METADATA

    # --- Rule 2: unclassified ---

    def test_unclassified_no_keywords_demoted(self):
        paper = make_paper(
            category=Category.UNCLASSIFIED,
            category_confidence=0.0,
            year=2024,
        )
        paper.unclassified_reason = "no_keywords_anywhere"
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.PENDING
        assert reason == PaperReason.UNCLASSIFIED_NO_KEYWORDS

    def test_unclassified_no_evidence_demoted(self):
        paper = make_paper(
            category=Category.UNCLASSIFIED,
            category_confidence=0.0,
            year=2024,
        )
        paper.unclassified_reason = "no_evidence_extractable"
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.PENDING
        assert reason == PaperReason.UNCLASSIFIED_NO_KEYWORDS

    def test_unclassified_llm_demoted(self):
        paper = make_paper(
            category=Category.UNCLASSIFIED,
            category_confidence=0.0,
            year=2024,
        )
        paper.unclassified_reason = "llm_unparseable"
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.PENDING
        assert reason == PaperReason.UNCLASSIFIED_LLM

    # --- Rule 3: low_confidence ---

    def test_low_confidence_fallback_path_demoted(self):
        paper = make_paper(
            category=Category.REFERENCING,
            category_confidence=0.4,
            year=2024,
        )
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.PENDING
        assert reason == PaperReason.LOW_CONFIDENCE

    # --- Happy path: verified (source-agnostic) ---

    def test_s2_full_metadata_high_confidence_verified(self):
        paper = make_paper(
            source=DiscoverySource.S2_CITATION,
            category=Category.USES_NNSIGHT,
            category_confidence=0.85,
            year=2024,
        )
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.VERIFIED
        assert reason is None

    def test_openalex_full_metadata_high_confidence_verified(self):
        """OpenAlex papers with good data and high confidence go to verified."""
        paper = make_paper(
            source=DiscoverySource.OPENALEX_FULLTEXT,
            category=Category.USES_NNSIGHT,
            category_confidence=0.85,
            year=2024,
        )
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.VERIFIED
        assert reason is None

    def test_github_full_metadata_high_confidence_verified(self):
        paper = make_paper(
            source=DiscoverySource.GITHUB_DEPENDENT,
            category=Category.USES_NNSIGHT,
            category_confidence=0.85,
            year=2024,
        )
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.VERIFIED
        assert reason is None

    # --- Stub_metadata takes priority over unclassified ---

    def test_year_zero_openalex_reason_is_stub_metadata(self):
        """With openalex_source rule removed, year=0 OpenAlex paper hits stub_metadata."""
        paper = make_paper(
            source=DiscoverySource.OPENALEX_FULLTEXT,
            year=0,
            category=Category.REFERENCING,
            category_confidence=0.85,
        )
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.PENDING
        assert reason == PaperReason.STUB_METADATA


# ---------------------------------------------------------------------------
# New band-based gating (replaces the legacy 0.7 float threshold)
# ---------------------------------------------------------------------------

class TestDecideBucketBands:
    """_decide_bucket should gate on Confidence band, not the legacy float."""

    def test_high_band_goes_to_verified(self):
        from ndif_citations.models import Confidence
        paper = make_paper(
            year=2024,
            abstract="A real abstract that is long enough.",
            arxiv_id="2407.14561",
            category=Category.USES_NDIF,
            category_confidence_band=Confidence.HIGH,
        )
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.VERIFIED
        assert reason is None

    def test_medium_band_goes_to_pending_with_medium_reason(self):
        from ndif_citations.models import Confidence
        paper = make_paper(
            year=2024,
            abstract="A real abstract that is long enough.",
            arxiv_id="2407.14561",
            category=Category.USES_NDIF,
            category_confidence_band=Confidence.MEDIUM,
        )
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.PENDING
        assert reason == PaperReason.MEDIUM_CONFIDENCE

    def test_low_band_goes_to_pending_low_confidence(self):
        from ndif_citations.models import Confidence
        paper = make_paper(
            year=2024,
            abstract="A real abstract that is long enough.",
            arxiv_id="2407.14561",
            category=Category.USES_NDIF,
            category_confidence_band=Confidence.LOW,
        )
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.PENDING
        assert reason == PaperReason.LOW_CONFIDENCE

    def test_certain_band_goes_to_verified(self):
        from ndif_citations.models import Confidence
        paper = make_paper(
            year=2024,
            abstract="A real abstract that is long enough.",
            arxiv_id="2407.14561",
            category=Category.REFERENCING,
            category_confidence_band=Confidence.CERTAIN,
        )
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.VERIFIED
        assert reason is None
