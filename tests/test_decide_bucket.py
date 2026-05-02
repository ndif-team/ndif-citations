"""Tests for _decide_bucket demotion rules (US-D3)."""
import pytest

from ndif_citations.models import Bucket, Category, DiscoverySource, PaperReason
from ndif_citations.process import _decide_bucket
from tests.conftest import make_paper


class TestDecideBucket:
    # --- Rule 1: openalex_source ---

    def test_openalex_paper_high_confidence_demoted(self):
        paper = make_paper(
            source=DiscoverySource.OPENALEX_FULLTEXT,
            category=Category.USES_NNSIGHT,
            category_confidence=0.85,
        )
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.PENDING
        assert reason == PaperReason.OPENALEX_SOURCE

    # --- Rule 2: stub_metadata ---

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

    def test_no_abstract_no_pdf_url_demoted(self):
        paper = make_paper(
            abstract=None,
            category=Category.REFERENCING, category_confidence=0.85,
            year=2024,
        )
        paper.pdf_url = None
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.PENDING
        assert reason == PaperReason.STUB_METADATA

    # --- Rule 3: unclassified_no_keywords ---

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

    # --- Rule 4: low_confidence ---

    def test_low_confidence_fallback_path_demoted(self):
        paper = make_paper(
            category=Category.REFERENCING,
            category_confidence=0.4,
            year=2024,
        )
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.PENDING
        assert reason == PaperReason.LOW_CONFIDENCE

    # --- Happy path: verified ---

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

    # --- First-rule-wins precedence ---

    def test_openalex_with_year_zero_reason_is_openalex_source(self):
        paper = make_paper(
            source=DiscoverySource.OPENALEX_FULLTEXT,
            year=0,
            category=Category.REFERENCING,
            category_confidence=0.85,
        )
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.PENDING
        assert reason == PaperReason.OPENALEX_SOURCE  # Rule 1 fires first
