"""Tests for the low-confidence dedup fix in write_outputs (US-F8)."""
from ndif_citations.models import DetailCategory, PipelineRun
from tests.conftest import make_paper


def _build_low_confidence_list(papers):
    """Replicate the logic from output.py:write_outputs for low_confidence."""
    run = PipelineRun()
    run.low_confidence = [
        f'"{p.title}" -- classified as "{p.detail_category.value}" (confidence: {p.category_confidence:.2f})'
        for p in papers
        if p.category_confidence < 0.7 and p.detail_category != DetailCategory.UNCLASSIFIED
    ]
    return run.low_confidence


class TestLowConfidenceDedup:
    def test_unclassified_excluded_from_low_confidence(self):
        unclassified = make_paper(
            detail_category=DetailCategory.UNCLASSIFIED,
            category_confidence=0.0,
            title="Unclassified Paper",
        )
        result = _build_low_confidence_list([unclassified])
        assert result == []

    def test_non_unclassified_low_confidence_included(self):
        low_conf = make_paper(
            detail_category=DetailCategory.REFERENCING,
            category_confidence=0.4,
            title="Low Confidence Paper",
        )
        result = _build_low_confidence_list([low_conf])
        assert len(result) == 1
        assert "Low Confidence Paper" in result[0]

    def test_high_confidence_excluded(self):
        high = make_paper(
            detail_category=DetailCategory.USES_NDIF,
            category_confidence=0.85,
        )
        result = _build_low_confidence_list([high])
        assert result == []

    def test_mixed_list(self):
        papers = [
            make_paper(title="A", detail_category=DetailCategory.UNCLASSIFIED, category_confidence=0.0),
            make_paper(title="B", detail_category=DetailCategory.REFERENCING, category_confidence=0.4),
            make_paper(title="C", detail_category=DetailCategory.USES_NNSIGHT, category_confidence=0.85),
        ]
        result = _build_low_confidence_list(papers)
        # Only "B" (non-UNCLASSIFIED + confidence < 0.7) should appear
        assert len(result) == 1
        assert "B" in result[0]
        assert "A" not in " ".join(result)
        assert "C" not in " ".join(result)
