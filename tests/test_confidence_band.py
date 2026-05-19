"""Tests for the Confidence enum, band ↔ float migration, and the
_compute_confidence_band rule.
"""
from __future__ import annotations

import pytest

from ndif_citations.models import (
    Category,
    Confidence,
    DiscoveredPaper,
)


class TestConfidenceFloatBridge:
    """Round-trip between the float (legacy) and the band (new)."""

    def test_band_to_float_mapping(self):
        from ndif_citations.models import _BAND_TO_FLOAT
        assert _BAND_TO_FLOAT[Confidence.CERTAIN] == 1.0
        assert _BAND_TO_FLOAT[Confidence.HIGH] == 0.85
        assert _BAND_TO_FLOAT[Confidence.MEDIUM] == 0.55
        assert _BAND_TO_FLOAT[Confidence.LOW] == 0.30
        assert _BAND_TO_FLOAT[Confidence.NONE] == 0.0

    def test_legacy_float_load_derives_band_high(self):
        """Loading an old paper with category_confidence=0.85 → band HIGH."""
        paper = DiscoveredPaper(
            title="X", abstract="Y",
            category=Category.USES_NDIF,
            category_confidence=0.85,
        )
        assert paper.category_confidence_band == Confidence.HIGH

    def test_legacy_float_load_derives_band_medium(self):
        paper = DiscoveredPaper(
            title="X", abstract="Y",
            category=Category.USES_NNSIGHT,
            category_confidence=0.55,
        )
        assert paper.category_confidence_band == Confidence.MEDIUM

    def test_legacy_float_load_derives_band_low(self):
        paper = DiscoveredPaper(
            title="X", abstract="Y",
            category_confidence=0.40,
        )
        assert paper.category_confidence_band == Confidence.LOW

    def test_legacy_float_load_derives_band_none(self):
        paper = DiscoveredPaper(
            title="X", abstract="Y",
            category=Category.UNCLASSIFIED,
            category_confidence=0.0,
        )
        assert paper.category_confidence_band == Confidence.NONE

    def test_manual_override_with_any_confidence_promotes_to_certain(self):
        """A paper with manual_override=True should be CERTAIN regardless of stored float."""
        paper = DiscoveredPaper(
            title="X", abstract="Y",
            category=Category.USES_NDIF,
            category_confidence=0.40,
            manual_override=True,
        )
        assert paper.category_confidence_band == Confidence.CERTAIN


class TestNewPaperReason:
    def test_medium_confidence_reason_exists(self):
        from ndif_citations.models import PaperReason
        assert PaperReason.MEDIUM_CONFIDENCE.value == "medium_confidence"


class TestComputeBand:
    """Pure-function rule: signal + tier + window_count + context_source -> band."""

    def test_pre_filter_negative_is_certain(self):
        from ndif_citations.process import _compute_confidence_band
        band = _compute_confidence_band(
            signal="pre_filter:negative_evidence",
            linked_paper_tier=None,
            surviving_window_count=0,
            context_source="pdf",
            category=Category.REFERENCING,
        )
        assert band == Confidence.CERTAIN

    def test_pre_filter_table_or_acks_is_medium(self):
        from ndif_citations.process import _compute_confidence_band
        for signal in ("pre_filter:comparison_table", "pre_filter:acks_only_thank_you"):
            band = _compute_confidence_band(
                signal=signal,
                linked_paper_tier=None,
                surviving_window_count=0,
                context_source="pdf",
                category=Category.REFERENCING,
            )
            assert band == Confidence.MEDIUM, f"{signal!r} → {band}"

    def test_llm_with_tier_1_or_2_cross_link_is_high(self):
        from ndif_citations.process import _compute_confidence_band
        for tier in (1, 2):
            band = _compute_confidence_band(
                signal="llm",
                linked_paper_tier=tier,
                surviving_window_count=1,
                context_source="abstract",
                category=Category.USES_NNSIGHT,
            )
            assert band == Confidence.HIGH, f"tier={tier} → {band}"

    def test_llm_with_multi_window_pdf_is_high(self):
        from ndif_citations.process import _compute_confidence_band
        band = _compute_confidence_band(
            signal="llm",
            linked_paper_tier=None,
            surviving_window_count=3,
            context_source="pdf",
            category=Category.USES_NDIF,
        )
        assert band == Confidence.HIGH

    def test_llm_with_single_window_pdf_is_medium(self):
        from ndif_citations.process import _compute_confidence_band
        band = _compute_confidence_band(
            signal="llm",
            linked_paper_tier=None,
            surviving_window_count=1,
            context_source="pdf",
            category=Category.USES_NDIF,
        )
        assert band == Confidence.MEDIUM

    def test_llm_with_abstract_only_is_medium(self):
        from ndif_citations.process import _compute_confidence_band
        band = _compute_confidence_band(
            signal="llm",
            linked_paper_tier=None,
            surviving_window_count=1,
            context_source="abstract",
            category=Category.USES_NDIF,
        )
        assert band == Confidence.MEDIUM

    def test_keyword_fallback_is_low(self):
        from ndif_citations.process import _compute_confidence_band
        band = _compute_confidence_band(
            signal="keyword_fallback",
            linked_paper_tier=None,
            surviving_window_count=0,
            context_source="pdf",
            category=Category.USES_NDIF,
        )
        assert band == Confidence.LOW

    def test_unclassified_is_none(self):
        from ndif_citations.process import _compute_confidence_band
        band = _compute_confidence_band(
            signal="llm",
            linked_paper_tier=None,
            surviving_window_count=0,
            context_source="none",
            category=Category.UNCLASSIFIED,
        )
        assert band == Confidence.NONE
