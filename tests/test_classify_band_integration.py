"""Integration tests: classify_category emits the correct Confidence band
across all decision paths in process.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ndif_citations.models import Category, Confidence
from ndif_citations.process import classify_category
from tests.conftest import make_paper


@pytest.fixture
def llm_client_uses_ndif():
    """Mock LLM client returning 'uses_ndif'."""
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="uses_ndif"))]
    )
    return client


class TestClassifyBandPath:
    """Each path through classify_category should produce a known band."""

    def test_no_pdf_no_keywords_in_abstract_is_none(self, tmp_path):
        paper = make_paper(abstract="Totally unrelated abstract about cats.")
        cat, conf, band = classify_category(paper, tmp_path, pdf_path=None)
        assert cat == Category.UNCLASSIFIED
        assert band == Confidence.NONE

    def test_no_pdf_no_abstract_is_none(self, tmp_path):
        paper = make_paper(abstract=None)
        cat, conf, band = classify_category(paper, tmp_path, pdf_path=None)
        assert cat == Category.UNCLASSIFIED
        assert band == Confidence.NONE

    def test_pre_filter_negative_evidence_is_certain(self, tmp_path, monkeypatch):
        # Patch extract_ndif_context to return text with negative evidence
        monkeypatch.setattr(
            "ndif_citations.process.extract_ndif_context",
            lambda *a, **kw: "Our approach is an alternative to NDIF and avoids it entirely.",
        )
        pdf = tmp_path / "fake.pdf"
        pdf.write_bytes(b"%PDF")  # exists()
        paper = make_paper()
        cat, conf, band = classify_category(paper, tmp_path, pdf_path=pdf)
        assert cat == Category.REFERENCING
        assert band == Confidence.CERTAIN

    def test_llm_with_tier_2_link_is_high(self, tmp_path, monkeypatch, llm_client_uses_ndif):
        monkeypatch.setattr(
            "ndif_citations.process.extract_ndif_context",
            lambda *a, **kw: "We use nnsight to investigate model internals.",
        )
        monkeypatch.setattr(
            "ndif_citations.process._get_llm_client",
            lambda: llm_client_uses_ndif,
        )
        pdf = tmp_path / "fake.pdf"
        pdf.write_bytes(b"%PDF")
        paper = make_paper(linked_paper_tier=2)
        cat, conf, band = classify_category(paper, tmp_path, pdf_path=pdf)
        assert cat == Category.USES_NDIF
        assert band == Confidence.HIGH

    def test_llm_with_single_window_pdf_is_medium(self, tmp_path, monkeypatch, llm_client_uses_ndif):
        monkeypatch.setattr(
            "ndif_citations.process.extract_ndif_context",
            lambda *a, **kw: "We mention nnsight once in passing.",
        )
        monkeypatch.setattr(
            "ndif_citations.process._get_llm_client",
            lambda: llm_client_uses_ndif,
        )
        pdf = tmp_path / "fake.pdf"
        pdf.write_bytes(b"%PDF")
        paper = make_paper(linked_paper_tier=None)
        cat, conf, band = classify_category(paper, tmp_path, pdf_path=pdf)
        # Single window, no tier -> MEDIUM
        assert band == Confidence.MEDIUM

    def test_keyword_fallback_when_llm_unavailable_is_low(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "ndif_citations.process.extract_ndif_context",
            lambda *a, **kw: "import nnsight is used in our pipeline.",
        )
        monkeypatch.setattr(
            "ndif_citations.process._get_llm_client",
            lambda: None,  # No client → fallback
        )
        pdf = tmp_path / "fake.pdf"
        pdf.write_bytes(b"%PDF")
        paper = make_paper()
        cat, conf, band = classify_category(paper, tmp_path, pdf_path=pdf)
        assert band == Confidence.LOW
