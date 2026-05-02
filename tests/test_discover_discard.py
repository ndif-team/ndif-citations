"""Tests for the zero_pdf_hits discard rule (US-D2)."""
from pathlib import Path

import pytest

from ndif_citations.models import Bucket, PaperReason
from ndif_citations.process import _check_discard_zero_pdf_hits
from tests.conftest import make_paper

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "papers"


def _make_pdf(tmp_path: Path, text: str) -> Path:
    """Write text to a fake PDF file and patch extract_text_from_pdf via monkeypatching."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")
    return pdf


class TestCheckDiscardZeroPdfHits:
    def test_pdf_with_keywords_not_discarded(self, monkeypatch, tmp_path):
        text = "We use nnsight to extract activations. " * 150  # > 1000 chars
        monkeypatch.setattr(
            "ndif_citations.process.extract_text_from_pdf",
            lambda path: text,
        )
        pdf = _make_pdf(tmp_path, text)
        paper = make_paper()
        discarded = _check_discard_zero_pdf_hits(paper, pdf)
        assert discarded is False
        assert paper.bucket == Bucket.VERIFIED

    def test_pdf_with_5000_chars_no_keywords_discarded(self, monkeypatch, tmp_path):
        text = "A paper about transformers and attention heads. " * 120  # > 1000 chars, no NDIF
        monkeypatch.setattr(
            "ndif_citations.process.extract_text_from_pdf",
            lambda path: text,
        )
        pdf = _make_pdf(tmp_path, text)
        paper = make_paper()
        discarded = _check_discard_zero_pdf_hits(paper, pdf)
        assert discarded is True
        assert paper.bucket == Bucket.DISCARDED
        assert paper.reason == PaperReason.ZERO_PDF_HITS
        assert paper.reason_detail is not None
        assert "chars" in paper.reason_detail

    def test_short_pdf_200_chars_no_keywords_not_discarded(self, monkeypatch, tmp_path):
        text = "Short text."  # well under 1000 chars
        monkeypatch.setattr(
            "ndif_citations.process.extract_text_from_pdf",
            lambda path: text,
        )
        pdf = _make_pdf(tmp_path, text)
        paper = make_paper()
        discarded = _check_discard_zero_pdf_hits(paper, pdf)
        assert discarded is False
        assert paper.bucket == Bucket.VERIFIED

    def test_no_pdf_not_discarded(self):
        paper = make_paper()
        discarded = _check_discard_zero_pdf_hits(paper, None)
        assert discarded is False
        assert paper.bucket == Bucket.VERIFIED

    def test_zero_pdf_hits_fixture_has_no_keywords(self):
        text = (FIXTURE_DIR / "zero_pdf_hits.txt").read_text()
        assert len(text) > 1000
        lower = text.lower()
        # None of the main keywords should appear
        for kw in ("nnsight", "ndif", "national deep inference"):
            assert kw not in lower, f"Keyword '{kw}' found in zero_pdf_hits fixture"
