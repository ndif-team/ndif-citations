"""Tests for extract_ndif_context (utils.py)."""
import tempfile
from pathlib import Path

import pytest

from ndif_citations.utils import extract_ndif_context


def _write_tmp_text(text: str) -> Path:
    """Write text to a temp file and return its Path (won't be auto-deleted)."""
    f = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, mode="w")
    f.write(text)
    f.close()
    return Path(f.name)


class TestExtractNdifContext:
    def test_keyword_found_returns_context_window(self, monkeypatch):
        monkeypatch.setattr(
            "ndif_citations.utils.extract_text_from_pdf",
            lambda path: "We ran experiments using nnsight to trace activations.",
        )
        result = extract_ndif_context(Path("/fake/path.pdf"))
        assert "nnsight" in result.lower()
        # Must not be the sentinel
        assert "No direct mentions" not in result

    def test_no_keywords_returns_sentinel(self, monkeypatch):
        monkeypatch.setattr(
            "ndif_citations.utils.extract_text_from_pdf",
            lambda path: "This paper discusses transformers and attention mechanisms.",
        )
        result = extract_ndif_context(Path("/fake/path.pdf"))
        assert result.startswith("No direct mentions")

    def test_empty_pdf_returns_extraction_failure_message(self, monkeypatch):
        monkeypatch.setattr(
            "ndif_citations.utils.extract_text_from_pdf",
            lambda path: "",
        )
        result = extract_ndif_context(Path("/fake/path.pdf"))
        assert "No text could be extracted" in result

    def test_multiple_occurrences_returns_up_to_max_excerpts(self, monkeypatch):
        # 6 occurrences but MAX_CONTEXT_EXCERPTS=5 by default
        text = " ... nnsight ... " * 6
        monkeypatch.setattr(
            "ndif_citations.utils.extract_text_from_pdf",
            lambda path: text,
        )
        result = extract_ndif_context(Path("/fake/path.pdf"))
        # At most 5 excerpts separated by "---"
        assert result.count("---") <= 4  # 5 excerpts → 4 separators

    def test_case_insensitive_match(self, monkeypatch):
        monkeypatch.setattr(
            "ndif_citations.utils.extract_text_from_pdf",
            lambda path: "We used NNsight library for activation patching.",
        )
        result = extract_ndif_context(Path("/fake/path.pdf"))
        assert "No direct mentions" not in result
        assert "NNsight" in result or "nnsight" in result.lower()

    def test_ndif_keyword_match(self, monkeypatch):
        monkeypatch.setattr(
            "ndif_citations.utils.extract_text_from_pdf",
            lambda path: "Models were hosted on the NDIF cluster at Northeastern.",
        )
        result = extract_ndif_context(Path("/fake/path.pdf"))
        assert "NDIF" in result

    def test_context_window_respected(self, monkeypatch):
        # Keyword buried in long text; window=10 should return short slices (not the full 207-char text)
        text = "x" * 100 + "nnsight" + "y" * 100
        monkeypatch.setattr(
            "ndif_citations.utils.extract_text_from_pdf",
            lambda path: text,
        )
        result = extract_ndif_context(Path("/fake/path.pdf"), window=10)
        # Each excerpt is ~27 chars; multiple keyword variants may match the same occurrence
        # but result is far shorter than the full 207-char text
        assert len(result) < len(text)
