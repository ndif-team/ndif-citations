"""Tests for the compute_context helper extracted from classify_category."""
from __future__ import annotations

from ndif_citations import process
from tests.conftest import make_paper

# compute_context calls get_cached_pdf via a local import from ndif_citations.pdf_cache,
# so we patch it at the source module (matching the pattern used by other process tests).
_PDF_CACHE_PATH = "ndif_citations.pdf_cache.get_cached_pdf"


def test_compute_context_returns_windows_source_signal(monkeypatch, tmp_path):
    """compute_context returns (windows, source, signal) with abstract fallback."""
    paper = make_paper(abstract="we use nnsight to trace activations. " * 5)
    monkeypatch.setattr(_PDF_CACHE_PATH, lambda p, o: None)  # force the no-PDF path
    windows, source, signal = process.compute_context(paper, tmp_path)
    assert isinstance(windows, list)
    assert source in ("pdf", "abstract", "none")


def test_compute_context_no_abstract_no_pdf_returns_empty(monkeypatch, tmp_path):
    """No PDF and no abstract → empty windows, source='none'."""
    paper = make_paper(abstract=None)
    monkeypatch.setattr(_PDF_CACHE_PATH, lambda p, o: None)
    windows, source, signal = process.compute_context(paper, tmp_path)
    assert windows == []
    assert source == "none"


def test_compute_context_abstract_no_keywords_returns_empty(monkeypatch, tmp_path):
    """Abstract with no NDIF/nnsight keywords → empty windows."""
    paper = make_paper(abstract="A paper about transformers and attention mechanisms only.")
    monkeypatch.setattr(_PDF_CACHE_PATH, lambda p, o: None)
    windows, source, signal = process.compute_context(paper, tmp_path)
    assert windows == []
    assert source == "none"


def test_compute_context_pdf_path_used_when_provided(monkeypatch, tmp_path):
    """compute_context uses an already-resolved pdf_path directly (no get_cached_pdf call)."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(
        "ndif_citations.process.extract_ndif_context",
        lambda path, window=500: "We use nnsight for all our experiments.",
    )
    paper = make_paper(abstract=None)
    windows, source, signal = process.compute_context(paper, tmp_path, pdf_path=pdf)
    assert source == "pdf"
    assert len(windows) >= 1


def test_compute_context_dedups_identical_windows(monkeypatch, tmp_path):
    """Byte-identical windows are deduplicated; distinct windows are preserved."""
    # Two identical windows + one distinct window: only 2 unique windows should survive.
    dup_window = "We perform experiments using nnsight to trace model activations."
    distinct_window = "We also ran ndif experiments on the cluster infrastructure."
    blob = f"{dup_window}\n---\n{dup_window}\n---\n{distinct_window}"

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(
        "ndif_citations.process.extract_ndif_context",
        lambda path, window=500: blob,
    )
    monkeypatch.setattr(
        "ndif_citations.pdf_cache.get_cached_pdf",
        lambda p, out: pdf,
    )

    paper = make_paper(abstract=None)
    windows, source, signal = process.compute_context(paper, tmp_path)

    # No exact duplicates remain.
    assert len(windows) == len(set(windows)), "Duplicate windows were not removed"
    # The duplicate window survived exactly once.
    assert sum(1 for w in windows if w == dup_window) == 1
    # The distinct window also survived.
    assert any("ndif" in w.lower() for w in windows), "Distinct ndif window was dropped"
    assert source == "pdf"
