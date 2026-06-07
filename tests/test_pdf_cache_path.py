"""Tests for the cached_pdf_path() helper in pdf_cache.py.

Verifies:
- Returns None when no cached file exists (no network/download).
- Returns the correct path when the file is present.
- Correct filename naming for arxiv_id / doi / title-based keys.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ndif_citations.pdf_cache import cached_pdf_path
from tests.conftest import make_paper


# ---------------------------------------------------------------------------
# 1. arxiv_id naming
# ---------------------------------------------------------------------------

def test_cached_pdf_path_returns_none_on_miss(tmp_path: Path):
    """When no PDF is cached, returns None — no download attempt."""
    paper = make_paper(arxiv_id="2401.1", doi=None)
    result = cached_pdf_path(paper, tmp_path)
    assert result is None


def test_cached_pdf_path_no_network_on_miss(tmp_path: Path):
    """cached_pdf_path() must NOT make any requests on a cache miss."""
    paper = make_paper(arxiv_id="2401.1", doi=None)
    with patch("ndif_citations.pdf_cache.requests.get") as mock_get, \
         patch("ndif_citations.pdf_cache.requests.head") as mock_head:
        cached_pdf_path(paper, tmp_path)
        mock_get.assert_not_called()
        mock_head.assert_not_called()


def test_cached_pdf_path_returns_path_on_hit_arxiv(tmp_path: Path):
    """When arxiv PDF file exists, returns the Path to it."""
    paper = make_paper(arxiv_id="2401.1", doi=None)
    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()
    expected = pdfs_dir / "arxiv-2401.1.pdf"
    expected.write_bytes(b"%PDF-1.4 test")

    result = cached_pdf_path(paper, tmp_path)
    assert result == expected


# ---------------------------------------------------------------------------
# 2. doi naming
# ---------------------------------------------------------------------------

def test_cached_pdf_path_returns_none_doi_on_miss(tmp_path: Path):
    """When no PDF is cached for a DOI paper, returns None."""
    paper = make_paper(arxiv_id=None, doi="10.1234/test.abc")
    result = cached_pdf_path(paper, tmp_path)
    assert result is None


def test_cached_pdf_path_returns_path_on_hit_doi(tmp_path: Path):
    """When doi PDF file exists, returns the Path to it."""
    from ndif_citations.utils import slugify

    paper = make_paper(arxiv_id=None, doi="10.1234/test.abc")
    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()
    expected = pdfs_dir / f"doi-{slugify('10.1234/test.abc')}.pdf"
    expected.write_bytes(b"%PDF-1.4 test")

    result = cached_pdf_path(paper, tmp_path)
    assert result == expected


# ---------------------------------------------------------------------------
# 3. title naming (fallback)
# ---------------------------------------------------------------------------

def test_cached_pdf_path_returns_none_title_on_miss(tmp_path: Path):
    """When no PDF is cached for a title-based paper, returns None."""
    paper = make_paper(arxiv_id=None, doi=None, title="Some Test Paper Without IDs")
    result = cached_pdf_path(paper, tmp_path)
    assert result is None


def test_cached_pdf_path_returns_path_on_hit_title(tmp_path: Path):
    """When title-based PDF file exists, returns the Path to it."""
    from ndif_citations.utils import slugify

    title = "Some Test Paper Without IDs"
    paper = make_paper(arxiv_id=None, doi=None, title=title)
    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()
    expected = pdfs_dir / f"{slugify(title[:50])}.pdf"
    expected.write_bytes(b"%PDF-1.4 test")

    result = cached_pdf_path(paper, tmp_path)
    assert result == expected


# ---------------------------------------------------------------------------
# 4. Does NOT create pdfs/ directory on miss
# ---------------------------------------------------------------------------

def test_cached_pdf_path_does_not_mkdir_on_miss(tmp_path: Path):
    """cached_pdf_path() must not create the pdfs/ directory when there is no cache."""
    paper = make_paper(arxiv_id="2401.1", doi=None)
    pdfs_dir = tmp_path / "pdfs"
    assert not pdfs_dir.exists()
    cached_pdf_path(paper, tmp_path)
    assert not pdfs_dir.exists()
