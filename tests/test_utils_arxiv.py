"""Tests for arXiv ID normalisation and extraction utilities (utils.py)."""
import pytest

from ndif_citations.utils import (
    extract_arxiv_id_from_doi,
    extract_arxiv_id_from_url,
    looks_like_pdf_url,
    normalize_arxiv_id,
)


class TestNormalizeArxivId:
    def test_strips_version_suffix(self):
        assert normalize_arxiv_id("2407.14561v2") == "2407.14561"

    def test_strips_v1(self):
        assert normalize_arxiv_id("2407.14561v1") == "2407.14561"

    def test_strips_abs_url_prefix(self):
        assert normalize_arxiv_id("https://arxiv.org/abs/2407.14561") == "2407.14561"

    def test_strips_pdf_url_prefix(self):
        assert normalize_arxiv_id("https://arxiv.org/pdf/2407.14561") == "2407.14561"

    def test_strips_http_prefix(self):
        assert normalize_arxiv_id("http://arxiv.org/abs/2407.14561") == "2407.14561"

    def test_strips_pdf_extension(self):
        assert normalize_arxiv_id("2407.14561.pdf") == "2407.14561"

    def test_strips_version_and_pdf_together(self):
        assert normalize_arxiv_id("https://arxiv.org/pdf/2407.14561v3.pdf") == "2407.14561"

    def test_preserves_valid_id(self):
        assert normalize_arxiv_id("2407.14561") == "2407.14561"

    def test_preserves_five_digit_suffix(self):
        assert normalize_arxiv_id("2301.12345") == "2301.12345"

    def test_strips_whitespace(self):
        assert normalize_arxiv_id("  2407.14561  ") == "2407.14561"


class TestExtractArxivIdFromUrl:
    def test_abs_path(self):
        assert extract_arxiv_id_from_url("https://arxiv.org/abs/2407.14561") == "2407.14561"

    def test_pdf_path(self):
        assert extract_arxiv_id_from_url("https://arxiv.org/pdf/2407.14561") == "2407.14561"

    def test_with_version(self):
        assert extract_arxiv_id_from_url("https://arxiv.org/abs/2407.14561v2") == "2407.14561"

    def test_non_arxiv_url_returns_none(self):
        assert extract_arxiv_id_from_url("https://openreview.net/forum?id=abc") is None

    def test_doi_url_returns_none(self):
        assert extract_arxiv_id_from_url("https://doi.org/10.1234/some.doi") is None

    def test_empty_string_returns_none(self):
        assert extract_arxiv_id_from_url("") is None


class TestExtractArxivIdFromDoi:
    def test_standard_arxiv_doi(self):
        assert extract_arxiv_id_from_doi("10.48550/arXiv.2504.14107") == "2504.14107"

    def test_lowercase_arxiv_in_doi(self):
        assert extract_arxiv_id_from_doi("10.48550/arxiv.2504.14107") == "2504.14107"

    def test_mixed_case(self):
        assert extract_arxiv_id_from_doi("10.48550/ArXiv.2407.14561") == "2407.14561"

    def test_non_arxiv_doi_returns_none(self):
        assert extract_arxiv_id_from_doi("10.1234/some.random.doi") is None

    def test_empty_string_returns_none(self):
        assert extract_arxiv_id_from_doi("") is None

    def test_strips_version_in_doi(self):
        assert extract_arxiv_id_from_doi("10.48550/arXiv.2407.14561v2") == "2407.14561"


class TestLooksLikePdfUrl:
    def test_dot_pdf_extension(self):
        assert looks_like_pdf_url("https://example.com/paper.pdf") is True

    def test_slash_pdf_path_segment(self):
        assert looks_like_pdf_url("https://arxiv.org/pdf/2407.14561") is True

    def test_bare_doi_redirect_returns_false(self):
        assert looks_like_pdf_url("https://doi.org/10.1234/some.doi") is False

    def test_dx_doi_redirect_returns_false(self):
        assert looks_like_pdf_url("https://dx.doi.org/10.1234/some.doi") is False

    def test_openreview_landing_page_is_true(self):
        # Not a DOI redirect — should pass through
        assert looks_like_pdf_url("https://openreview.net/forum?id=abc") is True

    def test_empty_string_returns_false(self):
        assert looks_like_pdf_url("") is False
