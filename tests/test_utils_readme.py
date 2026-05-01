"""Tests for README parsing utilities (utils.py)."""
from ndif_citations.utils import extract_bibtex_arxiv_ids, parse_readme_sections


class TestParseReadmeSections:
    def test_single_section(self):
        text = "# Installation\nRun pip install\n"
        sections = parse_readme_sections(text)
        assert "installation" in sections
        assert "pip install" in sections["installation"]

    def test_pre_header_content_under_empty_key(self):
        text = "Some intro text\n# Usage\ncontent"
        sections = parse_readme_sections(text)
        assert "" in sections
        assert "Some intro text" in sections[""]

    def test_header_key_is_lowercased(self):
        text = "# My SECTION\nbody"
        sections = parse_readme_sections(text)
        assert "my section" in sections

    def test_multiple_levels(self):
        text = "## Citation\ncite here\n### Sub\nsub body"
        sections = parse_readme_sections(text)
        assert "citation" in sections
        assert "sub" in sections

    def test_empty_readme(self):
        sections = parse_readme_sections("")
        assert sections == {"": ""}

    def test_no_headers(self):
        text = "just plain text\nno headers"
        sections = parse_readme_sections(text)
        assert sections == {"": "just plain text\nno headers"}

    def test_duplicate_headers_last_wins(self):
        text = "# Foo\nfirst\n# Foo\nsecond"
        sections = parse_readme_sections(text)
        assert "second" in sections["foo"]


class TestExtractBibtexArxivIds:
    def test_eprint_field(self):
        text = """
@article{smith2024test,
  title={Test},
  eprint={2407.14561},
  archivePrefix={arXiv}
}
"""
        ids = extract_bibtex_arxiv_ids(text)
        assert ids == ["2407.14561"]

    def test_arxiv_field(self):
        text = """
@misc{foo,
  arxiv={2301.12345}
}
"""
        ids = extract_bibtex_arxiv_ids(text)
        assert ids == ["2301.12345"]

    def test_embedded_url(self):
        text = """
@article{foo,
  url={https://arxiv.org/abs/2407.00001}
}
"""
        ids = extract_bibtex_arxiv_ids(text)
        assert ids == ["2407.00001"]

    def test_deduplicates(self):
        text = """
@article{a, eprint={2407.14561}}
@article{b, eprint={2407.14561}}
"""
        ids = extract_bibtex_arxiv_ids(text)
        assert ids == ["2407.14561"]

    def test_preserves_order(self):
        text = """
@article{a, eprint={2407.00001}}
@article{b, eprint={2407.00002}}
"""
        ids = extract_bibtex_arxiv_ids(text)
        assert ids == ["2407.00001", "2407.00002"]

    def test_strips_version(self):
        text = """
@article{foo, eprint={2407.14561v2}}
"""
        ids = extract_bibtex_arxiv_ids(text)
        assert ids == ["2407.14561"]

    def test_no_bibtex_returns_empty(self):
        assert extract_bibtex_arxiv_ids("Just a plain README with no BibTeX.") == []

    def test_unbalanced_brace_skipped(self):
        # Malformed BibTeX (unclosed brace) — should not crash
        text = "@article{broken, eprint={2407.11111}"
        ids = extract_bibtex_arxiv_ids(text)
        # Either finds it or not — just must not raise
        assert isinstance(ids, list)
