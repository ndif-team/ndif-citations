"""Tests for slugify, is_duplicate, and generate_bibtex (utils.py)."""
from ndif_citations.utils import generate_bibtex, is_duplicate, slugify


class TestSlugify:
    def test_basic_title(self):
        assert slugify("Language Models Use Trigonometry") == "Language-Models-Use-Trigonometry"

    def test_collapses_multiple_spaces(self):
        assert slugify("Hello   World") == "Hello-World"

    def test_strips_punctuation(self):
        result = slugify("Hello, World!")
        assert "," not in result
        assert "!" not in result

    def test_preserves_casing(self):
        result = slugify("NDIF and NNsight")
        assert "NDIF" in result
        assert "NNsight" in result

    def test_strips_leading_trailing_whitespace(self):
        assert slugify("  Hello World  ") == "Hello-World"

    def test_single_word(self):
        assert slugify("Attention") == "Attention"

    def test_hyphen_preserved(self):
        # Hyphens in source are kept (they're in \w complement exclusion)
        result = slugify("Self-Attention")
        assert "Self" in result
        assert "Attention" in result


class TestIsDuplicate:
    def test_identical_titles(self):
        assert is_duplicate("Hello World", "Hello World") is True

    def test_case_insensitive(self):
        assert is_duplicate("Hello World", "hello world") is True

    def test_completely_different(self):
        assert is_duplicate("Attention Is All You Need", "Quantum Computing Basics") is False

    def test_threshold_boundary_below(self):
        # Very different titles should be below 90%
        assert is_duplicate("ABCDEF", "GHIJKL") is False

    def test_minor_edit_above_threshold(self):
        # One char difference in a long title is above 90%
        title = "A Very Long Title About Language Models and Transformers"
        close = "A Very Long Title About Language Models and Transformer"
        assert is_duplicate(title, close) is True

    def test_strips_whitespace_in_comparison(self):
        assert is_duplicate("  Hello World  ", "Hello World") is True


class TestGenerateBibtex:
    def test_basic_structure(self):
        result = generate_bibtex(
            title="Test Paper",
            authors="Smith, John",
            year=2024,
            venue="arXiv",
            url="https://arxiv.org/abs/2407.00001",
            arxiv_id="2407.00001",
        )
        assert "@article{" in result
        assert "title={Test Paper}" in result
        assert "author={Smith, John}" in result
        assert "year={2024}" in result
        assert "eprint={2407.00001}" in result
        assert "archivePrefix={arXiv}" in result

    def test_cite_key_format(self):
        # cite key = lastname + year + first_title_word
        result = generate_bibtex(
            title="Attention Mechanisms",
            authors="Johnson, Alice",
            year=2025,
            venue="",
            url="",
        )
        assert "@article{johnson2025attention," in result

    def test_omits_missing_doi(self):
        result = generate_bibtex(
            title="Test",
            authors="Doe",
            year=2024,
            venue="",
            url="",
            doi=None,
        )
        assert "doi=" not in result

    def test_omits_missing_venue(self):
        result = generate_bibtex(
            title="Test",
            authors="Doe",
            year=2024,
            venue="",
            url="",
        )
        assert "journal=" not in result

    def test_includes_doi_when_present(self):
        result = generate_bibtex(
            title="Test",
            authors="Doe",
            year=2024,
            venue="",
            url="",
            doi="10.1234/test",
        )
        assert "doi={10.1234/test}" in result
