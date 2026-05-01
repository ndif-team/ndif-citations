"""Tests for _detect_linked_paper and _arxiv_id_year (discover.py)."""
from ndif_citations.discover import _arxiv_id_year, _detect_linked_paper


class TestArxivIdYear:
    def test_standard_id(self):
        assert _arxiv_id_year("2407.14561") == 24

    def test_older_id(self):
        assert _arxiv_id_year("1901.00001") == 19

    def test_year_2030(self):
        assert _arxiv_id_year("3001.99999") == 30

    def test_malformed_returns_zero(self):
        assert _arxiv_id_year("not-an-id") == 0

    def test_empty_returns_zero(self):
        assert _arxiv_id_year("") == 0

    def test_empty_returns_zero(self):
        assert _arxiv_id_year("") == 0


class TestDetectLinkedPaper:
    def test_returns_none_for_empty_readme(self):
        assert _detect_linked_paper("", []) is None

    def test_returns_none_for_no_ids(self):
        assert _detect_linked_paper("Just some text", []) is None

    def test_tier1_bibtex_wins(self):
        readme = """
## Citation
@article{foo2024,
  eprint={2407.14561},
  archivePrefix={arXiv}
}
"""
        # BibTeX tier should win over a Citation-section URL
        ids = ["2407.14561"]
        result = _detect_linked_paper(readme, ids)
        assert result == "https://arxiv.org/abs/2407.14561"

    def test_tier2_citation_section(self):
        readme = """
## Citation
Please cite https://arxiv.org/abs/2407.99999
"""
        result = _detect_linked_paper(readme, ["2407.99999"])
        assert result == "https://arxiv.org/abs/2407.99999"

    def test_tier1_beats_tier2(self):
        readme = """
## Citation
Please cite https://arxiv.org/abs/2407.00002

@article{bar,
  eprint={2407.00001}
}
"""
        result = _detect_linked_paper(readme, ["2407.00001", "2407.00002"])
        # BibTeX (tier 1) wins
        assert result == "https://arxiv.org/abs/2407.00001"

    def test_tier3_single_post2020_id(self):
        readme = "Check out our work at https://arxiv.org/abs/2301.55555"
        result = _detect_linked_paper(readme, ["2301.55555"])
        assert result == "https://arxiv.org/abs/2301.55555"

    def test_tier4_most_recent_id(self):
        readme = "Related papers: 2301.00001 and 2401.00002"
        result = _detect_linked_paper(readme, ["2301.00001", "2401.00002"])
        # Tier 4: most recent (highest YYMM prefix)
        assert result == "https://arxiv.org/abs/2401.00002"

    def test_tier5_only_pre2020_returns_none(self):
        # All IDs are pre-2020 (year prefix < 20) → None
        result = _detect_linked_paper("some text", ["1901.00001", "1812.99999"])
        assert result is None

    def test_citation_section_beats_single_id_fallback(self):
        readme = """
## Citation
https://arxiv.org/abs/2407.00001

Also see https://arxiv.org/abs/2407.00002
"""
        result = _detect_linked_paper(readme, ["2407.00001", "2407.00002"])
        # Tier 2 (citation section, first match) beats tier 3
        assert result == "https://arxiv.org/abs/2407.00001"
