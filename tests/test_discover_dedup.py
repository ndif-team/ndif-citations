"""Tests for _merge_paper_data and deduplicate_papers (discover.py)."""
from ndif_citations.discover import _merge_paper_data, deduplicate_papers
from ndif_citations.models import DiscoverySource
from tests.conftest import make_paper


class TestMergePaperData:
    def test_fills_missing_arxiv_id(self):
        primary = make_paper(arxiv_id=None)
        secondary = make_paper(arxiv_id="2407.99999")
        _merge_paper_data(primary, secondary)
        assert primary.arxiv_id == "2407.99999"

    def test_does_not_overwrite_existing_arxiv_id(self):
        primary = make_paper(arxiv_id="2407.00001")
        secondary = make_paper(arxiv_id="2407.99999")
        _merge_paper_data(primary, secondary)
        assert primary.arxiv_id == "2407.00001"

    def test_fills_missing_doi(self):
        primary = make_paper(arxiv_id=None, doi=None)
        secondary = make_paper(arxiv_id=None, doi="10.1234/test")
        _merge_paper_data(primary, secondary)
        assert primary.doi == "10.1234/test"

    def test_fills_missing_affiliations(self):
        primary = make_paper()
        primary.affiliations = ""
        secondary = make_paper()
        secondary.affiliations = "MIT, Stanford"
        _merge_paper_data(primary, secondary)
        assert primary.affiliations == "MIT, Stanford"

    def test_does_not_overwrite_existing_affiliations(self):
        primary = make_paper()
        primary.affiliations = "Harvard"
        secondary = make_paper()
        secondary.affiliations = "MIT"
        _merge_paper_data(primary, secondary)
        assert primary.affiliations == "Harvard"

    def test_fills_missing_abstract(self):
        primary = make_paper(abstract=None)
        secondary = make_paper(abstract="A good abstract.")
        _merge_paper_data(primary, secondary)
        assert primary.abstract == "A good abstract."

    def test_fills_missing_pdf_url(self):
        primary = make_paper()
        primary.pdf_url = None
        secondary = make_paper()
        secondary.pdf_url = "https://arxiv.org/pdf/2407.00001.pdf"
        _merge_paper_data(primary, secondary)
        assert primary.pdf_url == "https://arxiv.org/pdf/2407.00001.pdf"

    def test_fills_github_repo_url(self):
        primary = make_paper()
        primary.github_repo_url = None
        secondary = make_paper()
        secondary.github_repo_url = "https://github.com/owner/repo"
        _merge_paper_data(primary, secondary)
        assert primary.github_repo_url == "https://github.com/owner/repo"

    def test_fills_missing_venue(self):
        primary = make_paper(venue="")
        secondary = make_paper(venue="ICLR 2025")
        _merge_paper_data(primary, secondary)
        assert primary.venue == "ICLR 2025"

    def test_fills_missing_year(self):
        primary = make_paper(year=0)
        secondary = make_paper(year=2024)
        _merge_paper_data(primary, secondary)
        assert primary.year == 2024


class TestDeduplicatePapers:
    def test_same_arxiv_id_merges(self):
        p1 = make_paper(title="Paper A", arxiv_id="2407.00001", source=DiscoverySource.S2_CITATION)
        p2 = make_paper(title="Paper A alt", arxiv_id="2407.00001", source=DiscoverySource.OPENALEX_FULLTEXT)
        result = deduplicate_papers([p1, p2])
        assert len(result) == 1

    def test_same_doi_merges(self):
        p1 = make_paper(title="Paper B", arxiv_id=None, doi="10.1234/test", source=DiscoverySource.S2_CITATION)
        p2 = make_paper(title="Paper B alt", arxiv_id=None, doi="10.1234/test", source=DiscoverySource.OPENALEX_FULLTEXT)
        result = deduplicate_papers([p1, p2])
        assert len(result) == 1

    def test_title_similarity_merges(self):
        p1 = make_paper(title="Attention Is All You Need", arxiv_id=None, doi=None)
        p2 = make_paper(title="Attention Is All You Need.", arxiv_id=None, doi=None)
        result = deduplicate_papers([p1, p2])
        assert len(result) == 1

    def test_different_papers_not_merged(self):
        p1 = make_paper(title="Paper Alpha", arxiv_id="2407.00001")
        p2 = make_paper(title="Paper Beta", arxiv_id="2407.00002")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 2

    def test_s2_wins_over_openalex(self):
        s2 = make_paper(title="Paper", arxiv_id="2407.00001", authors="S2 Author",
                        source=DiscoverySource.S2_CITATION)
        oa = make_paper(title="Paper", arxiv_id="2407.00001", authors="OA Author",
                        source=DiscoverySource.OPENALEX_FULLTEXT)
        result = deduplicate_papers([s2, oa])
        assert result[0].authors == "S2 Author"

    def test_excluded_titles_filtered(self):
        excluded = make_paper(
            title="NNsight and NDIF: Democratizing Access to Open-Weight Foundation Model Internals",
            arxiv_id="2407.14561",
        )
        other = make_paper(title="Other Paper", arxiv_id="2407.00002")
        result = deduplicate_papers([excluded, other])
        assert len(result) == 1
        assert result[0].arxiv_id == "2407.00002"
