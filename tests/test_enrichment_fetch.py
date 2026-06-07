from ndif_citations import enrichment
from ndif_citations.enrichment import fetch_records
from tests.conftest import make_paper


def test_fetch_records_from_openalex(monkeypatch):
    p = make_paper(arxiv_id="2401.00001", abstract="snippet …")
    fake_paper = make_paper(abstract="A" * 600, venue="NeurIPS 2024")
    monkeypatch.setattr(enrichment, "_openalex_fetch_work", lambda ident, by="id": {"stub": True})
    monkeypatch.setattr(enrichment, "_openalex_work_to_discovered", lambda work: fake_paper)
    monkeypatch.setattr(enrichment, "query_arxiv_api", lambda ids: {})
    recs = fetch_records(p)
    oa = [r for r in recs if r.source == "openalex"]
    assert oa and oa[0].fields["abstract"] == "A" * 600


def test_openalex_record_tries_arxiv_doi_first(monkeypatch):
    # arXiv papers must be looked up via their arXiv DOI (reliable) before the
    # landing_page_url filter (which misses many).
    p = make_paper(arxiv_id="2407.14561")
    p.openalex_id = None
    seen = []
    def _fake(ident, by="id"):
        seen.append(ident)
        return None
    monkeypatch.setattr(enrichment, "_openalex_fetch_work", _fake)
    enrichment._openalex_record(p)
    assert "doi:10.48550/arXiv.2407.14561" in seen
    assert seen.index("doi:10.48550/arXiv.2407.14561") < \
        next(i for i, s in enumerate(seen) if "landing_page_url" in s)


def test_fetch_records_arxiv_authors(monkeypatch):
    p = make_paper(arxiv_id="2401.00002")
    monkeypatch.setattr(enrichment, "_openalex_fetch_work", lambda ident, by="id": None)
    monkeypatch.setattr(enrichment, "query_arxiv_api",
                        lambda ids: {"2401.00002": {"authors": ["A. One", "B. Two"], "affiliations": ["MIT"]}})
    recs = fetch_records(p)
    ax = [r for r in recs if r.source == "arxiv"]
    assert ax and ax[0].fields["authors"] == "A. One, B. Two"
