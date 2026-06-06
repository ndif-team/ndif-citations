from ndif_citations import enrichment
from ndif_citations.enrichment import title_similarity, resolve_identifiers
from tests.conftest import make_paper


def test_title_similarity_threshold():
    assert title_similarity("Attention Is All You Need", "attention is all you need!") >= 0.90
    assert title_similarity("Attention Is All You Need", "A totally different paper") < 0.90


def test_resolve_parses_arxiv_from_url(monkeypatch):
    p = make_paper(arxiv_id=None, doi=None, url="https://arxiv.org/abs/2401.12345")
    result = resolve_identifiers(p)
    assert result.resolved is True and result.via_title is False
    assert p.arxiv_id == "2401.12345"


def test_resolve_adopts_openalex_id_on_high_title_match(monkeypatch):
    p = make_paper(arxiv_id=None, doi=None, url="https://example.com/x", title="Sparse Probing of LLMs")
    work = {"id": "https://openalex.org/W123", "title": "Sparse Probing of LLMs",
            "ids": {"doi": "https://doi.org/10.1/abc"}}
    monkeypatch.setattr(enrichment, "_openalex_fetch_work", lambda ident, by="id": work)
    result = resolve_identifiers(p)
    assert result.resolved and result.via_title is True
    assert p.openalex_id == "https://openalex.org/W123" and p.doi == "10.1/abc"


def test_resolve_rejects_low_title_match(monkeypatch):
    p = make_paper(arxiv_id=None, doi=None, url="https://example.com/x", title="Sparse Probing of LLMs")
    work = {"id": "https://openalex.org/W999", "title": "Unrelated Physics Paper", "ids": {}}
    monkeypatch.setattr(enrichment, "_openalex_fetch_work", lambda ident, by="id": work)
    result = resolve_identifiers(p)
    assert result.resolved is False and (p.openalex_id in (None, ""))
