from __future__ import annotations

from ndif_citations.manual_add import seed_from_url, seed_from_pdf
from ndif_citations.models import DiscoverySource

def test_seed_from_url_extracts_arxiv():
    p = seed_from_url("https://arxiv.org/abs/2401.12345")
    assert p.arxiv_id == "2401.12345"
    assert p.url == "https://arxiv.org/abs/2401.12345"
    assert p.source == DiscoverySource.MANUAL_ADD

def test_seed_from_pdf_uses_provided_fields():
    p = seed_from_pdf(title="A Paywalled Paper", arxiv_id=None, doi="10.1/x")
    assert p.title == "A Paywalled Paper"
    assert p.doi == "10.1/x"
    assert p.source == DiscoverySource.MANUAL_ADD

def test_seed_from_url_non_arxiv_url_has_no_arxiv():
    p = seed_from_url("https://openreview.net/forum?id=abc123")
    assert p.arxiv_id is None
    assert p.url == "https://openreview.net/forum?id=abc123"
    assert p.source == DiscoverySource.MANUAL_ADD

def test_seed_from_pdf_blank_ids_collapse_to_none():
    p = seed_from_pdf(title="T", arxiv_id="", doi="")
    assert p.arxiv_id is None
    assert p.doi is None
