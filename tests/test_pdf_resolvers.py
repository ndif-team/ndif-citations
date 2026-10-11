"""PDF resolvers (spec §4.4) — all HTTP mocked."""
import requests

from ndif_citations import pdf_cache, utils
from tests.conftest import make_paper

ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2509.08454v3</id>
<title>Behind the Scenes: Mechanistic
 Interpretability of LoRA</title></entry></feed>"""


class _R:
    def __init__(self, text, status=200):
        self.text, self.status_code = text, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def test_openreview_forum_url_becomes_pdf():
    p = make_paper(arxiv_id=None, url="https://openreview.net/forum?id=qErnjDVfdN")
    assert pdf_cache._derived_pdf_urls(p) == ["https://openreview.net/pdf?id=qErnjDVfdN"]


def test_acl_anthology_url_becomes_pdf():
    p = make_paper(arxiv_id=None, url="https://aclanthology.org/2024.acl-long.123/")
    assert pdf_cache._derived_pdf_urls(p) == ["https://aclanthology.org/2024.acl-long.123.pdf"]


def test_resolver_tries_derived_before_unpaywall(monkeypatch):
    p = make_paper(arxiv_id=None, doi="10.1/x", url="https://openreview.net/forum?id=abc")
    monkeypatch.setattr(pdf_cache, "_test_pdf_url", lambda u, timeout=10: u.endswith("pdf?id=abc"))
    monkeypatch.setattr(pdf_cache, "query_unpaywall",
                        lambda doi: (_ for _ in ()).throw(AssertionError("unpaywall must not run")))
    assert pdf_cache.resolve_pdf_url(p) == "https://openreview.net/pdf?id=abc"


def test_arxiv_title_search_matches(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **k: _R(ATOM))
    assert utils.find_arxiv_pdf_by_title(
        "Behind the Scenes: Mechanistic Interpretability of LoRA") == "https://arxiv.org/pdf/2509.08454v3"


def test_arxiv_title_search_rejects_weak_match(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **k: _R(ATOM))
    assert utils.find_arxiv_pdf_by_title("A Totally Different Paper Title") is None


def test_resolver_never_sets_arxiv_id(monkeypatch):
    p = make_paper(arxiv_id=None, doi=None, url="https://example.org/paper")
    monkeypatch.setattr(pdf_cache, "_test_pdf_url", lambda u, timeout=10: True)
    monkeypatch.setattr("ndif_citations.utils.find_arxiv_pdf_by_title",
                        lambda t: "https://arxiv.org/pdf/2509.08454v3")
    assert pdf_cache.resolve_pdf_url(p) == "https://arxiv.org/pdf/2509.08454v3"
    assert p.arxiv_id is None
