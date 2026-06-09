"""F-011 — the seed paper must never leak into the catalog / gate candidates.

deduplicate_papers used to exclude the seed only by exact title-set membership
OR exact arXiv match. A Google-Scholar variant of the seed arrives id-less, with
an empty abstract and a mangled title (a trailing ``. doi: 10.48550`` fragment),
so both guards missed it and it surfaced as a review-gate candidate. The guard is
now tolerant: normalized-title + doi + arxiv signals.
"""
from __future__ import annotations

from ndif_citations import config
from ndif_citations.discover import deduplicate_papers
from ndif_citations.models import DiscoverySource

from tests.conftest import make_paper

SEED_TITLE = "NNsight and NDIF: Democratizing Access to Open-Weight Foundation Model Internals"


def _titles(papers) -> list[str]:
    return [p.title for p in papers]


def test_idless_scholar_seed_variant_is_excluded():
    # The exact variant that leaked: Scholar source, no arxiv_id, empty abstract,
    # title with a trailing "doi: 10.48550" fragment.
    variant = make_paper(
        title=f"{SEED_TITLE}. doi: 10.48550",
        arxiv_id=None,
        doi=None,
        abstract="",
        source=DiscoverySource.SCHOLAR,
    )
    result = deduplicate_papers([variant])
    assert result == []


def test_seed_by_exact_arxiv_id_still_excluded():
    seed = make_paper(title="Some scraped title", arxiv_id=config.SEED_ARXIV_ID)
    assert deduplicate_papers([seed]) == []


def test_seed_by_exact_title_still_excluded():
    seed = make_paper(title=SEED_TITLE, arxiv_id=None, doi=None)
    assert deduplicate_papers([seed]) == []


def test_seed_by_doi_variant_excluded():
    # arXiv-as-DOI form, no arxiv_id set.
    seed = make_paper(
        title="garbled seed row",
        arxiv_id=None,
        doi=f"10.48550/arXiv.{config.SEED_ARXIV_ID}",
        source=DiscoverySource.SCHOLAR,
    )
    assert deduplicate_papers([seed]) == []


def test_legitimate_citing_paper_is_kept():
    # A real citing paper merely mentions NDIF — must NOT be swept up by the guard.
    paper = make_paper(
        title="Probing Language Models with NNsight: A Case Study",
        arxiv_id="2501.12345",
    )
    result = deduplicate_papers([paper])
    assert _titles(result) == ["Probing Language Models with NNsight: A Case Study"]
