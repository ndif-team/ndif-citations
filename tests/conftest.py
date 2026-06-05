"""Shared fixtures and factory functions for the test suite."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

# Path to the mini fixture files shipped with the test suite.
_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_MINI_PAPERS = _FIXTURES_DIR / "mini-research-papers-full.json"
_MINI_REPOS = _FIXTURES_DIR / "mini-github-repos-full.json"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Suppress all rate_limit_sleep calls in every test — keeps the suite fast."""
    monkeypatch.setattr("ndif_citations.utils.time.sleep", lambda s: None)
    monkeypatch.setattr("ndif_citations.process.rate_limit_sleep", lambda s, label="": None)


@pytest.fixture()
def fixture_state(tmp_path: Path) -> Path:
    """Copy the mini fixture files into a fresh tmp_path/output tree.

    Creates the same subdirectory structure that the real pipeline uses:
      output/
        research-papers-full.json   (from mini-research-papers-full.json)
        github-repos-full.json      (from mini-github-repos-full.json)
        images/
        raw/

    Returns the ``output/`` Path so tests can pass it directly to pipeline functions.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "images").mkdir()
    (output_dir / "raw").mkdir()

    shutil.copy(_MINI_PAPERS, output_dir / "research-papers-full.json")
    shutil.copy(_MINI_REPOS, output_dir / "github-repos-full.json")

    return output_dir


from ndif_citations.models import (
    Bucket,
    Category,
    DiscoveredPaper,
    DiscoveredRepo,
    DiscoverySource,
)


def make_paper(
    title: str = "Test Paper",
    arxiv_id: str | None = "2401.00001",
    doi: str | None = None,
    venue: str = "arXiv 2024",
    abstract: str | None = "A test abstract for the unit test suite.",
    description: str = "",
    category: Category = Category.REFERENCING,
    bucket: Bucket = Bucket.VERIFIED,
    category_confidence: float = 0.0,
    manual_override: bool = False,
    has_summary: bool = False,
    has_classification: bool = False,
    has_thumbnail: bool = False,
    has_affiliations: bool = False,
    content_hash: str = "",
    source: DiscoverySource = DiscoverySource.S2_CITATION,
    **kwargs,
) -> DiscoveredPaper:
    """Factory for DiscoveredPaper with sensible defaults."""
    paper = DiscoveredPaper(
        title=title,
        arxiv_id=arxiv_id,
        doi=doi,
        venue=venue,
        abstract=abstract,
        description=description,
        category=category,
        bucket=bucket,
        category_confidence=category_confidence,
        manual_override=manual_override,
        has_summary=has_summary,
        has_classification=has_classification,
        has_thumbnail=has_thumbnail,
        has_affiliations=has_affiliations,
        source=source,
        **kwargs,
    )
    if content_hash:
        paper.content_hash = content_hash
    return paper


def make_repo(
    owner: str = "test-owner",
    repo: str = "test-repo",
    stars: int | None = 0,
    description: str | None = None,
    category: Category = Category.USES_NNSIGHT,
    manual_override: bool = False,
    has_classification: bool = True,
    content_hash: str = "",
    parent_full_name: str | None = None,
    linked_paper_url: str | None = None,
    linked_paper_tier: int | None = None,
    **kwargs,
) -> DiscoveredRepo:
    """Factory for DiscoveredRepo with sensible defaults."""
    r = DiscoveredRepo(
        owner=owner,
        repo=repo,
        url=f"https://github.com/{owner}/{repo}",
        stars=stars,
        description=description,
        category=category,
        manual_override=manual_override,
        has_classification=has_classification,
        parent_full_name=parent_full_name,
        linked_paper_url=linked_paper_url,
        linked_paper_tier=linked_paper_tier,
        **kwargs,
    )
    if content_hash:
        r.content_hash = content_hash
    return r
