"""Deterministic pipeline fakes for integration / parity tests.

Usage
-----
    import ndif_citations.discover as discover_mod
    install_pipeline_fakes(monkeypatch, discover_mod)

``install_pipeline_fakes`` patches all discovery, enrichment, and processing
functions **on the supplied target_module** so that pipeline tests never hit
the network.  The same function is also used by the orchestrator (Task 1.6)
when it is finally created — just call it with the orchestrator module instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ndif_citations.models import (
    Category,
    Confidence,
    DiscoveredPaper,
    DiscoveredRepo,
    DiscoverySource,
)


# ---------------------------------------------------------------------------
# Fixture arxiv_id that matches a paper already in mini-research-papers-full.json
# so that merge logic sees it as an existing record.
# ---------------------------------------------------------------------------
EXISTING_ARXIV_ID = "2602.16080"  # "Activation Steering via Generative Causal Mediation"


def fake_discover_papers() -> list[DiscoveredPaper]:
    """Return 2 deterministic DiscoveredPapers.

    * Paper 0: arxiv_id matches the first verified entry in mini-research-papers-full.json
      (exercises the "already seen" / update path).
    * Paper 1: genuinely new paper not present in any fixture.
    """
    existing = DiscoveredPaper(
        title="Activation Steering via Generative Causal Mediation",
        arxiv_id=EXISTING_ARXIV_ID,
        authors="Aruna Sankaranarayanan, Amir Zur, Atticus Geiger, Dylan Hadfield-Menell",
        venue="ICLR 2026",
        year=2026,
        abstract="Where should we intervene in a language model to localize and control behaviors. We use nnsight.",
        source=DiscoverySource.S2_CITATION,
        category=Category.USES_NNSIGHT,
    )
    new_paper = DiscoveredPaper(
        title="Fake New Paper for Test Harness",
        arxiv_id="9999.99999",
        authors="Test Author",
        venue="TestConf 2099",
        year=2099,
        abstract="A completely fabricated paper used only in the test harness. Uses nnsight.",
        source=DiscoverySource.S2_CITATION,
        category=Category.USES_NNSIGHT,
    )
    return [existing, new_paper]


def fake_discover_repos() -> list[DiscoveredRepo]:
    """Return 2 deterministic DiscoveredRepos."""
    repo_a = DiscoveredRepo(
        owner="callummcdougall",
        repo="ARENA_3.0",
        url="https://github.com/callummcdougall/ARENA_3.0",
        stars=1060,
        description=None,
        category=Category.USES_NNSIGHT,
    )
    repo_b = DiscoveredRepo(
        owner="fake-harness-owner",
        repo="fake-harness-repo",
        url="https://github.com/fake-harness-owner/fake-harness-repo",
        stars=42,
        description="Fake repo for the test harness",
        category=Category.USES_NNSIGHT,
    )
    return [repo_a, repo_b]


def install_pipeline_fakes(monkeypatch: Any, target_module: Any) -> None:
    """Patch all discovery, enrichment, and processing callables on *target_module*.

    This is intentionally generic so the same helper works for both the
    ``discover`` module (used by this smoke test) and the future orchestrator
    module (Task 1.6), which will re-export the same names.

    Patched on *target_module*
    --------------------------
    * discover_s2_citations(raw_dir)         -> list[DiscoveredPaper]
    * discover_openalex(raw_dir)             -> list[DiscoveredPaper]
    * discover_scholar(raw_dir, ...)         -> list[DiscoveredPaper]
    * discover_github_dependents(raw_dir)    -> list[DiscoveredRepo]
    * enrich_papers(papers, raw_dir=None)    -> list[DiscoveredPaper]  (identity)
    * enrich_repos_from_github_api(repos)    -> (list[DiscoveredRepo], dict[str,int])

    Patched on *ndif_citations.process*
    ------------------------------------
    * generate_summary(paper)                -> str
    * classify_category(paper, ...)          -> (Category, float, Confidence)
    * extract_thumbnail(paper, ...)          -> None
    * get_cached_pdf(paper, ...)             -> None  (from pdf_cache)
    """
    import ndif_citations.process as process_mod
    import ndif_citations.pdf_cache as pdf_cache_mod

    # -- Discovery fakes (patched onto the caller-supplied target module) ------

    def _fake_s2(raw_dir: Path | None = None) -> list[DiscoveredPaper]:
        return fake_discover_papers()

    def _fake_openalex(raw_dir: Path | None = None) -> list[DiscoveredPaper]:
        return []  # minimal — avoids duplicating results in parity tests

    def _fake_scholar(
        raw_dir: Path | None = None, force_refresh: bool = False
    ) -> list[DiscoveredPaper]:
        return []

    def _fake_github_dependents(raw_dir: Path | None = None) -> list[DiscoveredRepo]:
        return fake_discover_repos()

    def _fake_enrich_papers(
        papers: list[DiscoveredPaper], raw_dir: Path | None = None
    ) -> list[DiscoveredPaper]:
        return papers  # pass-through

    def _fake_enrich_repos(
        repos: list[DiscoveredRepo],
    ) -> tuple[list[DiscoveredRepo], dict[str, int]]:
        removal_counts: dict[str, int] = {"404": 0, "rename_redirect": 0, "archived": 0}
        return repos, removal_counts

    monkeypatch.setattr(target_module, "discover_s2_citations", _fake_s2)
    monkeypatch.setattr(target_module, "discover_openalex", _fake_openalex)
    monkeypatch.setattr(target_module, "discover_scholar", _fake_scholar)
    monkeypatch.setattr(target_module, "discover_github_dependents", _fake_github_dependents)

    # enrich_papers lives in ndif_citations.extract; patch target_module if it
    # re-exports the name (e.g. the future orchestrator), otherwise fall back
    # to the canonical home module so the call site is still patched.
    import ndif_citations.extract as extract_mod

    if hasattr(target_module, "enrich_papers"):
        monkeypatch.setattr(target_module, "enrich_papers", _fake_enrich_papers)
    else:
        monkeypatch.setattr(extract_mod, "enrich_papers", _fake_enrich_papers)

    # enrich_repos_from_github_api lives in ndif_citations.discover.
    import ndif_citations.discover as _discover_mod

    if hasattr(target_module, "enrich_repos_from_github_api"):
        monkeypatch.setattr(target_module, "enrich_repos_from_github_api", _fake_enrich_repos)
    else:
        monkeypatch.setattr(_discover_mod, "enrich_repos_from_github_api", _fake_enrich_repos)

    # -- Process fakes (always patched on the process module) ------------------

    def _fake_generate_summary(paper: DiscoveredPaper) -> str:
        return f"Fake summary for: {paper.title}"

    def _fake_classify_category(
        paper: DiscoveredPaper,
        output_dir: Path,
        pdf_path: Path | None = None,
    ) -> tuple[Category, float, Confidence]:
        return Category.USES_NNSIGHT, 0.85, Confidence.HIGH

    def _fake_extract_thumbnail(
        paper: DiscoveredPaper,
        output_dir: Path,
        pdf_path: Path | None = None,
    ) -> str | None:
        return None

    def _fake_get_cached_pdf(
        paper: DiscoveredPaper,
        output_dir: Path,
    ) -> Path | None:
        return None

    monkeypatch.setattr(process_mod, "generate_summary", _fake_generate_summary)
    monkeypatch.setattr(process_mod, "classify_category", _fake_classify_category)
    monkeypatch.setattr(process_mod, "extract_thumbnail", _fake_extract_thumbnail)
    monkeypatch.setattr(pdf_cache_mod, "get_cached_pdf", _fake_get_cached_pdf)
