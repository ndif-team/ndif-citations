"""Smoke tests verifying that the test-harness helpers work correctly.

These tests do NOT run the full pipeline — they just confirm that:
* fixture_state copies the mini fixtures into a writable temp tree
* fake_discover_papers / fake_discover_repos return the correct shapes
* install_pipeline_fakes patches the expected callables on a target module
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import ndif_citations.discover as discover_mod
from ndif_citations.models import (
    Category,
    Confidence,
    DiscoveredPaper,
    DiscoveredRepo,
)
from tests.helpers.fakes import (
    EXISTING_ARXIV_ID,
    fake_discover_papers,
    fake_discover_repos,
    install_pipeline_fakes,
)


# ---------------------------------------------------------------------------
# fixture_state
# ---------------------------------------------------------------------------

class TestFixtureState:
    def test_output_dir_created(self, fixture_state: Path):
        assert fixture_state.is_dir()
        assert fixture_state.name == "output"

    def test_subdirs_created(self, fixture_state: Path):
        assert (fixture_state / "images").is_dir()
        assert (fixture_state / "raw").is_dir()

    def test_papers_file_exists_and_is_valid(self, fixture_state: Path):
        papers_file = fixture_state / "research-papers-full.json"
        assert papers_file.exists()
        data = json.loads(papers_file.read_text())
        assert set(data.keys()) == {"pending", "verified", "discarded"}
        assert len(data["verified"]) == 3
        assert len(data["pending"]) == 1
        assert len(data["discarded"]) == 1

    def test_repos_file_exists_and_is_valid(self, fixture_state: Path):
        repos_file = fixture_state / "github-repos-full.json"
        assert repos_file.exists()
        data = json.loads(repos_file.read_text())
        assert isinstance(data, list)
        assert len(data) == 3

    def test_existing_arxiv_id_in_papers_fixture(self, fixture_state: Path):
        """EXISTING_ARXIV_ID in fakes.py must match a paper in the mini fixture."""
        data = json.loads((fixture_state / "research-papers-full.json").read_text())
        all_ids = [p.get("arxiv_id") for p in data["verified"]]
        assert EXISTING_ARXIV_ID in all_ids


# ---------------------------------------------------------------------------
# fake_discover_papers
# ---------------------------------------------------------------------------

class TestFakeDiscoverPapers:
    def test_returns_list_of_two(self):
        papers = fake_discover_papers()
        assert isinstance(papers, list)
        assert len(papers) == 2

    def test_all_are_discovered_paper_instances(self):
        for p in fake_discover_papers():
            assert isinstance(p, DiscoveredPaper)

    def test_first_paper_matches_existing_arxiv_id(self):
        papers = fake_discover_papers()
        assert papers[0].arxiv_id == EXISTING_ARXIV_ID

    def test_second_paper_is_genuinely_new(self):
        papers = fake_discover_papers()
        assert papers[1].arxiv_id != EXISTING_ARXIV_ID
        assert papers[1].arxiv_id == "9999.99999"

    def test_papers_have_required_fields(self):
        for p in fake_discover_papers():
            assert p.title
            assert p.abstract


# ---------------------------------------------------------------------------
# fake_discover_repos
# ---------------------------------------------------------------------------

class TestFakeDiscoverRepos:
    def test_returns_list_of_two(self):
        repos = fake_discover_repos()
        assert isinstance(repos, list)
        assert len(repos) == 2

    def test_all_are_discovered_repo_instances(self):
        for r in fake_discover_repos():
            assert isinstance(r, DiscoveredRepo)

    def test_repos_have_required_fields(self):
        for r in fake_discover_repos():
            assert r.owner
            assert r.repo
            assert r.url.startswith("https://github.com/")


# ---------------------------------------------------------------------------
# install_pipeline_fakes — patching on discover_mod
# ---------------------------------------------------------------------------

class TestInstallPipelineFakes:
    def test_discover_s2_citations_patched(self, monkeypatch):
        install_pipeline_fakes(monkeypatch, discover_mod)
        result = discover_mod.discover_s2_citations(raw_dir=None)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(p, DiscoveredPaper) for p in result)

    def test_discover_openalex_patched(self, monkeypatch):
        install_pipeline_fakes(monkeypatch, discover_mod)
        result = discover_mod.discover_openalex(raw_dir=None)
        assert isinstance(result, list)

    def test_discover_scholar_patched(self, monkeypatch):
        install_pipeline_fakes(monkeypatch, discover_mod)
        result = discover_mod.discover_scholar(raw_dir=None)
        assert isinstance(result, list)

    def test_discover_github_dependents_patched(self, monkeypatch):
        install_pipeline_fakes(monkeypatch, discover_mod)
        result = discover_mod.discover_github_dependents(raw_dir=None)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, DiscoveredRepo) for r in result)

    def test_enrich_papers_is_identity(self, monkeypatch, tmp_path):
        import ndif_citations.extract as extract_mod
        install_pipeline_fakes(monkeypatch, discover_mod)
        papers = fake_discover_papers()
        # enrich_papers lives in extract; fakes.py patches it there when
        # target_module doesn't expose the name.
        result = extract_mod.enrich_papers(papers, raw_dir=tmp_path)
        assert result is papers

    def test_enrich_repos_returns_tuple(self, monkeypatch):
        install_pipeline_fakes(monkeypatch, discover_mod)
        repos = fake_discover_repos()
        # enrich_repos_from_github_api IS on discover_mod — patched there directly.
        kept, counts = discover_mod.enrich_repos_from_github_api(repos)
        assert isinstance(kept, list)
        assert isinstance(counts, dict)
        assert set(counts.keys()) == {"404", "rename_redirect", "archived"}

    def test_process_generate_summary_patched(self, monkeypatch):
        import ndif_citations.process as process_mod
        install_pipeline_fakes(monkeypatch, discover_mod)
        paper = fake_discover_papers()[0]
        summary = process_mod.generate_summary(paper)
        assert isinstance(summary, str)
        assert paper.title in summary

    def test_process_classify_category_patched(self, monkeypatch, tmp_path):
        import ndif_citations.process as process_mod
        install_pipeline_fakes(monkeypatch, discover_mod)
        paper = fake_discover_papers()[0]
        cat, conf_float, band = process_mod.classify_category(paper, tmp_path)
        assert isinstance(cat, Category)
        assert isinstance(conf_float, float)
        assert isinstance(band, Confidence)

    def test_process_extract_thumbnail_patched(self, monkeypatch, tmp_path):
        import ndif_citations.process as process_mod
        install_pipeline_fakes(monkeypatch, discover_mod)
        paper = fake_discover_papers()[0]
        result = process_mod.extract_thumbnail(paper, tmp_path)
        assert result is None

    def test_pdf_cache_get_cached_pdf_patched(self, monkeypatch, tmp_path):
        import ndif_citations.pdf_cache as pdf_cache_mod
        install_pipeline_fakes(monkeypatch, discover_mod)
        paper = fake_discover_papers()[0]
        result = pdf_cache_mod.get_cached_pdf(paper, tmp_path)
        assert result is None
