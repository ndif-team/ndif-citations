"""Tests for the `add` CLI command (Task 1.9 — bug fix regression guard).

Bug fixed: `add()` was calling `process_papers(papers, out)` with
`list[DiscoveredPaper]` instead of `list[RoutingDecision]`, causing a crash.

The two tests here:
  1. Regression guard — asserts `process_papers` receives `list[RoutingDecision]`.
  2. Integration — asserts `research-papers-full.json` exists and contains
     the added paper after a successful `add` invocation.

What is faked / why
-------------------
* ``semanticscholar.SemanticScholar`` — patched to return a fake S2 paper so
  the S2 metadata-lookup block in `add()` doesn't hit the network.
* ``ndif_citations.extract.enrich_papers`` — patched to identity (pass-through)
  so the enrichment phase doesn't need an arXiv or PDF cache.
* ``generate_summary``, ``classify_category``, ``extract_thumbnail`` on
  ``ndif_citations.process`` — patched via ``install_pipeline_fakes`` on a
  throw-away module object so the LLM / PDF steps are skipped.
* ``get_cached_pdf`` on ``ndif_citations.pdf_cache`` — patched to return None
  (no PDF available) so the process loop doesn't try to read a disk file.

`add()` uses *lazy local imports* for its dependencies (matching the rest of
cli.py).  The patches must therefore be applied to the **source modules**, not
to `cli`, because the local ``from ndif_citations.X import Y`` binds names
from the source module at call time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from ndif_citations.cli import cli
from ndif_citations.models import Category, Confidence, DiscoveredPaper, DiscoverySource
from ndif_citations.router import RoutingDecision
from tests.helpers.fakes import install_pipeline_fakes


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TEST_ARXIV_ID = "9999.00001"
_TEST_URL = f"https://arxiv.org/abs/{_TEST_ARXIV_ID}"


def _make_fake_s2_paper() -> Any:
    """Return a mock that mimics the SemanticScholar paper object."""
    obj = MagicMock()
    obj.title = "Fake Added Paper"
    obj.authors = []           # no authors — simplest valid value
    obj.abstract = "A test abstract."
    obj.venue = "TestConf 2099"
    obj.publicationDate = "2099-01-01"
    obj.externalIds = {}
    obj.paperId = "fake-s2-id"
    obj.openAccessPdf = None
    return obj


def _install_s2_fake(monkeypatch: Any) -> None:
    """Patch SemanticScholar so the S2 lookup block in add() doesn't hit the network."""
    import semanticscholar

    fake_sch = MagicMock()
    fake_sch.get_paper.return_value = _make_fake_s2_paper()
    monkeypatch.setattr(semanticscholar, "SemanticScholar", lambda **kw: fake_sch)


def _install_process_fakes(monkeypatch: Any) -> None:
    """Patch LLM / PDF helpers on their home modules so process_papers succeeds."""
    import ndif_citations.process as process_mod
    import ndif_citations.pdf_cache as pdf_cache_mod

    monkeypatch.setattr(
        process_mod, "generate_summary",
        lambda paper: "Fake summary for add test."
    )
    monkeypatch.setattr(
        process_mod, "classify_category",
        lambda paper, output_dir, pdf_path=None: (Category.USES_NNSIGHT, 0.85, Confidence.HIGH)
    )
    monkeypatch.setattr(
        process_mod, "extract_thumbnail",
        lambda paper, output_dir, pdf_path=None: None
    )
    monkeypatch.setattr(
        pdf_cache_mod, "get_cached_pdf",
        lambda paper, output_dir: None
    )


def _install_enrich_fake(monkeypatch: Any) -> None:
    """Patch enrich_papers to identity (pass-through) on its home module."""
    import ndif_citations.extract as extract_mod

    monkeypatch.setattr(
        extract_mod, "enrich_papers",
        lambda papers, raw_dir=None: papers,
    )


# ---------------------------------------------------------------------------
# Test 1 — regression guard: process_papers receives list[RoutingDecision]
# ---------------------------------------------------------------------------

def test_add_routes_before_processing(monkeypatch, tmp_path):
    """process_papers must receive list[RoutingDecision], not list[DiscoveredPaper].

    This is the regression guard for the bug fixed in Task 1.9.
    """
    _install_s2_fake(monkeypatch)
    _install_enrich_fake(monkeypatch)
    _install_process_fakes(monkeypatch)

    # Spy on process_papers: record the type of the first positional arg,
    # then delegate to a minimal implementation that returns processed papers.
    import ndif_citations.process as process_mod

    received_first_arg: list[Any] = []

    original_process_papers = process_mod.process_papers

    def _spy_process_papers(decisions, output_dir, **kwargs):
        received_first_arg.append(decisions)
        # Return processed DiscoveredPaper objects (the real return type)
        return [d.paper for d in decisions]

    monkeypatch.setattr(process_mod, "process_papers", _spy_process_papers)

    out = tmp_path / "output"
    out.mkdir()
    (out / "images").mkdir()
    (out / "pdfs").mkdir()

    runner = CliRunner()
    result = runner.invoke(cli, ["add", _TEST_URL, "-o", str(out)], catch_exceptions=False)

    assert result.exit_code == 0, result.output

    # Spy was called exactly once
    assert len(received_first_arg) == 1, "process_papers should be called exactly once"

    decisions_arg = received_first_arg[0]
    assert isinstance(decisions_arg, list), "process_papers first arg must be a list"
    assert len(decisions_arg) >= 1, "decisions list must be non-empty"
    assert isinstance(
        decisions_arg[0], RoutingDecision
    ), (
        f"process_papers must receive list[RoutingDecision], "
        f"got {type(decisions_arg[0]).__name__}"
    )


# ---------------------------------------------------------------------------
# Test 2 — integration: output file exists and contains the added paper
# ---------------------------------------------------------------------------

def test_add_writes_paper_to_output(monkeypatch, tmp_path):
    """After `add <url>`, research-papers-full.json exists and contains the paper."""
    _install_s2_fake(monkeypatch)
    _install_enrich_fake(monkeypatch)
    _install_process_fakes(monkeypatch)

    out = tmp_path / "output"
    out.mkdir()
    (out / "images").mkdir()
    (out / "pdfs").mkdir()

    runner = CliRunner()
    result = runner.invoke(cli, ["add", _TEST_URL, "-o", str(out)], catch_exceptions=False)

    assert result.exit_code == 0, result.output

    full_json = out / "research-papers-full.json"
    assert full_json.exists(), "research-papers-full.json must be written by add"

    data = json.loads(full_json.read_text())
    # 3-bucket structure
    assert "pending" in data and "verified" in data and "discarded" in data, (
        "Output must use 3-bucket structure"
    )

    all_papers = data["pending"] + data["verified"] + data["discarded"]
    assert len(all_papers) >= 1, "At least one paper must be in the output"

    # The added paper should be identifiable by its arxiv_id
    arxiv_ids = [p.get("arxiv_id") for p in all_papers]
    assert _TEST_ARXIV_ID in arxiv_ids, (
        f"arxiv_id={_TEST_ARXIV_ID!r} not found in output; got: {arxiv_ids}"
    )
