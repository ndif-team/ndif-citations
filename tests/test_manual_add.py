"""Tests for ``ndif_citations.manual_add.add_paper_by_url`` (Task 4.5 — Part A).

What is faked / why
-------------------
* ``semanticscholar.SemanticScholar`` — patched so the S2 metadata-lookup block
  does not hit the network.
* ``ndif_citations.extract.enrich_papers`` — patched to identity (pass-through)
  so the enrichment phase doesn't need network or a PDF cache.
* ``generate_summary``, ``classify_category``, ``extract_thumbnail`` on
  ``ndif_citations.process``, and ``get_cached_pdf`` on ``ndif_citations.pdf_cache``
  — patched so LLM / PDF steps are skipped.

Test cases
----------
1. Adding a genuinely new arXiv URL returns ``{added: True, ...}`` and the paper
   appears in ``research-papers-full.json``.
2. Adding an arXiv ID that already exists in the fixture (``2602.16080``) returns
   ``{added: False, ...}`` — duplicate suppressed by merge logic.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from ndif_citations.models import Category, Confidence


# ---------------------------------------------------------------------------
# Shared fake helpers (mirrors the pattern in test_cli_add.py)
# ---------------------------------------------------------------------------

_NEW_ARXIV_ID = "9999.00002"
_NEW_URL = f"https://arxiv.org/abs/{_NEW_ARXIV_ID}"

# An arXiv ID that IS already in mini-research-papers-full.json
_EXISTING_ARXIV_ID = "2602.16080"
_EXISTING_URL = f"https://arxiv.org/abs/{_EXISTING_ARXIV_ID}"


def _make_fake_s2_paper(arxiv_id: str = _NEW_ARXIV_ID) -> Any:
    """Return a mock that mimics a SemanticScholar paper object."""
    obj = MagicMock()
    obj.title = f"Fake Paper {arxiv_id}"
    obj.authors = []
    obj.abstract = f"Abstract for {arxiv_id}."
    obj.venue = "TestConf 2099"
    obj.publicationDate = "2099-01-01"
    obj.externalIds = {}
    obj.paperId = f"fake-s2-{arxiv_id}"
    obj.openAccessPdf = None
    return obj


def _install_s2_fake(monkeypatch: Any, arxiv_id: str = _NEW_ARXIV_ID) -> None:
    import semanticscholar

    fake_sch = MagicMock()
    fake_sch.get_paper.return_value = _make_fake_s2_paper(arxiv_id)
    monkeypatch.setattr(semanticscholar, "SemanticScholar", lambda **kw: fake_sch)


def _install_enrich_fake(monkeypatch: Any) -> None:
    import ndif_citations.extract as extract_mod

    monkeypatch.setattr(
        extract_mod, "enrich_papers",
        lambda papers, raw_dir=None: papers,
    )


def _install_process_fakes(monkeypatch: Any) -> None:
    import ndif_citations.pdf_cache as pdf_cache_mod
    import ndif_citations.process as process_mod

    monkeypatch.setattr(
        process_mod, "generate_summary",
        lambda paper: "Fake summary for manual_add test.",
    )
    monkeypatch.setattr(
        process_mod, "classify_category",
        lambda paper, output_dir, pdf_path=None: (Category.USES_NNSIGHT, 0.85, Confidence.HIGH),
    )
    monkeypatch.setattr(
        process_mod, "extract_thumbnail",
        lambda paper, output_dir, pdf_path=None: None,
    )
    monkeypatch.setattr(
        pdf_cache_mod, "get_cached_pdf",
        lambda paper, output_dir: None,
    )


def _all_fakes(monkeypatch: Any, arxiv_id: str = _NEW_ARXIV_ID) -> None:
    _install_s2_fake(monkeypatch, arxiv_id)
    _install_enrich_fake(monkeypatch)
    _install_process_fakes(monkeypatch)


# ---------------------------------------------------------------------------
# Test 1 — new paper → added=True, appears in output JSON
# ---------------------------------------------------------------------------

def test_add_new_paper_returns_added_true(monkeypatch, fixture_state):
    """add_paper_by_url with a genuinely new arXiv ID returns added=True."""
    _all_fakes(monkeypatch)

    from ndif_citations.manual_add import add_paper_by_url

    result = add_paper_by_url(fixture_state, _NEW_URL)

    assert isinstance(result, dict), "add_paper_by_url must return a dict"
    assert result["added"] is True, f"expected added=True, got {result}"
    assert result["new_papers"] > 0, f"expected new_papers>0, got {result}"
    assert result["merge_key"], "merge_key should be non-empty"
    assert result["title"], "title should be non-empty"

    # Verify the paper appears in the output JSON
    full_json = fixture_state / "research-papers-full.json"
    assert full_json.exists(), "research-papers-full.json must be written"

    data = json.loads(full_json.read_text())
    all_papers = data["pending"] + data["verified"] + data["discarded"]
    arxiv_ids = [p.get("arxiv_id") for p in all_papers]
    assert _NEW_ARXIV_ID in arxiv_ids, (
        f"arxiv_id={_NEW_ARXIV_ID!r} not found in output; got: {arxiv_ids}"
    )


# ---------------------------------------------------------------------------
# Test 2 — already-present paper → added=False
# ---------------------------------------------------------------------------

def test_add_existing_paper_returns_added_false(monkeypatch, fixture_state):
    """add_paper_by_url with an already-present arXiv ID returns added=False."""
    _all_fakes(monkeypatch, arxiv_id=_EXISTING_ARXIV_ID)

    from ndif_citations.manual_add import add_paper_by_url

    result = add_paper_by_url(fixture_state, _EXISTING_URL)

    assert isinstance(result, dict), "add_paper_by_url must return a dict"
    assert result["added"] is False, f"expected added=False for existing paper, got {result}"
    assert result["new_papers"] == 0, f"expected new_papers=0, got {result}"


# ---------------------------------------------------------------------------
# Test 3 — cancel_check is forwarded to process_papers
# ---------------------------------------------------------------------------

def test_add_paper_cancel_check_forwarded(monkeypatch, fixture_state):
    """cancel_check passed to add_paper_by_url is forwarded to process_papers."""
    _all_fakes(monkeypatch)

    import ndif_citations.process as process_mod

    received_cancel_check = []

    original_process = process_mod.process_papers

    def _spy(decisions, output_dir, **kwargs):
        received_cancel_check.append(kwargs.get("cancel_check"))
        return original_process(decisions, output_dir, **kwargs)

    monkeypatch.setattr(process_mod, "process_papers", _spy)

    from ndif_citations.manual_add import add_paper_by_url

    sentinel = lambda: False  # noqa: E731
    add_paper_by_url(fixture_state, _NEW_URL, cancel_check=sentinel)

    assert len(received_cancel_check) == 1
    assert received_cancel_check[0] is sentinel, (
        "cancel_check was not forwarded to process_papers"
    )
