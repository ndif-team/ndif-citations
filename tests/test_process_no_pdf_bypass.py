"""Integration tests for process_papers no-PDF bypass fix (US-B2)."""
import pytest

from ndif_citations.models import Category
from ndif_citations.process import process_papers
from ndif_citations.router import ProcessingBucket, RoutingDecision
from tests.conftest import make_paper
from tests.helpers.llm import MockLLMClient


def _make_classify_decision(paper):
    return RoutingDecision(
        paper=paper,
        bucket=ProcessingBucket.NEW,
        existing_paper=None,
        processing_needed={"summary": False, "classify": True, "thumbnail": False, "affiliations": False},
    )


def _inject_mock(monkeypatch, mock):
    monkeypatch.setattr("ndif_citations.process._get_llm_client", lambda: mock)


def test_no_pdf_abstract_with_keywords_reaches_llm(monkeypatch, tmp_path):
    """Paper with no PDF but keywords in abstract must not be silently bypassed to UNCLASSIFIED."""
    mock = MockLLMClient()
    mock.expect("uses_nnsight")
    _inject_mock(monkeypatch, mock)
    monkeypatch.setattr(
        "ndif_citations.pdf_cache.get_cached_pdf",
        lambda paper, output_dir: None,
    )

    paper = make_paper(abstract="We used import nnsight to extract activations.")
    decision = _make_classify_decision(paper)

    results = process_papers([decision], tmp_path, skip_llm=False)

    assert len(results) == 1
    result = results[0]
    assert result.category == Category.USES_NNSIGHT
    assert result.category != Category.UNCLASSIFIED
    assert len(mock.record_calls()) == 1


def test_no_pdf_no_keywords_in_abstract_returns_unclassified_with_reason(monkeypatch, tmp_path):
    """Paper with no PDF and no keywords in abstract must be UNCLASSIFIED with reason set."""
    mock = MockLLMClient()
    _inject_mock(monkeypatch, mock)
    monkeypatch.setattr(
        "ndif_citations.pdf_cache.get_cached_pdf",
        lambda paper, output_dir: None,
    )

    paper = make_paper(abstract="A paper about transformers.")
    decision = _make_classify_decision(paper)

    results = process_papers([decision], tmp_path, skip_llm=False)

    assert len(results) == 1
    result = results[0]
    assert result.category == Category.UNCLASSIFIED
    assert result.unclassified_reason == "no_keywords_anywhere"
    mock.assert_no_calls()
