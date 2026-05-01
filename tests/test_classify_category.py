"""Tests for classify_category (US-F4, US-F6, US-F7)."""
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ndif_citations.models import DetailCategory, DiscoveredPaper
from ndif_citations.process import classify_category
from tests.conftest import make_paper
from tests.helpers.llm import MockLLMClient

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "papers"
FAKE_OUTPUT = Path("/tmp/ndif_test_output")


def _inject_mock(monkeypatch, mock: MockLLMClient):
    """Replace _get_llm_client so classify_category uses our mock."""
    monkeypatch.setattr("ndif_citations.process._get_llm_client", lambda: mock)


def _setup_pdf_context(monkeypatch, tmp_path: Path, context_text: str) -> Path:
    """Create a fake PDF file on disk and patch extract_ndif_context to return context_text.

    classify_category only calls extract_ndif_context when pdf_path.exists() is True,
    so we need both a real path and a patched extractor.
    """
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")  # minimal content so .exists() is True
    monkeypatch.setattr(
        "ndif_citations.process.extract_ndif_context",
        lambda path, window=500: context_text,
    )
    return pdf


# ---------------------------------------------------------------------------
# Category round-trips via mock LLM
# ---------------------------------------------------------------------------

class TestClassifyCategoryMockLLM:
    def test_uses_ndif_reply(self, monkeypatch, tmp_path):
        mock = MockLLMClient()
        mock.expect("uses_ndif")
        _inject_mock(monkeypatch, mock)
        pdf = _setup_pdf_context(monkeypatch, tmp_path, "We ran experiments on NDIF cluster.")
        paper = make_paper()
        cat, conf = classify_category(paper, FAKE_OUTPUT, pdf_path=pdf)
        assert cat == DetailCategory.USES_NDIF
        assert conf == pytest.approx(0.85)

    def test_uses_nnsight_reply(self, monkeypatch, tmp_path):
        mock = MockLLMClient()
        mock.expect("uses_nnsight")
        _inject_mock(monkeypatch, mock)
        pdf = _setup_pdf_context(monkeypatch, tmp_path, "import nnsight; model = nnsight.LanguageModel(...)")
        paper = make_paper()
        cat, conf = classify_category(paper, FAKE_OUTPUT, pdf_path=pdf)
        assert cat == DetailCategory.USES_NNSIGHT
        assert conf == pytest.approx(0.85)

    def test_referencing_reply(self, monkeypatch, tmp_path):
        mock = MockLLMClient()
        mock.expect("referencing")
        _inject_mock(monkeypatch, mock)
        pdf = _setup_pdf_context(monkeypatch, tmp_path, "NNsight is listed in related work.")
        paper = make_paper()
        cat, conf = classify_category(paper, FAKE_OUTPUT, pdf_path=pdf)
        assert cat == DetailCategory.REFERENCING
        assert conf == pytest.approx(0.85)

    def test_unclassified_reply(self, monkeypatch, tmp_path):
        mock = MockLLMClient()
        mock.expect("unclassified")
        _inject_mock(monkeypatch, mock)
        pdf = _setup_pdf_context(monkeypatch, tmp_path, "nnsight is mentioned somewhere.")
        paper = make_paper()
        cat, conf = classify_category(paper, FAKE_OUTPUT, pdf_path=pdf)
        assert cat == DetailCategory.UNCLASSIFIED
        assert conf == pytest.approx(0.0)
        assert paper.unclassified_reason == "llm_returned_unclassified"

    def test_garbage_reply_returns_unclassified_with_unparseable_reason(self, monkeypatch, tmp_path):
        mock = MockLLMClient()
        mock.expect("yes, definitely!")
        _inject_mock(monkeypatch, mock)
        pdf = _setup_pdf_context(monkeypatch, tmp_path, "nnsight in methods section.")
        paper = make_paper()
        cat, conf = classify_category(paper, FAKE_OUTPUT, pdf_path=pdf)
        assert cat == DetailCategory.UNCLASSIFIED
        assert conf == pytest.approx(0.0)
        assert paper.unclassified_reason == "llm_unparseable"

    def test_prompt_contains_fixture_snippet(self, monkeypatch, tmp_path):
        mock = MockLLMClient()
        mock.expect("uses_ndif")
        _inject_mock(monkeypatch, mock)
        context = "We used NDIF cluster for all inference."
        pdf = _setup_pdf_context(monkeypatch, tmp_path, context)
        paper = make_paper(title="My Test Paper")
        classify_category(paper, FAKE_OUTPUT, pdf_path=pdf)
        calls = mock.record_calls()
        assert len(calls) == 1
        user_msg = calls[0][-1]["content"]  # last message is the user message
        assert "My Test Paper" in user_msg
        assert "NDIF cluster" in user_msg


# ---------------------------------------------------------------------------
# Early-return paths (no LLM called)  — US-F7
# ---------------------------------------------------------------------------

class TestClassifyCategoryEarlyReturn:
    def test_no_pdf_no_abstract_no_llm_call(self, monkeypatch):
        mock = MockLLMClient()
        _inject_mock(monkeypatch, mock)
        paper = make_paper(abstract=None)
        # No pdf_path passed → context = "" → enters early-return branch
        cat, conf = classify_category(paper, FAKE_OUTPUT, pdf_path=None)
        assert cat == DetailCategory.UNCLASSIFIED
        assert conf == pytest.approx(0.0)
        assert paper.unclassified_reason == "no_evidence_extractable"
        mock.assert_no_calls()

    def test_abstract_present_but_no_keywords_no_llm_call(self, monkeypatch, tmp_path):
        mock = MockLLMClient()
        _inject_mock(monkeypatch, mock)
        pdf = _setup_pdf_context(monkeypatch, tmp_path,
                                 "No direct mentions of NDIF or nnsight found in the paper text.")
        paper = make_paper(abstract="A paper about transformers and attention.")
        cat, conf = classify_category(paper, FAKE_OUTPUT, pdf_path=pdf)
        assert cat == DetailCategory.UNCLASSIFIED
        assert conf == pytest.approx(0.0)
        assert paper.unclassified_reason == "no_keywords_anywhere"
        mock.assert_no_calls()

    def test_pdf_no_mentions_and_no_abstract_is_no_evidence(self, monkeypatch, tmp_path):
        mock = MockLLMClient()
        _inject_mock(monkeypatch, mock)
        pdf = _setup_pdf_context(monkeypatch, tmp_path,
                                 "No direct mentions of NDIF or nnsight found in the paper text.")
        paper = make_paper(abstract=None)
        cat, conf = classify_category(paper, FAKE_OUTPUT, pdf_path=pdf)
        assert cat == DetailCategory.UNCLASSIFIED
        assert paper.unclassified_reason == "no_evidence_extractable"
        mock.assert_no_calls()


# ---------------------------------------------------------------------------
# LLM_API_KEY unset path — falls back to keyword rules at confidence 0.4
# ---------------------------------------------------------------------------

class TestClassifyCategoryFallback:
    def test_no_client_uses_keyword_rules(self, monkeypatch, tmp_path):
        monkeypatch.setattr("ndif_citations.process._get_llm_client", lambda: None)
        pdf = _setup_pdf_context(monkeypatch, tmp_path, "We hosted models on ndif cluster.")
        paper = make_paper()
        cat, conf = classify_category(paper, FAKE_OUTPUT, pdf_path=pdf)
        assert cat == DetailCategory.USES_NDIF
        assert conf == pytest.approx(0.4)

    def test_no_client_nnsight_fallback(self, monkeypatch, tmp_path):
        monkeypatch.setattr("ndif_citations.process._get_llm_client", lambda: None)
        pdf = _setup_pdf_context(monkeypatch, tmp_path, "import nnsight; nnsight.trace(model)")
        paper = make_paper()
        cat, conf = classify_category(paper, FAKE_OUTPUT, pdf_path=pdf)
        assert cat == DetailCategory.USES_NNSIGHT
        assert conf == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# DEBUG log assertions — US-F6
# ---------------------------------------------------------------------------

class TestClassifyCategoryDebugLogs:
    def test_context_source_logged(self, monkeypatch, caplog, tmp_path):
        mock = MockLLMClient()
        mock.expect("referencing")
        _inject_mock(monkeypatch, mock)
        pdf = _setup_pdf_context(monkeypatch, tmp_path, "NNsight is mentioned in related work.")
        paper = make_paper()
        with caplog.at_level(logging.DEBUG, logger="ndif_citations.process"):
            classify_category(paper, FAKE_OUTPUT, pdf_path=pdf)
        assert any("context_source" in r.message for r in caplog.records)

    def test_llm_raw_reply_logged(self, monkeypatch, caplog, tmp_path):
        mock = MockLLMClient()
        mock.expect("referencing")
        _inject_mock(monkeypatch, mock)
        pdf = _setup_pdf_context(monkeypatch, tmp_path, "nnsight mentioned.")
        paper = make_paper()
        with caplog.at_level(logging.DEBUG, logger="ndif_citations.process"):
            classify_category(paper, FAKE_OUTPUT, pdf_path=pdf)
        assert any("referencing" in r.message for r in caplog.records)

    def test_no_client_path_logged(self, monkeypatch, caplog, tmp_path):
        monkeypatch.setattr("ndif_citations.process._get_llm_client", lambda: None)
        pdf = _setup_pdf_context(monkeypatch, tmp_path, "import nnsight.")
        paper = make_paper()
        with caplog.at_level(logging.DEBUG, logger="ndif_citations.process"):
            classify_category(paper, FAKE_OUTPUT, pdf_path=pdf)
        assert any("fallback" in r.message.lower() for r in caplog.records)
