"""Tests for _fallback_classification and _fallback_summary (process.py)."""
from ndif_citations.models import Category
from ndif_citations.process import _fallback_classification, _fallback_summary


class TestFallbackClassification:
    def test_ndif_cluster_signal(self):
        assert _fallback_classification("experiments hosted on ndif cluster") == Category.USES_NDIF

    def test_ndif_url_signal(self):
        assert _fallback_classification("we used ndif.us for inference") == Category.USES_NDIF

    def test_ndif_infrastructure_signal(self):
        assert _fallback_classification("leveraging ndif infrastructure") == Category.USES_NDIF

    def test_import_nnsight_signal(self):
        assert _fallback_classification("import nnsight from the library") == Category.USES_NNSIGHT

    def test_nnsight_trace_signal(self):
        assert _fallback_classification("using nnsight.trace to hook") == Category.USES_NNSIGHT

    def test_from_nnsight_signal(self):
        assert _fallback_classification("from nnsight import Tracer") == Category.USES_NNSIGHT

    def test_bare_mention_falls_back_to_referencing(self):
        # "nnsight" mentioned but no strong signal phrase → REFERENCING
        assert _fallback_classification("we compare against nnsight-based approaches") == Category.REFERENCING

    def test_ndif_bare_mention_falls_back_to_referencing(self):
        assert _fallback_classification("NDIF is a related project") == Category.REFERENCING

    def test_empty_context(self):
        assert _fallback_classification("") == Category.REFERENCING

    def test_case_insensitive_ndif(self):
        assert _fallback_classification("HOSTED ON NDIF CLUSTER") == Category.USES_NDIF

    def test_case_insensitive_nnsight(self):
        assert _fallback_classification("IMPORT NNSIGHT") == Category.USES_NNSIGHT


class TestFallbackSummary:
    def test_returns_first_two_sentences(self):
        abstract = "First sentence. Second sentence. Third sentence."
        result = _fallback_summary(abstract)
        assert "First sentence." in result
        assert "Second sentence." in result
        assert "Third sentence." not in result

    def test_single_sentence(self):
        result = _fallback_summary("Only one sentence.")
        assert result == "Only one sentence."

    def test_joined_with_space(self):
        result = _fallback_summary("Sentence one. Sentence two.")
        assert result == "Sentence one. Sentence two."

    def test_strips_whitespace(self):
        result = _fallback_summary("  Padded sentence.  Second one.  ")
        assert not result.startswith(" ")

    def test_empty_string(self):
        result = _fallback_summary("")
        assert result == ""


# ---------------------------------------------------------------------------
# generate_summary LLM gate (no bad-abstract LLM calls)
# ---------------------------------------------------------------------------

class TestGenerateSummaryGate:
    def _inject_mock(self, monkeypatch, mock):
        monkeypatch.setattr("ndif_citations.process._get_llm_client", lambda: mock)

    def test_none_abstract_skips_llm_logs_warning(self, monkeypatch, caplog):
        import logging
        from ndif_citations.process import generate_summary
        from tests.helpers.llm import MockLLMClient
        from tests.conftest import make_paper
        mock = MockLLMClient()
        self._inject_mock(monkeypatch, mock)
        paper = make_paper(abstract=None)
        with caplog.at_level(logging.WARNING, logger="ndif_citations.process"):
            result = generate_summary(paper)
        assert result == ""
        mock.assert_no_calls()
        assert any("missing" in r.message for r in caplog.records)

    def test_comma_abstract_skips_llm_logs_warning(self, monkeypatch, caplog):
        import logging
        from ndif_citations.process import generate_summary
        from tests.helpers.llm import MockLLMClient
        from tests.conftest import make_paper
        mock = MockLLMClient()
        self._inject_mock(monkeypatch, mock)
        paper = make_paper(abstract=",")
        with caplog.at_level(logging.WARNING, logger="ndif_citations.process"):
            result = generate_summary(paper)
        assert result == ""
        mock.assert_no_calls()
        assert any("malformed" in r.message or "short" in r.message for r in caplog.records)

    def test_good_abstract_calls_llm(self, monkeypatch):
        from ndif_citations.process import generate_summary
        from tests.helpers.llm import MockLLMClient
        from tests.conftest import make_paper
        mock = MockLLMClient()
        mock.expect("A good summary of the paper.")
        self._inject_mock(monkeypatch, mock)
        paper = make_paper(abstract="We propose a new method for interpreting language models using activation patching.")
        result = generate_summary(paper)
        assert result == "A good summary of the paper."
        assert len(mock.record_calls()) == 1

    def test_warning_includes_paper_title(self, monkeypatch, caplog):
        import logging
        from ndif_citations.process import generate_summary
        from tests.helpers.llm import MockLLMClient
        from tests.conftest import make_paper
        mock = MockLLMClient()
        self._inject_mock(monkeypatch, mock)
        paper = make_paper(title="My Important Paper", abstract=None)
        with caplog.at_level(logging.WARNING, logger="ndif_citations.process"):
            generate_summary(paper)
        assert any("My Important Paper" in r.message for r in caplog.records)
