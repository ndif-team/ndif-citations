"""LLM call hardening (spec §4.1-4.2): extra_body, empty reply = failure, breaker."""
from pathlib import Path

import pytest

from ndif_citations import config, process
from ndif_citations.models import Category
from tests.conftest import make_paper

ABSTRACT = "First sentence about interpretability. Second sentence here. Third one."


class _SpyClient:
    """Duck-types client.chat.completions.create; records kwargs; replies in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply

        class _O:
            pass

        msg, choice, resp = _O(), _O(), _O()
        msg.content = reply
        choice.message = msg
        resp.choices = [choice]
        return resp


def _use(monkeypatch, spy):
    monkeypatch.setattr(process, "_get_llm_client", lambda: spy)


def test_summary_sends_extra_body(monkeypatch):
    spy = _SpyClient(["A concise summary."])
    _use(monkeypatch, spy)
    monkeypatch.setattr(config, "LLM_EXTRA_BODY", {"chat_template_kwargs": {"enable_thinking": False}})
    assert process.generate_summary(make_paper(abstract=ABSTRACT)) == "A concise summary."
    assert spy.calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_empty_extra_body_sends_none(monkeypatch):
    spy = _SpyClient(["ok"])
    _use(monkeypatch, spy)
    monkeypatch.setattr(config, "LLM_EXTRA_BODY", {})
    process.generate_summary(make_paper(abstract=ABSTRACT))
    assert spy.calls[0]["extra_body"] is None


def test_empty_reply_counts_as_failure(monkeypatch):
    spy = _SpyClient([None])  # reasoning model spent the budget thinking
    _use(monkeypatch, spy)
    out = process.generate_summary(make_paper(abstract=ABSTRACT))
    assert out == "First sentence about interpretability. Second sentence here."
    assert process._consecutive_llm_failures == 1


def test_three_consecutive_failures_raise(monkeypatch):
    _use(monkeypatch, _SpyClient([RuntimeError("HTTP 410")] * 3))
    paper = make_paper(abstract=ABSTRACT)
    process.generate_summary(paper)
    process.generate_summary(paper)
    with pytest.raises(process.LLMUnavailableError, match="3 consecutive LLM failures"):
        process.generate_summary(paper)


def test_success_resets_the_counter(monkeypatch):
    _use(monkeypatch, _SpyClient([RuntimeError("a"), RuntimeError("b"), "fine", RuntimeError("c")]))
    paper = make_paper(abstract=ABSTRACT)
    for _ in range(4):
        process.generate_summary(paper)  # must not raise
    assert process._consecutive_llm_failures == 1


def test_classify_failures_trip_the_breaker(monkeypatch, tmp_path):
    _use(monkeypatch, _SpyClient([RuntimeError("down")] * 3))
    monkeypatch.setattr(process, "compute_context",
                        lambda paper, out, pdf_path=None: (["we use nnsight to patch"], "pdf", None))
    paper = make_paper()
    process.classify_category(paper, tmp_path)
    process.classify_category(paper, tmp_path)
    with pytest.raises(process.LLMUnavailableError):
        process.classify_category(paper, tmp_path)


def test_classify_sends_extra_body_and_parses(monkeypatch, tmp_path):
    spy = _SpyClient(["uses_nnsight"])
    _use(monkeypatch, spy)
    monkeypatch.setattr(process, "compute_context",
                        lambda paper, out, pdf_path=None: (["we use nnsight to patch"], "pdf", None))
    cat, _, _ = process.classify_category(make_paper(), tmp_path)
    assert cat == Category.USES_NNSIGHT
    assert "extra_body" in spy.calls[0]


def test_breaker_propagates_out_of_process_papers(monkeypatch, tmp_path):
    """process_papers must not swallow LLMUnavailableError: the run aborts, nothing is written."""
    from ndif_citations.router import ProcessingBucket, RoutingDecision
    _use(monkeypatch, _SpyClient([RuntimeError("HTTP 410")] * 3))
    needs = {"summary": True, "classify": False, "thumbnail": False, "affiliations": False}
    decisions = [
        RoutingDecision(paper=make_paper(title=f"P{i}", arxiv_id=f"2401.0000{i}", abstract=ABSTRACT),
                        bucket=ProcessingBucket.NEW, existing_paper=None, processing_needed=needs)
        for i in range(3)
    ]
    with pytest.raises(process.LLMUnavailableError):
        process.process_papers(decisions, tmp_path)
