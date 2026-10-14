"""Auto-publish gate (spec §4.7)."""
import pytest

from ndif_citations import config
from ndif_citations.models import Bucket, Category, Confidence
from ndif_citations.process import _decide_bucket, _passes_auto_publish_rule
from tests.conftest import make_paper


def _p(category, *, band=Confidence.HIGH, source="pdf", signal="llm", windows=("we use nnsight to patch",)):
    # year=2025: make_paper's year=0 default would make _decide_bucket return STUB_METADATA first.
    p = make_paper(category=category, bucket=Bucket.PENDING, year=2025,
                   abstract="A sufficiently long abstract here.")
    p.category_confidence_band = band
    p.context_source = source
    p.classification_signal = signal
    p.ndif_context_windows = list(windows)
    return p


def test_uses_nnsight_with_phrase_passes():
    assert _passes_auto_publish_rule(_p(Category.USES_NNSIGHT)) is True


def test_uses_nnsight_without_phrase_fails():
    assert _passes_auto_publish_rule(_p(Category.USES_NNSIGHT, windows=["Libraries such as NNsight provide"])) is False


def test_uses_ndif_never_passes():
    assert _passes_auto_publish_rule(_p(Category.USES_NDIF, windows=["we used nnsight on NDIF"])) is False


def test_referencing_with_pdf_passes():
    assert _passes_auto_publish_rule(_p(Category.REFERENCING, windows=["see NNsight (Fiotto-Kaufman)"])) is True


def test_band_none_never_publishes():
    assert _passes_auto_publish_rule(_p(Category.REFERENCING, band=Confidence.NONE)) is False


def test_abstract_only_evidence_never_publishes():
    assert _passes_auto_publish_rule(_p(Category.USES_NNSIGHT, source="abstract")) is False


def test_keyword_fallback_signal_never_publishes_usage():
    assert _passes_auto_publish_rule(_p(Category.USES_NNSIGHT, signal="keyword_fallback")) is False


def test_kill_switch(monkeypatch):
    monkeypatch.setattr(config, "AUTO_PUBLISH_RULES", False)
    assert _passes_auto_publish_rule(_p(Category.USES_NNSIGHT)) is False


def test_decide_bucket_publishes_rule_passers(monkeypatch):
    monkeypatch.setattr(config, "AUTO_VERIFY", False)
    assert _decide_bucket(_p(Category.USES_NNSIGHT)) == (Bucket.VERIFIED, None)


def test_decide_bucket_holds_uses_ndif(monkeypatch):
    monkeypatch.setattr(config, "AUTO_VERIFY", False)
    bucket, reason = _decide_bucket(_p(Category.USES_NDIF))
    assert bucket == Bucket.PENDING and reason.value == "needs_review"
