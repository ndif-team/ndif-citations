"""Test evidence fields in DiscoveredPaper model."""

from ndif_citations.models import DiscoveredPaper


def test_evidence_fields_default_and_roundtrip():
    p = DiscoveredPaper(title="X")
    assert p.ndif_context_windows == [] and p.context_source == "none"
    p.ndif_context_windows = ["...nnsight..."]; p.context_source = "pdf"
    d = p.model_dump(mode="json")
    assert d["ndif_context_windows"] == ["...nnsight..."] and d["context_source"] == "pdf"
    assert DiscoveredPaper.model_validate(d).context_source == "pdf"
