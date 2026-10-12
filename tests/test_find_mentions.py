"""Mention matching (spec §4.3)."""
from pathlib import Path

from ndif_citations import process
from ndif_citations.models import Confidence
from ndif_citations.utils import extract_ndif_context, find_mentions
from tests.conftest import make_paper


def test_one_match_per_mention_for_overlapping_keywords():
    text = "We import nnsight and visit nnsight.net for docs."
    assert len(find_mentions(text)) == 2


def test_case_variants_count_once():
    assert len(find_mentions("NNsight is great; nnsight rocks.")) == 2


def test_endif_and_indifferent_are_not_ndif():
    assert find_mentions("<![endif]--> we are indifferent to this") == []


def test_bare_ndif_and_ndif_us_match():
    assert len(find_mentions("Experiments ran on NDIF via ndif.us remote execution.")) == 2


def test_glued_footnote_mention_still_counts():
    assert len(find_mentions("implemented with nnsight3 (see footnote)")) == 1


def test_extract_bibliography_excerpts_come_last(monkeypatch):
    body = "x" * 600 + " we use nnsight for patching " + "y" * 600
    bib = " Nnsight and ndif: Democratizing access to open-weight foundation model internals. "
    monkeypatch.setattr("ndif_citations.utils.extract_text_from_pdf", lambda p: bib + "z" * 1200 + body)
    excerpts = extract_ndif_context(Path("/fake.pdf")).split("\n---\n")
    assert "we use nnsight" in excerpts[0]
    assert "democratizing access" in excerpts[-1].lower()


def test_sentinels_unchanged(monkeypatch):
    monkeypatch.setattr("ndif_citations.utils.extract_text_from_pdf", lambda p: "")
    assert extract_ndif_context(Path("/f.pdf")) == "No text could be extracted from the PDF."
    monkeypatch.setattr("ndif_citations.utils.extract_text_from_pdf", lambda p: "nothing relevant")
    assert extract_ndif_context(Path("/f.pdf")) == "No direct mentions of NDIF or nnsight found in the paper text."


def test_zero_hit_discard_ignores_endif(monkeypatch, tmp_path):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(process, "extract_text_from_pdf", lambda p: "<![endif]> " + "w " * 800)
    paper = make_paper()
    assert process._check_discard_zero_pdf_hits(paper, pdf) is True


def test_no_text_sentinel_falls_back_to_abstract(monkeypatch, tmp_path):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(process, "extract_ndif_context",
                        lambda path, window=500: "No text could be extracted from the PDF.")
    paper = make_paper(abstract="We build on NNsight to study circuits.")
    windows, source, _ = process.compute_context(paper, tmp_path, pdf_path=pdf)
    assert source == "abstract" and windows


def test_stored_pdf_evidence_reused_without_pdf(tmp_path):
    paper = make_paper()
    paper.context_source = "pdf"
    paper.ndif_context_windows = ["we use nnsight to trace activations"]
    windows, source, signal = process.compute_context(paper, tmp_path, pdf_path=Path("/missing.pdf"))
    assert (windows, source, signal) == (["we use nnsight to trace activations"], "pdf", None)


def test_stored_empty_prefiltered_evidence_reused(tmp_path):
    paper = make_paper()
    paper.context_source = "pdf"
    paper.ndif_context_windows = []
    paper.classification_signal = "pre_filter:negative_evidence"
    assert process.compute_context(paper, tmp_path, pdf_path=Path("/missing.pdf")) == (
        [], "pdf", "pre_filter:negative_evidence")


def test_process_papers_hands_stored_pdf_evidence_to_classify(tmp_path, monkeypatch):
    """The router passes the FRESH discovered copy (context_source "none"); its stored PDF evidence must follow."""
    from ndif_citations.models import Category, Confidence
    from ndif_citations.router import ProcessingBucket, RoutingDecision
    existing = make_paper(year=2025, category=Category.UNCLASSIFIED)
    existing.context_source = "pdf"
    existing.ndif_context_windows = ["we use nnsight to trace activations"]
    fresh = make_paper(year=2025)
    seen = []
    monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf", lambda p, out: None)
    monkeypatch.setattr(process, "classify_category", lambda paper, out, pdf_path=None: (
        seen.append((paper.context_source, list(paper.ndif_context_windows)))
        or (Category.USES_NNSIGHT, 0.9, Confidence.HIGH)))
    needs = {"summary": False, "classify": True, "thumbnail": False, "affiliations": False}
    process.process_papers([RoutingDecision(fresh, ProcessingBucket.FILL_GAPS, existing, needs)], tmp_path)
    assert seen == [("pdf", ["we use nnsight to trace activations"])]


def test_redact_emails():
    from ndif_citations.utils import redact_emails
    assert redact_emails("Contact jane.doe+x@uni.edu or a@b.io.") == "Contact [email] or [email]."
    assert redact_emails("no address here") == "no address here"
    assert redact_emails("pip install nnsight@0.4.2") == "pip install nnsight@0.4.2"
