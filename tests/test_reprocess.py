"""Core tests for the targeted force-reprocess recipe (Task 4.3, Part A).

These exercise ``ndif_citations.reprocess.reprocess_papers`` directly (no server
layer). The critical case is ``test_reprocess_summary_overwrites_curated``: it
proves the §3.3 recipe DEFEATS the FILL_GAPS protective hydration/guard — a
curated (manual_override=True) paper's description IS overwritten by the fresh
LLM value, which the naive FILL_GAPS path would have kept.

Fixture state (mini-research-papers-full.json):
  verified  (3): "Activation Steering..."  (arxiv:2602.16080, mo=False)
                 "ADAG..."                  (arxiv:2604.07615, mo=False, referencing)
                 "Behind the Scenes..."     (title-key, mo=True, has description+category+affs)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ndif_citations import reprocess
from ndif_citations.models import Category
from ndif_citations.output import load_existing_papers

# manual_override=True paper with a non-empty curated description.
CURATED_TITLE = (
    "Behind the Scenes: Mechanistic Interpretability of LoRA-based Image Captioning"
)
# Confirm the exact merge_key by loading the fixture in the test body instead of
# hard-coding the full lowercase title here.


def _find(out: Path, predicate):
    for p in load_existing_papers(out):
        if predicate(p):
            return p
    raise AssertionError("paper not found")


def _curated_id(out: Path) -> str:
    p = _find(out, lambda p: p.manual_override and p.title.startswith("Behind the Scenes"))
    return p.merge_key()


# ---------------------------------------------------------------------------
# 1. CRITICAL: reprocess overwrites a curated (manual_override) field.
# ---------------------------------------------------------------------------

def test_reprocess_summary_overwrites_curated(monkeypatch, fixture_state):
    import ndif_citations.process as process_mod

    out = fixture_state
    pid = _curated_id(out)

    before = _find(out, lambda p: p.merge_key() == pid)
    assert before.manual_override is True
    assert before.description  # non-empty curated value
    old_description = before.description
    old_category = before.category
    old_affiliations = before.affiliations

    # Fake LLM summary returns a brand-new deterministic value.
    monkeypatch.setattr(process_mod, "generate_summary", lambda paper: "NEW")
    # No PDF needed for a summary-only reprocess, but be safe.
    import ndif_citations.pdf_cache as pdf_cache_mod
    monkeypatch.setattr(pdf_cache_mod, "get_cached_pdf", lambda paper, out: None)

    result = reprocess.reprocess_papers(out, [pid], ["summary"])
    assert result == {"reprocessed": [pid], "fields": ["summary"]}

    after = _find(out, lambda p: p.merge_key() == pid)
    # The curated description WAS overwritten (proves the recipe defeats guard).
    assert after.description == "NEW"
    assert after.description != old_description
    # manual_override restored to True (paper stays curated).
    assert after.manual_override is True
    assert after.has_summary is True
    # Other fields untouched.
    assert after.category == old_category
    assert after.affiliations == old_affiliations


# ---------------------------------------------------------------------------
# 2. Reprocess only the requested field; others untouched.
# ---------------------------------------------------------------------------

def test_reprocess_only_requested_field(monkeypatch, fixture_state):
    import ndif_citations.process as process_mod
    import ndif_citations.pdf_cache as pdf_cache_mod

    out = fixture_state
    pid = _curated_id(out)

    before = _find(out, lambda p: p.merge_key() == pid)
    old_category = before.category
    old_image = before.image
    old_affiliations = before.affiliations

    monkeypatch.setattr(process_mod, "generate_summary", lambda paper: "ONLY-SUMMARY")
    # If classify/thumbnail/affiliations were (wrongly) re-run they'd need these:
    monkeypatch.setattr(
        process_mod,
        "classify_category",
        lambda paper, out, pdf_path=None: (_ for _ in ()).throw(
            AssertionError("classify_category should NOT be called")
        ),
    )
    monkeypatch.setattr(
        process_mod,
        "extract_thumbnail",
        lambda paper, out, pdf_path=None: (_ for _ in ()).throw(
            AssertionError("extract_thumbnail should NOT be called")
        ),
    )
    monkeypatch.setattr(pdf_cache_mod, "get_cached_pdf", lambda paper, out: None)

    reprocess.reprocess_papers(out, [pid], ["summary"])

    after = _find(out, lambda p: p.merge_key() == pid)
    assert after.description == "ONLY-SUMMARY"
    assert after.category == old_category
    assert after.image == old_image
    assert after.affiliations == old_affiliations


# ---------------------------------------------------------------------------
# 3. Invalid field → ValueError (mapped to 422 at the endpoint).
# ---------------------------------------------------------------------------

def test_reprocess_invalid_field_raises(fixture_state):
    out = fixture_state
    pid = _curated_id(out)
    with pytest.raises(ValueError):
        reprocess.reprocess_papers(out, [pid], ["bogus"])


def test_reprocess_empty_fields_raises(fixture_state):
    out = fixture_state
    pid = _curated_id(out)
    with pytest.raises(ValueError):
        reprocess.reprocess_papers(out, [pid], [])


# ---------------------------------------------------------------------------
# 4. Unknown paper id → ValueError.
# ---------------------------------------------------------------------------

def test_reprocess_unknown_id_raises(fixture_state):
    out = fixture_state
    with pytest.raises(ValueError):
        reprocess.reprocess_papers(out, ["arxiv:0000.00000"], ["summary"])


# ---------------------------------------------------------------------------
# 5. Reprocess classify on a curated paper clears + re-runs classification.
# ---------------------------------------------------------------------------

def test_reprocess_classify_overwrites_curated(monkeypatch, fixture_state):
    import ndif_citations.process as process_mod
    import ndif_citations.pdf_cache as pdf_cache_mod
    from ndif_citations.models import Confidence

    out = fixture_state
    pid = _curated_id(out)

    before = _find(out, lambda p: p.merge_key() == pid)
    old_description = before.description

    def _fake_classify(paper, output_dir, pdf_path=None):
        return Category.USES_NDIF, 0.85, Confidence.HIGH

    monkeypatch.setattr(process_mod, "classify_category", _fake_classify)
    monkeypatch.setattr(pdf_cache_mod, "get_cached_pdf", lambda paper, out: None)

    reprocess.reprocess_papers(out, [pid], ["classify"])

    after = _find(out, lambda p: p.merge_key() == pid)
    assert after.category == Category.USES_NDIF
    assert after.has_classification is True
    assert after.manual_override is True
    # summary left alone
    assert after.description == old_description
