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
from ndif_citations.events import RunCancelled
from ndif_citations.models import Category
from ndif_citations.output import load_existing_papers
from ndif_citations.utils import slugify

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


# ---------------------------------------------------------------------------
# 6. Stale on-disk PNG is deleted before re-extraction (Fix 1 regression guard).
# ---------------------------------------------------------------------------

def test_reextract_thumbnail_deletes_stale_and_regenerates(monkeypatch, fixture_state):
    """reprocess_papers must unlink the stale PNG so extract_thumbnail actually runs.

    Regression guard: if the stale PNG is NOT deleted, process_papers sees
    `image_path.exists() == True` and skips `extract_thumbnail` entirely —
    re-extraction silently keeps the old image.
    """
    import ndif_citations.process as process_mod
    import ndif_citations.pdf_cache as pdf_cache_mod

    out = fixture_state
    pid = _curated_id(out)
    paper = _find(out, lambda p: p.merge_key() == pid)

    # Pre-create a stale PNG on disk for this paper.
    slug = slugify(paper.title)
    stale_png = out / "images" / f"{slug}.png"
    stale_png.parent.mkdir(parents=True, exist_ok=True)
    stale_png.write_bytes(b"STALE")

    # Give the paper an existing image path so _clear_field has something to clear.
    paper.image = f"/images/{slug}.png"

    # Fake PDF path so the thumbnail branch in process_papers is entered.
    fake_pdf = out / "pdfs" / "fake.pdf"
    fake_pdf.parent.mkdir(parents=True, exist_ok=True)
    fake_pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(pdf_cache_mod, "get_cached_pdf", lambda p, o: fake_pdf)

    # Spy: records calls and returns a new image path.
    extract_calls: list[str] = []

    def _spy_extract(paper, output_dir, pdf_path=None):
        extract_calls.append(paper.title)
        new_path = output_dir / "images" / f"{slugify(paper.title)}.png"
        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
        return f"/images/{slugify(paper.title)}.png"

    monkeypatch.setattr(process_mod, "extract_thumbnail", _spy_extract)

    reprocess.reprocess_papers(out, [pid], ["thumbnail"])

    # The spy MUST have been called — proving extraction ran, not silently skipped.
    assert extract_calls, (
        "extract_thumbnail was NOT called: the stale PNG was not deleted and "
        "process_papers skipped re-extraction (the Fix 1 no-op bug)"
    )


# ---------------------------------------------------------------------------
# 7. Cancel during reprocess does not persist the partially-cleared paper.
# ---------------------------------------------------------------------------

def test_reprocess_cancel_does_not_persist(monkeypatch, fixture_state):
    """A cancel_check that fires must propagate RunCancelled and leave the
    on-disk research-papers-full.json byte-identical to the pre-call snapshot.

    This guards the correct-by-construction cancel safety: write_outputs is
    never reached when process_papers raises RunCancelled.
    """
    import ndif_citations.process as process_mod
    import ndif_citations.pdf_cache as pdf_cache_mod

    out = fixture_state
    pid = _curated_id(out)

    json_path = out / "research-papers-full.json"
    snapshot = json_path.read_bytes()

    # cancel_check always returns True → process_papers raises RunCancelled immediately.
    monkeypatch.setattr(process_mod, "generate_summary", lambda paper: "SHOULD NOT APPEAR")
    monkeypatch.setattr(pdf_cache_mod, "get_cached_pdf", lambda p, o: None)

    with pytest.raises(RunCancelled):
        reprocess.reprocess_papers(out, [pid], ["summary"], cancel_check=lambda: True)

    # The JSON file must be byte-identical to the snapshot — no cleared-but-not-
    # refilled paper was written to disk.
    assert json_path.read_bytes() == snapshot, (
        "research-papers-full.json was modified despite RunCancelled — "
        "write_outputs must not be called when process_papers is cancelled"
    )
