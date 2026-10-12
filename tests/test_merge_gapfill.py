"""Merge fills gaps; flags derived at load; discards stick (spec §4.5)."""
import json

from ndif_citations.models import Bucket, Category, Confidence, PaperReason, PipelineRun
from ndif_citations.output import _update_existing, load_existing_papers, merge_papers
from tests.conftest import make_paper


def _write_catalog(out, papers):
    data = {"pending": [], "verified": [], "discarded": []}
    for p in papers:
        data[p.bucket.value].append(p.to_full_dict())
    (out / "research-papers-full.json").write_text(json.dumps(data, default=str))


def test_flags_derived_from_fields_at_load(tmp_path):
    p = make_paper(description="A summary.", image="/images/x.png", category=Category.USES_NNSIGHT,
                   has_summary=False, has_thumbnail=False, has_classification=False)
    _write_catalog(tmp_path, [p])
    loaded = load_existing_papers(tmp_path)[0]
    assert (loaded.has_summary, loaded.has_thumbnail, loaded.has_classification) == (True, True, True)


def test_gapfill_adopts_description_only_when_empty():
    existing = make_paper(description="")
    new = make_paper(description="Fresh summary.")
    assert _update_existing(existing, new) is True
    assert existing.description == "Fresh summary."
    curated = make_paper(description="Curated.")
    _update_existing(curated, new)
    assert curated.description == "Curated."


def test_gapfill_adopts_classification_when_classification_ran():
    existing = make_paper(category=Category.UNCLASSIFIED)
    new = make_paper(category=Category.USES_NNSIGHT)
    new.category_confidence_band = Confidence.HIGH
    new.classification_signal = "llm"
    new.context_source = "pdf"
    new.ndif_context_windows = ["we use nnsight"]
    _update_existing(existing, new)
    assert existing.category == Category.USES_NNSIGHT
    assert existing.category_confidence_band == Confidence.HIGH
    assert existing.ndif_context_windows == ["we use nnsight"] and existing.context_source == "pdf"


def test_gapfill_never_adopts_default_category_when_classification_did_not_run():
    existing = make_paper(category=Category.UNCLASSIFIED)
    new = make_paper(category=Category.REFERENCING)   # model default; processing stopped early
    new.category_confidence_band = Confidence.NONE
    new.classification_signal = None
    _update_existing(existing, new)
    assert existing.category == Category.UNCLASSIFIED


def test_gapfill_propagates_zero_hit_discard_to_pending_paper():
    existing = make_paper(category=Category.UNCLASSIFIED, bucket=Bucket.PENDING)
    new = make_paper(bucket=Bucket.DISCARDED)
    new.reason = PaperReason.ZERO_PDF_HITS
    new.reason_detail = "Extracted 52000 chars; 0 matches"
    assert _update_existing(existing, new) is True
    assert existing.bucket == Bucket.DISCARDED and existing.reason == PaperReason.ZERO_PDF_HITS


def test_gapfill_never_touches_locked_papers():
    existing = make_paper(description="", manual_override=True)
    assert _update_existing(existing, make_paper(description="x")) is False
    assert existing.description == ""


def test_merge_keeps_auto_discards_discarded():
    existing = make_paper(bucket=Bucket.DISCARDED)
    existing.reason = PaperReason.ZERO_PDF_HITS
    rediscovered = make_paper()
    merged, _ = merge_papers([existing], [rediscovered], PipelineRun())
    assert merged[0].bucket == Bucket.DISCARDED


def test_gapfill_band_none_signal_llm_does_not_adopt_default_category():
    # Task 4 hand-off copies signal="llm" onto the fresh copy; a zero-hit discard fires before
    # classify, so the copy keeps defaults (REFERENCING, band NONE) and must not be adopted.
    existing = make_paper(category=Category.UNCLASSIFIED)
    new = make_paper(category=Category.REFERENCING)
    new.category_confidence_band = Confidence.NONE
    new.classification_signal = "llm"
    _update_existing(existing, new)
    assert existing.category == Category.UNCLASSIFIED


def test_zero_hit_discard_does_not_touch_verified_paper():
    existing = make_paper(bucket=Bucket.VERIFIED, category=Category.USES_NNSIGHT)
    new = make_paper(bucket=Bucket.DISCARDED)
    new.reason = PaperReason.ZERO_PDF_HITS
    _update_existing(existing, new)
    assert existing.bucket == Bucket.VERIFIED


def test_gapfill_windows_only_branch_adopts_windows_and_source():
    existing = make_paper(category=Category.USES_NNSIGHT)
    existing.ndif_context_windows = []
    new = make_paper(category=Category.USES_NNSIGHT)
    new.ndif_context_windows = ["we use nnsight"]
    new.context_source = "pdf"
    assert _update_existing(existing, new) is True
    assert existing.ndif_context_windows == ["we use nnsight"] and existing.context_source == "pdf"
