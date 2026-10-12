"""reprocess bucket safety + thumbnail temp-swap (spec §4.5)."""
import json

import pytest

from ndif_citations import process
from ndif_citations.models import Bucket, Category, Confidence
from ndif_citations.output import load_existing_papers
from ndif_citations.reprocess import reprocess_papers
from tests.conftest import make_paper


def _write(out, papers):
    data = {"pending": [], "verified": [], "discarded": []}
    for p in papers:
        data[p.bucket.value].append(p.to_full_dict())
    (out / "research-papers-full.json").write_text(json.dumps(data, default=str))
    (out / "images").mkdir(exist_ok=True)


def _fake_classify(category, band, windows):
    def _f(paper, out, pdf_path=None):
        paper.classification_signal = "llm"
        paper.context_source = "pdf"
        paper.ndif_context_windows = windows
        return category, 0.85, band
    return _f


@pytest.fixture()
def pdf(monkeypatch, tmp_path):
    f = tmp_path / "fake.pdf"
    f.write_bytes(b"%PDF-1.4 tiny")  # < 1000 chars of text → never zero-hit discarded
    monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf", lambda p, out: f)
    return f


def _only(out):
    return load_existing_papers(out)[0]


# Every make_paper below passes year=2025: with the default year=0, _decide_bucket returns
# STUB_METADATA first and these tests would pass or fail for the wrong reason.


def test_locked_pending_uses_ndif_stays_pending_after_classify(tmp_path, monkeypatch, pdf):
    p = make_paper(year=2025, bucket=Bucket.PENDING, manual_override=True, category=Category.USES_NDIF)
    _write(tmp_path, [p])
    monkeypatch.setattr(process, "classify_category",
                        _fake_classify(Category.USES_NDIF, Confidence.HIGH, ["ran on NDIF remotely"]))
    reprocess_papers(tmp_path, [p.merge_key()], ["classify"])
    assert _only(tmp_path).bucket == Bucket.PENDING


def test_pending_paper_passing_rule_is_verified(tmp_path, monkeypatch, pdf):
    p = make_paper(year=2025, bucket=Bucket.PENDING, manual_override=True)
    _write(tmp_path, [p])
    monkeypatch.setattr(process, "classify_category",
                        _fake_classify(Category.USES_NNSIGHT, Confidence.HIGH, ["we use nnsight to patch"]))
    reprocess_papers(tmp_path, [p.merge_key()], ["classify"])
    assert _only(tmp_path).bucket == Bucket.VERIFIED


def test_human_verified_paper_keeps_bucket_on_medium_result(tmp_path, monkeypatch, pdf):
    p = make_paper(year=2025, bucket=Bucket.VERIFIED, manual_override=True, category=Category.USES_NNSIGHT)
    _write(tmp_path, [p])
    monkeypatch.setattr(process, "classify_category",
                        _fake_classify(Category.USES_NNSIGHT, Confidence.MEDIUM, ["nnsight"]))
    reprocess_papers(tmp_path, [p.merge_key()], ["classify"])
    assert _only(tmp_path).bucket == Bucket.VERIFIED


def test_verified_paper_with_changed_category_goes_to_review(tmp_path, monkeypatch, pdf):
    p = make_paper(year=2025, bucket=Bucket.VERIFIED, manual_override=True, category=Category.USES_NNSIGHT)
    _write(tmp_path, [p])
    monkeypatch.setattr(process, "classify_category",
                        _fake_classify(Category.USES_NDIF, Confidence.HIGH, ["ran remotely on NDIF"]))
    reprocess_papers(tmp_path, [p.merge_key()], ["classify"])
    got = _only(tmp_path)
    assert got.bucket == Bucket.PENDING and got.reason.value == "needs_review"
    assert "uses_nnsight → uses_ndif" in got.reason_detail


def test_edit_on_pending_uses_ndif_keeps_it_pending(tmp_path):
    from ndif_citations.server.services import papers_svc
    p = make_paper(year=2025, bucket=Bucket.PENDING, manual_override=True, category=Category.USES_NDIF)
    p.context_source = "pdf"
    _write(tmp_path, [p])
    papers_svc.edit_paper(tmp_path, p.merge_key(), {"venue": "ICLR 2026"})
    assert _only(tmp_path).bucket == Bucket.PENDING


def test_edit_never_publishes_a_pending_paper(tmp_path):
    """A locked paper's band reloads as CERTAIN (models.model_post_init), so the 2nd edit used to publish."""
    from ndif_citations.server.services import papers_svc
    p = make_paper(year=2025, bucket=Bucket.PENDING, category=Category.REFERENCING)
    p.context_source, p.classification_signal = "pdf", "llm"
    p.category_confidence_band = Confidence.MEDIUM
    p.ndif_context_windows = ["see NNsight (Fiotto-Kaufman)"]
    _write(tmp_path, [p])
    for venue in ("ICLR 2026", "ICML 2026"):
        papers_svc.edit_paper(tmp_path, p.merge_key(), {"venue": venue})
        assert _only(tmp_path).bucket == Bucket.PENDING


def test_cli_edit_never_publishes_pending(tmp_path):
    from click.testing import CliRunner
    from ndif_citations.cli import cli
    p = make_paper(year=2025, bucket=Bucket.PENDING, manual_override=True, category=Category.USES_NDIF)
    _write(tmp_path, [p])
    r = CliRunner().invoke(cli, ["edit", p.arxiv_id, "-o", str(tmp_path), "--set", "venue=ICLR 2026", "--yes"])
    assert r.exit_code == 0, r.output
    assert _only(tmp_path).bucket == Bucket.PENDING


def test_cli_reclassify_sends_changed_published_category_to_review(tmp_path, monkeypatch, pdf):
    from click.testing import CliRunner
    from ndif_citations.cli import cli
    p = make_paper(year=2025, bucket=Bucket.VERIFIED, category=Category.USES_NNSIGHT)  # unlocked
    _write(tmp_path, [p])
    monkeypatch.setattr(process, "classify_category",
                        _fake_classify(Category.USES_NDIF, Confidence.HIGH, ["ran remotely on NDIF"]))
    r = CliRunner().invoke(cli, ["reclassify", "--ids", p.arxiv_id, "-o", str(tmp_path)])
    assert r.exit_code == 0, r.output
    got = _only(tmp_path)
    assert got.bucket == Bucket.PENDING and "uses_nnsight → uses_ndif" in got.reason_detail


def test_edit_to_unclassified_unpublishes(tmp_path):
    from ndif_citations.server.services import papers_svc
    p = make_paper(year=2025, bucket=Bucket.VERIFIED, manual_override=True, category=Category.USES_NNSIGHT)
    _write(tmp_path, [p])
    papers_svc.edit_paper(tmp_path, p.merge_key(), {"category": "unclassified"})
    assert _only(tmp_path).bucket == Bucket.PENDING
    assert json.loads((tmp_path / "research-papers.json").read_text()) == []


def test_edit_on_verified_keeps_verified(tmp_path):
    from ndif_citations.server.services import papers_svc
    p = make_paper(year=2025, bucket=Bucket.VERIFIED, manual_override=True, category=Category.USES_NNSIGHT)
    _write(tmp_path, [p])
    papers_svc.edit_paper(tmp_path, p.merge_key(), {"venue": "ICLR 2026"})
    assert _only(tmp_path).bucket == Bucket.VERIFIED


def test_summary_reprocess_never_changes_bucket(tmp_path, monkeypatch):
    p = make_paper(year=2025, bucket=Bucket.PENDING, manual_override=True, category=Category.USES_NDIF,
                   abstract="A long enough abstract about interpretability infrastructure.")
    _write(tmp_path, [p])
    monkeypatch.setattr(process, "generate_summary", lambda paper: "New summary.")
    reprocess_papers(tmp_path, [p.merge_key()], ["summary"])
    got = _only(tmp_path)
    assert got.bucket == Bucket.PENDING and got.description == "New summary."


def test_thumbnail_kept_when_no_pdf(tmp_path, monkeypatch):
    p = make_paper(year=2025, bucket=Bucket.VERIFIED, manual_override=True, image="/images/Test-Paper.png")
    _write(tmp_path, [p])
    png = tmp_path / "images" / "Test-Paper.png"
    png.write_bytes(b"old-png")
    monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf", lambda paper, out: None)
    reprocess_papers(tmp_path, [p.merge_key()], ["thumbnail"])
    assert png.read_bytes() == b"old-png"
    assert _only(tmp_path).image == "/images/Test-Paper.png"
    assert [f.name for f in (tmp_path / "images").iterdir()] == ["Test-Paper.png"]  # nothing parked in images/


def test_thumbnail_restored_when_processing_raises(tmp_path, monkeypatch):
    p = make_paper(year=2025, bucket=Bucket.VERIFIED, manual_override=True, image="/images/Test-Paper.png")
    _write(tmp_path, [p])
    png = tmp_path / "images" / "Test-Paper.png"
    png.write_bytes(b"old-png")

    def boom(*a, **k):
        raise RuntimeError("surya crashed")

    monkeypatch.setattr("ndif_citations.reprocess.process_papers", boom)
    with pytest.raises(RuntimeError, match="surya crashed"):
        reprocess_papers(tmp_path, [p.merge_key()], ["thumbnail"])
    assert png.read_bytes() == b"old-png"
    assert [f.name for f in (tmp_path / "images").iterdir()] == ["Test-Paper.png"]
