import json
from pathlib import Path
from click.testing import CliRunner
from ndif_citations.cli import cli


def _cat(out, papers):
    out.mkdir(parents=True, exist_ok=True)
    (out / "research-papers-full.json").write_text(
        json.dumps({"verified": papers, "pending": [], "discarded": []})
    )


def _boom(*a, **k):  # pragma: no cover - only fires on unexpected call
    raise AssertionError("should not be called")


def test_backfill_thumbnails_sets_image_and_writes(tmp_path, monkeypatch):
    out = tmp_path / "output"
    _cat(out, [{"title": "P", "arxiv_id": "2401.1"}])
    monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf",
                        lambda paper, o: tmp_path / "fake.pdf")
    monkeypatch.setattr("ndif_citations.process.extract_thumbnail",
                        lambda paper, o, pdf_path=None: "/images/P.png")
    res = CliRunner().invoke(cli, ["backfill-thumbnails", "-o", str(out)])
    assert res.exit_code == 0, res.output
    data = json.loads((out / "research-papers-full.json").read_text())
    assert data["verified"][0]["image"] == "/images/P.png"
    assert data["verified"][0]["has_thumbnail"] is True


def test_backfill_thumbnails_skips_idless_paper(tmp_path, monkeypatch):
    out = tmp_path / "output"
    _cat(out, [{"title": "No identifier here"}])  # no arxiv_id, no doi
    # Neither PDF resolution nor extraction should be attempted for id-less papers.
    monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf", _boom)
    monkeypatch.setattr("ndif_citations.process.extract_thumbnail", _boom)
    before = (out / "research-papers-full.json").read_text()
    res = CliRunner().invoke(cli, ["backfill-thumbnails", "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert (out / "research-papers-full.json").read_text() == before
    assert "skipped" in res.output.lower()


def test_backfill_thumbnails_dry_run_does_no_work_and_no_write(tmp_path, monkeypatch):
    out = tmp_path / "output"
    _cat(out, [{"title": "P", "arxiv_id": "2401.1"}])
    # Dry-run must classify candidates without downloading or rendering anything.
    monkeypatch.setattr("ndif_citations.pdf_cache.cached_pdf_path", lambda paper, o: None)
    monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf", _boom)
    monkeypatch.setattr("ndif_citations.process.extract_thumbnail", _boom)
    before = (out / "research-papers-full.json").read_text()
    res = CliRunner().invoke(cli, ["backfill-thumbnails", "-o", str(out), "--dry-run"])
    assert res.exit_code == 0, res.output
    assert (out / "research-papers-full.json").read_text() == before


def test_backfill_thumbnails_skips_paper_with_existing_image(tmp_path, monkeypatch):
    out = tmp_path / "output"
    _cat(out, [{"title": "P", "arxiv_id": "2401.1", "image": "/images/P.png"}])
    monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf", _boom)
    monkeypatch.setattr("ndif_citations.process.extract_thumbnail", _boom)
    before = (out / "research-papers-full.json").read_text()
    res = CliRunner().invoke(cli, ["backfill-thumbnails", "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert (out / "research-papers-full.json").read_text() == before


def test_backfill_thumbnails_isolates_per_paper_errors(tmp_path, monkeypatch):
    out = tmp_path / "output"
    _cat(out, [
        {"title": "Bad", "arxiv_id": "2401.1"},
        {"title": "Good", "arxiv_id": "2401.2"},
    ])
    monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf",
                        lambda paper, o: tmp_path / "fake.pdf")

    def fake_extract(paper, o, pdf_path=None):
        if paper.title == "Bad":
            raise RuntimeError("fitz blew up")
        return "/images/Good.png"

    monkeypatch.setattr("ndif_citations.process.extract_thumbnail", fake_extract)
    res = CliRunner().invoke(cli, ["backfill-thumbnails", "-o", str(out)])
    assert res.exit_code == 0, res.output
    data = json.loads((out / "research-papers-full.json").read_text())
    by_title = {p["title"]: p for p in data["verified"]}
    assert by_title["Good"]["image"] == "/images/Good.png"
    assert not by_title["Bad"].get("image")


def test_backfill_thumbnails_no_pdf_available(tmp_path, monkeypatch):
    out = tmp_path / "output"
    _cat(out, [{"title": "P", "arxiv_id": "2401.1"}])
    monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf", lambda paper, o: None)
    monkeypatch.setattr("ndif_citations.process.extract_thumbnail", _boom)
    before = (out / "research-papers-full.json").read_text()
    res = CliRunner().invoke(cli, ["backfill-thumbnails", "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert (out / "research-papers-full.json").read_text() == before


def test_backfill_thumbnails_ids_filter(tmp_path, monkeypatch):
    out = tmp_path / "output"
    _cat(out, [
        {"title": "Wanted", "arxiv_id": "2401.1"},
        {"title": "Other", "arxiv_id": "2401.2"},
    ])
    calls = []
    monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf",
                        lambda paper, o: tmp_path / "fake.pdf")

    def fake_extract(paper, o, pdf_path=None):
        calls.append(paper.title)
        return f"/images/{paper.title}.png"

    monkeypatch.setattr("ndif_citations.process.extract_thumbnail", fake_extract)
    res = CliRunner().invoke(cli, ["backfill-thumbnails", "-o", str(out), "--ids", "2401.1"])
    assert res.exit_code == 0, res.output
    assert calls == ["Wanted"]
