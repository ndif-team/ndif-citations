"""Network commands, all HTTP mocked (spec §6.3)."""
import io
import json

import pytest
import requests
from PIL import Image

from ndif_citations.models import Bucket, Category
from ndif_citations.output import load_existing_papers
from ndif_citations.review_commands import ReviewContext, run_commands
from ndif_citations.review_render import render_issue
from tests.conftest import make_paper


class _Stream:
    def __init__(self, data: bytes, status=200):
        self.data, self.status_code = data, status
        self.headers = {"Content-Length": str(len(data))}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self.data), chunk_size):
            yield self.data[i:i + chunk_size]

    def close(self):
        pass


def _write(out, papers):
    (out / "images").mkdir(parents=True, exist_ok=True)
    data = {"pending": [], "verified": [], "discarded": []}
    for p in papers:
        data[p.bucket.value].append(p.to_full_dict())
    (out / "research-papers-full.json").write_text(json.dumps(data, default=str))


def _ctx(out, paper, comment):
    _, body, _ = render_issue(paper)
    return ReviewContext(out, 12, body, "github-actions[bot]", {"review"}, comment, "alice")


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, "JPEG")   # any image → re-encoded as PNG
    return buf.getvalue()


@pytest.fixture()
def pending(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "x")   # preflight requires a key; conftest stubs the live check
    p = make_paper(title="Net Paper", arxiv_id="2502.00002", bucket=Bucket.PENDING, category=Category.USES_NNSIGHT)
    _write(tmp_path, [p])
    return tmp_path, p


def test_pdf_requires_https(pending):
    out, p = pending
    r = run_commands(_ctx(out, p, "/pdf http://example.org/a.pdf"))
    assert "Only https:// links are accepted" in r.reply and not r.changed


def test_pdf_refuses_non_pdf(pending, monkeypatch):
    out, p = pending
    monkeypatch.setattr(requests, "get", lambda url, **k: _Stream(b"<html>not a pdf</html>"))
    r = run_commands(_ctx(out, p, "/pdf https://example.org/a.pdf"))
    assert "isn't a PDF" in r.reply and not r.changed


def test_pdf_happy_path_reprocesses(pending, monkeypatch):
    out, p = pending
    monkeypatch.setattr(requests, "get", lambda url, **k: _Stream(b"%PDF-1.7 fake body"))
    seen = {}
    monkeypatch.setattr("ndif_citations.reprocess.reprocess_papers",
                        lambda o, ids, fields, **k: seen.update(ids=ids, fields=fields) or {})
    r = run_commands(_ctx(out, p, "/pdf https://example.org/a.pdf"))
    assert r.changed and seen["fields"] == ["classify", "thumbnail"]
    got = next(x for x in load_existing_papers(out) if x.merge_key() == p.merge_key())
    assert got.pdf_url == "https://example.org/a.pdf"


def test_pdf_keeps_existing_thumbnail(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "x")
    p = make_paper(title="Has Figure", arxiv_id="2502.00003", bucket=Bucket.PENDING,
                   category=Category.USES_NNSIGHT, image="/images/curator.png")
    _write(tmp_path, [p])
    (tmp_path / "images" / "curator.png").write_bytes(b"curator")
    monkeypatch.setattr(requests, "get", lambda url, **k: _Stream(b"%PDF-1.7 fake body"))
    seen = {}
    monkeypatch.setattr("ndif_citations.reprocess.reprocess_papers",
                        lambda o, ids, fields, **k: seen.update(fields=fields) or {})
    run_commands(_ctx(tmp_path, p, "/pdf https://example.org/a.pdf"))
    assert seen["fields"] == ["classify"]


def test_thumbnail_reextract_reports_no_new_figure(pending, monkeypatch):
    out, p = pending
    monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf", lambda paper, o: out / "x.pdf")
    monkeypatch.setattr("ndif_citations.reprocess.reprocess_papers", lambda o, ids, fields, **k: {})
    assert "No new figure" in run_commands(_ctx(out, p, "/thumbnail")).reply


def test_llm_down_blocks_pdf(pending, monkeypatch):
    out, p = pending
    monkeypatch.setattr("ndif_citations.key_validation.test_llm",
                        lambda *a, **k: {"ok": False, "detail": "HTTP 410"})
    monkeypatch.setenv("LLM_API_KEY", "x")
    r = run_commands(_ctx(out, p, "/pdf https://example.org/a.pdf"))
    assert "LLM is unavailable" in r.reply and not r.changed


def test_thumbnail_from_attachment(pending, monkeypatch):
    out, p = pending
    monkeypatch.setattr(requests, "get", lambda url, **k: _Stream(_png()))
    r = run_commands(_ctx(out, p, '/thumbnail\n<img src="https://github.com/user-attachments/assets/abc-123" alt="fig">'))
    assert r.changed
    assert next(x for x in load_existing_papers(out) if x.merge_key() == p.merge_key()).image.endswith(".png")


def test_thumbnail_rejects_foreign_host(pending, monkeypatch):
    out, p = pending
    monkeypatch.setattr(requests, "get", lambda url, **k: pytest.fail("must not download"))
    r = run_commands(_ctx(out, p, "![x](https://evil.example/a.png)\n/thumbnail"))
    assert "GitHub attachment" in r.reply and not r.changed


def test_retry_without_pdf_changes_nothing(pending, monkeypatch):
    out, p = pending
    monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf", lambda paper, o: None)
    monkeypatch.setattr("ndif_citations.reprocess.reprocess_papers", lambda *a, **k: pytest.fail("no reprocess"))
    r = run_commands(_ctx(out, p, "/retry"))
    assert "no PDF available" in r.reply and not r.changed


def test_summarize_reprocesses_summary(pending, monkeypatch):
    out, p = pending
    seen = {}
    monkeypatch.setattr("ndif_citations.reprocess.reprocess_papers",
                        lambda o, ids, fields, **k: seen.update(fields=fields) or {})
    assert run_commands(_ctx(out, p, "/summarize")).changed and seen["fields"] == ["summary"]


class _Redirected(_Stream):
    def __init__(self, data, url, history=()):
        super().__init__(data)
        self.url, self.history = url, list(history)


def test_pdf_rejects_non_https_final_url(pending, monkeypatch):
    out, p = pending
    monkeypatch.setattr(requests, "get", lambda url, **k: _Redirected(b"%PDF-1.7 x", "http://example.org/a.pdf"))
    r = run_commands(_ctx(out, p, "/pdf https://example.org/a.pdf"))
    assert "redirects to a non-https address" in r.reply and not r.changed


def test_pdf_rejects_non_https_redirect_hop(pending, monkeypatch):
    out, p = pending
    hop = _Redirected(b"", "http://mid.example/a.pdf")
    monkeypatch.setattr(requests, "get",
                        lambda url, **k: _Redirected(b"%PDF-1.7 x", "https://example.org/b.pdf", [hop]))
    r = run_commands(_ctx(out, p, "/pdf https://example.org/a.pdf"))
    assert "redirects to a non-https address" in r.reply and not r.changed


def test_thumbnail_downscales_and_flattens_alpha(pending, monkeypatch):
    out, p = pending
    buf = io.BytesIO()
    Image.new("RGBA", (3200, 1600), (255, 0, 0, 0)).save(buf, "PNG")  # fully transparent
    monkeypatch.setattr(requests, "get", lambda url, **k: _Stream(buf.getvalue()))
    r = run_commands(_ctx(out, p, "/thumbnail\n![x](https://github.com/user-attachments/assets/abc)"))
    assert r.changed
    got = next(x for x in load_existing_papers(out) if x.merge_key() == p.merge_key())
    img = Image.open(out / got.image.lstrip("/"))
    assert max(img.size) <= 1600 and img.mode == "RGB" and img.getpixel((0, 0)) == (255, 255, 255)
