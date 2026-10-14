"""Curator quotes + bytes image save (spec §6.3)."""
import io
import json

import pytest
from PIL import Image

from ndif_citations.models import Bucket, DiscoveredPaper
from ndif_citations.server.services import papers_svc
from tests.conftest import make_paper


def _write(out, papers):
    (out / "images").mkdir(parents=True, exist_ok=True)
    data = {"pending": [], "verified": [], "discarded": []}
    for p in papers:
        data[p.bucket.value].append(p.to_full_dict())
    (out / "research-papers-full.json").write_text(json.dumps(data, default=str))


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, "PNG")
    return buf.getvalue()


def test_field_defaults_empty_and_loads_legacy_json():
    assert DiscoveredPaper(title="T").curator_quotes == []
    assert DiscoveredPaper.model_validate({"title": "T"}).curator_quotes == []


def test_add_quote_redacts_and_appends(tmp_path):
    p = make_paper(bucket=Bucket.PENDING)
    _write(tmp_path, [p])
    papers_svc.add_curator_quote(tmp_path, p.merge_key(), "We use NNsight (ask a@b.io).")
    saved = json.loads((tmp_path / "research-papers-full.json").read_text())["pending"][0]
    assert saved["curator_quotes"] == ["We use NNsight (ask [email])."]


def test_add_quote_rejects_overlong(tmp_path):
    p = make_paper(bucket=Bucket.PENDING)
    _write(tmp_path, [p])
    with pytest.raises(ValueError):
        papers_svc.add_curator_quote(tmp_path, p.merge_key(), "x" * 2001)


def test_save_image_bytes_requires_png(tmp_path):
    p = make_paper(bucket=Bucket.PENDING)
    _write(tmp_path, [p])
    with pytest.raises(ValueError):
        papers_svc.save_image_bytes(tmp_path, p.merge_key(), b"GIF89a....")
    papers_svc.save_image_bytes(tmp_path, p.merge_key(), _png())
    saved = json.loads((tmp_path / "research-papers-full.json").read_text())["pending"][0]
    assert saved["image"].endswith(".png")
