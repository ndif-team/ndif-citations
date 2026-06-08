import json
import pytest
from ndif_citations.server.services import papers_svc

_PDF = b"%PDF-1.4\nfake\n"

def _cat(out, papers):
    out.mkdir(parents=True, exist_ok=True)
    (out / "research-papers-full.json").write_text(
        json.dumps({"verified": papers, "pending": [], "discarded": []}))

def test_attach_pdf_writes_and_reports_has_pdf(tmp_path):
    out = tmp_path / "output"
    _cat(out, [{"title": "P", "arxiv_id": "2401.1"}])
    from ndif_citations.models import DiscoveredPaper
    paper_id = DiscoveredPaper(title="P", arxiv_id="2401.1").merge_key()
    result = papers_svc.attach_pdf(out, paper_id, _PDF)
    assert (out / "pdfs" / "arxiv-2401.1.pdf").read_bytes() == _PDF
    assert result["has_pdf"] is True

def test_attach_pdf_unknown_id_raises(tmp_path):
    out = tmp_path / "output"
    _cat(out, [{"title": "P", "arxiv_id": "2401.1"}])
    with pytest.raises(KeyError):
        papers_svc.attach_pdf(out, "arxiv:9999.9", _PDF)

def test_attach_pdf_rejects_non_pdf(tmp_path):
    out = tmp_path / "output"
    _cat(out, [{"title": "P", "arxiv_id": "2401.1"}])
    from ndif_citations.models import DiscoveredPaper
    paper_id = DiscoveredPaper(title="P", arxiv_id="2401.1").merge_key()
    with pytest.raises(ValueError):
        papers_svc.attach_pdf(out, paper_id, b"nope")
