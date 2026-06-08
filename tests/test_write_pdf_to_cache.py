import pytest
from ndif_citations.pdf_cache import write_pdf_to_cache
from tests.conftest import make_paper

_PDF = b"%PDF-1.4\n%fake pdf bytes\n"

def test_writes_to_cache_path(tmp_path):
    paper = make_paper(title="A Paper", arxiv_id="2401.00001")
    path = write_pdf_to_cache(paper, _PDF, tmp_path)
    assert path == tmp_path / "pdfs" / "arxiv-2401.00001.pdf"
    assert path.read_bytes() == _PDF

def test_rejects_non_pdf(tmp_path):
    paper = make_paper(title="A Paper", arxiv_id="2401.00001")
    with pytest.raises(ValueError):
        write_pdf_to_cache(paper, b"<html>not a pdf</html>", tmp_path)

def test_rejects_oversize(tmp_path):
    paper = make_paper(title="A Paper", arxiv_id="2401.00001")
    with pytest.raises(ValueError):
        write_pdf_to_cache(paper, b"%PDF-" + b"x" * (50 * 1024 * 1024 + 1), tmp_path)

def test_overwrites_existing(tmp_path):
    paper = make_paper(title="A Paper", arxiv_id="2401.00001")
    write_pdf_to_cache(paper, _PDF, tmp_path)
    write_pdf_to_cache(paper, b"%PDF-1.7\nnewer\n", tmp_path)
    assert (tmp_path / "pdfs" / "arxiv-2401.00001.pdf").read_bytes() == b"%PDF-1.7\nnewer\n"
