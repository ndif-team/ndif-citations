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

def test_accepts_exact_50mb(tmp_path):
    paper = make_paper(title="A Paper", arxiv_id="2401.00001")
    # Exactly 50 MB must be accepted (the cap is a strict '>').
    data = b"%PDF-" + b"x" * (50 * 1024 * 1024 - 5)  # total == exactly 50 MB
    path = write_pdf_to_cache(paper, data, tmp_path)
    assert path.exists()
    assert len(path.read_bytes()) == 50 * 1024 * 1024

def test_rejects_empty_bytes(tmp_path):
    paper = make_paper(title="A Paper", arxiv_id="2401.00001")
    with pytest.raises(ValueError):
        write_pdf_to_cache(paper, b"", tmp_path)
