import json
from click.testing import CliRunner
from ndif_citations.cli import cli

def _cat(out, papers):
    out.mkdir(parents=True, exist_ok=True)
    (out / "research-papers-full.json").write_text(
        json.dumps({"verified": papers, "pending": [], "discarded": []}))

def test_attach_pdf_cli_writes_file(tmp_path):
    out = tmp_path / "output"
    _cat(out, [{"title": "P", "arxiv_id": "2401.1"}])
    pdf = tmp_path / "local.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfake\n")
    res = CliRunner().invoke(cli, ["attach-pdf", "2401.1", str(pdf), "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert (out / "pdfs" / "arxiv-2401.1.pdf").read_bytes() == b"%PDF-1.4\nfake\n"

def test_attach_pdf_cli_unknown_id(tmp_path):
    out = tmp_path / "output"
    _cat(out, [{"title": "P", "arxiv_id": "2401.1"}])
    pdf = tmp_path / "local.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfake\n")
    res = CliRunner().invoke(cli, ["attach-pdf", "9999.9", str(pdf), "-o", str(out)])
    assert res.exit_code == 0
    assert "not found" in res.output.lower()
