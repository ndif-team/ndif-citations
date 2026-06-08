"""Tests for the attach-pdf CLI command."""
from click.testing import CliRunner
from ndif_citations.cli import cli
from ndif_citations.models import PipelineRun
from ndif_citations.output import write_outputs
from tests.conftest import make_paper


def _write_catalog(out, papers):
    out.mkdir(parents=True, exist_ok=True)
    write_outputs(papers, out, PipelineRun())


def test_attach_pdf_cli_writes_file(tmp_path):
    out = tmp_path / "output"
    _write_catalog(out, [make_paper(title="P", arxiv_id="2401.1")])
    pdf = tmp_path / "local.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfake\n")
    res = CliRunner().invoke(cli, ["attach-pdf", "2401.1", str(pdf), "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert (out / "pdfs" / "arxiv-2401.1.pdf").read_bytes() == b"%PDF-1.4\nfake\n"


def test_attach_pdf_cli_unknown_id(tmp_path):
    out = tmp_path / "output"
    _write_catalog(out, [make_paper(title="P", arxiv_id="2401.1")])
    pdf = tmp_path / "local.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfake\n")
    res = CliRunner().invoke(cli, ["attach-pdf", "9999.9", str(pdf), "-o", str(out)])
    assert res.exit_code == 0
    assert "not found" in res.output.lower()


def test_attach_pdf_cli_bad_file(tmp_path):
    out = tmp_path / "output"
    _write_catalog(out, [make_paper(title="P", arxiv_id="2401.1")])
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not-a-pdf")
    res = CliRunner().invoke(cli, ["attach-pdf", "2401.1", str(bad), "-o", str(out)])
    assert res.exit_code == 0
    assert "not a pdf" in res.output.lower()
    assert not (out / "pdfs" / "arxiv-2401.1.pdf").exists()
