"""Tests for `python -m ndif_citations debug <id>` CLI command (US-F5)."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from ndif_citations.cli import cli
from ndif_citations.models import DetailCategory
from tests.conftest import make_paper

SECTION_HEADERS = [
    "1. Identifiers",
    "2. PDF cache check",
    "3. Text extraction",
    "4. Keyword hits",
    "5. Abstract scan",
    "6. Classification",
    "7. Final verdict",
]


def _write_full_json(tmp_path: Path, papers: list) -> Path:
    """Write a research-papers-full.json to tmp_path/output/."""
    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)
    data = [p.to_full_dict() for p in papers]
    full_json = out / "research-papers-full.json"
    full_json.write_text(json.dumps(data, indent=2, default=str))
    return out


class TestDebugCommand:
    def test_all_seven_sections_present(self, monkeypatch, tmp_path):
        paper = make_paper(
            title="Test Debug Paper",
            arxiv_id="2407.99999",
            detail_category=DetailCategory.USES_NNSIGHT,
            category_confidence=0.85,
        )
        out = _write_full_json(tmp_path, [paper])
        # Patch get_cached_pdf to return None (no PDF in test env)
        monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf", lambda p, d: None)

        runner = CliRunner()
        result = runner.invoke(cli, ["debug", "2407.99999", "--output-dir", str(out)])
        assert result.exit_code == 0, result.output
        for section in SECTION_HEADERS:
            assert section in result.output, f"Missing section: {section!r}\n{result.output}"

    def test_paper_id_not_in_dataset_warns(self, monkeypatch, tmp_path):
        out = _write_full_json(tmp_path, [])  # empty dataset
        monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf", lambda p, d: None)
        runner = CliRunner()
        result = runner.invoke(cli, ["debug", "9999.99999", "--output-dir", str(out)])
        assert result.exit_code == 0
        assert "not found" in result.output.lower() or "WARNING" in result.output

    def test_title_appears_in_output(self, monkeypatch, tmp_path):
        paper = make_paper(title="My Unique Test Paper", arxiv_id="2407.11111")
        out = _write_full_json(tmp_path, [paper])
        monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf", lambda p, d: None)
        runner = CliRunner()
        result = runner.invoke(cli, ["debug", "2407.11111", "--output-dir", str(out)])
        assert "My Unique Test Paper" in result.output

    def test_does_not_modify_output_files(self, monkeypatch, tmp_path):
        paper = make_paper(arxiv_id="2407.22222", title="Immutable Paper")
        out = _write_full_json(tmp_path, [paper])
        full_json = out / "research-papers-full.json"
        original_mtime = full_json.stat().st_mtime
        monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf", lambda p, d: None)
        runner = CliRunner()
        runner.invoke(cli, ["debug", "2407.22222", "--output-dir", str(out)])
        assert full_json.stat().st_mtime == original_mtime, "debug command modified the output file!"

    def test_out_file_option_writes_to_file(self, monkeypatch, tmp_path):
        paper = make_paper(arxiv_id="2407.33333", title="Out File Paper")
        out = _write_full_json(tmp_path, [paper])
        trace_file = tmp_path / "trace.txt"
        monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf", lambda p, d: None)
        runner = CliRunner()
        runner.invoke(cli, ["debug", "2407.33333", "--output-dir", str(out), "--out", str(trace_file)])
        assert trace_file.exists()
        content = trace_file.read_text()
        assert "2407.33333" in content or "Out File Paper" in content
