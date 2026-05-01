"""Tests for `python -m ndif_citations reclassify` CLI command (US-Q5)."""
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from ndif_citations.cli import cli
from ndif_citations.models import DetailCategory, DiscoveredPaper
from tests.conftest import make_paper
from tests.helpers.llm import MockLLMClient


def _write_full_json(tmp_path: Path, papers: list) -> Path:
    """Write research-papers-full.json to tmp_path/output/ and return the output dir."""
    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)
    data = [p.to_full_dict() for p in papers]
    full_json = out / "research-papers-full.json"
    full_json.write_text(json.dumps(data, indent=2, default=str))
    return out


def _inject_mock(monkeypatch, mock: MockLLMClient) -> None:
    monkeypatch.setattr("ndif_citations.process._get_llm_client", lambda: mock)


def _no_pdf(monkeypatch) -> None:
    monkeypatch.setattr("ndif_citations.pdf_cache.get_cached_pdf", lambda p, d: None)


def _fake_pdf(monkeypatch, tmp_path: Path, context_text: str) -> Path:
    """Create a fake PDF and patch extract_ndif_context to return context_text."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(
        "ndif_citations.process.extract_ndif_context",
        lambda path, window=500: context_text,
    )
    monkeypatch.setattr(
        "ndif_citations.pdf_cache.get_cached_pdf",
        lambda p, d: pdf,
    )
    return pdf


class TestReclassifyDryRun:
    def test_dry_run_does_not_write_files(self, monkeypatch, tmp_path):
        paper = make_paper(
            arxiv_id="2604.00001",
            title="Test Paper",
            detail_category=DetailCategory.USES_NNSIGHT,
        )
        out = _write_full_json(tmp_path, [paper])
        full_json = out / "research-papers-full.json"
        original_mtime = full_json.stat().st_mtime

        mock = MockLLMClient()
        mock.expect("referencing")
        _inject_mock(monkeypatch, mock)
        _fake_pdf(monkeypatch, tmp_path, "Removing the dependency on nnsight.")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["reclassify", "--ids", "2604.00001", "--dry-run", "--output-dir", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert full_json.stat().st_mtime == original_mtime, "dry-run wrote to disk!"

    def test_dry_run_prints_changes(self, monkeypatch, tmp_path):
        paper = make_paper(
            arxiv_id="2604.00002",
            title="ADAG Paper",
            detail_category=DetailCategory.USES_NNSIGHT,
        )
        out = _write_full_json(tmp_path, [paper])
        _inject_mock(monkeypatch, mock := MockLLMClient())
        _fake_pdf(monkeypatch, tmp_path, "Removing the dependency on nnsight and replacing with hooks.")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["reclassify", "--ids", "2604.00002", "--dry-run", "--output-dir", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert "uses_nnsight" in result.output
        assert "referencing" in result.output


class TestReclassifyIdsFilter:
    def test_only_specified_paper_is_processed(self, monkeypatch, tmp_path):
        paper_a = make_paper(arxiv_id="2604.00010", title="Paper A",
                             detail_category=DetailCategory.USES_NNSIGHT)
        paper_b = make_paper(arxiv_id="2604.00011", title="Paper B",
                             detail_category=DetailCategory.USES_NNSIGHT)
        out = _write_full_json(tmp_path, [paper_a, paper_b])

        mock = MockLLMClient()
        _inject_mock(monkeypatch, mock)
        _fake_pdf(monkeypatch, tmp_path, "Removing the dependency on nnsight.")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["reclassify", "--ids", "2604.00010", "--output-dir", str(out)]
        )
        assert result.exit_code == 0, result.output

        # Only paper_a should change; paper_b should be untouched
        data = json.loads((out / "research-papers-full.json").read_text())
        by_id = {d["arxiv_id"]: d for d in data}
        assert by_id["2604.00010"]["detail_category"] == "referencing"
        assert by_id["2604.00011"]["detail_category"] == "uses_nnsight"

    def test_unknown_id_warns_and_continues(self, monkeypatch, tmp_path):
        paper = make_paper(arxiv_id="2604.00020", title="Known Paper",
                           detail_category=DetailCategory.REFERENCING)
        out = _write_full_json(tmp_path, [paper])
        _inject_mock(monkeypatch, MockLLMClient())
        _no_pdf(monkeypatch)

        runner = CliRunner()
        result = runner.invoke(
            cli, ["reclassify", "--ids", "9999.99999", "--output-dir", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert "WARNING" in result.output or "not found" in result.output.lower()


class TestReclassifyManualOverride:
    def test_manual_override_paper_is_skipped(self, monkeypatch, tmp_path):
        paper = make_paper(
            arxiv_id="2604.00030",
            title="Protected Paper",
            detail_category=DetailCategory.USES_NNSIGHT,
            manual_override=True,
        )
        out = _write_full_json(tmp_path, [paper])
        _inject_mock(monkeypatch, mock := MockLLMClient())
        _fake_pdf(monkeypatch, tmp_path, "Removing the dependency on nnsight.")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["reclassify", "--ids", "2604.00030", "--output-dir", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert "SKIP" in result.output or "manual_override" in result.output.lower()

        # Paper should be unchanged in the output file
        data = json.loads((out / "research-papers-full.json").read_text())
        assert data[0]["detail_category"] == "uses_nnsight"
        assert data[0]["manual_override"] is True


class TestReclassifyOutputDiff:
    def test_changed_paper_shows_old_and_new_category(self, monkeypatch, tmp_path):
        paper = make_paper(
            arxiv_id="2604.00040",
            title="Changed Paper",
            detail_category=DetailCategory.USES_NNSIGHT,
        )
        out = _write_full_json(tmp_path, [paper])
        _inject_mock(monkeypatch, MockLLMClient())
        _fake_pdf(monkeypatch, tmp_path, "Removing the dependency on nnsight.")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["reclassify", "--ids", "2604.00040", "--output-dir", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert "uses_nnsight" in result.output
        assert "referencing" in result.output

    def test_unchanged_paper_shows_no_changes(self, monkeypatch, tmp_path):
        paper = make_paper(
            arxiv_id="2604.00050",
            title="Already Correct",
            detail_category=DetailCategory.REFERENCING,
            category_confidence=0.85,
        )
        out = _write_full_json(tmp_path, [paper])
        mock = MockLLMClient()
        mock.expect("referencing")
        _inject_mock(monkeypatch, mock)
        # Provide a context window that triggers the LLM and returns referencing
        _fake_pdf(monkeypatch, tmp_path, "NNsight is listed in related work only.")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["reclassify", "--ids", "2604.00050", "--output-dir", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert "No classification changes" in result.output


class TestReclassifyDOIResolution:
    def test_doi_resolution(self, monkeypatch, tmp_path):
        paper = make_paper(
            arxiv_id=None,
            doi="10.1234/abc",
            title="DOI Paper",
            detail_category=DetailCategory.USES_NNSIGHT,
        )
        out = _write_full_json(tmp_path, [paper])
        _inject_mock(monkeypatch, MockLLMClient())
        _fake_pdf(monkeypatch, tmp_path, "Removing the dependency on nnsight.")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["reclassify", "--ids", "10.1234/abc", "--output-dir", str(out)]
        )
        assert result.exit_code == 0, result.output

        data = json.loads((out / "research-papers-full.json").read_text())
        assert data[0]["detail_category"] == "referencing"
