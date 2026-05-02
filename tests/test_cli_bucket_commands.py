"""Tests for promote / demote / discard CLI commands (US-D8)."""
import json
import os

import pytest
from click.testing import CliRunner

from ndif_citations.cli import cli
from ndif_citations.models import Bucket, Category, PaperReason, PipelineRun
from ndif_citations.output import write_outputs
from tests.conftest import make_paper


def _write_papers(tmp_path, papers):
    """Write papers to the 3-bucket JSON structure for CLI commands to load."""
    run = PipelineRun()
    write_outputs(papers, tmp_path, run)


class TestPromoteCommand:
    def test_promote_pending_paper(self, tmp_path):
        paper = make_paper(
            arxiv_id="2401.00001", title="Pending Paper", bucket=Bucket.PENDING, year=2024
        )
        paper.reason = PaperReason.OPENALEX_SOURCE
        _write_papers(tmp_path, [paper])

        runner = CliRunner()
        result = runner.invoke(cli, ["promote", "2401.00001", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output

        data = json.loads((tmp_path / "research-papers-full.json").read_text())
        verified = data["verified"]
        assert len(verified) == 1
        assert verified[0]["bucket"] == "verified"
        assert verified[0]["manual_override"] is True
        assert verified[0]["reason"] is None

    def test_promote_dry_run_no_file_change(self, tmp_path):
        paper = make_paper(arxiv_id="2401.00001", bucket=Bucket.PENDING, year=2024)
        _write_papers(tmp_path, [paper])
        mtime_before = os.path.getmtime(tmp_path / "research-papers-full.json")

        runner = CliRunner()
        result = runner.invoke(cli, ["promote", "2401.00001", "--output-dir", str(tmp_path), "--dry-run"])
        assert result.exit_code == 0
        assert os.path.getmtime(tmp_path / "research-papers-full.json") == mtime_before

    def test_promote_unknown_id_warns(self, tmp_path):
        paper = make_paper(arxiv_id="2401.00001", year=2024)
        _write_papers(tmp_path, [paper])

        runner = CliRunner()
        result = runner.invoke(cli, ["promote", "9999.99999", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "WARNING" in result.output or "not found" in result.output.lower()


class TestDemoteCommand:
    def test_demote_verified_paper(self, tmp_path):
        paper = make_paper(
            arxiv_id="2401.00001", title="Verified Paper", bucket=Bucket.VERIFIED,
            category=Category.REFERENCING, year=2024
        )
        _write_papers(tmp_path, [paper])

        runner = CliRunner()
        result = runner.invoke(cli, [
            "demote", "2401.00001",
            "--reason", "manual_demote",
            "--output-dir", str(tmp_path),
        ])
        assert result.exit_code == 0, result.output

        data = json.loads((tmp_path / "research-papers-full.json").read_text())
        pending = data["pending"]
        assert len(pending) == 1
        assert pending[0]["bucket"] == "pending"
        assert pending[0]["reason"] == "manual_demote"
        assert pending[0]["manual_override"] is True

    def test_demote_dry_run_no_file_change(self, tmp_path):
        paper = make_paper(arxiv_id="2401.00001", year=2024)
        _write_papers(tmp_path, [paper])
        mtime_before = os.path.getmtime(tmp_path / "research-papers-full.json")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "demote", "2401.00001", "--reason", "manual_demote",
            "--output-dir", str(tmp_path), "--dry-run",
        ])
        assert result.exit_code == 0
        assert os.path.getmtime(tmp_path / "research-papers-full.json") == mtime_before

    def test_demote_unknown_id_warns(self, tmp_path):
        paper = make_paper(arxiv_id="2401.00001", year=2024)
        _write_papers(tmp_path, [paper])

        runner = CliRunner()
        result = runner.invoke(cli, [
            "demote", "9999.99999", "--reason", "manual_demote",
            "--output-dir", str(tmp_path),
        ])
        assert result.exit_code == 0
        assert "WARNING" in result.output or "not found" in result.output.lower()


class TestDiscardCommand:
    def test_discard_paper_sets_reason_and_override(self, tmp_path):
        paper = make_paper(
            arxiv_id="2401.00001", title="False Positive", bucket=Bucket.VERIFIED, year=2024
        )
        _write_papers(tmp_path, [paper])

        runner = CliRunner()
        result = runner.invoke(cli, [
            "discard", "2401.00001",
            "--detail", "OpenAlex false positive — SVD survey",
            "--output-dir", str(tmp_path),
        ])
        assert result.exit_code == 0, result.output

        data = json.loads((tmp_path / "research-papers-full.json").read_text())
        discarded = data["discarded"]
        assert len(discarded) == 1
        assert discarded[0]["bucket"] == "discarded"
        assert discarded[0]["reason"] == "manual_discard"
        assert discarded[0]["reason_detail"] == "OpenAlex false positive — SVD survey"
        assert discarded[0]["manual_override"] is True

    def test_discard_dry_run_no_file_change(self, tmp_path):
        paper = make_paper(arxiv_id="2401.00001", year=2024)
        _write_papers(tmp_path, [paper])
        mtime_before = os.path.getmtime(tmp_path / "research-papers-full.json")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "discard", "2401.00001", "--output-dir", str(tmp_path), "--dry-run",
        ])
        assert result.exit_code == 0
        assert os.path.getmtime(tmp_path / "research-papers-full.json") == mtime_before

    def test_discard_unknown_id_warns(self, tmp_path):
        paper = make_paper(arxiv_id="2401.00001", year=2024)
        _write_papers(tmp_path, [paper])

        runner = CliRunner()
        result = runner.invoke(cli, [
            "discard", "9999.99999", "--output-dir", str(tmp_path),
        ])
        assert result.exit_code == 0
        assert "WARNING" in result.output or "not found" in result.output.lower()
