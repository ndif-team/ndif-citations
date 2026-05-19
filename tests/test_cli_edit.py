"""Tests for the `edit` CLI command."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from ndif_citations.cli import cli
from ndif_citations.models import Bucket, Category, DiscoveredPaper
from tests.conftest import make_paper


def _write_state(output_dir: Path, papers: list[DiscoveredPaper]) -> None:
    """Write a minimal research-papers-full.json with the 3-bucket structure."""
    state = {"pending": [], "verified": [], "discarded": []}
    for p in papers:
        state[p.bucket.value].append(p.to_full_dict())
    (output_dir / "research-papers-full.json").write_text(
        json.dumps(state, default=str)
    )


class TestEditCommand:
    def test_edit_single_field_via_menu(self, tmp_path):
        paper = make_paper(
            title="Original",
            arxiv_id="2407.14561",
            venue="ICLR 2025",
            year=2025,
            bucket=Bucket.VERIFIED,
            category=Category.USES_NDIF,
            has_classification=True,
        )
        _write_state(tmp_path, [paper])

        # Stdin: choose field 4 (Venue), input new value, q to save, y to confirm
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["edit", "2407.14561", "--output-dir", str(tmp_path)],
            input="4\nICLR 2025 (Spotlight)\nq\ny\n",
        )
        assert result.exit_code == 0, result.output

        state = json.loads((tmp_path / "research-papers-full.json").read_text())
        verified = state["verified"]
        assert len(verified) == 1
        assert verified[0]["venue"] == "ICLR 2025 (Spotlight)"
        assert verified[0]["manual_override"] is True

    def test_edit_abort_does_not_save(self, tmp_path):
        paper = make_paper(
            title="Original",
            arxiv_id="2407.14561",
            venue="ICLR 2025",
            year=2025,
            bucket=Bucket.VERIFIED,
            category=Category.USES_NDIF,
            has_classification=True,
        )
        _write_state(tmp_path, [paper])

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["edit", "2407.14561", "--output-dir", str(tmp_path)],
            input="4\nICLR 2025 (Spotlight)\na\n",
        )
        assert result.exit_code == 0

        state = json.loads((tmp_path / "research-papers-full.json").read_text())
        assert state["verified"][0]["venue"] == "ICLR 2025"
        assert state["verified"][0]["manual_override"] is False

    def test_edit_paper_not_found(self, tmp_path):
        _write_state(tmp_path, [])

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["edit", "9999.99999", "--output-dir", str(tmp_path)],
            input="",
        )
        assert "not found" in result.output.lower()

    def test_edit_with_set_flag_one_shot(self, tmp_path):
        """edit <id> --set venue=ICML\\ 2025 should bypass interactive prompt."""
        paper = make_paper(
            title="Original",
            arxiv_id="2407.14561",
            venue="ICLR 2025",
            year=2025,
            bucket=Bucket.VERIFIED,
            category=Category.USES_NDIF,
            has_classification=True,
        )
        _write_state(tmp_path, [paper])

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["edit", "2407.14561",
             "--set", "venue=ICML 2025",
             "--set", "category=uses_nnsight",
             "--output-dir", str(tmp_path),
             "--yes"],
        )
        assert result.exit_code == 0, result.output

        state = json.loads((tmp_path / "research-papers-full.json").read_text())
        all_papers = state["verified"] + state["pending"] + state["discarded"]
        target = [p for p in all_papers if p.get("arxiv_id") == "2407.14561"][0]
        assert target["venue"] == "ICML 2025"
        assert target["category"] == "uses_nnsight"
        assert target["manual_override"] is True

    def test_edit_dry_run_does_not_write(self, tmp_path):
        paper = make_paper(
            title="Original",
            arxiv_id="2407.14561",
            venue="ICLR 2025",
            year=2025,
            bucket=Bucket.VERIFIED,
            category=Category.USES_NDIF,
            has_classification=True,
        )
        _write_state(tmp_path, [paper])

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["edit", "2407.14561",
             "--set", "venue=ICML 2025",
             "--dry-run",
             "--output-dir", str(tmp_path),
             "--yes"],
        )
        assert result.exit_code == 0

        state = json.loads((tmp_path / "research-papers-full.json").read_text())
        assert state["verified"][0]["venue"] == "ICLR 2025"  # unchanged
