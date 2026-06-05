"""Parity test for the orchestrator-backed `cli.run` (Task 1.8).

The golden snapshot in ``tests/fixtures/golden/`` was captured from the LEGACY
inline ``cli.run`` (before it was re-pointed at ``orchestrator.run_pipeline``),
using ``install_pipeline_fakes`` on the source modules and the mini fixtures —
see the throwaway capture script documented in the Task 1.8 report.

This test drives the NEW ``cli.run`` (via ``CliRunner``) on a fresh copy of the
same fixtures, normalizes the same volatile field (``date_discovered``), and
asserts the resulting ``research-papers.json`` + ``research-papers-full.json``
EQUAL the committed golden. That proves the orchestrator-backed CLI is a
behavior-preserving swap of the legacy baseline.

It also asserts the CliRunner output contains the key phase markers so the
``_render_event`` renderer is exercised end to end.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from click.testing import CliRunner

import ndif_citations.orchestrator as orchestrator
from ndif_citations.cli import cli
from tests.helpers.fakes import install_pipeline_fakes

_FIXTURES = Path(__file__).parent / "fixtures"
_GOLDEN = _FIXTURES / "golden"


def _normalize_full(data: dict) -> dict:
    """Zero out the only volatile field (date_discovered) in the full JSON.

    Mirrors the normalization used when the golden was captured. ``run_date`` is
    not serialized into either output file, and the website JSON has no datetime
    fields, so this is the complete volatile-field set.
    """
    for bucket in ("pending", "verified", "discarded"):
        for p in data.get(bucket, []):
            if "date_discovered" in p:
                p["date_discovered"] = ""
    return data


def test_cli_run_matches_golden(monkeypatch, tmp_path):
    """NEW cli.run output equals the legacy-captured golden (volatile fields normalized).

    The new ``cli.run`` drives ``orchestrator.run_pipeline``, and the orchestrator
    imports every pipeline callable at MODULE TOP (its own bindings — distinct from
    ``ndif_citations.discover``'s). So the fakes are installed on the ORCHESTRATOR
    module (``install_pipeline_fakes(monkeypatch, orchestrator)``), not on
    ``discover``. (The legacy ``cli.run`` used in-function ``from ... import``, so
    the golden was captured by patching ``discover`` instead — see the Task 1.8
    report.) Both reach the identical fakes; only the patch target differs because
    the orchestrator binds the names eagerly.
    """
    install_pipeline_fakes(monkeypatch, orchestrator)

    # Fresh fixture_state copy (inline so we control the exact out dir passed to -o).
    out = tmp_path / "output"
    out.mkdir()
    (out / "images").mkdir()
    (out / "raw").mkdir()
    shutil.copy(_FIXTURES / "mini-research-papers-full.json", out / "research-papers-full.json")
    shutil.copy(_FIXTURES / "mini-github-repos-full.json", out / "github-repos-full.json")

    runner = CliRunner()
    result = runner.invoke(
        cli, ["run", "-o", str(out), "--skip-github"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output

    # --- Output-file parity ---
    website = json.loads((out / "research-papers.json").read_text())
    full = _normalize_full(json.loads((out / "research-papers-full.json").read_text()))

    golden_website = json.loads((_GOLDEN / "research-papers.json").read_text())
    golden_full = json.loads((_GOLDEN / "research-papers-full.json").read_text())

    assert website == golden_website, "research-papers.json diverged from legacy golden"
    assert full == golden_full, "research-papers-full.json diverged from legacy golden"

    # --- Renderer exercised: key phase markers present in console output ---
    out_text = result.output
    assert "Phase 1:" in out_text
    assert "Discovery" in out_text
    assert "Phase 2.5:" in out_text
    assert "Routing" in out_text
    # routing summary line ("Papers — N to process, M skipped")
    assert "to process" in out_text and "skipped" in out_text
    # final report still rendered directly by print_report
    assert "Run Complete" in out_text
