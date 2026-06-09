"""F-013 — a live pipeline run must snapshot the catalog before overwriting it.

A run finalizes by overwriting output/research-papers-full.json in place. Before
this fix nothing backed it up first (only the re-enrich/backfill/csv tools did),
so a run's mutations could not be cleanly identified or reverted. finalize_stage
now snapshots the catalog to output/backups/<name>.<ts>.pre-run.json first.
"""
from __future__ import annotations

import json
from pathlib import Path

from ndif_citations import orchestrator
from ndif_citations.output import backup_outputs


def _list_pre_run_backups(output_dir: Path) -> list[Path]:
    return sorted((output_dir / "backups").glob("research-papers-full.*.pre-run.json"))


# ---------------------------------------------------------------------------
# Unit: backup_outputs snapshots the existing catalog, no-ops when absent
# ---------------------------------------------------------------------------

def test_backup_outputs_snapshots_existing_catalog(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    catalog = output_dir / "research-papers-full.json"
    original = {"pending": [], "verified": [{"title": "Keep me"}], "discarded": []}
    catalog.write_text(json.dumps(original))

    created = backup_outputs(output_dir)

    backups = _list_pre_run_backups(output_dir)
    assert len(backups) == 1
    assert backups[0] in created
    # The snapshot holds the OLD contents (taken before any overwrite).
    assert json.loads(backups[0].read_text()) == original


def test_backup_outputs_noop_when_catalog_absent(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    created = backup_outputs(output_dir)

    assert created == []
    assert not (output_dir / "backups").exists()


# ---------------------------------------------------------------------------
# Integration: a real run backs up the pre-run catalog before overwriting
# ---------------------------------------------------------------------------

def test_run_pipeline_backs_up_catalog_before_overwrite(monkeypatch, fixture_state: Path):
    from tests.helpers.fakes import install_pipeline_fakes

    catalog = fixture_state / "research-papers-full.json"
    pre_run_contents = json.loads(catalog.read_text())

    install_pipeline_fakes(monkeypatch, orchestrator)
    orchestrator.run_pipeline(fixture_state, mode="incremental", skip_github=True)

    backups = _list_pre_run_backups(fixture_state)
    assert backups, "a .pre-run.json backup must be written before the run overwrites the catalog"
    # The backup is the PRE-run snapshot: it matches the original and lacks the
    # brand-new paper the run just added (which the live file now contains).
    backed_up = json.loads(backups[-1].read_text())
    assert backed_up == pre_run_contents
    live_ids = {
        p.get("arxiv_id")
        for bucket in ("pending", "verified", "discarded")
        for p in json.loads(catalog.read_text())[bucket]
    }
    backup_ids = {
        p.get("arxiv_id")
        for bucket in ("pending", "verified", "discarded")
        for p in backed_up[bucket]
    }
    assert "9999.99999" in live_ids
    assert "9999.99999" not in backup_ids
