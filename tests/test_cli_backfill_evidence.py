import json
from click.testing import CliRunner
from ndif_citations.cli import cli

def _cat(out, papers):
    out.mkdir(parents=True, exist_ok=True)
    (out / "research-papers-full.json").write_text(json.dumps({"verified": papers, "pending": [], "discarded": []}))

def test_backfill_evidence_populates(tmp_path, monkeypatch):
    out = tmp_path / "output"
    _cat(out, [{"title": "P", "arxiv_id": "2401.1", "abstract": "we use nnsight"}])
    monkeypatch.setattr("ndif_citations.process.compute_context",
                        lambda paper, o: (["window about nnsight"], "abstract", None))
    res = CliRunner().invoke(cli, ["backfill-evidence", "-o", str(out)])
    assert res.exit_code == 0, res.output
    data = json.loads((out / "research-papers-full.json").read_text())
    assert data["verified"][0]["ndif_context_windows"] == ["window about nnsight"]
    assert data["verified"][0]["context_source"] == "abstract"

def test_backfill_evidence_dry_run_writes_nothing(tmp_path, monkeypatch):
    out = tmp_path / "output"
    _cat(out, [{"title": "P", "arxiv_id": "2401.1", "abstract": "we use nnsight"}])
    monkeypatch.setattr("ndif_citations.process.compute_context",
                        lambda paper, o: (["w"], "abstract", None))
    before = (out / "research-papers-full.json").read_text()
    res = CliRunner().invoke(cli, ["backfill-evidence", "-o", str(out), "--dry-run"])
    assert res.exit_code == 0, res.output
    assert (out / "research-papers-full.json").read_text() == before
