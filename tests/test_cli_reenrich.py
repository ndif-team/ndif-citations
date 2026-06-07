import json
from click.testing import CliRunner
from ndif_citations.cli import cli
from ndif_citations import enrichment


def _write_catalog(out, papers):
    out.mkdir(parents=True, exist_ok=True)
    (out / "research-papers-full.json").write_text(json.dumps(
        {"verified": papers, "pending": [], "discarded": []}))


def test_reenrich_dry_run_writes_nothing(tmp_path, monkeypatch):
    out = tmp_path / "output"
    _write_catalog(out, [{"title": "P", "arxiv_id": "2401.1", "abstract": "snippet …",
                          "source": "scholar"}])
    monkeypatch.setattr(enrichment, "resolve_identifiers", lambda p: enrichment.ResolveResult(resolved=False))
    monkeypatch.setattr(enrichment, "fetch_records",
                        lambda p: [enrichment.Record("openalex", {"abstract": "Full. " * 60})])
    before = (out / "research-papers-full.json").read_text()
    res = CliRunner().invoke(cli, ["re-enrich", "-o", str(out), "--dry-run"])
    assert res.exit_code == 0, res.output
    assert (out / "research-papers-full.json").read_text() == before  # unchanged


def test_reenrich_rejects_unknown_fields(tmp_path):
    out = tmp_path / "output"
    _write_catalog(out, [{"title": "P", "arxiv_id": "2401.1", "abstract": "x", "source": "scholar"}])
    res = CliRunner().invoke(cli, ["re-enrich", "-o", str(out), "--fields", "venue,bogus"])
    assert res.exit_code != 0
    assert "venue" in res.output and "bogus" in res.output  # both flagged as unknown


def test_reenrich_applies_and_is_idempotent(tmp_path, monkeypatch):
    out = tmp_path / "output"
    _write_catalog(out, [{"title": "P", "arxiv_id": "2401.1", "abstract": "snippet …",
                          "source": "scholar"}])
    monkeypatch.setattr(enrichment, "resolve_identifiers", lambda p: enrichment.ResolveResult(resolved=False))
    monkeypatch.setattr(enrichment, "fetch_records",
                        lambda p: [enrichment.Record("openalex", {"abstract": "Full sentence. " * 60})])
    r1 = CliRunner().invoke(cli, ["re-enrich", "-o", str(out)])
    assert r1.exit_code == 0, r1.output
    data = json.loads((out / "research-papers-full.json").read_text())
    assert data["verified"][0]["abstract"].startswith("Full sentence.")
    r2 = CliRunner().invoke(cli, ["re-enrich", "-o", str(out)])
    assert "0 updated" in r2.output or "unchanged" in r2.output.lower()
