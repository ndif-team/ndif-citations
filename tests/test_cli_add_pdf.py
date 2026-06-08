"""Tests for the add-pdf CLI command + the synchronous seeded-add helper."""
from __future__ import annotations
from click.testing import CliRunner
from ndif_citations.cli import cli
from ndif_citations.models import PipelineRun


def test_add_pdf_invokes_manual_add(tmp_path, monkeypatch):
    out = tmp_path / "output"; out.mkdir()
    (out / "research-papers-full.json").write_text('{"verified":[],"pending":[],"discarded":[]}')
    pdf = tmp_path / "p.pdf"; pdf.write_bytes(b"%PDF-1.4\nx\n")
    called = {}
    def fake_run(o, seed_papers, pdf_bytes=None, **k):
        called["title"] = seed_papers[0].title; called["pdf"] = pdf_bytes
        from ndif_citations import orchestrator
        return orchestrator.FinalizeResult(merged_papers=seed_papers, merged_repos=[], run_stats=PipelineRun())
    monkeypatch.setattr("ndif_citations.manual_add.run_manual_add_seed", fake_run, raising=False)
    res = CliRunner().invoke(cli, ["add-pdf", str(pdf), "--title", "Paywalled", "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert called["title"] == "Paywalled"
    assert called["pdf"] == b"%PDF-1.4\nx\n"


def test_add_pdf_rejects_non_pdf_file(tmp_path):
    out = tmp_path / "output"; out.mkdir()
    (out / "research-papers-full.json").write_text('{"verified":[],"pending":[],"discarded":[]}')
    bad = tmp_path / "bad.pdf"; bad.write_bytes(b"not a pdf")
    res = CliRunner().invoke(cli, ["add-pdf", str(bad), "--title", "X", "-o", str(out)])
    assert res.exit_code == 0
    assert "not a pdf" in res.output.lower()


def test_run_manual_add_seed_caches_pdf_under_enriched_id(tmp_path, monkeypatch):
    """PDF is cached AFTER enrich using the POST-enrich arXiv id (not the seed's blank id)."""
    from ndif_citations import orchestrator, manual_add
    from ndif_citations.models import DiscoveredPaper, DiscoverySource
    out = tmp_path / "output"; out.mkdir()
    seed = DiscoveredPaper(title="P", source=DiscoverySource.MANUAL_ADD)  # no arxiv_id yet
    def fake_enrich(o, d, **k):
        d.papers[0].arxiv_id = "2404.12121"  # enrichment resolves an id
        return orchestrator.EnrichResult(papers=d.papers, repos=[], removal_counts={}, existing_repos=[])
    monkeypatch.setattr(orchestrator, "enrich_stage", fake_enrich)
    monkeypatch.setattr(orchestrator, "route_stage",
                        lambda o, e, **k: orchestrator.RouteResult(paper_decisions=[], repo_decisions=[], existing_papers=[], enrich=e))
    monkeypatch.setattr(orchestrator, "process_stage", lambda o, r, **k: (0, 0))
    monkeypatch.setattr(orchestrator, "finalize_stage",
                        lambda o, r, rs, **k: orchestrator.FinalizeResult(merged_papers=[], merged_repos=[], run_stats=rs))
    manual_add.run_manual_add_seed(out, [seed], pdf_bytes=b"%PDF-1.4\nz\n")
    assert (out / "pdfs" / "arxiv-2404.12121.pdf").read_bytes() == b"%PDF-1.4\nz\n"  # post-enrich id
