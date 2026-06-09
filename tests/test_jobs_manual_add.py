import time
import pytest
from ndif_citations import orchestrator
from ndif_citations.jobs import JobRunner
from ndif_citations.models import DiscoveredPaper, DiscoverySource, PipelineRun
from ndif_citations.router import RoutingDecision, ProcessingBucket


def _wait(pred, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def _new_decision(paper):
    return RoutingDecision(
        paper=paper,
        bucket=ProcessingBucket.NEW,
        existing_paper=None,
        processing_needed={"summary": True, "classify": True, "thumbnail": True, "affiliations": True},
    )


def _fake_stages(monkeypatch):
    def fake_enrich(o, d, **k):
        return orchestrator.EnrichResult(papers=d.papers, repos=[], removal_counts={}, existing_repos=[])
    def fake_route(o, e, **k):
        return orchestrator.RouteResult(
            paper_decisions=[_new_decision(p) for p in e.papers],
            repo_decisions=[], existing_papers=[], enrich=e,
        )
    def fake_process(o, r, **k):
        return (0, 0)
    def fake_finalize(o, r, run_stats, *, completed=None, **k):
        from ndif_citations.output import write_outputs
        papers = [d.paper for d in r.paper_decisions]
        write_outputs(papers, o, run_stats)
        return orchestrator.FinalizeResult(merged_papers=papers, merged_repos=[], run_stats=run_stats)
    monkeypatch.setattr(orchestrator, "enrich_stage", fake_enrich)
    monkeypatch.setattr(orchestrator, "route_stage", fake_route)
    monkeypatch.setattr(orchestrator, "process_stage", fake_process)
    monkeypatch.setattr(orchestrator, "finalize_stage", fake_finalize)


def test_manual_add_runs_ungated_to_done(tmp_path, monkeypatch):
    """manual-add runs straight to 'done' without parking at awaiting_review."""
    out = tmp_path / "output"; out.mkdir()
    (out / "research-papers-full.json").write_text('{"verified":[],"pending":[],"discarded":[]}')
    _fake_stages(monkeypatch)
    seed = DiscoveredPaper(title="Manually Added", arxiv_id="2401.55555", source=DiscoverySource.MANUAL_ADD)
    runner = JobRunner()
    runner.start_manual_add(out, seed)
    assert _wait(lambda: runner.status().state == "done"), runner.status().state
    assert runner.status().state != "awaiting_review", "manual-add must never park at the gate"


def test_manual_add_caches_pdf_after_enrich(tmp_path, monkeypatch):
    out = tmp_path / "output"; out.mkdir()
    (out / "research-papers-full.json").write_text('{"verified":[],"pending":[],"discarded":[]}')
    _fake_stages(monkeypatch)
    seed = DiscoveredPaper(title="Paywalled", arxiv_id="2402.99999", source=DiscoverySource.MANUAL_ADD)
    runner = JobRunner()
    pdf = b"%PDF-1.4\nmanual\n"  # MUST start with %PDF- magic — write_pdf_to_cache rejects otherwise
    runner.start_manual_add(out, seed, pdf_bytes=pdf)
    # PDF is cached inside run_manual_add_seed (after enrich_stage); verify after run completes.
    assert _wait(lambda: runner.status().state == "done"), runner.status().state
    assert (out / "pdfs" / "arxiv-2402.99999.pdf").read_bytes() == pdf
