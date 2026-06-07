from ndif_citations import enrichment
from ndif_citations.enrichment import enrich_paper, Record
from tests.conftest import make_paper


def _stub_records(*records):
    return lambda paper: list(records)


def test_enrich_replaces_broken_abstract_and_records_provenance(monkeypatch):
    p = make_paper(arxiv_id="2401.1", abstract="snippet about models …", manual_override=False)
    monkeypatch.setattr(enrichment, "resolve_identifiers", lambda paper: enrichment.ResolveResult(resolved=False))
    monkeypatch.setattr(enrichment, "fetch_records",
                        _stub_records(Record("openalex", {"abstract": "Full abstract. " * 40})))
    cs = enrich_paper(p)
    assert p.abstract.startswith("Full abstract.")
    assert p.enrichment_provenance.get("abstract") == "openalex"
    assert cs.changes  # non-empty change set


def test_enrich_locked_paper_fills_only_empty(monkeypatch):
    p = make_paper(arxiv_id="2401.2", abstract="Existing curated full abstract. " * 30,
                   affiliations="", manual_override=True)
    monkeypatch.setattr(enrichment, "resolve_identifiers", lambda paper: enrichment.ResolveResult(resolved=False))
    monkeypatch.setattr(enrichment, "fetch_records",
                        _stub_records(Record("openalex",
                            {"abstract": "DIFFERENT full abstract. " * 30, "affiliations": "MIT, Stanford"})))
    before_abstract = p.abstract
    enrich_paper(p)
    assert p.abstract == before_abstract          # locked, non-empty: untouched
    assert p.affiliations == "MIT, Stanford"        # locked, was empty: filled


def test_enrich_locked_broken_nonempty_is_untouched(monkeypatch):
    # A curator-locked paper with a truncated (broken) BUT non-empty abstract must
    # not be replaced — manual_override protects even low-quality curated values.
    p = make_paper(arxiv_id="2401.5", abstract="Curator-truncated abstract …", manual_override=True)
    monkeypatch.setattr(enrichment, "resolve_identifiers", lambda paper: enrichment.ResolveResult(resolved=False))
    monkeypatch.setattr(enrichment, "fetch_records",
                        _stub_records(Record("openalex", {"abstract": "Full replacement abstract. " * 40})))
    enrich_paper(p)
    assert p.abstract == "Curator-truncated abstract …"
    assert "abstract" not in p.enrichment_provenance


def test_enrich_clean_paper_unchanged(monkeypatch):
    good = "A clean complete abstract. " * 40
    p = make_paper(arxiv_id="2401.3", abstract=good)
    monkeypatch.setattr(enrichment, "resolve_identifiers", lambda paper: enrichment.ResolveResult(resolved=False))
    monkeypatch.setattr(enrichment, "fetch_records",
                        _stub_records(Record("openalex", {"abstract": "Other complete abstract. " * 40})))
    enrich_paper(p)
    assert p.abstract == good                       # regression guard holds


def test_enrich_dry_run_does_not_mutate(monkeypatch):
    p = make_paper(arxiv_id="2401.4", abstract="snippet …", manual_override=False)
    monkeypatch.setattr(enrichment, "resolve_identifiers", lambda paper: enrichment.ResolveResult(resolved=False))
    monkeypatch.setattr(enrichment, "fetch_records",
                        _stub_records(Record("openalex", {"abstract": "Full abstract. " * 40})))
    cs = enrich_paper(p, dry_run=True)
    assert cs.changes                               # preview reports the change
    assert p.abstract == "snippet …"                # but original is untouched
    assert p.enrichment_provenance == {}            # and provenance not written
