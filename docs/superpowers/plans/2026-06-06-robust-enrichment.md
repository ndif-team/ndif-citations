# Robust Metadata Enrichment Implementation Plan (Plan A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconcile every paper's metadata (abstract, authors, affiliations, venue, year, identifiers) from authoritative sources (OpenAlex/arXiv/Crossref/S2) instead of keeping thin Google-Scholar data, via a smart compare-and-replace engine — wired into the pipeline `enrich` stage and exposed as a no-LLM `re-enrich` repair command.

**Architecture:** A new pure, unit-testable module `enrichment.py` holds the reconciliation engine (`is_broken`, `SOURCE_TRUST`, `reconcile_field`, `reconcile_paper`) plus the I/O orchestration (`resolve_identifiers`, `fetch_records`, `enrich_paper`) that reuses existing query helpers. `extract.enrich_papers` calls `enrich_paper` (forward path); a new `re-enrich` Click command runs it over the existing catalog (repair path). Curator-locked papers (`manual_override`) are fill-gaps-only.

**Tech Stack:** Python 3, Pydantic v2 (`DiscoveredPaper`), Click CLI, pytest + monkeypatch. Reuses `extract._openalex_fetch_work`, `discover._openalex_work_to_discovered`, `utils.query_arxiv_api/query_crossref/query_s2_publication_venue/extract_arxiv_id_from_url/rate_limit_sleep`, `output.load_existing_papers/write_outputs`, `config.get_output_dir`.

**Spec:** `docs/superpowers/specs/2026-06-06-robust-enrichment-design.md`

**Pre-flight:** `cd ndif-citations && pip install -e ".[dev]"`. Confirm baseline: `pytest tests/ -q` → 859 passed. Branch `feat/citations-web-app`.

---

### Task 1: Model — `enrichment_provenance` field

**Files:**
- Modify: `src/ndif_citations/models.py` (class `DiscoveredPaper`, near the other `Field(default_factory=...)` fields, ~line 149)
- Test: `tests/test_enrichment_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_enrichment_model.py
from ndif_citations.models import DiscoveredPaper


def test_enrichment_provenance_defaults_empty_and_roundtrips():
    p = DiscoveredPaper(title="X")
    assert p.enrichment_provenance == {}
    p.enrichment_provenance["abstract"] = "openalex"
    dumped = p.model_dump(mode="json")
    assert dumped["enrichment_provenance"] == {"abstract": "openalex"}
    restored = DiscoveredPaper.model_validate(dumped)
    assert restored.enrichment_provenance == {"abstract": "openalex"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_enrichment_model.py -q`
Expected: FAIL (`enrichment_provenance` not a field / KeyError in dumped).

- [ ] **Step 3: Add the field**

In `models.py`, inside `class DiscoveredPaper`, alongside the other defaulted fields:

```python
    # Provenance of enrichment-repaired fields: field name -> winning source
    # (e.g. {"abstract": "openalex", "authors": "arxiv"}). Mirrors `venue_source`.
    enrichment_provenance: dict[str, str] = Field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_enrichment_model.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite (no regressions from the new field)**

Run: `pytest tests/ -q`
Expected: all pass (existing serialization tests tolerate the new defaulted field).

- [ ] **Step 6: Commit**

```bash
git add src/ndif_citations/models.py tests/test_enrichment_model.py
git commit -m "feat(model): add enrichment_provenance to DiscoveredPaper"
```

---

### Task 2: `enrichment.py` — `is_broken` quality predicates (pure)

**Files:**
- Create: `src/ndif_citations/enrichment.py`
- Test: `tests/test_enrichment_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_enrichment_engine.py
import pytest
from ndif_citations.enrichment import is_broken


@pytest.mark.parametrize("value,expected", [
    ("", True),
    ("Short snippet about a model …", True),          # ellipsis
    ("Short snippet about a model ...", True),         # ascii ellipsis
    ("x" * 200, True),                                  # under 280-char floor
    ("A full abstract. " * 30, False),                  # long, complete
])
def test_is_broken_abstract(value, expected):
    assert is_broken("abstract", value) is expected


@pytest.mark.parametrize("value,expected", [
    ("", True),
    ("J. Smith, A. …", True),
    ("Jane Smith, Alan Turing", False),
])
def test_is_broken_authors(value, expected):
    assert is_broken("authors", value) is expected


def test_is_broken_venue_and_affiliations_and_year():
    assert is_broken("venue", "") is True
    assert is_broken("venue", "arXiv") is True          # weak (via venue._WEAK_VENUE_RE)
    assert is_broken("venue", "NeurIPS 2024") is False
    assert is_broken("affiliations", "") is True
    assert is_broken("affiliations", "MIT") is False
    assert is_broken("year", 0) is True
    assert is_broken("year", 2024) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_enrichment_engine.py -q`
Expected: FAIL (`ModuleNotFoundError: ndif_citations.enrichment`).

- [ ] **Step 3: Implement `is_broken`**

```python
# src/ndif_citations/enrichment.py
"""Authoritative-source metadata reconciliation engine + orchestration.

Pure helpers (is_broken / reconcile_field / reconcile_paper) take already-fetched
values and decide the best one; the fetch/orchestration helpers reuse existing
query functions. See docs/superpowers/specs/2026-06-06-robust-enrichment-design.md.
"""
from __future__ import annotations

from ndif_citations.venue import _WEAK_VENUE_RE

_ELLIPSIS = ("…", "...")
_ABSTRACT_MIN = 280


def is_broken(field: str, value) -> bool:
    """True if `value` for `field` is empty/truncated/low-quality."""
    if field == "year":
        return not value
    text = (value or "").strip() if isinstance(value, str) else ""
    if not text:
        return True
    if field == "abstract":
        return text.endswith(_ELLIPSIS) or len(text) < _ABSTRACT_MIN
    if field == "authors":
        return text.endswith(_ELLIPSIS) or "…" in text or text.rstrip().endswith("et al")
    if field == "venue":
        return bool(_WEAK_VENUE_RE.match(text))
    if field == "affiliations":
        return False  # non-empty affiliations are acceptable
    return False
```

(If `venue._WEAK_VENUE_RE` is not importable, confirm its name with `grep -n "_WEAK_VENUE_RE" src/ndif_citations/venue.py`; the spec references it as the existing weak-venue matcher.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_enrichment_engine.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ndif_citations/enrichment.py tests/test_enrichment_engine.py
git commit -m "feat(enrichment): is_broken field-quality predicates"
```

---

### Task 3: `enrichment.py` — `SOURCE_TRUST` + `reconcile_field` (pure)

**Files:**
- Modify: `src/ndif_citations/enrichment.py`
- Test: `tests/test_enrichment_engine.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_enrichment_engine.py
from ndif_citations.enrichment import reconcile_field, Candidate


def _c(value, source): return Candidate(value=value, source=source)

def test_reconcile_replaces_broken_with_valid():
    r = reconcile_field("abstract", _c("snippet …", "scholar"),
                         [_c("A" * 600, "openalex")])
    assert r.changed and r.value == "A" * 600 and r.source == "openalex"

def test_reconcile_keeps_good_over_lower_trust():
    good = "A clean full abstract. " * 30
    r = reconcile_field("abstract", _c(good, "openalex"),
                         [_c("Another full abstract. " * 30, "scholar")])
    assert r.changed is False and r.source == "openalex"

def test_reconcile_tie_on_validity_prefers_most_trusted():
    a = "Full abstract alpha. " * 30
    b = "Full abstract bravo. " * 30  # same length tier, different source
    r = reconcile_field("abstract", _c(a, "s2"), [_c(b, "openalex")])
    assert r.value == b and r.source == "openalex"

def test_reconcile_no_candidates_unchanged():
    r = reconcile_field("authors", _c("Jane Smith, Alan Turing", "s2"), [])
    assert r.changed is False

def test_reconcile_low_confidence_flag_propagates():
    r = reconcile_field("abstract", _c("snippet …", "scholar"),
                        [_c("A" * 600, "openalex")], low_confidence_sources={"openalex"})
    assert r.changed and r.low_confidence is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_enrichment_engine.py -q`
Expected: FAIL (`reconcile_field` / `Candidate` not defined).

- [ ] **Step 3: Implement `Candidate`, `Resolution`, `SOURCE_TRUST`, `reconcile_field`**

```python
# add to src/ndif_citations/enrichment.py
from dataclasses import dataclass


SOURCE_TRUST: dict[str, int] = {
    "openalex": 4, "crossref": 3, "arxiv": 3, "s2": 3,
    "manual_add": 2, "scholar": 1, "unknown": 0,
}


@dataclass(frozen=True)
class Candidate:
    value: object
    source: str


@dataclass(frozen=True)
class Resolution:
    value: object
    source: str
    changed: bool
    low_confidence: bool


def _trust(source: str) -> int:
    return SOURCE_TRUST.get(source, 0)


def _completeness(field: str, value) -> int:
    if field in ("abstract", "authors") and isinstance(value, str):
        return len(value)
    return 1


def _score(field: str, c: Candidate):
    # higher tuple wins: valid first, then trust, then completeness
    return (0 if is_broken(field, c.value) else 1, _trust(c.source), _completeness(field, c.value))


def reconcile_field(field, current: Candidate, candidates: list[Candidate],
                    low_confidence_sources: set[str] | None = None) -> Resolution:
    low_confidence_sources = low_confidence_sources or set()
    pool = [current] + [c for c in candidates
                        if (c.value not in (None, "", 0))]
    winner = max(pool, key=lambda c: _score(field, c))
    # Regression guard: never downgrade a non-broken current to an equal-or-worse value.
    if not is_broken(field, current.value) and _score(field, winner) <= _score(field, current):
        winner = current
    changed = winner.value != current.value
    low_conf = changed and winner.source in low_confidence_sources
    return Resolution(value=winner.value, source=winner.source, changed=changed, low_confidence=low_conf)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_enrichment_engine.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ndif_citations/enrichment.py tests/test_enrichment_engine.py
git commit -m "feat(enrichment): reconcile_field compare-and-replace engine"
```

---

### Task 4: `enrichment.py` — title similarity + `resolve_identifiers` (mocked)

**Files:**
- Modify: `src/ndif_citations/enrichment.py`
- Test: `tests/test_enrichment_resolve.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_enrichment_resolve.py
from ndif_citations import enrichment
from ndif_citations.enrichment import title_similarity, resolve_identifiers
from tests.conftest import make_paper


def test_title_similarity_threshold():
    assert title_similarity("Attention Is All You Need", "attention is all you need!") >= 0.90
    assert title_similarity("Attention Is All You Need", "A totally different paper") < 0.90


def test_resolve_parses_arxiv_from_url(monkeypatch):
    p = make_paper(arxiv_id=None, doi=None, url="https://arxiv.org/abs/2401.12345")
    # No OpenAlex needed — URL parse should win.
    changed = resolve_identifiers(p)
    assert changed is True and p.arxiv_id == "2401.12345"


def test_resolve_adopts_openalex_id_on_high_title_match(monkeypatch):
    p = make_paper(arxiv_id=None, doi=None, url="https://example.com/x", title="Sparse Probing of LLMs")
    work = {"id": "https://openalex.org/W123", "title": "Sparse Probing of LLMs",
            "ids": {"doi": "https://doi.org/10.1/abc"}}
    monkeypatch.setattr(enrichment, "_openalex_fetch_work", lambda ident, by="id": work)
    changed = resolve_identifiers(p)
    assert changed and p.openalex_id == "https://openalex.org/W123" and p.doi == "10.1/abc"


def test_resolve_rejects_low_title_match(monkeypatch):
    p = make_paper(arxiv_id=None, doi=None, url="https://example.com/x", title="Sparse Probing of LLMs")
    work = {"id": "https://openalex.org/W999", "title": "Unrelated Physics Paper", "ids": {}}
    monkeypatch.setattr(enrichment, "_openalex_fetch_work", lambda ident, by="id": work)
    changed = resolve_identifiers(p)
    assert changed is False and p.openalex_id in (None, "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_enrichment_resolve.py -q`
Expected: FAIL (`title_similarity` / `resolve_identifiers` not defined).

- [ ] **Step 3: Implement**

```python
# add to src/ndif_citations/enrichment.py
import re
from difflib import SequenceMatcher

from ndif_citations.extract import _openalex_fetch_work  # reused OpenAlex fetcher
from ndif_citations.utils import extract_arxiv_id_from_url

TITLE_MATCH_THRESHOLD = 0.90


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (t or "").lower()).strip()


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_title(a), _norm_title(b)).ratio()


def resolve_identifiers(paper) -> bool:
    """Resolve+persist missing ids. Returns True if anything was resolved.
    Sets paper._enrichment_via_title = True when an id was adopted via title.search
    (used to mark downstream fetches low-confidence)."""
    if paper.arxiv_id or paper.doi:
        return False
    # 1. cheap: parse arxiv id from URLs
    for u in (paper.url, paper.pdf_url):
        axid = extract_arxiv_id_from_url(u) if u else None
        if axid:
            paper.arxiv_id = axid
            return True
    # 2. OpenAlex title.search
    work = _openalex_fetch_work(f"title.search:{paper.title[:100]}", by="filter")
    if not work:
        return False
    if title_similarity(paper.title, work.get("title") or "") < TITLE_MATCH_THRESHOLD:
        return False
    paper.openalex_id = work.get("id") or paper.openalex_id
    ids = work.get("ids") or {}
    doi = (ids.get("doi") or "").replace("https://doi.org/", "")
    if doi and not paper.doi:
        paper.doi = doi
    object.__setattr__(paper, "_enrichment_via_title", True)
    return True
```

Note: `_openalex_fetch_work(f"title.search:...", by="filter")` mirrors `extract.py:_enrich_affiliations_from_openalex`. Importing `_openalex_fetch_work` into `enrichment.py` makes `monkeypatch.setattr(enrichment, "_openalex_fetch_work", ...)` work in tests.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_enrichment_resolve.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ndif_citations/enrichment.py tests/test_enrichment_resolve.py
git commit -m "feat(enrichment): identifier resolution (URL parse + OpenAlex title match >=0.90)"
```

---

### Task 5: `enrichment.py` — `fetch_records` (mocked helpers)

**Files:**
- Modify: `src/ndif_citations/enrichment.py`
- Test: `tests/test_enrichment_fetch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_enrichment_fetch.py
from ndif_citations import enrichment
from ndif_citations.enrichment import fetch_records
from ndif_citations.discover import _openalex_work_to_discovered  # noqa: F401 (patched target)
from tests.conftest import make_paper


def test_fetch_records_from_openalex(monkeypatch):
    p = make_paper(arxiv_id="2401.00001", abstract="snippet …", source=None)
    fake_paper = make_paper(abstract="A" * 600, venue="NeurIPS 2024")
    monkeypatch.setattr(enrichment, "_openalex_fetch_work", lambda ident, by="id": {"stub": True})
    monkeypatch.setattr(enrichment, "_openalex_work_to_discovered", lambda work: fake_paper)
    monkeypatch.setattr(enrichment, "query_arxiv_api", lambda ids: {})
    recs = fetch_records(p)
    # one record tagged source="openalex" carrying the extracted fields
    oa = [r for r in recs if r.source == "openalex"]
    assert oa and oa[0].fields["abstract"] == "A" * 600


def test_fetch_records_arxiv_authors(monkeypatch):
    p = make_paper(arxiv_id="2401.00002")
    monkeypatch.setattr(enrichment, "_openalex_fetch_work", lambda ident, by="id": None)
    monkeypatch.setattr(enrichment, "query_arxiv_api",
                        lambda ids: {"2401.00002": {"authors": ["A. One", "B. Two"], "affiliations": ["MIT"]}})
    recs = fetch_records(p)
    ax = [r for r in recs if r.source == "arxiv"]
    assert ax and ax[0].fields["authors"] == "A. One, B. Two"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_enrichment_fetch.py -q`
Expected: FAIL (`fetch_records` not defined).

- [ ] **Step 3: Implement `Record` + `fetch_records`**

```python
# add to src/ndif_citations/enrichment.py
from ndif_citations.discover import _openalex_work_to_discovered
from ndif_citations.utils import query_arxiv_api, rate_limit_sleep
from ndif_citations import config

_MANAGED_FIELDS = ("abstract", "authors", "affiliations", "venue", "year")


@dataclass(frozen=True)
class Record:
    source: str
    fields: dict


def _openalex_record(paper) -> Record | None:
    work = None
    if paper.openalex_id:
        work = _openalex_fetch_work(paper.openalex_id.replace("https://openalex.org/", ""), by="id")
    if not work and paper.arxiv_id:
        work = _openalex_fetch_work(
            f"locations.landing_page_url:https://arxiv.org/abs/{paper.arxiv_id}", by="filter")
    if not work and paper.doi:
        work = _openalex_fetch_work(f"doi:{paper.doi}", by="filter")
    if not work:
        return None
    d = _openalex_work_to_discovered(work)
    if d is None:
        return None
    return Record(source="openalex", fields={
        "abstract": d.abstract or "", "authors": d.authors or "",
        "affiliations": d.affiliations or "", "venue": d.venue or "", "year": d.year or 0,
    })


def fetch_records(paper) -> list[Record]:
    """Gather authority records for a paper (reuses existing query helpers)."""
    records: list[Record] = []
    try:
        rate_limit_sleep(config.OPENALEX_RATE_LIMIT_SLEEP, "OpenAlex enrich")
        oa = _openalex_record(paper)
        if oa:
            records.append(oa)
    except Exception:  # network failure on one source must not abort the paper
        pass
    if paper.arxiv_id:
        try:
            rate_limit_sleep(0.3, "arXiv enrich")
            ax = query_arxiv_api([paper.arxiv_id]).get(paper.arxiv_id) or {}
            authors = ", ".join(ax.get("authors") or [])
            affils = ", ".join(ax.get("affiliations") or [])
            if authors or affils:
                records.append(Record(source="arxiv",
                                       fields={"authors": authors, "affiliations": affils}))
        except Exception:
            pass
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_enrichment_fetch.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ndif_citations/enrichment.py tests/test_enrichment_fetch.py
git commit -m "feat(enrichment): fetch_records from OpenAlex + arXiv (reuses existing clients)"
```

---

### Task 6: `enrichment.py` — `reconcile_paper` + `enrich_paper`

**Files:**
- Modify: `src/ndif_citations/enrichment.py`
- Test: `tests/test_enrichment_paper.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_enrichment_paper.py
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


def test_enrich_clean_paper_unchanged(monkeypatch):
    good = "A clean complete abstract. " * 40
    p = make_paper(arxiv_id="2401.3", abstract=good)
    monkeypatch.setattr(enrichment, "resolve_identifiers", lambda paper: enrichment.ResolveResult(resolved=False))
    monkeypatch.setattr(enrichment, "fetch_records",
                        _stub_records(Record("openalex", {"abstract": "Other complete abstract. " * 40})))
    enrich_paper(p)
    assert p.abstract == good                       # regression guard holds
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_enrichment_paper.py -q`
Expected: FAIL (`enrich_paper` / `reconcile_paper` not defined).

- [ ] **Step 3: Implement**

```python
# add to src/ndif_citations/enrichment.py
from ndif_citations.extract import detect_peer_review, detect_venue_type


@dataclass
class ChangeSet:
    changes: dict  # field -> (old, new, source, low_confidence)


def reconcile_paper(paper, records: list[Record], *, locked: bool,
                    low_confidence: bool, fields: tuple[str, ...] = _MANAGED_FIELDS) -> ChangeSet:
    low_conf_sources = {r.source for r in records} if low_confidence else set()
    changes: dict = {}
    for field in fields:
        current_value = getattr(paper, field)
        cands = [Candidate(r.fields[field], r.source) for r in records if field in r.fields]
        if not cands:
            continue
        res = reconcile_field(field, Candidate(current_value, paper.source.value if paper.source else "unknown"),
                              cands, low_confidence_sources=low_conf_sources)
        if not res.changed:
            continue
        if locked and (current_value not in (None, "", 0)):
            continue  # fill-gaps only for curator-locked papers
        setattr(paper, field, res.value)
        paper.enrichment_provenance[field] = res.source
        changes[field] = (current_value, res.value, res.source, res.low_confidence)
    return ChangeSet(changes=changes)


def enrich_paper(paper, *, dry_run: bool = False, fields: tuple[str, ...] = _MANAGED_FIELDS) -> ChangeSet:
    """Resolve ids -> fetch authorities -> reconcile -> apply. In dry_run, ALL work
    (including identifier resolution) happens on a deep copy so `paper` is untouched."""
    target = paper.model_copy(deep=True) if dry_run else paper
    resolved = resolve_identifiers(target)   # ResolveResult(resolved, via_title)
    records = fetch_records(target)
    cs = reconcile_paper(target, records, locked=target.manual_override,
                         low_confidence=resolved.via_title, fields=fields)
    if cs.changes and not dry_run:
        paper.peer_reviewed = detect_peer_review(paper.venue)
        paper.venue_type = detect_venue_type(paper.venue)
        paper.has_affiliations = bool(paper.affiliations)
    return cs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_enrichment_paper.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ndif_citations/enrichment.py tests/test_enrichment_paper.py
git commit -m "feat(enrichment): reconcile_paper + enrich_paper orchestration"
```

---

### Task 7: Wire into `extract.enrich_papers` (forward path)

**Files:**
- Modify: `src/ndif_citations/extract.py` (`enrich_papers`, ~line 28; add the per-paper reconcile after the existing venue/affiliation passes, before the per-paper post-processing loop)
- Test: `tests/test_extract_enrichment.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_extract_enrichment.py
from ndif_citations import extract as extract_mod
from tests.conftest import make_paper


def test_enrich_papers_invokes_enrich_paper(monkeypatch):
    called = []
    monkeypatch.setattr("ndif_citations.enrichment.enrich_paper",
                        lambda paper, **kw: called.append(paper.title))
    # neutralize the network passes already in enrich_papers
    monkeypatch.setattr(extract_mod, "enrich_via_external_apis", lambda papers: None)
    monkeypatch.setattr(extract_mod, "_enrich_affiliations_from_openalex", lambda papers, raw: None)
    papers = [make_paper(title="P1"), make_paper(title="P2")]
    extract_mod.enrich_papers(papers)
    assert set(called) == {"P1", "P2"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extract_enrichment.py -q`
Expected: FAIL (enrich_paper never called).

- [ ] **Step 3: Wire it in**

In `extract.py`, inside `enrich_papers`, after `_enrich_affiliations_from_openalex(...)` and before the per-paper post-processing loop, add:

```python
    # Step 2.5: reconcile abstract/authors/ids/etc from authoritative sources.
    from ndif_citations import enrichment
    for paper in papers:
        try:
            enrichment.enrich_paper(paper)
        except Exception as e:
            logger.warning(f"enrich_paper failed for {paper.title!r}: {e}")
```

(Import inside the function to avoid a circular import: `enrichment` imports from `extract`.)

- [ ] **Step 4: Run test + full suite (parity: clean papers unchanged)**

Run: `pytest tests/test_extract_enrichment.py -q && pytest tests/ -q`
Expected: PASS (the regression guard keeps clean s2/openalex fixtures unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/ndif_citations/extract.py tests/test_extract_enrichment.py
git commit -m "feat(enrich): reconcile metadata from authoritative sources in enrich_papers"
```

---

### Task 8: `re-enrich` CLI command (repair path)

**Files:**
- Modify: `src/ndif_citations/cli.py` (new `@cli.command()` after `reclassify`, ~line 480)
- Test: `tests/test_cli_reenrich.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_reenrich.py
import json
from click.testing import CliRunner
from ndif_citations.cli import cli
from ndif_citations import enrichment


def _write_catalog(out, papers):
    (out).mkdir(parents=True, exist_ok=True)
    (out / "research-papers-full.json").write_text(json.dumps(
        {"verified": papers, "pending": [], "discarded": []}))


def test_reenrich_dry_run_writes_nothing(tmp_path, monkeypatch):
    out = tmp_path / "output"
    _write_catalog(out, [{"title": "P", "arxiv_id": "2401.1", "abstract": "snippet …",
                          "source": "scholar"}])
    monkeypatch.setattr(enrichment, "resolve_identifiers", lambda p: False)
    monkeypatch.setattr(enrichment, "fetch_records",
                        lambda p: [enrichment.Record("openalex", {"abstract": "Full. " * 60})])
    before = (out / "research-papers-full.json").read_text()
    res = CliRunner().invoke(cli, ["re-enrich", "-o", str(out), "--dry-run"])
    assert res.exit_code == 0
    assert (out / "research-papers-full.json").read_text() == before  # unchanged


def test_reenrich_applies_and_is_idempotent(tmp_path, monkeypatch):
    out = tmp_path / "output"
    _write_catalog(out, [{"title": "P", "arxiv_id": "2401.1", "abstract": "snippet …",
                          "source": "scholar"}])
    monkeypatch.setattr(enrichment, "resolve_identifiers", lambda p: False)
    monkeypatch.setattr(enrichment, "fetch_records",
                        lambda p: [enrichment.Record("openalex", {"abstract": "Full sentence. " * 60})])
    r1 = CliRunner().invoke(cli, ["re-enrich", "-o", str(out)])
    assert r1.exit_code == 0
    data = json.loads((out / "research-papers-full.json").read_text())
    assert data["verified"][0]["abstract"].startswith("Full sentence.")
    # second run: nothing left to change
    r2 = CliRunner().invoke(cli, ["re-enrich", "-o", str(out)])
    assert "0 updated" in r2.output or "unchanged" in r2.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_reenrich.py -q`
Expected: FAIL (`No such command 're-enrich'`).

- [ ] **Step 3: Implement the command**

```python
# add to src/ndif_citations/cli.py (mirror the reclassify command's structure)
@cli.command(name="re-enrich")
@click.option("--ids", default=None, help="Comma-separated arXiv IDs, DOIs, or URLs to re-enrich")
@click.option("--output-dir", "-o", default=None, help="Custom output directory")
@click.option("--fields", default=None,
              help="Comma-separated subset of: abstract,authors,affiliations,venue,year")
@click.option("--dry-run", is_flag=True, help="Print changes without writing files")
def re_enrich(ids, output_dir, fields, dry_run):
    """Reconcile metadata (abstract/authors/affiliations/venue/year + identifiers) from
    authoritative sources for existing papers. No LLM, no discovery. Respects manual_override."""
    from ndif_citations.output import load_existing_papers, write_outputs
    from ndif_citations import config as cfg, enrichment

    _setup_logging(verbose=True)
    out = cfg.get_output_dir(output_dir)
    papers = load_existing_papers(out)
    if not papers:
        console.print(f"[bold red]No papers found in {out}[/bold red]")
        return

    field_tuple = enrichment._MANAGED_FIELDS
    if fields:
        field_tuple = tuple(f.strip() for f in fields.split(",") if f.strip())

    targets = papers
    if ids:
        wanted = {x.strip() for x in ids.split(",") if x.strip()}
        targets = [p for p in papers if p.arxiv_id in wanted or p.doi in wanted or p.url in wanted]

    updated = 0
    needs_review: list[str] = []
    for p in targets:
        cs = enrichment.enrich_paper(p, dry_run=dry_run, fields=field_tuple)
        if cs.changes:
            updated += 1
            for f, (old, new, src, low) in cs.changes.items():
                tag = " [LOW-CONF]" if low else ""
                console.print(f"  {p.title[:50]!r}: {f} <- {src}{tag}")
        elif (not p.arxiv_id and not p.doi):
            needs_review.append(p.title)

    console.print(f"\n[bold]{updated} updated[/bold], {len(targets) - updated} unchanged.")
    if needs_review:
        console.print(f"[yellow]{len(needs_review)} need manual review (no resolvable identifier).[/yellow]")
    if not dry_run and updated:
        write_outputs(papers, out)
        console.print("[green]Catalog written.[/green]")
    elif dry_run:
        console.print("[dim]Dry run — no files written.[/dim]")
```

(Confirm `write_outputs(papers, out)` signature with `grep -n "def write_outputs" src/ndif_citations/output.py`; match `reclassify`'s call exactly.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_reenrich.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ndif_citations/cli.py tests/test_cli_reenrich.py
git commit -m "feat(cli): re-enrich command to repair existing-catalog metadata"
```

---

### Task 9: Full-suite gate + live dry-run verification

- [ ] **Step 1: Run the whole suite**

Run: `pytest tests/ -q`
Expected: all pass (859 prior + the new enrichment tests).

- [ ] **Step 2: Live dry-run against the real catalog (manual, read-only)**

```bash
cp output/research-papers-full.json output/backups/research-papers-full.pre-reenrich.json
python -m ndif_citations re-enrich --dry-run 2>&1 | tee /tmp/reenrich-dryrun.txt
```
Expected: a list of proposed abstract/authors/affiliations changes for the ~47 Scholar papers, plus a "need manual review" count for any unresolved title-only papers. **Eyeball a few** — the new abstracts should be full sentences (not "…"), and titles should match.

- [ ] **Step 3: STOP — report to the user**

Do not apply (`re-enrich` without `--dry-run`) until the user has reviewed the dry-run output and the manual-review list. Applying is the rollout step in the spec, gated on user approval.

---

## Self-Review

**Spec coverage:** identifier resolution (T4) · reconcile engine + regression guard + trust tiebreaker (T2/T3) · abstract/authors/affiliations/venue/year (managed fields, T6) · forward path (T7) · repair command (T8) · manual_override fill-gaps (T6) · ≥0.90 + low-confidence flag (T4/T6/T8) · provenance (T1/T6) · dry-run (T8) — all covered.

**Placeholder scan:** every code step has runnable code; commands have expected output; no TBD/TODO.

**Type consistency:** `Candidate(value, source)`, `Resolution(value, source, changed, low_confidence)`, `Record(source, fields)`, `ChangeSet(changes)`, `enrich_paper(paper, *, dry_run, fields)`, `reconcile_field(field, current, candidates, low_confidence_sources)`, `_MANAGED_FIELDS` — used consistently across tasks. `_openalex_fetch_work`/`_openalex_work_to_discovered`/`query_arxiv_api` imported into `enrichment` so tests patch `ndif_citations.enrichment.<name>`.

**Verification flags to confirm during implementation:** `venue._WEAK_VENUE_RE` name (T2), `write_outputs(papers, out)` signature (T8), and that `_openalex_work_to_discovered` is importable without side effects (T5).
