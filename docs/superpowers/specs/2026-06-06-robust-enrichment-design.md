# Robust Metadata Enrichment — Design

- **Date:** 2026-06-06
- **Branch (base):** `feat/citations-web-app`
- **Status:** Approved (brainstorm) → ready for implementation plan (**Plan A**, builds before the UX workspace plan)
- **Companion:** `2026-06-06-paper-curation-ux-design.md` (Plan B; its narrow `backfill-abstracts` is superseded by this spec)

> `docs/` is gitignored (maintainer's choice). This spec lives on disk, untracked.

## 1. Problem

The pipeline conflates **discovery source** with **metadata authority**. A paper discovered via
Google Scholar keeps Scholar's metadata, but Scholar only returns search-result *snippets*.
Measured across the live catalog (147 papers):

| Field | Scholar (n=50) | s2_citation (57) | openalex_fulltext (28) |
|---|---|---|---|
| abstract truncated ("…") | **47** | 1 | 0 |
| authors truncated | **10** | 0 | 0 |
| affiliations missing | **24** | 2 | 0 |
| **no DOI and no arxiv_id** | **30** | 1 | 0 |
| year missing | 6 | 0 | 0 |

The **30 id-less Scholar papers are the crux**: every authoritative lookup (S2 / Crossref / arXiv,
and OpenAlex's reliable paths) is keyed by an identifier, so id-less papers can't be enriched at
all. Resolve their identifiers and the rest (abstract, authors, affiliations, venue, year) becomes
fetchable. `_compute_missing` already flags some of this in the UI; the data itself is the fix.

## 2. Goals / Non-goals

**Goals**
- Treat Scholar (and any thin source) as **discovery-only**; reconcile every paper's metadata from
  authoritative sources (OpenAlex, arXiv, Crossref, S2) via a resolved identifier.
- A **smart, testable reconciliation heuristic**: compare current vs. candidate values and keep the
  best, using a per-field source-trust ranking as the tiebreaker — not blind overwrite.
- **Two delivery paths from one engine:** (a) wired into the pipeline `enrich` stage (future runs
  are clean); (b) a no-LLM, no-discovery `re-enrich` command to repair the existing catalog.
- Never clobber curator edits (`manual_override` = fill-gaps only).
- Flag low-confidence (title-resolved) changes for review.

**Non-goals (deferred)**
- Thumbnail/image enrichment (separate PDF-extraction flow; unaffected here).
- LLM re-summarization (`description` is the LLM's job; enrichment doesn't touch it).
- A web UI for re-enrich (CLI only this iteration; the UX plan may surface it later).
- New HTTP clients — reuse the existing `utils.py` / `discover.py` / `extract.py` query helpers.

## 3. The reconciliation engine (the "smart heuristic")

A pure, unit-testable core in a new module `enrichment.py`. No I/O — it takes the current value and
candidate values (already fetched) and decides the winner. This is the piece the user emphasized.

### 3.1 Per-field "broken" predicates — `is_broken(field, value) -> bool`
- `abstract`: empty, OR ends with `…`/`...`, OR `len < 280` (snippet-length floor).
- `authors`: empty, OR contains `…`/`...`, OR ends with `et al`/`, …`.
- `venue`: empty, OR matches the existing `venue._WEAK_VENUE_RE`.
- `affiliations`: empty.
- `year`: falsy (`0`/`None`).

### 3.2 Source-trust ranking — `SOURCE_TRUST: dict[str, int]`
Higher = more authoritative for metadata (configurable; defaults from the measured quality):
`openalex=4, crossref=3, arxiv=3, s2=3, manual_add=2, scholar=1, unknown=0`.
(Curator edits are protected by `manual_override`, *not* by this ranking.)

### 3.3 Reconcile one field — `reconcile_field(field, current, candidates) -> Resolution`
`candidates: list[Candidate]` where `Candidate = (value, source)`; `current` is also a `Candidate`
(its source = the paper's recorded source). Returns
`Resolution(value, source, changed: bool, low_confidence: bool)`.

Algorithm (compare-and-replace, trust as tiebreaker):
1. Drop candidates whose value is empty.
2. **Score** each candidate by the tuple `(not is_broken, SOURCE_TRUST[source], completeness)` where
   `completeness` = `len(value)` for `abstract`/`authors`, else `1`. Higher tuple wins
   (lexicographic).
3. Winner = max by score. `changed = winner.value != current.value`.
4. **Guard against regressions:** if `current` is *not* broken and the winner is not strictly better
   (same validity tier and not more trusted and not longer), keep `current` (`changed=False`). Never
   replace a good value with an equal-or-worse one.
5. `low_confidence = winner came from a title-resolved fetch` (set by the caller; see §4.2).

This makes "blindly take the most trusted source" a *special case* (when values tie on validity,
trust decides) while never downgrading a good value to a broken one.

### 3.4 Reconcile a paper — `reconcile_paper(paper, records, *, locked) -> ChangeSet`
`records` = the authority payloads fetched in §4. For each managed field, build the candidate list
from `records` + the paper's current value, call `reconcile_field`, and collect changes.
- If `locked` (`paper.manual_override`): **only** apply changes where the current value is *empty*
  (fill-gaps); never replace non-empty curator-owned fields.
- Returns a `ChangeSet` (per-field old→new, winning source, low_confidence) — applied by the caller
  and surfaced in reports. Records winning source into `paper.enrichment_provenance` (§5).

## 4. Fetch layer (reuses existing helpers)

### 4.1 Identifier resolution — `resolve_identifiers(paper) -> bool` (in `enrichment.py`)
For papers missing `arxiv_id`/`doi`:
1. Cheap: parse arxiv_id from `url`/`pdf_url` via existing `extract_arxiv_id_from_url`.
2. OpenAlex `title.search` (the cascade in `extract.py:_enrich_affiliations_from_openalex` already
   does this) → compute normalized title similarity (`difflib.SequenceMatcher` on lowercased,
   punctuation-stripped titles). If **≥ 0.90**, adopt `openalex_id` and pull `doi`/`arxiv_id` from
   the work's `ids`. Else **flag** (record in the run report; adopt nothing).
Persist resolved ids on the paper. Returns whether anything was resolved (and whether via title →
marks subsequent fetches `low_confidence`).

### 4.2 Authority fetch — `fetch_records(paper) -> list[Record]`
With the best identifier, gather (reusing existing query fns, rate-limited via `rate_limit_sleep`):
- **OpenAlex** work (by `openalex_id` → arxiv → doi → title) — **primary authority for abstract**
  (`_reconstruct_abstract`), plus authors, affiliations, year, venue, ids. Reuse the field
  extraction already in `_openalex_work_to_discovered` (call it on the fetched work and read its
  fields — no re-implementation).
- **arXiv API** (`query_arxiv_api([arxiv_id])`) for arxiv papers — authoritative **authors /
  affiliations** (note: this helper returns authors/affiliations/journal_ref/comment, **not** the
  abstract; OpenAlex remains the abstract source).
- **Crossref** (`query_crossref(doi)`) for venue/year when a DOI exists.
- **S2** (`query_s2_publication_venue(...)`) — venue (existing behavior).
Each `Record` carries its `source` tag so the engine can rank it. A title-resolved OpenAlex record
is tagged so its winning fields are reported `low_confidence`.

### 4.3 Orchestration
- `enrich_paper(paper, *, dry_run=False) -> ChangeSet`: `resolve_identifiers` → `fetch_records` →
  `reconcile_paper` → apply (unless dry_run) → re-derive `peer_reviewed`/`venue_type` (existing
  `detect_*`) and `has_*` flags.
- **Forward (pipeline):** call `enrich_paper` for each paper inside `extract.enrich_papers` (after
  the existing venue pass), so production runs reconcile abstract/authors/ids too. Behavior-preserving
  for already-clean (s2/openalex) papers — the regression guard (§3.3 step 4) keeps good values.
- **Repair (catalog):** new command (§6).

## 5. Model change (`models.py`, `DiscoveredPaper`)
- `enrichment_provenance: dict[str, str] = Field(default_factory=dict)` — maps a repaired field name
  → winning source (e.g. `{"abstract": "openalex", "authors": "arxiv"}`). Lightweight provenance,
  mirroring the existing `venue_source` precedent; auto-serializes via `model_dump()`, so it's also
  visible in the detail API for the UX plan's trust signals.

No other field shape changes (ids/abstract/authors/affiliations/year already exist).

## 6. Repair command (`cli.py`)
`ndif-citations re-enrich [--ids X,Y] [--fields abstract,authors,affiliations,venue,year,ids] [--dry-run]`
— **no LLM, no discovery.** Loads the catalog, runs `enrich_paper` (filtered to `--fields` if given)
over each (or `--ids`) paper, writes back. Output: a summary table
(`field: updated / filled / skipped-low-confidence / unchanged`) and a **manual-review list** of
papers whose identifiers couldn't be resolved confidently. `--dry-run` prints the change set without
writing. Always back up first (existing `output/backups/` convention).

## 7. Testing (TDD)

**Pure engine (`enrichment.py`) — no network:**
- `is_broken`: truncated/short abstract, ellipsis authors, weak venue, empty affiliations, year 0.
- `reconcile_field`: (a) broken current + valid candidate → replace; (b) good current + lower-trust
  valid candidate → keep (no regression); (c) tie on validity → most-trusted wins; (d) longer
  non-truncated abstract from equal source → replace; (e) no candidates → unchanged.
- `reconcile_paper`: locked paper fills only empty fields; unlocked replaces broken; provenance recorded.
- title similarity: ≥0.90 adopt, 0.89 reject; punctuation/case-insensitive.

**Fetch layer — mocked HTTP** (patch the existing query helpers): OpenAlex work → fields extracted;
arXiv abstract preferred for arxiv papers; id-less paper resolved via title.search adopts id and is
marked low_confidence; below-threshold title match resolves nothing.

**Command:** `re-enrich --dry-run` writes nothing; `--ids` scoping; idempotent on a clean fixture
(second run = all "unchanged"); never flips `manual_override`; `--fields` filter respected.

**Parity:** `enrich_papers` on already-clean s2/openalex fixtures leaves abstract/authors/venue
unchanged (regression guard holds) — extend existing enrich tests.

## 8. Rollout
1. Land `enrichment.py` + model field + wire into `enrich_papers` + `re-enrich` command + tests.
2. Back up the catalog; `ndif-citations re-enrich --dry-run` → review proposed changes + the
   manual-review (unresolved) list.
3. `ndif-citations re-enrich` to apply; spot-check a few formerly-truncated Scholar papers now have
   full abstracts/authors and resolved ids. (Restart `serve` is not needed — this is CLI/data; the
   server reads the JSON fresh per request.)
4. Hand off to **Plan B** (UX workspace), which now displays clean data.

## 9. Risks
- **Wrong title match corrupts data:** mitigated by ≥0.90 threshold + low-confidence flagging +
  `--dry-run` review + the regression guard (a wrong match rarely yields a *longer non-truncated*
  abstract for a *different* paper, but the manual-review list is the backstop).
- **API rate limits / failures:** reuse existing `rate_limit_sleep` + per-paper try/except so one
  failure doesn't abort the batch (record and continue).
- **OpenAlex/arXiv downtime:** `re-enrich` is re-runnable and idempotent; partial progress is safe.
- **Over-replacement:** the §3.3 regression guard + per-field `is_broken` ensure clean values are
  never downgraded.
