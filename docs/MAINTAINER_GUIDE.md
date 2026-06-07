# ndif-citations — Maintainer Guide

> A handoff document for whoever runs this pipeline next. Covers how it works end-to-end and how to keep the published results clean over time.
>
> **Audience:** the person who will run this pipeline weekly/monthly and curate its output. No deep Python expected.
>
> **Updated:** 2026-05-19

---

## Table of contents

1. [What this is](#1-what-this-is)
2. [TL;DR — the 30-second mental model](#2-tldr--the-30-second-mental-model)
3. [Quick start](#3-quick-start)
   - 3.1 [The web app](#31-the-web-app-alternative-to-the-cli)
4. [Pipeline walkthrough](#4-pipeline-walkthrough)
   - 4.1 [Phase 1 — Discovery](#41-phase-1--discovery)
   - 4.2 [Phase 2 — Enrichment](#42-phase-2--enrichment)
   - 4.3 [Phase 2.5 — Routing](#43-phase-25--routing)
   - 4.4 [Phase 3 — Processing](#44-phase-3--processing)
   - 4.5 [Phase 4 — Output & merge](#45-phase-4--output--merge)
5. [Data model](#5-data-model)
6. [Output files explained](#6-output-files-explained)
7. [CLI reference](#7-cli-reference)
8. [Maintaining the results — curator playbook](#8-maintaining-the-results--curator-playbook)
9. [Configuration knobs](#9-configuration-knobs)
10. [Updating the website](#10-updating-the-website)
11. [CI/CD](#11-cicd)
12. [Troubleshooting](#12-troubleshooting)
13. [Known issues & tech debt](#13-known-issues--tech-debt)
14. [Glossary](#14-glossary)

---

## 1. What this is

An automated Python CLI that **finds, classifies, and catalogs every academic paper and GitHub repo that cites or uses NDIF / NNsight**, then emits website-ready JSON/XLSX files.

It replaces a manual workflow (Google Scholar checks → spreadsheet copy-paste → figure screenshots → HTML edits) with a single command. The output of this pipeline is consumed by the NDIF website (`ndif-web-beta/packages/ndif.us/`).

**Three things it produces that you'll touch directly:**

- `output/research-papers.json` — verified papers (the website renders this)
- `output/github-repos.json` — all GitHub repos using nnsight (the website renders this)
- `output/research-data.xlsx` — 4-sheet workbook for grant reporting, NSF check-ins, and curator review

Everything else (`-full.json`, `pdfs/`, `images/`, `raw/`) is internal state.

---

## 2. TL;DR — the 30-second mental model

```
                ┌─────────────────────────────────────────────────────┐
                │  4 sources                                           │
                │  ┌────────────┐  ┌──────────┐  ┌─────────┐  ┌─────┐  │
                │  │   S2       │  │ OpenAlex │  │ Scholar │  │ GH  │  │
                │  │ citations  │  │ fulltext │  │ SerpAPI │  │ deps│  │
                │  └────────────┘  └──────────┘  └─────────┘  └─────┘  │
                └────────────────────────┬────────────────────────────┘
                                          │
                            Phase 1: Discovery + dedup
                                          │
                                          ▼
                            Phase 2: Enrich venue, affiliations, bibtex
                                          │
                                          ▼
                  Phase 2.5: Route → NEW / REPROCESS / FILL_GAPS / SKIP / PROTECTED
                                          │
                                          ▼
                Phase 3: LLM summary + classify + thumbnail + bucket decision
                                          │
                                          ▼
                Phase 4: Merge into existing state, write outputs, print report
                                          │
                                          ▼
        verified  ──►  research-papers.json  ──►  website
        pending   ──►  research-papers-full.json  (curator reviews)
        discarded ──►  research-papers-full.json  (kept for audit)
```

**Two pieces of mental scaffolding that make everything else make sense:**

1. **Buckets vs. categories are different things.**
   - **Bucket** = "where does this paper sit in the curation pipeline?" → `verified` / `pending` / `discarded`. Only `verified` ships to the website.
   - **Category** = "what is the paper's relationship to NDIF?" → `uses_ndif` / `uses_nnsight` / `referencing` / `unclassified`.

2. **`manual_override=True` is the curator's veto.** Once a paper or repo is flagged manual_override, the pipeline never overwrites non-empty curator-set fields again. It *will* fill in empty fields (description, thumbnail, affiliations) on subsequent runs via the FILL_GAPS path — this lets you set the things you care about and let the pipeline handle the rest. Use this flag whenever you fix something by hand (the `edit` CLI sets it automatically).

---

## 3. Quick start

```bash
cd ndif-citations
pip install -e .

# Configure API keys (only LLM keys required for `run`)
cp .env.example .env
# Edit .env

# Discovery only — no API costs, no LLM
python -m ndif_citations discover

# Full pipeline (typical run)
python -m ndif_citations run

# Re-run with everything from scratch
python -m ndif_citations run --fresh
```

**Required for `run`:** `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`.
**Optional but strongly recommended:** `S2_API_KEY` (10× faster), `GITHUB_TOKEN` (60→5000 req/hr), `OPENALEX_EMAIL` (polite pool), `SERPAPI_API_KEY` (Scholar discovery).

A full run on ~150 papers takes ~30 minutes on the NVIDIA Build free tier (rate-limited at 5 LLM req/min).

### 3.1 The web app (alternative to the CLI)

Everything in the curator playbook (§8) and the website update (§10) can also be done from a
local web UI — same pipeline, same JSON files, no database or auth:

```bash
python -m ndif_citations serve     # → http://127.0.0.1:8723 (FastAPI + React SPA; /docs = Swagger)
```

- **Runs** screen — start a run and watch live progress over SSE. Incremental runs **pause at a
  review gate before any LLM spend**: you approve/discard the discovered candidates first, then it
  processes only the approved set. Cancel anytime; the catalog is only written at finalize.
- **Papers / Repos** — the §8 playbook as point-and-click: inline-edit curated fields (sets
  `manual_override`), promote/demote/discard, batch-reprocess, fix thumbnails, exclude repos.
- **Settings → Publish** — the §10 website update with a **dry-run diff first**, then apply.
- One run at a time; mutating actions are blocked (`409`) while a run is in progress.

Operational notes: backend (Python) changes need a server restart; frontend changes need
`cd web && bun run build` (the server serves the prebuilt `web/dist`). The in-memory review-gate
state survives a browser refresh (`GET /api/runs/active`) but not a server restart.

---

## 4. Pipeline walkthrough

This section walks through what actually happens when you type `python -m ndif_citations run`. File:function references throughout — open them as you read.

### 4.1 Phase 1 — Discovery

**Goal:** produce a deduplicated `list[DiscoveredPaper]` and `list[DiscoveredRepo]` from 4 independent sources.

**File:** `src/ndif_citations/discover.py`

**Sources:**

| Source | Function | What it does |
|---|---|---|
| **Semantic Scholar** | `discover_s2_citations()` | Walks the citation graph of **two** seed S2 records: `ARXIV:2407.14561` (ICLR 2025) and `23b6a2c856a8...` (2024 preprint). They have *disjoint* citer sets in S2's index, so we union both. Each citer becomes a `DiscoveredPaper`. |
| **OpenAlex** | `discover_openalex()` | Full-text searches for `nnsight`, `"national deep inference fabric"`, and `ndif.us` against OpenAlex's index. Catches papers that mention NDIF in their text even when no formal citation parser linked them to the seed. |
| **Google Scholar** | `discover_scholar()` | Via SerpAPI. Two queries: (1) cited-by traversal of the seed's `cluster_id`, (2) `q=nnsight` keyword search. Cached 24 h in `raw/scholar_*_raw.json` to conserve the 250-call/month free tier. Recovers papers that S2/OpenAlex bibliography parsers missed. |
| **GitHub dependents** | `discover_github_dependents()` | HTML-scrapes `github.com/ndif-team/nnsight/network/dependents`. Per-page checkpointing to `raw/github_dependents_checkpoint/page_N.json` — if the scrape is interrupted, the next run resumes from the last successful page. Each repo becomes a `DiscoveredRepo` shell (no API calls yet). |

**After source collection:**

- `deduplicate_papers()` — dedup priority: arxiv_id → doi → 90% rapidfuzz title ratio. When duplicates merge across sources, S2 > OpenAlex > GitHub > Scholar for field priority, with one exception: **affiliations always prefer OpenAlex**, and **venue prefers whichever source produced a confidently-recognized venue token** (so a junk "ArXiv.org" from S2 yields to "ICLR 2025" from OpenAlex).
- `filter_by_min_year(MIN_PAPER_YEAR=2024)` — papers known to predate the NDIF release (July 2024) are dropped at discovery. Papers with `year=0` (unknown) pass through; they end up in `pending` later as `stub_metadata`.
- `EXCLUDED_PAPER_TITLES` set in `config.py` drops the seed paper itself.

### 4.2 Phase 2 — Enrichment

**Goal:** fill in venue, affiliations, BibTeX, peer-review status, and (for repos) GitHub API metadata + linked-paper detection.

**Files:** `src/ndif_citations/extract.py`, `src/ndif_citations/venue.py`, `src/ndif_citations/utils.py`, `src/ndif_citations/discover.py` (continued)

#### 4.2a Paper enrichment

`enrich_papers()` runs three sub-passes:

1. **`enrich_via_external_apis()`** — collects raw venue signals from CrossRef (for non-arXiv DOIs), S2 publicationVenue, OpenReview (fuzzy title search), and the arXiv Atom API (for `journal_ref` and `comment` fields). Authors get refreshed from the arXiv API (cleanest Unicode source). Affiliations from CrossRef ride along. Then per paper, `venue.resolve_venue()` picks the canonical venue.

2. **`_enrich_affiliations_from_openalex()`** — for any paper still missing affiliations, looks them up via OpenAlex. Priority: direct `openalex_id` → arXiv URL filter → DOI filter → title search.

3. **Per-paper post-processing** — `detect_peer_review()`, `detect_venue_type()`, `generate_bibtex()`, `_best_url()`. BibTeX is synthesized from structured metadata (not parsed from anywhere).

#### 4.2b Venue resolution cascade (`venue.resolve_venue`)

This is one of the most important pieces of the codebase to understand. It picks the canonical venue from up to 9 candidate sources:

```
1. DOI prefix decode    ──►  deterministic (ACL/EACL/NAACL/EMNLP/FAccT all have known prefixes)
2. arXiv journal_ref    ──►  parsed via "Accepted at X YYYY" regex
3. arXiv comment        ──►  same regex
4. OpenAlex display_name
5. S2 publicationVenue.name
6. CrossRef container_title
7. OpenReview venue
8. Existing paper.venue
9. ArXiv {year} fallback ◄── when everything else failed
```

**Each source goes through a confidence gate** (`is_confident_venue`) before being accepted. The gate checks the venue contains a recognized acronym/journal name from `data/known_venues.json` (or maps to one via the acronym_map). This is intentional: a paper with no confidently-identifiable venue ships as "ArXiv {year}" and gets routed to `pending` for human review, rather than carry a half-broken label.

**Year reconciliation** — when a high-confidence source (`doi_prefix` or `arxiv_comment_parsed`) returns a venue string containing an explicit year (e.g. "ACL 2025"), the paper's `year` field gets reconciled to that year. This fixes the common case where `paper.year` is the arXiv upload year (6–12 months ahead of the conference proceedings).

**`normalize_venue()` pipeline** strips ellipses, "Proceedings of (the) (Nth)" prefixes, "(Volume N: …)" tails, Poster/Spotlight/Oral suffixes, "(Cornell University)" parentheticals, and leading articles. It then substitutes long-form names for acronyms (longest first to avoid partial matches) and appends the year if missing.

**`is_preprint_sentinel()`** rejects mangled outputs: unbalanced parens, leading ordinals ("39th Conference on…"), bare conference shells, year-only strings, etc. Forces the cascade to fall through.

#### 4.2c Repo enrichment

`enrich_repos_from_github_api()` does five things per repo in one pass:

1. **Fetch via GitHub API** — stars, forks, last_commit, language, license, topics, parent.full_name (for forks).
2. **Staleness purge** — 404 → drop, rename redirect → drop (let the new name re-enter as NEW next scrape), archived → drop. Rate-limited (403/429) or transport errors → **keep with stale data**. Tracked in `removal_counts`.
3. **Single README fetch** — extracts all `arxiv.org/abs/…` URLs, runs the NDIF keyword classifier (regex + substring, with boilerplate-span filter so "join the NDIF Discord" doesn't trigger).
4. **Linked-paper detection** — `_detect_linked_paper()` runs a 4-tier priority:
   - **Tier 1** — BibTeX block in README (strongest signal: explicit citation entry)
   - **Tier 2** — arXiv ID under a Citation/Paper/Reference section header
   - **Tier 3** — exactly one post-2020 arXiv ID in the entire README
   - **Tier 4** — multiple post-2020 IDs → most recent (highest YYMM prefix)
   - No match → `(None, None)`
5. **Set `repo.has_classification=True`** even when README is unavailable (all nnsight dependents use the library by definition).

**Then back in `cli.py`:**

- `_unlink_shared_template_papers()` — cross-repo cleanup. When ≥ `SHARED_PAPER_THRESHOLD` (default 5) repos share the same `linked_paper_url`, only the highest-star repo keeps the link. The rest get cleared. This is how ARENA-course forks (22+ repos all pointing to the same paper) are detected.
- `_tag_repo_type()` — 7-rule decision tree:
  - **Course tiers 1-3:** parent in `KNOWN_COURSE_SOURCES` → course. Unlinked-by-shared-cleanup + 0 stars + no description → course. Name/description contains a `COURSE_NAME_PATTERNS` substring (ARENA / MATS / CBAI) → course.
  - **Research tiers 4-6:** category == `uses_ndif` → research. `linked_paper_url` is set → research. stars ≥ 6 AND has description → research.
  - **Experiment (default):** everything else.
- **Course repos clear their `linked_paper_url`** — course repos cite many papers, none canonical, so we strip the link.
- `link_repos_to_papers()` — propagates `repo.url → matched_paper.project_url` and the strongest linked_paper_tier → matched_paper.linked_paper_tier.

### 4.3 Phase 2.5 — Routing

**Goal:** decide which papers/repos need expensive LLM processing vs. can be carried over from the previous run.

**File:** `src/ndif_citations/router.py`

For each discovered paper, `_route_single_paper()` matches against the existing database by arxiv_id → doi → content_hash and assigns one of 5 buckets:

| Bucket | Criteria | Action |
|---|---|---|
| **NEW** | Not in DB | Full processing (summary + classify + thumbnail + affiliations) |
| **REPROCESS** | Match found AND (venue upgrade OR content_hash changed) | Full processing |
| **FILL_GAPS** | Match found, hash same, but missing fields (`has_summary` / `has_classification` / `has_thumbnail` / `has_affiliations` is False) — **OR** `manual_override=True` with any missing flag | Process only the missing fields. On `manual_override` papers, a guard in `process.py` skips writes whose field is already non-empty (defense in depth). |
| **SKIP** | Match found, hash same, all complete | Pass through unchanged |
| **PROTECTED** | Match found AND `manual_override=True` AND **all** `has_*` flags are True | Pass through, never touch curator-set fields. (If any flag is False, the router routes to FILL_GAPS instead so the pipeline can backfill.) |

Same scheme for repos (`route_repos`) keyed by `owner/repo`. **Venue upgrade detection** — `_is_venue_upgrade()` returns True when existing venue type is `preprint` and new venue type is `conference`/`journal`. This is how arXiv → ICLR transitions trigger a reprocess.

### 4.4 Phase 3 — Processing

**Goal:** run expensive operations (PDF download, LLM calls, thumbnail extraction) only for the routing buckets that need them.

**Files:** `src/ndif_citations/process.py`, `src/ndif_citations/pdf_cache.py`

#### 4.4a Per-paper processing (`process_papers`)

For each `RoutingDecision`:

1. **Skip SKIP and PROTECTED early** — for PROTECTED, copy `bucket`/`category`/`reason`/`description` from the existing paper (manual_override freezes these).
2. **Resolve PDF if needed** — `get_cached_pdf()` checks the local cache (`output/pdfs/arxiv-{id}.pdf` or `doi-{slug}.pdf`); on miss, tries existing pdf_url → arXiv direct (`arxiv.org/pdf/{id}.pdf`) → Unpaywall (via DOI) → CrossRef link array. Each URL is HEAD-tested first (with Range-GET fallback for HEAD-blocking servers like MDPI/ACM DL).
3. **Generate summary** (if `needs["summary"]` and LLM available) — `generate_summary()` calls the LLM with `SUMMARY_SYSTEM_PROMPT`. Falls back to first 2 sentences of abstract on failure.
4. **Discard check** — `_check_discard_zero_pdf_hits()` — if PDF has ≥1000 chars and zero NDIF keyword matches anywhere, the paper is moved to bucket=`discarded` with `reason=ZERO_PDF_HITS`. Short or unreadable PDFs are not discarded (might be scan failures).
5. **Classify category** — `classify_category()` is the most subtle path in the codebase. Returns `(category, float_confidence, band)` where `band` is a `Confidence` enum (`CERTAIN` / `HIGH` / `MEDIUM` / `LOW` / `NONE`) derived deterministically from the evidence type. The float is kept for backwards-compat callers and is just `_BAND_TO_FLOAT[band]`. See deep-dive below.
6. **Decide bucket** — `_decide_bucket()` returns `(bucket, reason)` based on ordered rules (band-based, no more magic 0.7 threshold):
   1. `stub_metadata` if year=0, no usable abstract, or no identifier (arxiv_id/doi/s2_id/openalex_id/url all missing).
   2. `unclassified_no_keywords` or `unclassified_llm` if category came back as UNCLASSIFIED.
   3. `medium_confidence` if `category_confidence_band == MEDIUM` (LLM verdict on thin evidence — single window, abstract-only, or weak pre-filter signal).
   4. `low_confidence` if `category_confidence_band == LOW` (keyword fallback, no LLM available).
   5. Otherwise (`HIGH` or `CERTAIN`) → `verified`.
7. **Extract thumbnail** — `extract_thumbnail()` scans all pages for figure captions via regex, scores each by mech-interp keyword tiers (`MECH_INTERP_TIER_1/2/3` in utils.py) + early-figure boost + page decay, rasterizes the winning page at 150 DPI, runs Surya layout detection (`batch_layout_detection`) to find the exact Figure bounding box, clusters fragmented boxes if needed, and crops. Saves to `output/images/{slugify(title)}.png`. Falls back to top 35% of page 1 (title/abstract area) if no good candidates exist.
8. **Extract affiliations** — `extract_affiliations_from_pdf()` is a pure-heuristic (no LLM) parser with three strategies: block-aware footer (IEEE template), anchored footnote block ("Correspondence to:" / "Preprint." anchors), inline block between authors and "Abstract" (ACL/EMNLP). Handles both prefix-marker (1Stanford 2MIT) and suffix-marker (Lab♡, Other♣) affiliation formats.

#### 4.4b Classification deep-dive (`classify_category`)

This is the most decision-rich function in the codebase. The flow:

```
1. extract_ndif_context(pdf)  ──►  list of context windows (500 chars around each NDIF keyword)
                                   │
2. fallback to abstract        ──►  if PDF has no mentions, check abstract for keywords
                                   │  if no keywords there either → UNCLASSIFIED (no_keywords_anywhere)
                                   ▼
3. PRE-FILTER PASS — _apply_prefilters() — runs on each context window:
   (a) Negative evidence regex      "removing nnsight", "rather than NDIF", "alternative to nnsight", etc.
   (b) Comparison table              window contains ≥3 of ✓✗∼ (likely a capability table row)
   (c) Acks-only thank-you           "thank/acknowledge NDIF" WITHOUT "implement/use/run nnsight"
   
   If ALL windows eliminated  ──►  REFERENCING, confidence 0.85-0.9, no LLM call
   If some survive             ──►  rebuild context and continue
                                   ▼
4. Select prompt based on which keyword family matched in the surviving context:
   - Only nnsight keywords matched → LIBRARY_PROMPT (uses_nnsight / referencing / unclassified)
   - Only NDIF keywords matched    → INFRASTRUCTURE_PROMPT (uses_ndif / referencing / unclassified)
   - Mixed or empty                 → UNIFIED_PROMPT (uses_ndif / uses_nnsight / referencing / unclassified)
                                   ▼
5. Augment prompt for tier-1/2 cross-links — _augment_prompt_with_tier():
   When linked_paper_tier ≤ 2 (BibTeX or Citation section), append a block telling the LLM
   "repositories that nnsight is a dependency of and that explicitly cite a paper
    overwhelmingly use that library — weight this evidence heavily." Confidence stays 0.85.
                                   ▼
6. Call LLM at temperature 0.1, max_tokens 20. Parse response substring-match:
   "uses_ndif" / "uses_nnsight" / "referencing" / "unclassified" (or unparseable → UNCLASSIFIED)
                                   ▼
7. Confidence is derived from a deterministic rule (`_compute_confidence_band`):

       CERTAIN (1.00) — pre_filter:negative_evidence (regex caught explicit non-use)
       HIGH    (0.85) — LLM verdict + (linked_paper_tier ≤ 2 OR ≥2 surviving windows)
       MEDIUM  (0.55) — LLM verdict on single window OR abstract-only context
                       OR pre_filter:comparison_table / acks_only_thank_you
       LOW     (0.30) — keyword fallback (LLM unavailable/errored)
       NONE    (0.00) — UNCLASSIFIED for any reason

   The legacy float `category_confidence` is set from `_BAND_TO_FLOAT[band]` so
   old downstream code keeps working. `manual_override=True` papers carry
   `Confidence.CERTAIN` regardless of stored float (set in `model_post_init`).

   (Note: actual LLM logprobs are still NOT used — band is derived from
   structural evidence properties, not the model's softmax. See known issues
   for the logprob improvement path.)
```

#### 4.4c Per-repo processing (`process_repos`)

Much simpler — classification happened in Phase 2 enrichment. This loop just recomputes `content_hash` for NEW/REPROCESS, falls back to `classify_repo()` for FILL_GAPS without classification, and passes through SKIP/PROTECTED.

### 4.5 Phase 4 — Output & merge

**Goal:** merge processed results into persistent state, write all output files, print the run report.

**File:** `src/ndif_citations/output.py`

#### 4.5a `merge_papers()`

For each newly-processed paper:

1. Match against existing by arxiv_id → doi → 90% title fuzz.
2. If matched: `_update_existing()` fills gaps (authors, affiliations, IDs, abstract) and applies **venue-confidence-aware merge logic**:
   - New is confident venue → take new (latest authoritative resolution).
   - New is ArXiv fallback AND existing is confident → keep existing, but re-normalize in case it was a stale long-form name (e.g. "International Conference on Machine Learning" → "ICML 2025").
   - New is ArXiv fallback AND existing is junk/truncated → take new (ArXiv is cleaner than an unrecognized stub).
3. **`manual_override=True` short-circuits everything** — `_update_existing` returns immediately without touching anything.
4. **Auto-recovery / auto-demotion** — after the update, re-run `_decide_bucket()` on the existing paper. If the new bucket differs from the old:
   - pending → verified = auto-promoted (logged + tracked in `PipelineRun.auto_promoted`).
   - verified → pending = auto-demoted (logged + tracked in `PipelineRun.auto_demoted`).

#### 4.5b `merge_repos()`

Different staleness model than papers: **repos absent from the current discovered list get purged** (unless they have manual_override=True). This is one-way — a transient GitHub failure that returns fewer repos can cause real repos to be removed from state. Mitigation: rate-limited/transport errors keep stale repos in `enrich_repos_from_github_api()`, so partial scrapes don't trigger purges. A *successful* scrape that returns fewer pages than expected, however, IS treated as authoritative.

#### 4.5c File writing

- `write_outputs()` → `research-papers.json` (verified only, sorted year-desc) + `research-papers-full.json` (3-bucket structure: `{"pending": [...], "verified": [...], "discarded": [...]}`).
- `_write_repos_outputs()` → `github-repos.json` (sorted by stars desc) + `github-repos-full.json`.
- `_write_xlsx()` → `research-data.xlsx` with sheets: **Papers** (verified), **Pending**, **Discarded**, **GitHub**. URLs are auto-hyperlinked.

#### 4.5d `print_report()`

Rich-formatted CLI summary: sources checked, dedup count, merge results, 3-bucket breakdown, auto-promoted/demoted titles, missing thumbnails, GitHub repo type/category breakdown, output file paths.

---

## 5. Data model

**File:** `src/ndif_citations/models.py`

### 5.1 `DiscoveredPaper`

The central model. Every phase mutates one. Key fields:

| Field | Meaning |
|---|---|
| `arxiv_id`, `doi`, `s2_paper_id`, `openalex_id` | Identifiers (in dedup priority order) |
| `category: Category` | `uses_ndif` / `uses_nnsight` / `referencing` / `unclassified` |
| `category_confidence_band: Confidence` | `CERTAIN` / `HIGH` / `MEDIUM` / `LOW` / `NONE` — categorical band tied to evidence type. Auto-migrated from the legacy `category_confidence: float` on load. |
| `category_confidence: float` | Legacy float (`_BAND_TO_FLOAT[band]`). Kept for backwards compat; new code should read the band. |
| `bucket: Bucket` | `pending` / `verified` / `discarded` — where it sits in curation |
| `reason: PaperReason` | Why it's in pending/discarded (`stub_metadata`, `medium_confidence`, `low_confidence`, `unclassified_no_keywords`, `unclassified_llm`, `openalex_source`, `zero_pdf_hits`, `manual_discard`, `manual_demote`) |
| `reason_detail` | Free-text supplement to `reason` (e.g. "manual_venue='Dissertation' json_venue='ICLR 2025'") |
| `manual_override: bool` | **THE CURATOR'S VETO** — when True, pipeline never overwrites bucket/category/reason/description |
| `content_hash` | `sha256(title + "::" + abstract)[:16]` — drives change detection in router |
| `linked_paper_tier` | 1=BibTeX, 2=Citation section, 3=single ID, 4=multi-most-recent. Set when a repo cross-links to this paper |
| `classification_signal` | When pre-filter classified without LLM: `pre_filter:negative_evidence` / `:comparison_table` / `:acks_only_thank_you` |
| `unclassified_reason` | When classification returned UNCLASSIFIED: `no_evidence_extractable` / `no_keywords_anywhere` / `llm_returned_unclassified` / `llm_unparseable` |
| `venue_source` | Audit field: which source produced this venue (`doi_prefix`, `arxiv_comment_parsed`, `openalex`, `s2`, `crossref`, `openreview`, `existing`, `manual`, `fallback`) |

**`to_website_dict()`** is what ships to the website. **`to_full_dict()`** is the full Pydantic dump.

### 5.2 `DiscoveredRepo`

Same idea for GitHub repos. Key fields:

- `owner`, `repo`, `url`, `description`, `stars`, `forks`, `last_commit`, `archived`, `is_fork`, `language`, `license`, `topics`
- `category: Category` — `uses_ndif` or `uses_nnsight`
- `classification_reason` — `ndif_keyword_match` or `github_dependent` (default)
- `repo_type` — `research` / `course` / `experiment`
- `parent_full_name` — for forks
- `linked_paper_url`, `linked_paper_tier`, `readme_arxiv_ids`
- `manual_override` — same veto behavior
- `content_hash` — `sha256(description + "::" + last_commit + "::" + archived)`

### 5.3 `PipelineRun`

Run-level stats. Populated as the pipeline runs, dumped into the CLI report. Tracks `s2_citations_found`, `openalex_found`, `scholar_found`, `github_dependents_found`, `new_papers`, `updated_papers`, `bucket_*` counts, `auto_promoted`/`auto_demoted` lists, `low_confidence` titles, `missing_thumbnails` titles, `errors`.

---

## 6. Output files explained

```
output/
├── research-papers.json           # ──► copy to website (verified, slim)
├── research-papers-full.json      # persistent 3-bucket state, all fields, all papers
├── github-repos.json              # ──► copy to website (slim)
├── github-repos-full.json         # persistent state, all fields, all repos
├── research-data.xlsx             # 4-sheet spreadsheet for review/reporting
├── images/                        # ──► copy to website (thumbnails as PNG)
│   └── {slugify(title)}.png
├── pdfs/                          # local PDF cache, never copied anywhere
│   ├── arxiv-{id}.pdf
│   └── doi-{slug}.pdf
└── raw/                           # raw API responses, for debugging
    ├── s2_citations_raw.json
    ├── openalex_raw.json
    ├── scholar_citedby_raw.json
    ├── scholar_keyword_*_raw.json
    ├── github_dependents_raw.json
    └── github_dependents_checkpoint/   # only present mid-scrape; deleted on success
```

**The two files the website actually reads** (after you copy them):

- `research-papers.json` → `ndif-web-beta/packages/ndif.us/data/research-papers.json`
- Images: `output/images/*.png` → `ndif-web-beta/packages/ndif.us/public/images/`
- `github-repos.json` → similar destination in the website repo (check current site code for exact path)

**Files that get committed/synced manually** — none of `output/` is in git. Each maintainer regenerates by running the pipeline.

**The 4 `.pre-*.bak.json` files in `output/`** are one-shot backups from the 2026-05-18 finale merger (when Emma's manual spreadsheets were folded in). Safe to delete or archive; nothing reads them.

---

## 7. CLI reference

| Command | What it does | When to use |
|---|---|---|
| `python -m ndif_citations run` | Full pipeline | Weekly/monthly normal run |
| `python -m ndif_citations run --fresh` | Full pipeline, ignore existing state | After a major schema change, or when you want to rebuild everything (will lose `manual_override` flags!) |
| `python -m ndif_citations run --skip-github` | Papers only | When GitHub API rate-limit is hot |
| `python -m ndif_citations run --skip-papers` | Repos only | When you just need to refresh stars/forks |
| `python -m ndif_citations discover` | Discovery only, no LLM | Preview what's out there without spending API credits |
| `python -m ndif_citations add <url>` | Add a single paper by URL | Curator wants to manually include a paper. **⚠ Currently broken — see [§13](#13-known-issues--tech-debt). Use the JSON edit workaround for now.** |
| `python -m ndif_citations edit <id>` | Interactive menu — override any of 16 curated fields on one paper (sets `manual_override=True`) | Curator wants to fix a paper's venue/category/description/etc. without hand-editing JSON. **The main curator tool.** |
| `python -m ndif_citations edit <id> --set venue="ICML 2025" --set category=uses_nnsight --yes` | One-shot, scriptable. Repeat `--set` for multiple fields, `--yes` skips confirm | Batch fixes via shell |
| `python -m ndif_citations edit <id> --set ... --dry-run` | Preview without writing | Always do this first for batch edits |
| `python -m ndif_citations debug <id>` | Read-only trace for a single paper | Diagnose why a specific paper got its bucket/category |
| `python -m ndif_citations reclassify` | Re-run classification on every paper | After a `config.py` change or pre-filter tweak |
| `python -m ndif_citations reclassify --ids 2407.14561,2504.14107` | Re-classify specific papers | Targeted fix |
| `python -m ndif_citations reclassify --dry-run` | Show changes without writing | Always do this first |
| `python -m ndif_citations promote <id>` | Move paper to verified + freeze with `manual_override=True` | Curator confirms a pending paper is good (bucket-only — use `edit` for field changes) |
| `python -m ndif_citations promote <id> --detail "Verified by Emma 2026-05-19"` | Same, with audit note | Always include context |
| `python -m ndif_citations demote <id> --reason low_confidence` | Move paper to pending + freeze | Curator wants a verified paper held |
| `python -m ndif_citations discard <id>` | Move paper to discarded + freeze | Curator confirms paper is irrelevant |

**Global flags:**
- `--verbose` / `-v` — debug logging (recommended for any troubleshooting)
- `--output-dir <path>` / `-o <path>` — use a custom output directory

**`<id>` accepts:** arXiv ID (`2407.14561`), DOI (`10.48550/arXiv.2407.14561`), or full URL (`https://arxiv.org/abs/2407.14561`).

---

## 8. Maintaining the results — curator playbook

This is the section you'll come back to.

### 8.1 The weekly/monthly run

**Routine cadence:**

```bash
cd ndif-citations
python -m ndif_citations run
```

When it finishes, read the report carefully. Pay attention to:

- **`* N NEW papers added`** — eyeball each. Click through to confirm relevance. Run `promote/demote/discard` as needed.
- **`Auto-promoted (N)`** — papers the pipeline moved pending → verified this run. Spot-check.
- **`Auto-demoted (N)`** — papers the pipeline moved verified → pending. **Always investigate** — usually means a metadata regression (year went to 0, abstract got blanked out by an API hiccup, etc.). If transient, just re-run.
- **`N pending`** breakdown by reason — your work queue.
- **`Papers needing manual thumbnails`** — list of titles. See §8.5.
- **`Errors (0)`** — should be 0 in a clean run. Investigate anything else.

### 8.2 Reviewing the pending bucket

Pending is where the curator earns their paycheck. Open the `Pending` sheet in `research-data.xlsx` or query `research-papers-full.json`:

```bash
python -c 'import json; d=json.load(open("output/research-papers-full.json")); print(len(d["pending"]), "pending"); [print(p["title"][:80], "|", p.get("reason")) for p in d["pending"]]'
```

**Reason-by-reason playbook:**

| Reason | What to check | Resolution |
|---|---|---|
| `stub_metadata` | Year=0, abstract missing, or no identifiers at all | Look up the paper manually. If real → fix metadata in `research-papers-full.json` and `promote`. If junk → `discard`. |
| `unclassified_no_keywords` | No NDIF/nnsight mention found in PDF or abstract | Read the paper. If genuinely tangential → `discard`. If you can verify it actually uses NDIF → `promote --detail "verified manually"`. |
| `unclassified_llm` | LLM couldn't decide | Read the paper. Set the right category via `promote` (and edit `category` in the JSON if you need to change it). |
| `medium_confidence` | LLM verdict on thin evidence — single context window OR abstract-only OR pre-filter caught a comparison-table/acks-only signal | Read the paper. If category is right → `promote` with `--detail "verified — single mention but real"`. If wrong → `edit <id> --set category=...` and the bucket auto-recovers. |
| `low_confidence` | Keyword fallback (LLM was unavailable/errored) | Re-run with LLM access — most LOW papers will reclassify HIGH/MEDIUM. Or `promote` with the right category if you've verified manually. |
| `openalex_source` | Found only by OpenAlex fulltext (no S2 citation graph backing) | OpenAlex has ~3% false-positive rate. Read the paper; if no real NDIF connection → `discard`. |
| `manual_demote` | Previously demoted by curator | Should not auto-resurface; if it did, investigate. |

### 8.3 Adding a paper that the pipeline missed

**Current workaround** (because `add` CLI is broken — see [§13](#13-known-issues--tech-debt)):

1. Open `output/research-papers-full.json`.
2. Decide which bucket: `verified` if you've fully reviewed it, `pending` otherwise.
3. Append a new paper object to that bucket array. Required fields: `title`, `arxiv_id` (or `doi`), `year`, `venue`, `authors`, `url`. Set `manual_override: true`.
4. Save. Next pipeline run will respect manual_override and leave it alone.
5. To get a thumbnail and summary: temporarily set `manual_override: false`, set `has_summary: false`, `has_classification: false`, `has_thumbnail: false`, run `python -m ndif_citations run`, then set `manual_override: true` after the run completes.

**Or, to bypass the JSON edit:** discover the paper via S2 (search for its title in Semantic Scholar; if it cites the NDIF seed, the next pipeline run will find it automatically).

### 8.4 Correcting a wrong venue / category / summary

The cleanest workflow is the `edit` command — interactive menu, no JSON poking:

```bash
python -m ndif_citations edit 2407.14561
# Shows numbered field list with current values.
# Pick "4" for Venue, type the corrected value, then "q" to save+confirm.
# manual_override is auto-flipped to True; has_* flags are re-derived;
# _decide_bucket is re-run so the paper auto-promotes/demotes to the right bucket.
```

**Scriptable for batch fixes:**

```bash
python -m ndif_citations edit 2407.14561 \
    --set venue="ICLR 2025 (Spotlight)" \
    --set category=uses_nnsight \
    --set description="A more curator-approved one-liner." \
    --dry-run    # preview first
python -m ndif_citations edit 2407.14561 \
    --set venue="ICLR 2025 (Spotlight)" \
    --set category=uses_nnsight \
    --yes        # apply
```

**After the edit:**
- `manual_override=True` is sticky — pipeline never overwrites these fields again.
- Empty fields (description, image, affiliations) you didn't touch will be filled in automatically on the next `run` via the `FILL_GAPS` path (the router sees `manual_override` + missing `has_*` flag and lets the pipeline backfill empty values only — the guard preserves anything you set).

### 8.5 Missing thumbnails

The report lists papers without successfully-extracted thumbnails. For each:

1. Check `output/pdfs/arxiv-{id}.pdf` — was the PDF downloaded?
2. If no PDF: the paper is paywalled or arXiv direct failed. You can manually save a screenshot as `output/images/{slugify(title)}.png` (use the same slugify rule: `re.sub(r'[^\w\s-]', '', title).replace(' ', '-')`).
3. If PDF exists but no thumbnail: Surya layout detection failed to find a Figure box. The fallback (top 35% of page 1) should have produced something. Check `output/images/` — maybe it's there with a slightly different slug.
4. To force re-extraction: delete the existing thumbnail (if any), set the paper's `has_thumbnail: false` in `research-papers-full.json`, re-run.

### 8.6 Updating `known_venues.json`

Whenever a new venue appears in the pipeline output that isn't recognized — e.g. a paper accepted to a niche workshop:

1. Open `data/known_venues.json`.
2. Add the venue under the `venues` key with its type:
   ```json
   "NewConfName": {
     "type": "conference",
     "aliases": ["The Long Form Name", "The Even Longer Form Name"]
   }
   ```
3. Types are: `"conference"`, `"workshop"`, `"journal"`, `"preprint"`.
4. Workshops can declare a `"parent"`: `"BlackboxNLP": { "type": "workshop", "parent": "EMNLP" }`.
5. Aliases get auto-folded into the `acronym_map` so the normalizer maps long-form to short.
6. After editing, re-run `python -m ndif_citations run` — the venue cascade will pick up the new entries automatically.

### 8.7 Adding a course source / pattern

When a new MATS-style course or new ARENA-style template starts showing up as ~5+ forks all citing the same paper:

1. Edit `src/ndif_citations/config.py`.
2. `KNOWN_COURSE_SOURCES` — add the exact `owner/repo` of the upstream template (e.g. `"someorg/ARENA_4.0"`).
3. `COURSE_NAME_PATTERNS` — add a substring that appears in course-fork names/descriptions (case-insensitive).
4. Re-run `python -m ndif_citations run`. Existing forks will be re-classified on next discovery.

### 8.8 Excluding a specific repo

Edit `config.py`, add the repo's `"owner/repo"` to `EXCLUDED_GITHUB_REPOS`. Re-run. The repo will be dropped from output.

### 8.9 When to use `--fresh`

**Rarely.** `--fresh` ignores existing state and rebuilds from scratch — which means **you lose every `manual_override` flag** unless they're explicitly preserved by a finale-merger-style script. Use cases:

- After a major schema change to `DiscoveredPaper` (rare — design choice was to migrate, not rebuild)
- You want to verify reproducibility of a complete run
- Existing state file is corrupted

Always back up `research-papers-full.json` first.

### 8.10 Backing up state

The pipeline doesn't auto-backup. Convention used previously (see `.pre-*.bak.json` files):

```bash
cp output/research-papers-full.json output/research-papers-full.pre-$(date +%Y-%m-%d).bak.json
```

before any risky operation (major config change, --fresh run, schema migration). Old backups can live in `output/backups/` to declutter — nothing reads them.

### 8.11 Reclassifying after a fix

If you change `config.py` (e.g. tighten `NDIF_README_NEGATIVE_PATTERNS`, add a keyword) or tweak the pre-filter regexes in `process.py`:

```bash
# Dry-run first
python -m ndif_citations reclassify --dry-run

# Apply if happy
python -m ndif_citations reclassify
```

`reclassify` skips papers with `manual_override=True`. It re-runs `classify_category()` but doesn't re-fetch PDFs (uses cached) or re-extract thumbnails.

---

## 9. Configuration knobs

**File:** `src/ndif_citations/config.py`

Knobs you'll actually touch (env vars first, then code constants):

### 9.1 Environment variables (`.env`)

| Var | Default | Effect |
|---|---|---|
| `LLM_API_KEY` | none | Required for `run`. Without it, LLM features use fallbacks (0.4 confidence). |
| `LLM_BASE_URL` | NVIDIA Build | OpenAI-compatible endpoint. |
| `LLM_MODEL` | `meta/llama-3.1-70b-instruct` | Model identifier. |
| `S2_API_KEY` | none | Speeds up S2 calls 6× (0.5 s/call vs 3 s). Free tier: 1000 req/sec/key. |
| `OPENALEX_EMAIL` | none | Polite-pool access — better OpenAlex rate limits + reliability. |
| `UNPAYWALL_EMAIL` | falls back to `OPENALEX_EMAIL` | Required for Unpaywall PDF lookup. |
| `GITHUB_TOKEN` | none | 60→5000 req/hr. Without it, activity fields are null for repos past the first ~30. |
| `SERPAPI_API_KEY` | none | Enables Google Scholar discovery. Free tier = 250 calls/month (~25 full runs). |

### 9.2 Constants in `config.py`

| Constant | Default | When you'd change it |
|---|---|---|
| `MIN_PAPER_YEAR` | `2024` | Move to `2025` if pre-2024 noise increases. |
| `SEED_S2_IDS` | 2 IDs | If a new "canonical" NDIF paper appears (e.g. published version of the 2024 preprint), add its S2 ID. |
| `OPENALEX_SEARCH_QUERIES` | `["nnsight", '"national deep inference fabric"', "ndif.us"]` | Add new exact-phrase queries when the project picks up new identifiers. |
| `NDIF_KEYWORDS` | 6 strings | Keywords matched in PDFs for context extraction + zero-hit discard. |
| `NDIF_README_KEYWORDS_REGEX/SUBSTR` | small lists | Keywords for *repo* README classification (narrower than paper keywords — no "nnsight" since all dependents use it). |
| `NDIF_README_NEGATIVE_PATTERNS` | `["NDIF Discord", "NDIF Pilot Program", "join the NDIF"]` | Boilerplate that shouldn't count as NDIF usage. Add as new patterns emerge. |
| `EXCLUDED_PAPER_TITLES` | `{seed-paper-title}` | Add titles you never want to appear. |
| `EXCLUDED_GITHUB_REPOS` | `{"ndif-team/nnsight"}` | Repos to drop entirely. |
| `KNOWN_COURSE_SOURCES` | `{"callummcdougall/ARENA_3.0"}` | Repos whose forks are course-tagged. |
| `COURSE_NAME_PATTERNS` | `["ARENA", "MATS", "CBAI"]` | Name/desc substrings that flag course origin. |
| `SHARED_PAPER_THRESHOLD` | `5` | Min repos sharing same `linked_paper_url` to trigger unlink. Lower → more aggressive unlinking. |
| `LLM_RATE_LIMIT_SLEEP` | `12.0` s | NVIDIA Build free tier is ~5 req/min. Lower this if you have a paid endpoint. |
| `S2_RATE_LIMIT_SLEEP` | `3.0` s (anon) | Drops to 0.5 s when `S2_API_KEY` is set. |
| `CONTEXT_WINDOW` | `500` chars | Window around each NDIF keyword. |
| `MAX_CONTEXT_EXCERPTS` | `5` | Cap on context windows fed to LLM. |

---

## 10. Updating the website

After a successful run:

```bash
# From ndif-citations/
cp output/research-papers.json ../ndif-web-beta/packages/ndif.us/data/research-papers.json
cp output/github-repos.json ../ndif-web-beta/packages/ndif.us/data/github-repos.json
cp output/images/*.png ../ndif-web-beta/packages/ndif.us/public/images/

cd ../ndif-web-beta
bun run build  # verify it builds clean
```

The website's `ResearchPaper` TypeScript interface must match `DiscoveredPaper.to_website_dict()` output. If you add a field, update both.

---

## 11. CI/CD

A template GitHub Actions workflow lives at `docs/github_actions_template.yml`.

**Critical setup detail (from `docs/CI_CD_NOTES.md`):** Surya layout detection downloads a ~200 MB Segformer model on first use. **You must cache `~/.cache/huggingface/hub` in CI**, keyed off `pyproject.toml`, or every run will spend 5+ minutes downloading.

```yaml
- name: Cache HuggingFace Models (Surya OCR)
  uses: actions/cache@v4
  with:
    path: ~/.cache/huggingface/hub
    key: ${{ runner.os }}-huggingface-${{ hashFiles('pyproject.toml') }}
    restore-keys: |
      ${{ runner.os }}-huggingface-
```

Cadence suggestion: nightly cron + push-to-main trigger.

---

## 12. Troubleshooting

### "No PDF found for X" warnings

Most papers behind paywalls have no open-access PDF. This is fine — the paper still gets classified from its abstract and falls into `pending` if there's no abstract either. Check `paper.pdf_url` in the full JSON. If it's null and you have a working PDF link, edit it in, set `has_thumbnail: false`, re-run.

### Pipeline hangs on LLM calls

Almost certainly rate-limiting. The free NVIDIA Build tier is ~5 req/min, and the pipeline sleeps 12 s between calls. For 100 papers this is ~25 minutes of LLM time. To verify: run with `--verbose` and watch the rate-limit-sleep logs.

### "GitHub API rate-limit hit" warning

Without `GITHUB_TOKEN`: 60 req/hr. With a token: 5000 req/hr. If you see this, set a token. Affected repos this run will be kept with stale data (won't be purged).

### `research-papers-full.json uses the old flat-list format` error

You're trying to load a pre-3-bucket-schema file. Either:
- Delete the file and run with `--fresh` to regenerate, OR
- Manually convert to `{"pending": [...], "verified": [...], "discarded": [...]}` structure.

### Thumbnail extraction fails silently

Surya layout detection requires the Segformer model. First run downloads ~200 MB. If your `~/.cache/huggingface/hub` is missing or unreadable, thumbnails fail without error. Run `python -c "from surya.model.detection.model import load_model; load_model()"` to force download.

### A correct paper keeps getting demoted to pending

Set `manual_override=True` on it (via `promote <id> --detail "..."`). The auto-demote logic respects that flag.

### Course forks keep showing up as research

Check (1) is the parent in `KNOWN_COURSE_SOURCES`? (2) do they share a `linked_paper_url` with ≥5 other repos? (3) does the name/description match any `COURSE_NAME_PATTERNS`? If none — add the parent or a name pattern to `config.py`.

### A repo shows the same paper as many others (template-pollution)

This is what `_unlink_shared_template_papers` is for. If a template-inherited link slips through, either lower `SHARED_PAPER_THRESHOLD` (e.g. from 5 to 3) or manually edit the repo entry in `github-repos-full.json` and set `linked_paper_url: null, manual_override: true`.

---

## 13. Known issues & tech debt

**As of 2026-05-19, these are unresolved and would mislead a new maintainer if undocumented:**

1. **`add <url>` CLI is broken.** Calls `process_papers(papers, out)` with `list[DiscoveredPaper]`, but `process_papers` now requires `list[RoutingDecision]`. Crashes immediately. See [`cli.py:416`](../src/ndif_citations/cli.py). Workaround in §8.3. **Fix:** wrap the paper in a routing decision or call `route_papers([paper], existing)` first.
2. **CSV writers are orphaned.** `_write_csv` ([`output.py:473`](../src/ndif_citations/output.py)) and `_write_repos_csv` ([`output.py:429`](../src/ndif_citations/output.py)) are defined but never called. README/CODEBASE_CONTEXT.md still advertise `research-papers.csv` and `github-repos.csv` — they don't get written. Either wire the calls into `write_outputs`/`_write_repos_outputs` or delete the functions and update docs.
3. **`__version__` drift.** `src/ndif_citations/__init__.py` says `"0.1.0"` while `pyproject.toml` says `"1.9.0"`. Bump on next release.
4. **Dead code in `utils.py`:** `calculate_image_score()`, `get_section_for_page()`, `download_pdf()` (superseded by `pdf_cache.py`). Safe to delete.
5. **`process_papers_legacy()`** in `process.py` is marked deprecated and unreferenced. Safe to delete.
6. **`download_pdf()` deprecated wrapper** at the bottom of `pdf_cache.py`. Safe to delete.
7. **Linked-paper tier docs inconsistency.** Code returns 4 tiers + `(None, None)` for no-match. README says "5-tier priority" listing `null` as tier 5. Wiki and CODEBASE_CONTEXT say 4 tiers. Pick one: either change code to return tier=5 for no-match, or update README to say "4 tiers + no-match". Recommended: latter.
8. **Repo `_unlink_shared_template_papers` mutations to existing-not-rediscovered repos get lost.** If a repo previously had a link that becomes a shared-template link this run, but the repo itself isn't re-discovered (e.g. GitHub temporarily returns fewer pages), the unlink doesn't persist because `merge_repos` re-loads existing repos from disk fresh. Edge case; unlikely to bite often.
9. **No retry on CrossRef/Unpaywall/arXiv API failures.** GitHub dependents scrape retries 3×, but these silently return `{}`. Inconsistent. Worth normalizing.
10. **LLM logprobs still not used.** Confidence is now banded (CERTAIN/HIGH/MEDIUM/LOW/NONE) from structural evidence — fixed as of 2026-05-19. Next improvement would be requesting `top_logprobs` from providers that expose them (NVIDIA Build may not) and deriving a real probability from the token distribution.
11. **Auto-demote-on-merge can churn.** When a paper temporarily loses its abstract (API hiccup), it auto-demotes to `stub_metadata` pending, then auto-promotes next run when the abstract returns. Curator-visible noise.
12. **`output/*.pre-*.bak.json`** — 4 backup files from the 2026-05-18 merger finale. Safe to move to `output/backups/` or delete.

**Files worth archiving:**
- `tasks/prd-*.md` — 4 shipped PRDs. Move to `tasks/archive/` so they don't look like current work.
- `scripts/merger/` — one-shot merger from 2026-05-18, won't run again. Move to `scripts/archive/`.
- `docs/PAPERS_NEEDING_BETTER_SUMMARIES.md` — likely stale.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **Bucket** | Curation state: `pending` / `verified` / `discarded`. Only `verified` ships to the website. |
| **Category** | NDIF relationship: `uses_ndif` / `uses_nnsight` / `referencing` / `unclassified`. |
| **Processing bucket** | Routing decision: `NEW` / `REPROCESS` / `FILL_GAPS` / `SKIP` / `PROTECTED`. Different concept from output bucket. |
| **Confidence band** | `CERTAIN` / `HIGH` / `MEDIUM` / `LOW` / `NONE` — categorical replacement for the old `category_confidence` float. Tied to evidence type (see §4.4b step 7). |
| **`manual_override`** | Curator's veto flag. When True, the pipeline never overwrites *non-empty* fields. Still permits filling empty fields via the FILL_GAPS path so curated state grows incrementally instead of staying frozen with gaps. The single most important field for maintenance. |
| **content_hash** | `sha256(title + "::" + abstract)[:16]` for papers, `sha256(description + "::" + last_commit + "::" + archived)[:16]` for repos. Drives change detection. |
| **Seed paper** | The NDIF paper itself — `ARXIV:2407.14561` (ICLR 2025) + the 2024 S2 preprint record. |
| **Linked paper tier** | 1=BibTeX, 2=Citation section, 3=single post-2020 ID, 4=most-recent of multiple. How confidently we identified the paper a repo cites. |
| **Confidence gate** | `is_confident_venue()` — a venue must contain a recognized acronym/journal name to be accepted. Otherwise the cascade falls through. |
| **Pre-filter pass** | Pattern-based context-window elimination before the LLM is called. Negative evidence, comparison tables, acks-only. Can classify without LLM. |
| **Course / research / experiment** | Repo types. Course = teaching material forks. Research = legitimate research repos. Experiment = personal exploration. |
| **Shared-template detection** | When ≥5 repos share the same linked_paper_url, all but the highest-star get the link cleared. Catches ARENA-style course propagation. |
| **Stub metadata** | A paper with year=0 OR no usable abstract OR no identifier. Goes to pending for human review. |
| **Auto-promote/demote** | Pipeline re-evaluates bucket on every merge. A pending paper whose metadata is filled in promotes automatically; a verified paper whose metadata regresses demotes automatically. Respects `manual_override`. |

---

## Appendix: where to start reading code

If you have to dig into the code (e.g. to fix one of the known issues):

1. **`src/ndif_citations/cli.py`** — start here. The `run()` function is the full pipeline entry point. Skim phases 1–4 to see the orchestration.
2. **`src/ndif_citations/models.py`** — `DiscoveredPaper`, `DiscoveredRepo`, enums. The data contract.
3. **`src/ndif_citations/router.py`** — short and self-contained. Read second.
4. **`src/ndif_citations/process.py`** — `classify_category` is where LLM logic lives. Read the pre-filter section carefully before changing anything.
5. **`src/ndif_citations/output.py`** — merge logic. `_update_existing` and `merge_repos` are subtle.
6. **`src/ndif_citations/discover.py`** — largest module (~1200 lines). Read it phase by phase: S2, OpenAlex, Scholar, GitHub.
7. **`src/ndif_citations/venue.py`** — venue cascade. Read alongside `extract.py:enrich_via_external_apis()`.
8. **`src/ndif_citations/utils.py`** — utility library. Affiliation heuristics here, plus all the API wrappers.

**Tests:** `tests/` has 27 test files (~80% coverage of helper logic). `pytest tests/` runs them all. No end-to-end test; verification of full runs is manual.
