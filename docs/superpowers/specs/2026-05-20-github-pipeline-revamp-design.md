# GitHub Pipeline Revamp — Design

**Date:** 2026-05-20 (rev. 3)
**Scope:** `ndif-citations/` GitHub track + site (`ndif-web-beta/`) cleanup
**Goal:** Simplify what we emit so a landing-page visitor can answer "is NNsight real, what's being built with it?" in one glance. Tighten the slim/full split, drop debug fields from the website JSON, and harden the scraper against silent data loss.

**Rev 3 changelog (2026-05-20):**
- Slim contract grew from 10 → 12 fields: added `is_fork: boolean` and `parent_full_name: string | null` so the site can show a "Fork of X" badge on the 9 reliably-detectable GitHub forks. Manual-clone detection remains out of scope.
- Dropped `"workshop"` from broadened `COURSE_NAME_PATTERNS` (too many false positives on workshop-paper research repos).
- Slim JSON ships all repos in state (including stale-but-protected within the 30-day age-out window). No per-emit filter for `last_seen == today`.

**Rev 2 changelog (2026-05-20):**
- Investigated and confirmed the §3.3 manual-clone hypothesis with live GitHub API spot-checks; reframed as a not-a-bug finding rather than an open action item.
- Dropped the pipeline-side `0.7 * existing` partial-scrape abort guard — moved to the curator frontend (out of scope here).
- Removed CSV outputs entirely (`_write_csv` / `_write_repos_csv` are orphans today); only XLSX persists.
- Clarified hybrid staleness policy: API-confirmed dead repos drop immediately; only scrape-absent repos use the 30-day soft age-out.
- Added: `_tag_repo_type()` honors `manual_override=True` so curator-set `repo_type` sticks across runs.
- Pinned `first_seen` / `last_seen` stamping location to `merge_repos()` in `output.py`.
- Acceptance criteria revised accordingly (dropped AC #4, replaced AC #5).

---

## 1. What the visitor sees (informing the schema)

Four real visitor types — researchers evaluating NNsight, grad students looking for examples, NSF/funders doing diligence, curious community. All answering one question: **"is this real?"** Per-card they need:

1. What is it (name + description)
2. Is it serious (stars, recent activity, linked paper)
3. Take me there (URL)
4. Glance signal (language, course badge)

They don't need: pipeline state, classification audit trails, raw arXiv ID lists, license, topics, NDIF-vs-NNsight (it's 3/231 vs 228/231 — useless distinction), `archived` (already excluded).

---

## 2. Schemas

### 2.1 Slim `github-repos.json` — 12 fields, ships to the site

```json
{
  "owner": "EleutherAI",
  "repo": "delphi",
  "url": "https://github.com/EleutherAI/delphi",
  "description": "Automated interpretability with sparse autoencoders.",
  "stars": 253,
  "forks": 58,
  "last_commit": "2026-04-27",
  "language": "Python",
  "linked_paper_url": "https://arxiv.org/abs/2410.13928",
  "is_course": false,
  "is_fork": false,
  "parent_full_name": null
}
```

**All 231 repos ship.** No pipeline-side trim. Visitors navigate via sort (stars / recent / A-Z) and search. The site already drops its "Show low-signal" and "Coursework" filters separately (see §4).

**Why a boolean for course, not the 3-way `repo_type`:**

- The site only ever reads `repo_type === "course"`. Pipeline-side, the 3-way (`research` / `course` / `experiment`) still drives `_tag_repo_type()` decision-making and reporting — but it's a pipeline internal, not a contract with the site.
- Boolean avoids string-union drift between `models.py`, JSON, and `data/github-repos.ts`.

**Dropped from slim** (relative to today's emission): `archived`, `license`, `topics`, `readme_arxiv_ids`, `linked_paper_tier`, `category`, `classification_reason`, `content_hash`, `manual_override`, `has_metadata`, `has_classification`, `processing_bucket`, `repo_type` (replaced by `is_course`). `is_fork` and `parent_full_name` are kept (per rev 3 — so the site can show "Fork of X" for the 9 reliably-detectable GitHub forks).

### 2.2 Full `github-repos-full.json` — persistence layer

Keeps everything currently emitted, plus two new fields:

- `first_seen: str` (ISO date) — when this repo first appeared in any pipeline run
- `last_seen: str` (ISO date) — last run that observed it

Continues to carry `repo_type` (3-way), all pipeline state flags, etc. Nothing dropped.

### 2.3 XLSX "GitHub" sheet — audit / grant reporting

Columns in this order:

| Column | Source |
|---|---|
| owner, repo, url | identity |
| description | metadata |
| stars, forks | impact |
| last_commit, first_seen | activity / trend |
| language, license | tech |
| topics (joined `"a, b, c"`) | tags |
| linked_paper_url | linkage |
| linked_paper_tier | audit |
| readme_arxiv_ids (joined) | audit |
| category | grant-slide breakdown |
| repo_type | full 3-way categorization |
| parent_full_name | fork chains |
| archived, is_fork | filterable |
| classification_reason | audit |
| manual_override | curation flag |

**Dropped from XLSX:** `content_hash`, `has_metadata`, `has_classification`, `processing_bucket` (no human reads these).

**CSV output is removed entirely.** `_write_csv` and `_write_repos_csv` in `output.py` are orphaned (defined but never called by `write_outputs` / `_write_repos_outputs`). Delete both functions in this PR (~80 LOC of dead code) and the unused `csv` import. Grant reporters export from XLSX if they need CSV.

---

## 3. Pipeline changes (`ndif-citations/`)

### 3.1 `models.py` — `DiscoveredRepo`

- Add `first_seen: Optional[str]`, `last_seen: Optional[str]` (ISO `YYYY-MM-DD`)
- Update `to_website_dict()` to emit exactly the 10 slim fields (currently emits the full state — that's the bug)
- Update `to_full_dict()` to include the new fields
- Add a derived property `is_course` → `self.repo_type == "course"` for cleanliness at the emit point

### 3.2 `output.py`

- Fix the slim/full leak: `to_website_dict()` must produce only the 10 fields. Today it emits internal state.
- Stamp `first_seen` and `last_seen` **inside `merge_repos()`** (single source of truth — `discover.py` and `process.py` don't touch these):
  - NEW repos: `first_seen = last_seen = today()`
  - Re-observed repos: `last_seen = today()`, `first_seen` preserved
  - On the first run after this change ships, backfill any existing repo missing the fields with `first_seen = last_seen = today()`
- Replace one-shot purge with a **hybrid policy**:
  - **API-confirmed dead** (404 / rename_redirect / archived from `enrich_repos_from_github_api`) → drop immediately. Positive evidence; no benefit to waiting.
  - **Absent from scrape, no API confirmation** (e.g., dependents page returned fewer pages than expected due to a flaky scrape) → keep with stale data; soft age-out by `last_seen` after 30 days.
- XLSX column order matches §2.3 above.

**Note on mass-deletion protection.** The earlier draft proposed a `len(discovered) < 0.7 * len(existing)` abort guard inside the pipeline. We've moved that responsibility to the curator frontend (out of scope for this PR), which will surface a warning before the maintainer commits/merges the new outputs. The pipeline itself never aborts; the hybrid age-out above already buys time for a partial scrape to recover on the next run.

### 3.3 `discover.py` — course-tag improvements

- Extend `COURSE_NAME_PATTERNS` (in `config.py`) to add: `"course project"`, `"coursework"`, `"exercises"`, `"capstone"`, `"homework"`, `"assignment"`, `"workshop"`. These come from a real audit of 2 missed course repos in the experiment bucket. Broaden cautiously and only against `name + description`.
- **`parent_full_name` gap — confirmed 2026-05-20 as not-a-bug.** Of 38 course-tagged repos in the current state, only 1 (`Gattocrucco/ARENA_3.0`) has `is_fork=True` + `parent_full_name` set. A live GitHub API spot-check of 5 random ARENA-named course repos (`84rt/ARENA`, `atharvanihalani/ARENA_3.0`, `iakshay/ARENA_3.0`, `sterfd/ARENA`, `tinuademargaret/ARENA_3.0`) returned `is_fork=False, parent=null` for all 5, with `pushed_at` dates spanning 2024-03 to 2025-12 (not stale cached responses). **These are manual clones** — `git clone https://github.com/callummcdougall/ARENA_3.0` followed by `git push` to a brand-new repo — which GitHub's API has no way to link back to the upstream. No code change at the `parent.full_name` capture site is needed. The existing `COURSE_NAME_PATTERNS` rule (Tier 3 in `_tag_repo_type`) catches 32/38 of them already; broadened patterns above cover the residual 2 known leaks. Recovering the rest would require manual-clone detection (README-hash / commit-ancestry comparison against ARENA_3.0), which is **out of scope** — see §6.

- **Honor `manual_override=True` in `_tag_repo_type()`.** Today the tagging function runs unconditionally on every repo, every run — including repos the curator has manually classified. Add an early-return at the top: if `repo.manual_override` is True, return `repo.repo_type` unchanged. Also update the in-place loop in `cli.py` (around the `for repo in all_for_cross: repo.repo_type = _tag_repo_type(...)` site) to skip `manual_override=True` repos, so a curator-set `repo_type='course'` on a manual ARENA clone sticks across runs. Since the new slim JSON derives `is_course` from `repo_type`, this gives the curator a reliable lever for false negatives the heuristics miss.

Acceptance for the course-tag work: (1) the 2 known leaks (`AntonKorznikov/TheiaSae`, `MichaelRipa/coding-exercises-metarepo`) classified as `course` after the broadened patterns ship; (2) zero regressions in the existing 38-repo course bucket; (3) curator-set `repo_type='course'` with `manual_override=True` persists across a `run --fresh` followed by a normal `run`.

### 3.4 Schema snapshot test

New file: `tests/test_output_schema.py` (the `tests/` dir already contains 27 test files and a `conftest.py` with `make_repo` / `make_paper` factories — use those).

- Test 1: build a `DiscoveredRepo` via `make_repo()`, call `to_website_dict()`, assert keys exactly match the 10-field set (`owner`, `repo`, `url`, `description`, `stars`, `forks`, `last_commit`, `language`, `linked_paper_url`, `is_course`).
- Test 2: load `output/github-repos.json` if present and assert every entry has exactly those 10 keys (skip if file absent so a clean clone doesn't fail).
- Test 3: `manual_override=True` + curator-set `repo_type='course'` survives a `_tag_repo_type()` call that would otherwise re-tag it.
- Test 4: `merge_repos()` stamps `first_seen` and `last_seen` correctly on NEW and re-observed repos.

---

## 4. Site changes (`ndif-web-beta/packages/ndif.us/`)

Already captured in `ndif-citations/tasks/notes-site-trim-changes.md`. Summary:

- `data/github-repos.ts`: tighten `GitHubRepo` to the 10 fields; `repo_type` → `is_course: boolean`; drop `is_featured`.
- `lib/repos.ts`: delete `isLowSignal()`; update `isCoursework()` → `r.is_course`; update `hasPaper()` → uses `!r.is_course`.
- `components/research/GitHubRepoList.tsx`: remove the "Show low-signal" checkbox, the `?low=1` URL param, the `coursework` filter chip + `CHIP_LABELS` entry + `chipMatch` branch + `Chip` union member. Keep search, sort, pagination, and the Course badge on cards.
- `components/research/ResearchTabs.tsx`: `codeCount` becomes `githubRepos.length` (drop the `isLowSignal` filter and its import).
- `components/research/GitHubRepoCard.tsx`: verify no reads of dropped fields (`archived`, `license`, `topics`, `linked_paper_tier`, `category`).
- `components/FeaturedCode.tsx`: drop the `is_featured` pinned-list logic; drop the `!r.archived` check (pipeline already excludes archived). Keep the existing rule — top 6 by stars with 2 slots reserved for `linked_paper_url !== null` repos.

**Site is a separate PR** that lands after the pipeline PR. The slim JSON change is forward-compatible because the site tolerates extra fields during the transition.

---

## 5. FeaturedCode ranking — decision recorded

**Rule:** top 6 by stars, 2 of 6 reserved for repos with a linked paper. Stars alone, no forks weighting. Forks correlate with coursework copies (ARENA has 685 forks, most are student copies of the same curriculum) and don't add useful signal at the top of the landing page.

---

## 6. Out of scope (deferred)

- Linked-paper validation (cross-check paper title against repo name/description before trusting tier-4 guesses)
- Faster GitHub fetching via rate-limit headers (current 2.0s sleep → ~720ms)
- Stars-over-time JSONL snapshot for trend graphs
- Manual curation override (`is_featured` flag) — dropped from FeaturedCode entirely in this revamp
- Papers track changes — this revamp is GitHub-only
- **Manual-clone detection** for ARENA derivatives that aren't GitHub forks (most of them). The `COURSE_NAME_PATTERNS` rule + curator `manual_override` is the supported path for these. See §3.3 for the 2026-05-20 investigation that confirmed this.
- **Pipeline-side mass-deletion guard** — moved to the curator frontend, which surfaces a diff warning before the maintainer merges the new outputs into the website repo.

---

## 7. Acceptance criteria

1. `python -m ndif_citations run --skip-papers` produces a slim `github-repos.json` where every entry has exactly the 10 keys listed in §2.1.
2. `github-repos-full.json` contains `first_seen` and `last_seen` for every repo (backfilled with today's date on first run).
3. `research-data.xlsx` "GitHub" sheet matches the column list in §2.3. No `research-data.csv` / `github-repos.csv` files are produced.
4. After a run with broadened `COURSE_NAME_PATTERNS`: (a) `AntonKorznikov/TheiaSae` and `MichaelRipa/coding-exercises-metarepo` are tagged `is_course=true`; (b) no regressions in the existing 38-repo course bucket. Manual-clone detection of ARENA copies is documented as a known limitation (§6, and in the maintainer guide's "Known issues" section).
5. Curator workflow round-trips: set `repo_type="course"` + `manual_override=true` on any repo by hand-editing `github-repos-full.json` → run `python -m ndif_citations run --skip-papers` → the manual `repo_type` survives (verified by `is_course=true` in the slim JSON).
6. API-confirmed dead repos (404 / rename / archived) still drop on the same run; scrape-absent repos survive for up to 30 days via the `last_seen` age-out.
7. `pytest tests/test_output_schema.py` passes (4 tests: slim-keys, persisted-keys, manual_override-protects-repo_type, first_seen/last_seen stamping).
8. Site builds (`bun run build`) after the companion site PR.
