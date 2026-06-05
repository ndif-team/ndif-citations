# NDIF Citations — Local Web App (CLI → localhost) — Design Spec

> **Date:** 2026-06-05
> **Status:** Draft for review
> **Goal:** Replace the curator's CLI workflow with a local, single-user web app that makes *runs* and *edits* far more flexible, without rewriting the pipeline.

---

## 1. Goal & success criteria

Turn the `ndif-citations` CLI pipeline into a **local single-user web app** (localhost, no auth, no cloud) that lets the curator:

1. **See** all existing data at startup — papers by bucket (`verified` / `pending` / `discarded`) and repos by type (`research` / `course` / `experiment`).
2. **Edit** any entry on the go — paper fields, images, repo tags — with one-click bucket moves.
3. **Re-process** targeted fields on selected items (summary / classify / thumbnail / affiliations) without a full run.
4. **Run** the pipeline transparently — live phase + per-item progress, an explicit **gate before LLM spend**, and visible LLM rate-limit countdowns.
5. **Publish** results to the live site (`ndif-web-beta`) with one action.

**Success =** the curator never has to touch the terminal or hand-edit JSON for the routine loop (run → review → curate → publish), and LLM budget is only spent on items they approve.

### Non-goals (YAGNI)
- No auth, no hosting, no multi-user, no cloud.
- **No database** — the existing JSON files remain the source of truth.
- No rewrite of scraping / venue / LLM / thumbnail logic — it's reused as a library.
- The existing **CLI keeps working** (both CLI and web call the same orchestrator).

---

## 2. Architecture overview

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (localhost:8723)                                    │
│  Vite + React + TS + Tailwind + shadcn/ui  (SPA)             │
│    • REST for reads/actions   • SSE for live run progress    │
└───────────────┬─────────────────────────────────────────────┘
                │  HTTP / SSE  (127.0.0.1 only)
┌───────────────▼─────────────────────────────────────────────┐
│  FastAPI app  (uvicorn, started by `ndif-citations serve`)   │
│  • serves built SPA static assets                            │
│  • /api/* routers  → service layer                           │
│  • JobRunner: ONE run at a time, background thread,          │
│    event queue, cancel token                                 │
└───────────────┬─────────────────────────────────────────────┘
                │  in-process import (same venv)
┌───────────────▼─────────────────────────────────────────────┐
│  ndif_citations pipeline (library)                           │
│  orchestrator.py (NEW)  → discover/route/process/finalize    │
│  discover • extract • router • process • output • config     │
│  reads/writes output/*.json  (unchanged persistence)         │
└──────────────────────────────────────────────────────────────┘
```

**Why this shape:** the pipeline is local-file-bound, key-bound (`.env`), heavy-dep (Surya 200 MB), and rate-limited (~30 min runs). A local in-process FastAPI service is the simplest thing that reuses every line of pipeline logic, streams progress, and needs zero infra. Models are already Pydantic v2 → FastAPI serializes them for free.

### Package layout
```
ndif-citations/
├── src/ndif_citations/
│   ├── orchestrator.py      # NEW — staged, event-emitting pipeline (CLI + web share this)
│   ├── events.py            # NEW — ProgressEvent types + a thread-local emit sink
│   ├── jobs.py              # NEW — in-process JobRunner (one run, thread, queue, cancel)
│   ├── settings_store.py    # NEW — runtime-editable settings.json bridge
│   ├── publish.py           # NEW — detect/validate ndif-web-beta + copy slim JSON + images
│   ├── server/
│   │   ├── app.py           # NEW — FastAPI app + static mount + SSE
│   │   ├── deps.py          # NEW — shared deps (output dir, run-lock guard)
│   │   ├── routers/         # NEW — papers, repos, runs, reprocess, publish, settings, venues, stats, images
│   │   └── services/        # NEW — load→mutate→write wrappers, force-reprocess, gate
│   ├── cli.py               # MODIFIED — `run`/`add` re-implemented on orchestrator; new `serve`
│   ├── config.py            # MODIFIED — load knobs from settings.json + reload_settings()
│   └── (discover/extract/router/process/output/utils/...)  # lightly instrumented for progress
├── web/                     # NEW — Vite React SPA (built to web/dist/, served by FastAPI)
├── settings.json            # NEW — runtime knobs (git-ignored; ships with .example)
└── output/
    ├── staging/             # NEW — discovered+routed candidates awaiting the gate
    ├── backups/             # NEW — auto pre-fresh / pre-publish snapshots
    └── runs/                # NEW — per-run stats + event log (history)
```

---

## 3. The pipeline refactor (shared by CLI + web)

The current orchestration lives inside `cli.py:run()` (lines 42–266) and is tangled with Rich console printing. We extract it into **`orchestrator.py`** as discrete, event-emitting stages. `cli.run()` becomes a thin consumer that renders events to the Rich console (CLI UX preserved); the web `JobRunner` consumes the same events into an SSE stream.

### 3.1 Stages (the gate-before-LLM seam)

| Stage | Functions (existing) | Cost | Emits |
|---|---|---|---|
| **discover** | `discover_s2_citations`, `discover_openalex`, `discover_scholar`, `discover_github_dependents`, `deduplicate_papers`, `filter_by_min_year` | cheap (scrape APIs) | source counts, dedup count |
| **enrich** | `enrich_papers`; repos: `enrich_repos_from_github_api`, `_unlink_shared_template_papers`, `_tag_repo_type`, `link_repos_to_papers` | cheap (metadata APIs) | per-item enrich progress |
| **route** | `route_papers(unique, existing)`, `route_repos(disc, existing)` | cheap (DB compare) | bucket counts (NEW/REPROCESS/FILL_GAPS/SKIP/PROTECTED) |
| **— GATE —** | *persist staging → await curator selection* | — | `awaiting_review` + candidate list |
| **process** | `process_papers(approved_decisions, out)`, `process_repos(repo_decisions)` | **expensive (LLM, PDF, Surya)** | per-paper step events + LLM countdowns |
| **finalize** | `merge_papers`, `merge_repos`, `write_outputs`, `_write_repos_outputs`, `_write_xlsx`, `print_report`/report event | cheap | merge results, output paths |

**Ordering constraints honored:** repo enrichment + `_unlink_shared_template_papers` + `_tag_repo_type` + `link_repos_to_papers` all run in **enrich** (they need the full repo set and both papers+repos together); routing needs existing-from-disk; merge precedes write.

**Run modes:**
- **`discover` (incremental, default):** runs discover→enrich→route, **pauses at GATE**. Curator picks which `NEW`/`REPROCESS` candidates to process (and may discard or edit-then-process). Only approved items hit `process`. Repos (cheap, no LLM) auto-flow but surface in the gate for a light review. Then finalize.
- **`fresh`:** `--fresh` semantics — **skips the gate** (deliberate full rebuild), auto-backs-up `*-full.json` first, processes everything. (Warns it drops `manual_override` unless preserved.)
- **skip toggles:** `skip_github` / `skip_papers` as today.

### 3.2 Progress events

`events.py` defines a small `ProgressEvent` set: `stage_start`, `stage_done`, `source_count`, `dedup`, `route_summary`, `item_start`, `item_step` (summary/classify/thumbnail/affiliations done), `rate_limit_wait {label, seconds, remaining}`, `awaiting_review`, `merge_result`, `report`, `error`, `cancelled`, `done`.

Injection strategy (from integration map):
- **Universal sleep hook:** `utils.rate_limit_sleep()` is the single choke point for *every* API/LLM sleep (S2, OpenAlex, Scholar, GitHub, CrossRef, arXiv, Unpaywall, **LLM 12 s**). It reads a **thread-local emit sink** (set by `JobRunner` at run start) and emits `rate_limit_wait` with a live countdown. No signature changes to the dozens of API helpers.
- **Per-item events:** explicit `emit()` calls added to the `process_papers` loop (`process.py:789`) after summary/classify/thumbnail/affiliations, and to the `process_repos` loop (`process.py:955`).
- **Rich decoupling:** `orchestrator` never prints; it only emits. `cli.run()` subscribes and renders to `console`. The thread-local sink + plain `logging` replace direct `console` use in the hot path (Rich console is not thread-safe).

### 3.3 Targeted force-reprocess  *(corrected after code review)*

`process_papers` honors a per-field `processing_needed` dict on each `RoutingDecision` (`{summary, classify, thumbnail, affiliations}`). **Gotcha (verified `process.py:822–840`):** on a `manual_override` `FILL_GAPS` paper a *protective hydration block* restores curated values from `existing_paper` **before** the per-field guards — `paper.category = existing.category` is **unconditional**, and `description`/`image`/`affiliations` use `existing.X or paper.X`. So the naive "clear the field and reroute" recipe is **defeated**: the value is restored from the existing record and then protected.

**Correct recipe** (the `reprocess_svc` works on a copy):
1. Load the paper; for each requested field *F*, clear *F* (`description`→"", `category`→`UNCLASSIFIED`, `image`→None, `affiliations`→"") **and** its `has_F` flag.
2. Set `manual_override = False` on the working copy **for the duration of the reprocess** → `is_protected_manual` is False → the hydration/guard block is skipped entirely and *F* re-runs purely from `processing_needed`.
3. Construct the `RoutingDecision` **directly** (not via `route_papers`, which would attach the un-cleared on-disk `existing_paper` and re-trigger hydration): `RoutingDecision(paper, ProcessingBucket.FILL_GAPS, existing_paper=paper, processing_needed={f: f in fields for f in ("summary","classify","thumbnail","affiliations")})`. Unrequested fields have `needs=False` → never touched.
4. Run through the **JobRunner** (serialized + streamed; the Surya global is never touched concurrently — §5).
5. Restore `manual_override = True`, re-derive `has_*`, re-run `_decide_bucket`, merge, write.

UI shows an `AlertDialog`: *"This overwrites your curated <field>. Continue?"*. **Caveat:** a forced re-classify still runs the zero-PDF-hit discard check; the service surfaces that outcome rather than silently discarding. **Test must assert the field value actually changes** (mock LLM returns a new value), not merely that routing produced `FILL_GAPS`.

### 3.4 Fix `add` (currently broken)
`cli.py:422` calls `process_papers(papers, out)` with `list[DiscoveredPaper]` (needs `list[RoutingDecision]`). Fix: `route_papers(papers, existing)` → `process_papers(decisions, out)` → `merge_papers` → `write_outputs`. The web "add paper by URL" uses the same fixed path.

### 3.5 Runtime-editable config
`config.py` reads env + constants **at import time**, and code accesses knobs as `config.X`. We:
- Add **`settings.json`** holding the runtime knobs: `EXCLUDED_GITHUB_REPOS`, `KNOWN_COURSE_SOURCES`, `COURSE_NAME_PATTERNS`, `NDIF_KEYWORDS`, `NDIF_README_*`, `MIN_PAPER_YEAR`, `SHARED_PAPER_THRESHOLD`, rate-limit sleeps, and `LLM_MODEL` / `LLM_BASE_URL`.
- `config.py` loads `settings.json` over its defaults at import, and exposes **`config.reload_settings()`** that re-reads the file and reassigns module attributes.
- The `JobRunner` calls `config.reload_settings()` **before each run**, so UI edits take effect without a server restart.
- **Secrets gotcha (verified):** `config.LLM_API_KEY` / `GITHUB_TOKEN` / `S2_API_KEY` are bound to module attributes **at import** (`config.py:46/95/102`) and read as `config.X` at call time. `load_dotenv(override=True)` updates `os.environ` but **not** those bound attributes. So `reload_settings()` must **also re-read the secret env vars into the module attributes** (`config.LLM_API_KEY = os.environ.get("LLM_API_KEY") or None`, etc.) after `load_dotenv(override=True)`. Without this, an edited `.env` key changes nothing.
- **Knob-access note (verified):** the codebase already reads every runtime knob via `config.X` (the only `from config import` of knobs is a *function-local* import in `utils.py`, which re-reads each call). So no broad call-site conversion is needed — the reload just works for these. Rate-limit values are *also* accepted as optional params to `process_papers`/`process_repos` (belt-and-suspenders).
- **`KNOWN_VENUES` reload (verified gap):** `config.KNOWN_VENUES` is a *derived dict* built at import from `data/known_venues.json` (`config.py:170–199`) and is **not** a settings.json key. The venue manager writes `known_venues.json` and must call a dedicated **`config.reload_venues()`** that re-runs the derivation block — `reload_settings()` alone does not rebuild it.

---

## 4. Backend API surface

All under `/api`, bound to `127.0.0.1`. Mutating endpoints return **409** while a run is active (single-writer invariant). Pydantic models reused directly.

**Data (papers)**
- `GET /papers?bucket&q&sort&page` → slim rows; `GET /papers/{id}` → full `DiscoveredPaper`
- `PATCH /papers/{id}` → edit `EDITABLE_FIELDS` (reuse `edit_schema` parsers) → sets `manual_override`, re-derives `has_*`, re-runs `_decide_bucket`, writes
- `POST /papers/{id}/bucket {bucket, reason?, detail?}` → promote/demote/discard (reuse cli logic)
- `POST /papers/{id}/reprocess {fields:[summary|classify|thumbnail|affiliations]}` → force path (§3.3)
- `POST /papers/{id}/image` (multipart upload) and `POST /papers/{id}/reextract-thumbnail`
- `POST /papers/add {url}` → fixed add path (§3.4)

**Data (repos)** — *net-new editing*
- `GET /repos?type&q&sort`; `GET /repos/{owner}/{repo}`
- `PATCH /repos/{owner}/{repo}` → override `repo_type`, `linked_paper_url`, `description`; sets `manual_override`
- `POST /repos/{owner}/{repo}/exclude` → add to `EXCLUDED_GITHUB_REPOS` (settings) + drop from state

**Runs**
- `POST /runs {mode: discover|fresh, skip_github, skip_papers}` → `{run_id}` (409 if one active)
- `GET /runs/{id}/events` → **SSE** stream of `ProgressEvent`s
- `POST /runs/{id}/gate {process_ids, discard_ids, edits}` → resume from GATE into `process`
- `POST /runs/{id}/cancel`; `GET /runs/{id}`; `GET /runs` (history from `output/runs/`)

**Publish / settings / misc**
- `GET /publish/target` (auto-detect + validate ndif-web-beta) · `PUT /publish/target {path}` · `POST /publish {dry_run}` (diff then publish). **Publish wraps the *existing* sync scripts** `ndif-web-beta/packages/ndif.us/scripts/sync-research-papers.mjs` + `sync-github-repos.mjs` (don't reinvent) — see §7.
- `GET/PUT /settings` · `GET/PUT /venues`
- `GET /stats` (dashboard counts) · `GET /images/{slug}` (serve `output/images/*`)

---

## 5. Run state machine & job model

```
IDLE → DISCOVERING → ENRICHING → ROUTING → AWAITING_REVIEW → PROCESSING → MERGING → DONE
                                              │ (fresh skips review)        │
                                              └──────────── auto ───────────┘
   any state ──cancel──► CANCELLED (merge completed items, keep staging)
   any state ──error───► ERROR (last good state preserved)
```

- **JobRunner** (`jobs.py`): a singleton holding `current_run` (id, state, started_at, event log) and a `threading.Lock`. Runs execute in a **daemon thread**; a `threading.Event` cancel token is polled between items and stages. Events go to an in-memory ring buffer + queue per run; SSE replays buffered events on (re)connect so a browser refresh re-syncs.
- **Staging:** after ROUTING, candidates (`decisions` + `repo_decisions` + discovered lists) are serialized to `output/staging/run-<id>.json` so a refresh/restart can resume the gate. Cleared on DONE/CANCELLED.
- **Cancel semantics (corrected):** `process_papers` mutates `decision.paper` **in place** and the merge consumes `[d.paper for d in paper_decisions]` — so unprocessed decisions still hold half-formed papers (empty summary, `UNCLASSIFIED`). A naive "merge all decisions on cancel" would inject junk into state. Mechanism: `process_papers`/`process_repos` accept a `cancel_check: Callable[[], bool]` checked at the **top of each item loop**; when set, they raise `RunCancelled` after appending only the completed items. The runner finalizes by merging **only the completed prefix** (`decisions[:completed]`); remaining approved items stay in staging for a later run; state = CANCELLED. Never partial-write mid-item.
- **Backups:** before `fresh` and before `publish`, snapshot `*-full.json` → `output/backups/<file>.<ts>.bak.json`.
- **Thread-safety:** one run at a time means the Surya global model and config reload are never touched concurrently. Edits/mutations are rejected (409) during a run.

---

## 6. Frontend

**Stack:** Vite + React + TS + Tailwind + shadcn/ui. **TanStack Query** (server state/caching), **TanStack Table** (data-dense tables under shadcn `Table`), **React Router** (routes), native `EventSource` for SSE. Built to `web/dist/`, served by FastAPI; `ndif-citations serve` opens the browser.

**Design system (Data-Dense Dashboard):**
- **Color:** primary `#1E40AF`, accent `#D97706`, neutral slate surfaces; full light+dark via CSS variables / shadcn theme.
- **Semantic tokens:**
  - *Bucket* — verified = green, pending = amber, discarded = slate (red on hover/destructive).
  - *Confidence band* — CERTAIN/HIGH = green, MEDIUM = amber, LOW = orange, NONE = slate.
  - *Run state* — running = blue, **LLM cooldown = amber (pulsing)**, error = red, done = green. *(The 12 s LLM sleeps are post-call throttle cooldowns, not "model is thinking" — label them "next LLM call in Xs", surfaced via the `rate_limit_wait` event.)*
  - Color is never the only signal (icon + label too — a11y).
- **Type:** Fira Sans (UI) + a mono (JetBrains Mono / Fira Code) for arXiv IDs, hashes, URLs, and the run log; **tabular figures** in numeric table columns.
- **Density:** compact rows, sticky header, column visibility toggles, virtualized lists (≥50 rows).

**Screens → shadcn components:**
1. **Dashboard** — KPI `Card`s (bucket counts, repo-type breakdown, last run), quick-actions; small bar/donut (recharts) for category/type distribution; “Run” and “Publish” entry buttons.
2. **Papers browser** — `Table`+TanStack, faceted filters (`Select`/`Popover` + `Command`), search `Input`, sort; row → detail `Sheet` (drawer) with every field, thumbnail, confidence/reason/source `Badge`s, links. Inline edit via `Dialog`/`Popover` forms; multi-select checkboxes → sticky action `Toolbar` (reprocess fields via `DropdownMenu`, bucket moves, all behind `AlertDialog` confirms). `Toast` (sonner) for results, with **Undo** on bucket moves.
3. **Repos browser** — same table pattern; edit `repo_type`/linked paper/description; exclude toggle.
4. **Run console** — `mode` selector + skip toggles → start; a phase **Stepper** (Discover→Enrich→Route→Review→Process→Merge); live **log feed** (mono, auto-scroll) from SSE; a prominent **rate-limit countdown** badge/progress; per-item progress (`x / N`, current title); `Cancel` button (`AlertDialog`). Respects `prefers-reduced-motion`.
5. **Review gate** — table of NEW/REPROCESS candidates with per-row **Process / Discard / Edit-then-process** (inline edit in a `Sheet`); select-all; “Process N selected (≈ est. LLM time)” primary button.
6. **Settings** — forms for knobs (`Input`/`Textarea`/`Switch`), LLM model/endpoint + rate-limit, publish-target picker; **Venue manager** (`Table` add/edit/remove with type + aliases).

**Accessibility/perf:** WCAG AA contrast in both themes, visible focus rings, keyboard nav, 150–300 ms transitions, skeletons for >300 ms loads, `aria-live` for toasts/log.

---

## 7. Error handling & edge cases

- **Single-writer:** all mutating endpoints + edits return 409 during an active run; UI disables edit controls and shows a “run in progress” banner.
- **Run failure:** pipeline already degrades gracefully (broad try/except + logger.warning); orchestrator catches per-stage, emits `error`, preserves last good state, leaves staging intact for retry.
- **Cancel:** see §5 (merge completed, keep staging).
- **Repo staleness:** `merge_repos` ages out repos not seen in >30 days unless `manual_override` (existing behavior). Surface aged-out repos in the report. A **papers-only** run (`skip_github`) must not call `_write_repos_outputs` at all (repos untouched). A partial GitHub scrape during a gated run is the existing one-way risk — unchanged, just made visible.
- **Repo edit-loss (verified gap):** `_tag_repo_type()` runs **unconditionally** on every repo each run (`cli.py:159`) and would overwrite a curator's `repo_type` edit. Fix: `_tag_repo_type` (and the linked-paper clear) must **skip `manual_override` repos** so curator repo edits stick. This is a small pipeline behavior change shipped with repo editing (Phase 4).
- **Publish safety (corrected):** publish **wraps the existing `sync-*.mjs` scripts** rather than a parallel Python copier. The site `import`s `public/data/*.json` at **build time**, so after publish the curator runs `bun run build` (a one-click "build & verify" button, or a clear instruction). The existing image sync is **additive-only** (skips files that already exist) — re-extracted thumbnails reuse the same slug filename and would silently *not* update; publish must **force-overwrite** changed images. Always target `public/data/` + `public/images/`, never `out/data/` (build output) and never the ignored `ndif-website/`. Dry-run diff first; backup the site's current `public/data/*.json` before copy.
- **Config reload:** `reload_settings()` + `load_dotenv(override=True)` before each run; bad settings.json → validation error surfaced, run blocked.
- **Image upload:** validate PNG, save as `output/images/{slugify(title)}.png`, set `paper.image`, set `has_thumbnail`.

---

## 8. Testing strategy

- **Pipeline (existing 631 tests — verified `pytest --co`):** must stay green after the orchestrator refactor (behavior-preserving extract + emit). **Test-harness gotcha (verified):** `tests/conftest.py` provides only `no_sleep`, `make_paper`, `make_repo`; `tests/helpers/` has only `llm.py` (MockLLMClient); `tests/test_smoke.py` is `assert True`. The parity test the spec wants needs a **new fixture-state copier + discovery/LLM fakes that don't exist yet** — building that harness is explicit scoped work (Phase 1, Task 1.0), not a `...` placeholder. Also: the autouse fixture currently stubs `rate_limit_sleep` to a no-op, which would bypass the new countdown emit — change it to patch `time.sleep` instead so `rate_limit_sleep` (and its emit) still runs.
- **Orchestrator/events:** unit-test stage sequencing, event emission, and the gate pause/resume on fakes (no network/LLM).
- **Force-reprocess:** unit tests that clearing field *F* + reroute yields `FILL_GAPS` with only *F* in `processing_needed`, and the guard protects other curated fields.
- **API:** FastAPI `TestClient` (httpx) for each router incl. the 409-during-run invariant, edit→bucket re-derivation, add-by-URL, publish dry-run, settings reload.
- **JobRunner:** one-run-at-a-time lock, cancel mid-batch merges completed items, staging resume.
- **Frontend:** `tsc` + eslint; a light Playwright smoke for the four critical flows (browse→edit→save, start run→see SSE events→cancel, gate select→process, publish dry-run). Mock the backend or run against a fixture output dir.

---

## 9. Build, run, launch

- `pip install -e .` (add deps: `fastapi`, `uvicorn[standard]`, `python-multipart`).
- Frontend dev: `cd web && bun install && bun run dev` (proxy `/api`→FastAPI). Prod build: `bun run build` → `web/dist/`.
- **`ndif-citations serve [--port 8723] [--no-open]`** → starts uvicorn, serves `web/dist/` + `/api`, opens `127.0.0.1:8723`. One command, one process.
- Surya 200 MB model still lazy-loads on first thumbnail (unchanged); document the warm-up.

---

## 10. Phasing (high-level — detailed plan follows in writing-plans)

1. **Foundation:** orchestrator extraction (behavior-preserving) + events + settings_store + fix `add`; keep CLI + tests green.
2. **JobRunner + run API + SSE:** background run, live events, cancel, history; CLI still works.
3. **Gate-before-LLM:** staging persist/resume, route→review→process split, gate endpoint.
4. **Data + curation API:** papers/repos read, edit, bucket, force-reprocess, image, add.
5. **Publish + settings + venues:** detect/validate/diff/copy; settings.json + venue manager.
6. **Frontend:** scaffold + design system → dashboard, browsers, curation, run console, gate, settings.
7. **Polish + tests:** Playwright smoke, a11y/dark-mode pass, docs (`serve`, maintainer guide update).

---

## Open questions / assumptions to confirm
- Export target is **`ndif-web-beta/packages/ndif.us/`** (`public/data/` + `public/images/`), **not** the ignored `ndif-website/`. *(assumed)*
- Default port `8723` (arbitrary; configurable). *(assumed)*
- On cancel, **merge already-processed items** rather than discard them. *(assumed)*
- Repos remain bucket-less (managed by `repo_type` + `manual_override`); no pending/verified for repos. *(confirmed by current model)*
