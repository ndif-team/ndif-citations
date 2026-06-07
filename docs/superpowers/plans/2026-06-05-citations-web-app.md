# NDIF Citations Local Web App — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Plan shape:** This is a **master plan** for a 7-phase, multi-subsystem project. Phase 1 is fully bite-sized (TDD steps with code) because it's the load-bearing refactor. Phases 2–7 are specified at **task + contract granularity** with exact files, interfaces, and exit criteria; each is expanded into full bite-sized steps at execution time (one fresh subagent plan per phase). Cross-phase **contracts** (§Contracts) are locked here so phases interlock.

**Goal:** Turn the `ndif-citations` CLI pipeline into a local single-user web app (FastAPI + Vite/React SPA at localhost) for browsing, editing, gated runs, targeted reprocessing, and one-click publish — without rewriting pipeline logic.

**Architecture:** A new `orchestrator.py` extracts the staged pipeline out of `cli.run()` and emits structured progress events; both the CLI and a FastAPI `JobRunner` (one run at a time, background thread) drive it. A React SPA (served by FastAPI) consumes REST + SSE. JSON files stay the source of truth; no DB.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, Pydantic v2 (existing models), Click (existing CLI); Vite + React + TS + Tailwind + shadcn/ui + TanStack Query/Table + React Router; SSE via `EventSource`.

**Reference spec:** `docs/superpowers/specs/2026-06-05-citations-web-app-design.md`

---

## File structure (locked)

**New Python modules** (`src/ndif_citations/`):
| File | Responsibility |
|---|---|
| `events.py` | `ProgressEvent` dataclass + thread-local emit sink (`set_sink`/`clear_sink`/`emit`) |
| `orchestrator.py` | Staged, event-emitting pipeline: `discover_stage`, `enrich_stage`, `route_stage`, `process_stage`, `finalize_stage`, `run_pipeline` |
| `settings_store.py` | Load/save `settings.json`; schema validation |
| `jobs.py` | `JobRunner` singleton: one run, background thread, event ring-buffer + queue, cancel token, staging persist/resume |
| `publish.py` | Detect/validate `ndif-web-beta`, diff, copy slim JSON + images, backups |
| `server/app.py` | FastAPI app, static SPA mount, CORS-off (same origin), 127.0.0.1 bind |
| `server/deps.py` | Shared deps: output dir, `require_no_active_run` guard |
| `server/routers/{papers,repos,runs,publish,settings,venues,stats,images}.py` | Endpoint groups |
| `server/services/{papers_svc,repos_svc,reprocess_svc,gate_svc}.py` | load→mutate→write wrappers, force-reprocess, gate apply |

**Modified Python:**
| File | Change |
|---|---|
| `cli.py` | `run`/`add` re-implemented on `orchestrator`; add `serve` command; keep all CLI UX via an event→console renderer |
| `config.py` | Load `settings.json` over defaults at import; add `reload_settings()`; convert runtime-knob direct imports to attribute access |
| `utils.py` | `rate_limit_sleep()` emits `rate_limit_wait` events via the sink (countdown) |
| `process.py` | `process_papers`/`process_repos` loops emit `item_*` events; accept optional rate-limit overrides |
| `pyproject.toml` | add `fastapi`, `uvicorn[standard]`, `python-multipart`; add `web` extra; bump version |

**New frontend** (`web/`): Vite React app → built to `web/dist/` (served by FastAPI).

**New runtime dirs/files:** `settings.json` (+ `.example`), `output/staging/`, `output/backups/`, `output/runs/`.

---

## Contracts (locked — every phase depends on these)

### C1. ProgressEvent + sink (`events.py`)
```python
from __future__ import annotations
import threading, time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

EventType = str  # "stage_start"|"stage_done"|"source_count"|"dedup"|"route_summary"
                 # |"item_start"|"item_step"|"rate_limit_wait"|"awaiting_review"
                 # |"merge_result"|"report"|"error"|"cancelled"|"done"|"log"

@dataclass
class ProgressEvent:
    type: EventType
    stage: Optional[str] = None      # "discover"|"enrich"|"route"|"process"|"finalize"
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    def to_dict(self) -> dict: return asdict(self)

_local = threading.local()
def set_sink(fn: Callable[[ProgressEvent], None]) -> None: _local.fn = fn
def clear_sink() -> None: _local.fn = None
def emit(type: EventType, stage: str | None = None, **data: Any) -> None:
    fn = getattr(_local, "fn", None)
    if fn is not None:
        fn(ProgressEvent(type=type, stage=stage, data=data))
```
A thread-local sink (not a contextvar) because runs execute in a worker **thread**. `emit()` is a no-op when no sink is set (so the library stays usable headless/in tests).

### C2. Orchestrator stage signatures (`orchestrator.py`)  *(corrected to match real control flow)*
```python
@dataclass
class DiscoverResult:
    papers: list[DiscoveredPaper]; repos: list[DiscoveredRepo]; run_stats: PipelineRun
@dataclass
class EnrichResult:                       # removal_counts is produced HERE (enrich_repos_from_github_api), not discover
    papers: list[DiscoveredPaper]; repos: list[DiscoveredRepo]
    removal_counts: dict[str, int]; existing_repos: list[DiscoveredRepo]   # loaded once, threaded forward
@dataclass
class RouteResult:
    paper_decisions: list[RoutingDecision]; repo_decisions: list[RepoRoutingDecision]
    existing_papers: list[DiscoveredPaper]; enrich: EnrichResult            # carry forward to avoid re-loading
@dataclass
class FinalizeResult:
    merged_papers: list[DiscoveredPaper]; merged_repos: list[DiscoveredRepo]; run_stats: PipelineRun

def discover_stage(out, *, skip_papers, skip_github, fresh) -> DiscoverResult: ...
def enrich_stage(out, d: DiscoverResult, *, skip_papers, skip_github, fresh) -> EnrichResult: ...
def route_stage(out, e: EnrichResult, *, skip_papers, skip_github, fresh) -> RouteResult: ...
# process_stage MUTATES decision.paper IN PLACE (matches process_papers reality); returns how many completed
# (for cancel). cancel_check is polled at each item boundary (see Task 1.7).
def process_stage(out, r: RouteResult, *, skip_papers, skip_github,
                  cancel_check: "Callable[[], bool] | None" = None) -> tuple[int, int]: ...
# finalize merges the DECISIONS (not a separate processed list): [d.paper for d in r.paper_decisions[:n_papers]]
# + check_venue_upgrades(enrich.papers, existing). Honors skip_github (then repos untouched).
def finalize_stage(out, r: RouteResult, run_stats, *, skip_papers, skip_github, fresh,
                   completed: tuple[int, int] | None = None) -> FinalizeResult: ...

def run_pipeline(out, *, mode: str, skip_papers=False, skip_github=False) -> FinalizeResult:
    """Full end-to-end (CLI + fresh). mode in {'fresh','incremental'}. Emits events. No gate pause."""
```
**Key realities encoded (verified):** (1) `process_papers` mutates `decision.paper` in place and the legacy merge consumes `[d.paper for d in decisions]` — so `finalize_stage` reads `RouteResult`, not a separate processed list; (2) `removal_counts` comes from enrich, not discover; (3) existing papers/repos are loaded once and threaded forward (the legacy flow loads them 3×). The **gate** = `JobRunner` runs `discover→enrich→route`, pauses, then `process→finalize` on a filtered `RouteResult`. `run_pipeline` is the ungated convenience for CLI + `fresh`.

### C3. settings.json schema (`settings_store.py`)
```json
{
  "min_paper_year": 2024,
  "shared_paper_threshold": 5,
  "excluded_github_repos": ["ndif-team/nnsight"],
  "known_course_sources": ["callummcdougall/ARENA_3.0"],
  "course_name_patterns": ["ARENA","MATS","CBAI"," course ","course project","coursework","exercises","capstone","homework","assignment"],
  "ndif_keywords": ["nnsight","NNsight","NDIF","ndif.us","nnsight.net","import nnsight"],
  "ndif_readme_keywords_regex": ["\\bNDIF\\b"],
  "ndif_readme_keywords_substr": ["ndif.us","NDIF cluster","hosted on NDIF"],
  "ndif_readme_negative_patterns": ["NDIF Discord","NDIF Pilot Program","join the NDIF"],
  "llm_model": "meta/llama-3.1-70b-instruct",
  "llm_base_url": "https://integrate.api.nvidia.com/v1",
  "llm_rate_limit_sleep": 12.0,
  "s2_rate_limit_sleep": 3.0,
  "github_rate_limit_sleep": 2.0,
  "publish_target": null
}
```
`config.reload_settings()` re-reads this and reassigns the matching module attributes. Secrets (`*_API_KEY`, `GITHUB_TOKEN`) stay in `.env` (refreshed via `load_dotenv(override=True)` before each run), never in settings.json.

### C4. JobRunner interface (`jobs.py`)
```python
class JobRunner:
    def start(self, *, mode: str, skip_papers=False, skip_github=False) -> str   # run_id; raises RunActiveError if busy
    def subscribe(self, run_id: str) -> Iterator[ProgressEvent]                   # replays buffered, then live (SSE)
    def submit_gate(self, run_id: str, *, process_ids: list[str], discard_ids: list[str], edits: dict[str, dict]) -> None
    def cancel(self, run_id: str) -> None
    def status(self, run_id: str | None = None) -> RunStatus
    @property
    def active(self) -> bool
```
Single global instance in `server/deps.py`. `RunStatus` includes `state` (the §5 state machine), `run_id`, `started_at`, counts, and (when `AWAITING_REVIEW`) the candidate list.

### C5. API row shapes
- `PaperRow` (list view): `id` (merge_key), `title`, `authors`, `venue`, `year`, `category`, `bucket`, `confidence_band`, `reason`, `source`, `has_image`, `manual_override`, `url`.
- `RepoRow`: `id` (`owner/repo`), `owner`, `repo`, `stars`, `forks`, `language`, `repo_type`, `category`, `linked_paper_url`, `last_commit`, `manual_override`.
- Detail endpoints return the full Pydantic model (`to_full_dict`). `id` for a paper = its `merge_key()` (`arxiv:…`/`doi:…`/`title:…`), URL-encoded.

---

## Phase sequencing & exit criteria

| Phase | Produces (independently testable) | Exit criteria |
|---|---|---|
| **1. Foundation** | test harness, `events`, `orchestrator`, `settings_store`, `config.reload_settings`/`reload_venues`, instrumented `rate_limit_sleep`/process loops, fixed `add` | All **631** existing tests pass (verified count); new harness/orchestrator/events/settings tests pass; **parity test**: `run_pipeline` output == legacy `run()` output on a fixture; CLI `run`/`discover`/`add` work |
| **2. JobRunner + Run API** | `jobs`, `server/app`, `/runs` + SSE, `cancel`, history, `serve` cmd | `POST /runs {mode:fresh}` streams events over SSE; cancel stops & merges completed; CLI still works; `ndif-citations serve` boots |
| **3. Gate-before-LLM** | staging persist/resume, route→review→process split, `/runs/{id}/gate` | Incremental run pauses at `AWAITING_REVIEW`; selecting a subset processes only those; restart mid-gate resumes from staging |
| **4. Data + curation API** | papers/repos read, `PATCH`, bucket, reprocess, image, add | Full curate-via-API incl. force-reprocess + 409-during-run; tests |
| **5. Publish + settings + venues** | `publish` (detect/validate/diff/copy + backup), `/settings`, `/venues` | Dry-run diff then publish copies slim JSON + images to ndif-web-beta; settings edit reloads; venue edits take effect |
| **6. Frontend** | Vite SPA: dashboard, browsers, curation, run console, gate, settings | All flows usable against the API in light+dark |
| **7. Polish + tests + docs** | Playwright smoke, a11y/perf pass, docs | Smoke green; maintainer-guide + README updated; `serve` documented |

---

## PHASE 1 — Foundation refactor (fully detailed)

**Why first:** everything (web runs, gate, reprocess) calls the orchestrator. The refactor must be **behavior-preserving** — proven by a parity test before we build on it.

### Task 1.0 — Build the test harness (prerequisite for the parity test)

> **Verified gap:** `tests/conftest.py` has only `no_sleep`/`make_paper`/`make_repo`; `tests/helpers/` has only `llm.py`; `tests/test_smoke.py` is `assert True`. The parity test (Task 1.8) needs a fixture-state copier and discovery/LLM fakes that **don't exist** — build them here, not as a `...` placeholder.

**Files:** Create `tests/helpers/fakes.py`, `tests/fixtures/mini-research-papers-full.json`; Modify `tests/conftest.py`.

- [ ] **Step 1:** Create `tests/fixtures/mini-research-papers-full.json` — a 3-bucket file with ~3 verified + 1 pending paper (hand-picked from `output/research-papers-full.json`, trimmed) and a sibling `mini-github-repos-full.json` (~3 repos).
- [ ] **Step 2:** Add a `fixture_state` fixture to `conftest.py` that copies those into a `tmp_path/output` and returns the dir.
- [ ] **Step 3:** In `tests/helpers/fakes.py`, add deterministic fakes: `fake_discover_papers()` (returns 2 `DiscoveredPaper`s — 1 already in fixture, 1 new), `fake_discover_repos()`, and `install_pipeline_fakes(monkeypatch, orchestrator_mod)` that patches `discover_s2_citations`/`discover_openalex`/`discover_scholar`/`discover_github_dependents`/`enrich_papers`/`enrich_repos_from_github_api` on the orchestrator module, plus `generate_summary`/`classify_category`/`extract_thumbnail`/`get_cached_pdf` on `process` (reuse `helpers/llm.MockLLMClient`).
- [ ] **Step 4:** Add `test_harness_smoke` that calls `install_pipeline_fakes` + asserts the fakes return the expected shapes (so the harness itself is covered).
- [ ] **Step 5:** Run `pytest tests/test_harness_smoke.py -v` → PASS.
- [ ] **Step 6: Commit** `git add -A && git commit -m "test: pipeline fakes + fixture-state harness"`

### Task 1.1 — Add web dependencies & scaffold

**Files:** Modify `pyproject.toml`; Create `settings.json.example`, `.gitignore` entries.

- [ ] **Step 1:** Add to `pyproject.toml` `dependencies`: `"fastapi>=0.110"`, `"uvicorn[standard]>=0.29"`, `"python-multipart>=0.0.9"`. Bump `version` to `2.0.0`.
- [ ] **Step 2:** Create `settings.json.example` with the C3 schema. Add `settings.json`, `output/staging/`, `output/backups/`, `output/runs/`, `web/dist/`, `web/node_modules/` to `.gitignore`.
- [ ] **Step 3:** `pip install -e .` — Expected: resolves with fastapi/uvicorn installed.
- [ ] **Step 4: Commit** `git add -A && git commit -m "chore: add web deps + settings scaffold"`

### Task 1.2 — `events.py` (ProgressEvent + sink)

**Files:** Create `src/ndif_citations/events.py`; Test `tests/test_events.py`.

- [ ] **Step 1: Write failing test**
```python
# tests/test_events.py
from ndif_citations import events
def test_emit_noop_without_sink():
    events.clear_sink(); events.emit("done")  # must not raise
def test_emit_to_sink():
    got = []
    events.set_sink(got.append)
    events.emit("item_step", stage="process", title="X", step="summary")
    events.clear_sink()
    assert got[0].type == "item_step" and got[0].stage == "process"
    assert got[0].data["title"] == "X" and got[0].data["step"] == "summary"
def test_sink_is_thread_local():
    import threading
    main_got, thread_got = [], []
    events.set_sink(main_got.append)
    def worker():
        events.emit("log", msg="bg")           # no sink in this thread
        events.set_sink(thread_got.append); events.emit("log", msg="bg2"); events.clear_sink()
    t = threading.Thread(target=worker); t.start(); t.join()
    events.emit("log", msg="main"); events.clear_sink()
    assert [e.data["msg"] for e in main_got] == ["main"]
    assert [e.data["msg"] for e in thread_got] == ["bg2"]
```
- [ ] **Step 2:** Run `pytest tests/test_events.py -v` → FAIL (no module).
- [ ] **Step 3:** Implement `events.py` exactly as Contract **C1**.
- [ ] **Step 4:** Run `pytest tests/test_events.py -v` → PASS.
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat: progress event sink"`

### Task 1.3 — Emit LLM-cooldown events (instrument `rate_limit_sleep`)

**Files:** Modify `src/ndif_citations/utils.py` (`rate_limit_sleep`, ~line 449); Modify `tests/conftest.py` (autouse sleep fixture); Test `tests/test_utils_misc.py` (append).

> **Verified semantics:** the LLM `rate_limit_sleep(12s)` calls fire *after* the API returns, success-path only (`process.py:135,432`) — post-call throttle cooldowns. Emit a `rate_limit_wait` so the UI shows "next LLM call in Xs". `rate_limit_sleep` is the single choke point for ALL API+LLM sleeps, so instrumenting it covers everything. **But** the autouse conftest fixture currently stubs `rate_limit_sleep` to a no-op (`conftest.py:8–10`), which would bypass the emit — fix that first.

- [ ] **Step 1: Fix the harness.** Change the autouse fixture to patch **`time.sleep`** (global no-op) instead of stubbing `ndif_citations.process.rate_limit_sleep`, so `rate_limit_sleep` (and its emit) still runs while tests stay fast. Run `pytest tests/ -q` and fix any test that relied on the old stub.
- [ ] **Step 2: Write failing test**
```python
def test_rate_limit_sleep_emits_wait():
    from ndif_citations import events, utils
    got = []; events.set_sink(got.append)
    utils.rate_limit_sleep(0.0, "LLM classify")   # time.sleep no-op'd by conftest
    events.clear_sink()
    waits = [e for e in got if e.type == "rate_limit_wait"]
    assert waits and waits[0].data["label"] == "LLM classify"
```
- [ ] **Step 3:** Run → FAIL.
- [ ] **Step 4:** In `rate_limit_sleep(seconds, label="")`: add `from ndif_citations import events` at top of utils.py, and **before** sleeping call `events.emit("rate_limit_wait", label=label, seconds=seconds)`. Keep `logger.debug`. (A per-second `remaining` tick loop is optional polish, deferred.)
- [ ] **Step 5:** Run → PASS; `pytest tests/ -q` → all green.
- [ ] **Step 6: Commit** `git commit -am "feat: emit LLM-cooldown rate_limit_wait events"`

### Task 1.4 — `settings_store.py` + `config.reload_settings()`

**Files:** Create `src/ndif_citations/settings_store.py`; Modify `config.py`; Test `tests/test_settings_store.py`.

- [ ] **Step 1: Write failing test**
```python
# tests/test_settings_store.py
import json
from ndif_citations import settings_store, config
def test_load_defaults_when_missing(tmp_path):
    s = settings_store.load(tmp_path / "settings.json")
    assert s["min_paper_year"] == 2024 and "ndif-team/nnsight" in s["excluded_github_repos"]
def test_save_and_reload_applies_to_config(tmp_path, monkeypatch):
    f = tmp_path / "settings.json"
    settings_store.save(f, {"min_paper_year": 2025, "llm_rate_limit_sleep": 2.0})
    monkeypatch.setattr(config, "_SETTINGS_FILE", f, raising=False)
    config.reload_settings()
    assert config.MIN_PAPER_YEAR == 2025 and config.LLM_RATE_LIMIT_SLEEP == 2.0
def test_validate_rejects_bad_types(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        settings_store.save(tmp_path / "s.json", {"min_paper_year": "two thousand"})
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `settings_store.load(path)->dict` (deep-merge over `DEFAULTS`), `save(path, partial)` (validate types against `DEFAULTS`, merge, write), `DEFAULTS` = C3 schema. In `config.py`: add `_SETTINGS_FILE = _PROJECT_ROOT / "settings.json"`, call private `_apply_settings(settings_store.load(_SETTINGS_FILE))` at import to override the listed constants, and define `def reload_settings(): _apply_settings(...)` **then re-read the secret env vars into module attributes** — `load_dotenv(override=True)` and `LLM_API_KEY = os.environ.get("LLM_API_KEY") or None` (same for `S2_API_KEY`, `GITHUB_TOKEN`, `SERPAPI_API_KEY`, `OPENALEX_EMAIL`). **Verified:** `os.environ` changes alone don't rebind these module attributes that call sites read as `config.X`.
- [ ] **Step 4:** Run `pytest tests/test_settings_store.py tests/test_known_venues_schema.py -v` → PASS.
- [ ] **Step 5: Commit** `git commit -am "feat: runtime-editable settings.json + config.reload_settings"`

### Task 1.5 — `config.reload_venues()` + verify knob access

> **Per review:** runtime knobs are already read as `config.X`; the only knob `from config import` is a *function-local* one in `utils.py` (re-reads each call → reload-safe). So the real reload gap is **`KNOWN_VENUES`** — a derived dict built at import (`config.py:170–199`), not a settings.json key.

**Files:** Modify `config.py`; Test `tests/test_known_venues_schema.py` (append).

- [ ] **Step 1:** `grep -rn "from ndif_citations.config import\|from .config import\|from .config import" src/ndif_citations` — confirm no *module-level* `from config import <C3 knob>` remains. If one is found, convert it to `config.X`.
- [ ] **Step 2: Write failing test**
```python
def test_reload_venues_picks_up_new_entry(tmp_path, monkeypatch):
    import json
    from ndif_citations import config
    f = tmp_path / "known_venues.json"
    f.write_text(json.dumps({"venues": {"ZZZConf": {"type": "conference", "aliases": ["Zeta Z Conf"]}}}))
    monkeypatch.setattr(config, "_VENUES_FILE", f, raising=False)
    config.reload_venues()
    assert "ZZZConf" in config.KNOWN_VENUES["conferences"]
    assert config.KNOWN_VENUES["acronym_map"].get("Zeta Z Conf") == "ZZZConf"
```
- [ ] **Step 3:** Run → FAIL.
- [ ] **Step 4:** Extract the `config.py:170–199` derivation into `def reload_venues(): ...` that rebuilds `KNOWN_VENUES` from `_VENUES_FILE`; call it once at import (no settings.json involvement — venues are file-backed).
- [ ] **Step 5:** Run → PASS; `pytest tests/test_known_venues_schema.py -v` → green.
- [ ] **Step 6: Commit** `git commit -am "feat: config.reload_venues for live venue edits"`

### Task 1.6 — Extract `orchestrator.py` stages (behavior-preserving)

**Files:** Create `src/ndif_citations/orchestrator.py`; Test `tests/test_orchestrator.py`.

> Extract the body of `cli.run()` (`cli.py:42–266`) verbatim in behavior into the C2 stage functions. Each stage wraps the **same** function calls in the same order, replaces `console.print(...)` with `events.emit(...)`, and returns the dataclasses in C2. No logic changes.

- [ ] **Step 1: Write failing test (event sequence on fakes)**
```python
# tests/test_orchestrator.py
from ndif_citations import orchestrator, events
def test_discover_stage_emits_and_returns(monkeypatch, tmp_path):
    from ndif_citations.models import DiscoveredPaper
    monkeypatch.setattr(orchestrator, "discover_s2_citations", lambda raw: [DiscoveredPaper(title="A", arxiv_id="1")])
    monkeypatch.setattr(orchestrator, "discover_openalex", lambda raw: [])
    monkeypatch.setattr(orchestrator, "discover_scholar", lambda raw, force_refresh=False: [])
    monkeypatch.setattr(orchestrator, "discover_github_dependents", lambda raw: [])
    got = []; events.set_sink(got.append)
    res = orchestrator.discover_stage(tmp_path, skip_papers=False, skip_github=True, fresh=False)
    events.clear_sink()
    assert len(res.papers) == 1
    assert any(e.type == "stage_start" and e.stage == "discover" for e in got)
    assert any(e.type == "source_count" for e in got)
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement the stages. Import the same functions `cli.run()` imports (lazy imports promoted to module top in orchestrator). Map each `console.print` to an `emit` (`source_count`, `dedup`, `route_summary`, `merge_result`, `report`). Mirror the `--skip-*`/`fresh` branching and the cross-repo ordering (enrich does `_unlink_shared_template_papers`/`_tag_repo_type`/`link_repos_to_papers`). Add `run_pipeline(out, *, mode, skip_papers, skip_github)` calling the five stages in sequence (no gate).
- [ ] **Step 4:** Run `pytest tests/test_orchestrator.py -v` → PASS.
- [ ] **Step 5: Commit** `git commit -am "feat: extract event-emitting pipeline orchestrator"`

### Task 1.7 — Per-item events in `process_papers`/`process_repos`

**Files:** Modify `src/ndif_citations/process.py` (loops at ~789, ~955); Test `tests/test_process_fallbacks.py` (append).

- [ ] **Step 1: Write failing test**
```python
def test_process_papers_emits_item_events(monkeypatch, tmp_path):
    from ndif_citations import process, events, router
    from ndif_citations.models import DiscoveredPaper
    monkeypatch.setattr(process, "generate_summary", lambda p: "s")
    monkeypatch.setattr(process, "classify_category", lambda p, o, pdf_path=None: (p.category, 0.85, p.category_confidence_band))
    monkeypatch.setattr(process, "get_cached_pdf", lambda p, o: None)
    monkeypatch.setattr(process, "extract_thumbnail", lambda *a, **k: None)
    paper = DiscoveredPaper(title="T", arxiv_id="9", abstract="a")
    decisions = router.route_papers([paper], [])
    got = []; events.set_sink(got.append)
    process.process_papers(decisions, tmp_path); events.clear_sink()
    assert any(e.type == "item_start" for e in got)
    assert any(e.type == "item_step" and e.data.get("step") == "summary" for e in got)
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Add an optional `cancel_check: Callable[[], bool] | None = None` param to `process_papers`/`process_repos`. At the **top of each item loop**: if `cancel_check and cancel_check()`, `raise RunCancelled()` (new sentinel in `events.py`) — *after* already-completed items are in `results`, so the caller can merge the completed prefix (the loop appends as it goes). Emit `item_start {idx,total,title,bucket}` per decision and `item_step {step}` after each of summary/classify/thumbnail/affiliations; in `process_repos` emit `item_start`/`item_step {step:"classify"}`. `cancel_check` defaults to `None` (CLI/tests unaffected); `events.emit` is a no-op without a sink.
- [ ] **Step 4:** Run `pytest tests/test_process_fallbacks.py -v` and `pytest tests/ -q` → green.
- [ ] **Step 5: Commit** `git commit -am "feat: per-item progress events in processing"`

### Task 1.8 — Re-point `cli.run` at the orchestrator (preserve CLI UX)

**Files:** Modify `src/ndif_citations/cli.py` (`run`, lines 42–266); Test `tests/test_smoke.py` (append parity test).

- [ ] **Step 1: Write failing parity test**
```python
def test_cli_run_parity_with_orchestrator(monkeypatch, fixture_state):
    """run_pipeline produces the same merged verified set as the legacy flow on a fixture."""
    from ndif_citations import orchestrator
    from tests.helpers.fakes import install_pipeline_fakes   # built in Task 1.0
    install_pipeline_fakes(monkeypatch, orchestrator)
    out = fixture_state                                       # tmp output dir w/ mini-*-full.json
    res = orchestrator.run_pipeline(out, mode="incremental", skip_github=True)
    titles = sorted(p.title for p in res.merged_papers if p.bucket.value == "verified")
    assert titles == sorted([...])   # snapshot from the fixture + the 1 new fake paper
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Replace the body of `cli.run()` with: build an event→console renderer (`def render(ev): console.print(...)` mapping event types back to today's Rich lines), `events.set_sink(render)`, call `orchestrator.run_pipeline(out, mode="fresh" if fresh else "incremental", skip_papers=..., skip_github=...)`, `events.clear_sink()`. Keep the `--skip-github`+`--skip-papers` mutual-exclusion guard. (CLI `run` stays ungated — the gate is web-only in Phase 3.)
- [ ] **Step 4:** Run `pytest tests/ -q` → all green incl. parity.
- [ ] **Step 5: Commit** `git commit -am "refactor: cli.run drives orchestrator via event renderer"`

### Task 1.9 — Fix the broken `add` command

**Files:** Modify `src/ndif_citations/cli.py` (`add`, ~line 422); Test `tests/test_cli_edit.py` or new `tests/test_cli_add.py`.

- [ ] **Step 1: Write failing test**
```python
def test_add_routes_before_processing(monkeypatch, tmp_path):
    from ndif_citations import cli, process, router
    from click.testing import CliRunner
    # stub enrich/process so no network; assert process_papers receives RoutingDecisions
    seen = {}
    real = process.process_papers
    monkeypatch.setattr(process, "process_papers", lambda dec, out: seen.setdefault("decisions", dec) or [d.paper for d in dec])
    ...  # invoke `add <arxiv-url>`; assert seen["decisions"][0].__class__.__name__ == "RoutingDecision"
```
- [ ] **Step 2:** Run → FAIL (current code passes `list[DiscoveredPaper]`).
- [ ] **Step 3:** In `add()`, after `enrich_papers`: `existing = load_existing_papers(out)`; `decisions = route_papers(papers, existing)`; `processed = process_papers(decisions, out)`; `merged, run_stats = merge_papers(existing, processed)`; `write_outputs(merged, out, run_stats)`. Update the report prints to read from `processed[0]`.
- [ ] **Step 4:** Run → PASS; `pytest tests/ -q` → green.
- [ ] **Step 5: Commit** `git commit -am "fix: add command routes papers before processing"`

**Phase 1 exit:** `pytest tests/ -q` fully green (**631** existing + new), parity test passes, `python -m ndif_citations run --skip-github`/`discover`/`add <url>` all work with unchanged CLI output.

---

## PHASE 2 — JobRunner + Run API + SSE (task+contract)

**Goal:** start a (fresh/ungated) run over HTTP, stream live events, cancel, view history; ship `serve`.

**Files:** Create `jobs.py`, `server/app.py`, `server/deps.py`, `server/routers/runs.py`; Modify `cli.py` (`serve`). Tests: `tests/test_jobs.py`, `tests/test_api_runs.py`.

**Tasks:**
1. `jobs.JobRunner` (C4): `start` spawns a daemon thread running `orchestrator.run_pipeline` with `events.set_sink(self._record)`; `_record` appends to a per-run ring buffer + a `queue.Queue` for live subscribers; `RunActiveError` if `active`. Persist final `RunStatus` + event log to `output/runs/<run_id>.json`.
2. Cancel: the runner owns a `threading.Event`; it passes `cancel_check=lambda: ev.is_set()` into `process_stage` (→ `process_papers`/`process_repos`, the Task 1.7 param). On set, the process loop raises `RunCancelled` **after** completed items are in `results`. The runner catches it and calls `finalize_stage(..., completed=(n_papers, n_repos))` so merge consumes only `decisions[:n]` (the completed prefix) — **never the half-formed unprocessed decisions** (they'd carry empty summary/`UNCLASSIFIED` and pollute state). State → `CANCELLED`; staging retains the remainder.
   - **Carried forward from the Phase 1 final review (verified gaps):** (a) `orchestrator.run_pipeline` accepts `process_stage`'s `cancel_check` only indirectly — it does **not** currently take/forward a `cancel_check` param. Add a small additive change so `run_pipeline(out, *, mode, skip_papers, skip_github, cancel_check=None)` forwards it to `process_stage`; otherwise the JobRunner must drive the five stages individually (also fine for the gate flow — Phase 3 needs per-stage control anyway). (b) `RunCancelled` is raised but **caught nowhere** in Phase 1 (by design — the CLI never cancels); the JobRunner owns the `try/except RunCancelled` and the cancel→`finalize_stage(completed=...)` path (the `completed` slice param already exists). (c) **Thread-locality contract:** the event sink is `threading.local()`, so the runner must call `events.set_sink(...)` **inside the worker thread** that runs the pipeline, not in the request handler.
3. `server/app.py`: FastAPI, mount `web/dist` at `/` (StaticFiles, html=True), routers under `/api`, bind `127.0.0.1`.
4. `runs.py`: `POST /runs`, `GET /runs/{id}/events` (SSE via `StreamingResponse`, `text/event-stream`, replays buffer then drains queue), `POST /runs/{id}/cancel`, `GET /runs/{id}`, `GET /runs`.
5. `cli.serve(--port 8723 --no-open)`: launch uvicorn programmatically, open browser unless `--no-open`.

**Exit:** `pytest tests/test_jobs.py tests/test_api_runs.py` green; manual: `ndif-citations serve` → `POST /api/runs {mode:fresh, skip_papers:true}` streams `stage_*`/`item_*`/`rate_limit_wait`/`done`; cancel mid-run finalizes completed; CLI unaffected.

---

## PHASE 3 — Gate-before-LLM (task+contract)

**Goal:** incremental runs pause for curator selection before LLM spend; resilient to restart.

**Files:** Modify `jobs.py` (staged execution + staging), Create `server/services/gate_svc.py`, Modify `server/routers/runs.py`. Tests: `tests/test_gate.py`, `tests/test_staging.py`.

**Tasks:**
1. JobRunner incremental path: run `discover_stage`→`enrich_stage`→`route_stage`, then **serialize** `RouteResult` (+ discovered repos, run_stats) to `output/staging/<run_id>.json`, emit `awaiting_review {candidates}` (NEW/REPROCESS papers as `PaperRow`s + new repos), set state `AWAITING_REVIEW`, and **block** the worker on a `threading.Event` until `submit_gate`.
2. `submit_gate(process_ids, discard_ids, edits)`: apply `edits` to the matching decisions' papers (reuse edit logic), drop `discard_ids` (and write them to discarded bucket via merge), filter `paper_decisions` to `process_ids` (others → carried over unchanged), unblock → `process_stage`→`finalize_stage`.
3. Resume: on `serve` startup, if `output/staging/*.json` exists, expose it via `GET /runs/active` so the UI can resume the gate (rehydrate `RouteResult`).
4. Repos: auto-flow through process (cheap), but include new repos in the `awaiting_review` payload for light review (select/exclude).

**Exit:** incremental run halts at `AWAITING_REVIEW`; selecting 2 of 5 candidates runs LLM on exactly 2; killing+restarting `serve` mid-gate rehydrates from staging; `pytest tests/test_gate.py tests/test_staging.py` green.

---

## PHASE 4 — Data + curation API (task+contract)

**Goal:** browse + full curation over HTTP, incl. force-reprocess; mutations blocked during runs.

**Files:** Create `server/routers/{papers,repos,images,stats}.py`, `server/services/{papers_svc,repos_svc,reprocess_svc}.py`. Tests: `tests/test_api_papers.py`, `tests/test_api_repos.py`, `tests/test_reprocess.py`.

**Tasks:**
1. `papers_svc`: `list_rows(bucket,q,sort)`, `get(id)`, `edit(id, fields)` (reuse `edit_schema` parsers → set `manual_override`, re-derive `has_*`, re-run `_decide_bucket`, `write_outputs`), `set_bucket(id, bucket, reason, detail)` (promote/demote/discard logic from cli), `add_by_url(url)` (fixed path).
2. `reprocess_svc.force(ids, fields)` — **corrected recipe (spec §3.3)**: load papers; for each, clear the requested field(s) **and** their `has_*` flags, set `manual_override=False` **on the working copy** (so the protective hydration block is skipped), and construct the `RoutingDecision` **directly** with `existing_paper=paper` and `processing_needed={f: f in fields ...}` — do **not** use `route_papers` (it would attach the un-cleared on-disk existing and re-trigger hydration). Always run through the **JobRunner** (never synchronously on the request thread) so SSE shows progress, the run-lock applies, and the Surya global is never touched concurrently. After processing: restore `manual_override=True`, re-derive `has_*`, re-run `_decide_bucket`, merge, write. Test asserts the field value **actually changes** (mock LLM returns a new value).
3. Image: `POST /papers/{id}/image` (validate PNG, save `output/images/{slug}.png`, set `image`+`has_thumbnail`), `POST /papers/{id}/reextract-thumbnail` (call `extract_thumbnail` on cached PDF via JobRunner).
4. `repos_svc`: `list_rows`, `get`, `edit` (repo_type/linked_paper_url/description + `manual_override`), `exclude` (append to settings `excluded_github_repos` + drop from `github-repos-full.json`).
4b. **Guard `_tag_repo_type` (verified edit-loss):** `_tag_repo_type` runs unconditionally each run (`cli.py:159`) and would overwrite a curator's `repo_type`. In `orchestrator.enrich_stage`, **skip re-tagging and the linked-paper clear for `manual_override` repos**. TDD: test that a `manual_override` repo with curator `repo_type="research"` keeps it after an enrich pass that would otherwise tag it `course`.
5. `deps.require_no_active_run` → 409 on all mutating routes when `JobRunner.active` (also blocks the `exclude`/edit endpoints from racing a partial-scrape merge).
6. `stats`: dashboard counts; `images`: serve `output/images/*`.

**Exit:** edit→bucket auto-derive verified by test; force-reprocess clears+reroutes+fills only target field (test); 409 during active run (test); `pytest tests/test_api_*.py tests/test_reprocess.py` green.

---

## PHASE 5 — Publish + settings + venues (task+contract)

**Goal:** one-click publish to the live site with a safety diff; edit knobs + venues from UI.

**Files:** Create `publish.py`, `server/routers/{publish,settings,venues}.py`. Tests: `tests/test_publish.py`, `tests/test_api_settings.py`.

**Tasks:**
1. `publish.detect_target()`: search siblings for `ndif-web-beta/packages/ndif.us` with `public/data/` + `public/images/`; validate; **refuse** `ndif-website/` (the ignored old repo) **and** `out/data/` (build output). Persist chosen path to settings `publish_target`.
2. `publish.diff(out, target)`: compare `output/research-papers.json`/`github-repos.json` (+ image filenames) against the target's `public/data/` — return added/changed/removed papers+repos and new/changed images.
3. `publish.apply(out, target)` — **wrap the existing scripts, don't reinvent (verified):** `ndif-web-beta/packages/ndif.us/scripts/sync-research-papers.mjs` + `sync-github-repos.mjs` already do the copy. Invoke them (`node`/`bun`) from `publish.apply` after backing up the target's current `public/data/*.json` to `output/backups/`. **Fix the additive-only image copy:** those scripts skip images that already exist, so a re-extracted thumbnail (same slug filename) won't update — `publish.apply` must **force-overwrite** changed images (compare bytes/mtime). The site `import`s `public/data/*.json` at **build time**, so `apply` returns a "run `bun run build` to see changes on the site" instruction (or a one-click build+verify). `POST /publish {dry_run}` returns the diff (dry) or applies. (If the scripts' logic must change, edit them in-repo rather than forking into Python.)
4. `settings`: `GET/PUT /settings` (via `settings_store` + `config.reload_settings`); `venues`: `GET/PUT /venues` (edit `known_venues.json`, validate against the existing schema test, then call `config.reload_venues()`).

**Exit:** dry-run diff lists exact changes; apply copies files + backs up; bad target rejected; settings PUT visible to next run; `pytest tests/test_publish.py tests/test_api_settings.py tests/test_known_venues_schema.py` green.

---

## PHASE 6 — Frontend (task+contract)

**Goal:** the "good face" — all flows, light+dark, data-dense, per the spec §6 design system.

**Files:** `web/` Vite project — `src/{routes,components,lib,api,hooks}`, `tailwind.config`, shadcn setup, design tokens.

**Tasks (each ships a usable screen against the live API):**
1. Scaffold Vite+React+TS+Tailwind+shadcn; theme tokens (blue/amber + semantic bucket/band/run-state colors), Fira Sans + mono, dark mode toggle; TanStack Query client; React Router; `EventSource` SSE hook; API client typed to C5 shapes.
2. **Dashboard** — KPI cards + category/type charts (recharts) + Run/Publish entry.
3. **Papers browser** — TanStack Table, faceted filters/search/sort, detail `Sheet`, inline edit forms, multi-select toolbar (bucket + reprocess via `DropdownMenu`, `AlertDialog` confirms, `Toast`+Undo).
4. **Repos browser** — same pattern; repo_type/linked-paper edit; exclude.
5. **Run console** — mode+skip selector, phase `Stepper`, SSE log feed (mono, auto-scroll), prominent rate-limit countdown, per-item `x/N`, cancel.
6. **Review gate** — candidate table, per-row Process/Discard/Edit-then-process, "Process N selected" with LLM-time estimate.
7. **Settings** — knob forms + LLM model/endpoint + publish target picker + venue manager table.

**Exit:** every backend capability reachable from the UI; light+dark verified; reduced-motion respected.

---

## PHASE 7 — Polish, tests, docs (task+contract)

**Tasks:**
1. Playwright smoke (via the available Playwright MCP / gstack `/browse`) for: browse→edit→save; start fresh run→see SSE→cancel; incremental→gate select→process; publish dry-run. Run against a fixture `--output-dir`.
2. A11y/dark pass (contrast, focus, `aria-live` on log/toasts), perf (virtualize ≥50 rows, skeletons).
3. Docs: update `docs/MAINTAINER_GUIDE.md` (new web workflow + `serve`), `README.md`, and `docs/CODEBASE_CONTEXT.md` (add web layer; fix the stale "no tests" line); `settings.json.example` documented.
4. Version bump + changelog.

**Exit:** smoke green; docs match shipped behavior.

---

## Risks & mitigations
- **Refactor regression (Phase 1):** mitigated by the parity test + keeping all **631** tests green; orchestrator is extract-not-rewrite. Parity needs a real test harness → built first in Task 1.0.
- **Force-reprocess silently no-ops (CRITICAL, found in review):** the `manual_override` hydration block restores cleared fields. Mitigated by the corrected recipe (clear field+flag, `manual_override=False` on the copy, direct `RoutingDecision`) **and a test that asserts the value actually changes**, not just that routing produced FILL_GAPS.
- **Publish divergence (CRITICAL, found in review):** existing `sync-*.mjs` already publish + the site imports JSON at build time + image sync is additive-only. Mitigated by wrapping the scripts, force-overwriting changed images, and prompting `bun run build`.
- **Thread-safety (Surya global, Rich console):** one-run-at-a-time lock + orchestrator emits (never prints) + mutations 409 during runs + **all** Surya/thumbnail work (incl. reprocess) routed through the JobRunner.
- **Config live-reload not seen:** knobs already read `config.X`; `reload_settings()` **re-reads secret env vars into module attributes** (not just `os.environ`); `reload_venues()` rebuilds the derived `KNOWN_VENUES`; rate limits also param-overridable.
- **Cancel consistency:** `cancel_check` raises after completed items are in `results`; finalize merges only the completed prefix (`decisions[:n]`); never the half-formed unprocessed decisions; staging keeps the remainder.
- **Repo edit-loss + staleness:** `_tag_repo_type` guarded to skip `manual_override` repos; papers-only runs don't touch repos; aged-out repos surfaced; edit/exclude blocked (409) during a run.

## Self-review notes
- **Spec coverage:** every spec section maps to a phase (browse→P4/P6, edit→P4/P6, reprocess→P4, runs/gate→P2/P3, publish→P5, settings/venues→P5, persistence→unchanged, docs→P7). ✓
- **Contract consistency:** `ProgressEvent`, stage signatures, settings keys, `JobRunner` methods, and row shapes are defined once in §Contracts and referenced by id throughout. ✓
- **Placeholders:** Phase 1 steps carry real code; Phases 2–7 are intentionally task+contract granularity (expanded to bite-sized at execution per the master-plan note). The test harness the parity/add tests need is now built explicitly in **Task 1.0** (not a `...` placeholder).
- **Critique applied (2026-06-05 adversarial review):** fixed the force-reprocess hydration trap (§3.3 / Phase 4.2), reconciled publish with the existing `sync-*.mjs` (Phase 5.3), re-baselined the test count 37→631 + added the harness task (1.0), corrected the orchestrator contract to the in-place-mutation reality (C2), reframed the LLM-cooldown event + fixed the conftest stub (1.3), narrowed config-reload to the real gaps — secrets + `KNOWN_VENUES` (1.4/1.5), specified cancel via completed-prefix merge (Phase 2.2 / 1.7), and added the `_tag_repo_type` manual-override guard (Phase 4.4b).
