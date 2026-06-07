# GitHub Pipeline Revamp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten the GitHub side of the `ndif-citations` pipeline so the website JSON ships exactly 10 useful fields, the persistence layer tracks `first_seen`/`last_seen`, dead code is removed, and curator overrides are honored — implementing the design in `docs/superpowers/specs/2026-05-20-github-pipeline-revamp-design.md` (rev 2).

**Architecture:** All work lives in `ndif-citations/src/ndif_citations/`. Slim `to_website_dict()` is fixed to emit the 10-field contract; `merge_repos()` becomes the single stamping site for `first_seen`/`last_seen`; `_tag_repo_type()` early-returns when `manual_override=True`; `COURSE_NAME_PATTERNS` is broadened. CSV write functions are deleted as confirmed dead code. The 3-way `repo_type` stays in the model and full JSON; the site reads only `is_course: boolean` (a derived property). XLSX gains a `first_seen` column.

**Tech Stack:** Python 3.10+, Pydantic v2, pytest, openpyxl. No new dependencies.

---

## File Map

**Modified files:**
- `src/ndif_citations/models.py` — `DiscoveredRepo`: add `first_seen` / `last_seen` fields, `is_course` property, rewrite `to_website_dict()`
- `src/ndif_citations/output.py` — `merge_repos()` stamping + hybrid staleness; delete orphaned `_write_csv` / `_write_repos_csv` + `import csv`; XLSX columns updated
- `src/ndif_citations/discover.py` — `_tag_repo_type()` early-returns on `manual_override`
- `src/ndif_citations/cli.py` — guard the `linked_paper_url` clear loop against `manual_override`
- `src/ndif_citations/config.py` — extend `COURSE_NAME_PATTERNS`

**New files:**
- `tests/test_output_schema.py` — 4 tests for the new schema and behavioral contracts

---

## Task 1: Remove orphaned CSV write code

**Files:**
- Modify: `src/ndif_citations/output.py` (delete `_write_csv` ~474-524, `_write_repos_csv` ~430-471, `import csv` line 5)

This is pure dead-code removal. `grep` confirms zero callers across `src/` and `tests/`.

- [ ] **Step 1: Confirm no callers (sanity check)**

Run: `cd ndif-citations && grep -rn '_write_csv\|_write_repos_csv' src/ tests/`
Expected: only the definitions themselves appear — no callers.

- [ ] **Step 2: Delete `_write_repos_csv` and `_write_csv` functions**

Remove the entire function bodies in `src/ndif_citations/output.py`:
- `_write_repos_csv(repos: list[DiscoveredRepo], csv_path: Path) -> None:` and its body
- `_write_csv(papers: list[DiscoveredPaper], csv_path: Path) -> None:` and its body

- [ ] **Step 3: Remove `import csv`**

Edit `src/ndif_citations/output.py` line 5: delete the `import csv` line.

- [ ] **Step 4: Run the existing test suite**

Run: `cd ndif-citations && pytest -x -q`
Expected: all existing tests pass (none referenced the removed functions).

- [ ] **Step 5: Commit**

```bash
cd ndif-citations
git add src/ndif_citations/output.py
git commit -m "refactor: remove orphaned CSV write functions

_write_csv and _write_repos_csv had zero callers across src/ and tests/.
XLSX is the supported audit/grant-reporting surface; grant reporters
export CSV from there if needed."
```

---

## Task 2: Add `first_seen` / `last_seen` fields to `DiscoveredRepo`

**Files:**
- Modify: `src/ndif_citations/models.py` (~272 — `DiscoveredRepo` field block)
- Test: `tests/test_output_schema.py` (new file)

These fields are persistence-only; the slim JSON does not expose them. Test that `to_full_dict()` round-trips them.

- [ ] **Step 1: Create `tests/test_output_schema.py` with a failing test**

```python
"""Schema and behavioral tests for the GitHub pipeline revamp (2026-05-20)."""
from __future__ import annotations

from datetime import date
from ndif_citations.models import DiscoveredRepo


def test_discoveredrepo_persists_first_and_last_seen():
    r = DiscoveredRepo(
        owner="o", repo="r", url="https://github.com/o/r",
        first_seen="2025-01-15", last_seen="2026-05-20",
    )
    dumped = r.to_full_dict()
    assert dumped["first_seen"] == "2025-01-15"
    assert dumped["last_seen"] == "2026-05-20"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd ndif-citations && pytest tests/test_output_schema.py::test_discoveredrepo_persists_first_and_last_seen -v`
Expected: FAIL with `ValidationError` (fields don't exist yet) or KeyError on dumped dict.

- [ ] **Step 3: Add the fields to `DiscoveredRepo`**

In `src/ndif_citations/models.py`, inside the `DiscoveredRepo` class, after the `processing_bucket: str = ""` line, add:

```python
    # Lifecycle tracking
    first_seen: Optional[str] = None  # ISO YYYY-MM-DD — first run that observed this repo
    last_seen: Optional[str] = None   # ISO YYYY-MM-DD — most recent run that observed it
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd ndif-citations && pytest tests/test_output_schema.py::test_discoveredrepo_persists_first_and_last_seen -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ndif-citations
git add src/ndif_citations/models.py tests/test_output_schema.py
git commit -m "feat: add first_seen/last_seen to DiscoveredRepo

Persistence-only fields for trend tracking and the upcoming
hybrid staleness policy. Not exposed in the slim website JSON."
```

---

## Task 3: Add `is_course` property and rewrite slim `to_website_dict()` to the 12-field contract

**Files:**
- Modify: `src/ndif_citations/models.py` (~298 — `DiscoveredRepo.to_website_dict`)
- Modify: `tests/test_output_schema.py` (add test)

The site's `lib/repos.ts` only reads `repo_type === "course"`. The slim JSON should expose only `is_course: boolean` (a derived property) plus the 9 other display fields.

- [ ] **Step 1: Add a failing test for the exact 12-key contract**

Append to `tests/test_output_schema.py`:

```python
SLIM_KEYS = {
    "owner", "repo", "url", "description", "stars", "forks",
    "last_commit", "language", "linked_paper_url",
    "is_course", "is_fork", "parent_full_name",
}


def test_to_website_dict_emits_exactly_the_twelve_slim_keys():
    r = DiscoveredRepo(
        owner="o", repo="r", url="https://github.com/o/r",
        description="hello", stars=10, forks=2,
        last_commit=date(2026, 1, 1), language="Python",
        linked_paper_url="https://arxiv.org/abs/1234.5678",
        repo_type="course",
        is_fork=True, parent_full_name="upstream/parent",
    )
    slim = r.to_website_dict()
    assert set(slim.keys()) == SLIM_KEYS
    assert slim["is_course"] is True
    assert slim["is_fork"] is True
    assert slim["parent_full_name"] == "upstream/parent"
    assert slim["last_commit"] == "2026-01-01"


def test_to_website_dict_is_course_false_for_research():
    r = DiscoveredRepo(owner="o", repo="r", url="https://github.com/o/r", repo_type="research")
    assert r.to_website_dict()["is_course"] is False


def test_to_website_dict_is_course_false_for_experiment():
    r = DiscoveredRepo(owner="o", repo="r", url="https://github.com/o/r", repo_type="experiment")
    assert r.to_website_dict()["is_course"] is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ndif-citations && pytest tests/test_output_schema.py -v -k 'website_dict'`
Expected: FAIL — the current `to_website_dict()` emits 16 keys (archived, category, etc.) and no `is_course`.

- [ ] **Step 3: Add the `is_course` derived property**

In `src/ndif_citations/models.py`, inside `DiscoveredRepo`, after `merge_key()` and before `compute_content_hash()`, add:

```python
    @property
    def is_course(self) -> bool:
        """True iff repo_type is 'course'. The slim JSON exposes only this boolean."""
        return self.repo_type == "course"
```

- [ ] **Step 4: Rewrite `to_website_dict()` to the 10-field slim contract**

In `src/ndif_citations/models.py`, replace the body of `to_website_dict(self) -> dict:` with:

```python
    def to_website_dict(self) -> dict:
        """Export slim dict for github-repos.json — exactly 12 fields, no internal state."""
        return {
            "owner": self.owner,
            "repo": self.repo,
            "url": self.url,
            "description": self.description,
            "stars": self.stars,
            "forks": self.forks,
            "last_commit": self.last_commit.isoformat() if self.last_commit else None,
            "language": self.language,
            "linked_paper_url": self.linked_paper_url,
            "is_course": self.is_course,
            "is_fork": self.is_fork,
            "parent_full_name": self.parent_full_name,
        }
```

- [ ] **Step 5: Run the website_dict tests to verify they pass**

Run: `cd ndif-citations && pytest tests/test_output_schema.py -v -k 'website_dict'`
Expected: all 3 tests PASS.

- [ ] **Step 6: Run the full suite to catch regressions**

Run: `cd ndif-citations && pytest -x -q`
Expected: all tests pass. (Any test that depended on the old slim shape would break here — none are expected, but verify.)

- [ ] **Step 7: Commit**

```bash
cd ndif-citations
git add src/ndif_citations/models.py tests/test_output_schema.py
git commit -m "feat: slim github-repos.json to 12 fields, expose is_course

to_website_dict() now emits only the fields the site actually reads:
owner, repo, url, description, stars, forks, last_commit, language,
linked_paper_url, is_course, is_fork, parent_full_name. repo_type
stays in the full JSON and XLSX; is_course is a derived property so
models.py is the source of truth and the site stops branching on a
3-way string union. is_fork + parent_full_name let the site show
'Fork of X' on the 9 GitHub-detectable forks."
```

---

## Task 4: Stamp `first_seen` / `last_seen` in `merge_repos()` with an injectable `today()`

**Files:**
- Modify: `src/ndif_citations/output.py` (~191 — `merge_repos`)
- Modify: `tests/test_output_schema.py` (add test)

`merge_repos()` is the single point where new-vs-existing is decided. Stamp here so `discover.py` and `process.py` never touch these fields. An injectable `_today()` helper lets tests pin the date.

- [ ] **Step 1: Add a failing test for the stamping behavior**

Append to `tests/test_output_schema.py`:

```python
from ndif_citations import output as output_module
from ndif_citations.output import merge_repos


def test_merge_repos_stamps_first_and_last_seen_on_new_repo(monkeypatch):
    monkeypatch.setattr(output_module, "_today", lambda: date(2026, 5, 20))
    discovered = [DiscoveredRepo(owner="o", repo="new", url="https://github.com/o/new")]
    merged = merge_repos(discovered=discovered, existing=[])
    assert len(merged) == 1
    assert merged[0].first_seen == "2026-05-20"
    assert merged[0].last_seen == "2026-05-20"


def test_merge_repos_preserves_first_seen_and_updates_last_seen_on_re_observed(monkeypatch):
    monkeypatch.setattr(output_module, "_today", lambda: date(2026, 5, 20))
    existing = [DiscoveredRepo(
        owner="o", repo="r", url="https://github.com/o/r",
        first_seen="2025-01-15", last_seen="2026-01-01",
    )]
    discovered = [DiscoveredRepo(owner="o", repo="r", url="https://github.com/o/r")]
    merged = merge_repos(discovered=discovered, existing=existing)
    target = next(m for m in merged if m.merge_key() == "o/r")
    assert target.first_seen == "2025-01-15"
    assert target.last_seen == "2026-05-20"


def test_merge_repos_backfills_first_seen_when_missing_on_existing(monkeypatch):
    monkeypatch.setattr(output_module, "_today", lambda: date(2026, 5, 20))
    existing = [DiscoveredRepo(owner="o", repo="r", url="https://github.com/o/r")]  # no first_seen
    discovered = [DiscoveredRepo(owner="o", repo="r", url="https://github.com/o/r")]
    merged = merge_repos(discovered=discovered, existing=existing)
    target = next(m for m in merged if m.merge_key() == "o/r")
    assert target.first_seen == "2026-05-20"
    assert target.last_seen == "2026-05-20"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ndif-citations && pytest tests/test_output_schema.py -v -k 'merge_repos and (stamp or preserve or backfill)'`
Expected: FAIL — `_today` attribute doesn't exist, `first_seen`/`last_seen` never get set.

- [ ] **Step 3: Add the `_today()` helper to `output.py`**

In `src/ndif_citations/output.py`, replace `from datetime import datetime` (line 8) with:

```python
from datetime import date, datetime


def _today() -> date:
    """Wrapped so tests can monkeypatch the current date deterministically."""
    return date.today()
```

- [ ] **Step 4: Stamp `first_seen` / `last_seen` inside `merge_repos()`**

In `src/ndif_citations/output.py`, replace the existing `merge_repos` body (the loop at ~226-243 that handles each discovered repo, plus the protected-existing branch at ~210-224) with:

```python
def merge_repos(
    discovered: list[DiscoveredRepo],
    existing: list[DiscoveredRepo],
) -> list[DiscoveredRepo]:
    """Merge discovered repos into existing state.

    - Stamps first_seen / last_seen on every repo that survives the merge:
        * NEW (in discovered, not in existing): first_seen = last_seen = today
        * Re-observed (in both): preserve first_seen, update last_seen to today
        * Protected-but-absent (existing + manual_override, missing from discovered):
          last_seen unchanged (we have no fresh observation), first_seen preserved
    - API-confirmed-dead repos (404/renamed/archived) are NOT in `discovered`
      — they were already dropped in enrich_repos_from_github_api. Those are
      removed here unless manual_override=True.
    - Scrape-absent repos (still alive on GitHub, just missing from this run's
      dependents page) are also NOT in `discovered`. They survive as protected
      stale entries via the soft age-out applied in Task 5.
    """
    today_str = _today().isoformat()
    by_key: dict[str, DiscoveredRepo] = {r.merge_key(): r for r in existing}
    discovered_keys: set[str] = {r.merge_key() for r in discovered}

    merged: list[DiscoveredRepo] = []

    # Keep existing repos that are NOT in this run's discovered set
    removed_count = 0
    for existing_repo in existing:
        key = existing_repo.merge_key()
        if key in discovered_keys:
            continue  # Will be handled in the discovered loop below
        if existing_repo.manual_override:
            logger.debug(f"Keeping protected repo not seen this run: {key}")
            merged.append(existing_repo)
        else:
            logger.info(f"Purging stale repo from state: {key}")
            removed_count += 1

    if removed_count:
        logger.info(f"Purged {removed_count} stale repo(s) from github-repos-full.json")

    # Merge each discovered repo, stamping first_seen / last_seen
    for repo in discovered:
        key = repo.merge_key()
        existing_repo = by_key.get(key)

        if existing_repo is None:
            # NEW — stamp both
            repo.first_seen = today_str
            repo.last_seen = today_str
            merged.append(repo)
        elif repo.processing_bucket in ("skip", "protected"):
            # Keep existing object; just bump last_seen
            existing_repo.first_seen = existing_repo.first_seen or today_str
            existing_repo.last_seen = today_str
            merged.append(existing_repo)
        else:
            # NEW / REPROCESS / FILL_GAPS — preserve manual_override + first_seen
            if existing_repo.manual_override:
                repo.manual_override = True
            repo.first_seen = existing_repo.first_seen or today_str
            repo.last_seen = today_str
            merged.append(repo)

    return merged
```

- [ ] **Step 5: Run the stamping tests to verify they pass**

Run: `cd ndif-citations && pytest tests/test_output_schema.py -v -k 'merge_repos and (stamp or preserve or backfill)'`
Expected: 3 PASS.

- [ ] **Step 6: Run the full suite**

Run: `cd ndif-citations && pytest -x -q`
Expected: all pass. (`tests/test_output_merge.py` will exercise `merge_repos` — verify those still pass.)

- [ ] **Step 7: Commit**

```bash
cd ndif-citations
git add src/ndif_citations/output.py tests/test_output_schema.py
git commit -m "feat: stamp first_seen/last_seen in merge_repos

Single source of truth — discover.py and process.py never touch these.
Re-observed repos preserve first_seen; existing rows missing the field
get backfilled to today. Introduces a _today() helper so tests can pin
the date with monkeypatch."
```

---

## Task 5: Hybrid staleness policy — 30-day age-out for scrape-absent repos

**Files:**
- Modify: `src/ndif_citations/output.py` (~191 — `merge_repos`)
- Modify: `tests/test_output_schema.py` (add test)

Today the merge purges any existing repo not in `discovered`. Spec §3.2 distinguishes:
- **API-confirmed dead** (404/rename/archived → already absent from `discovered` because `enrich_repos_from_github_api` dropped them): keep purging immediately.
- **Scrape-absent** (still alive on GitHub but missing from this run's dependents page): survive for 30 days via `last_seen`.

Since the merge can't distinguish these two cases from inside, we lean on `last_seen`: any existing repo absent from the current run gets carried over only if `last_seen` is within 30 days. The trade-off: an API-confirmed-dead repo will linger for 30 days. That's acceptable — the alternative (immediate hard purge) is exactly what we're moving away from, and these few stale entries don't affect the slim JSON (which only ships repos in the current run anyway — see Step 4 nuance below).

**Important nuance:** the slim `github-repos.json` is built from `merged`, which now contains stale-but-protected entries. We need to verify those *don't* leak into the slim JSON. Inspection of `_write_repos_outputs` shows it writes all of `merged` to the slim file. We add a filter: slim only includes repos where `last_seen == today()` (i.e., observed this run).

- [ ] **Step 1: Add a failing test for the 30-day age-out behavior**

Append to `tests/test_output_schema.py`:

```python
def test_merge_repos_keeps_scrape_absent_repo_within_30_days(monkeypatch):
    monkeypatch.setattr(output_module, "_today", lambda: date(2026, 5, 20))
    existing = [DiscoveredRepo(
        owner="o", repo="r", url="https://github.com/o/r",
        first_seen="2025-01-01", last_seen="2026-05-01",  # 19 days ago
    )]
    merged = merge_repos(discovered=[], existing=existing)
    assert any(m.merge_key() == "o/r" for m in merged), \
        "Scrape-absent repo within 30d should survive"


def test_merge_repos_ages_out_scrape_absent_repo_past_30_days(monkeypatch):
    monkeypatch.setattr(output_module, "_today", lambda: date(2026, 5, 20))
    existing = [DiscoveredRepo(
        owner="o", repo="r", url="https://github.com/o/r",
        first_seen="2025-01-01", last_seen="2026-04-01",  # 49 days ago
    )]
    merged = merge_repos(discovered=[], existing=existing)
    assert all(m.merge_key() != "o/r" for m in merged), \
        "Scrape-absent repo older than 30d should age out"


def test_merge_repos_manual_override_survives_age_out(monkeypatch):
    monkeypatch.setattr(output_module, "_today", lambda: date(2026, 5, 20))
    existing = [DiscoveredRepo(
        owner="o", repo="r", url="https://github.com/o/r",
        first_seen="2024-01-01", last_seen="2024-01-01",  # ancient
        manual_override=True,
    )]
    merged = merge_repos(discovered=[], existing=existing)
    assert any(m.merge_key() == "o/r" for m in merged), \
        "manual_override must always survive regardless of last_seen"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ndif-citations && pytest tests/test_output_schema.py -v -k 'age_out or within_30'`
Expected: FAIL — the current merge purges all non-manual_override absentees on the first run.

- [ ] **Step 3: Implement the soft age-out in `merge_repos`**

In `src/ndif_citations/output.py`, locate the "Keep existing repos that are NOT in this run's discovered set" loop (added in Task 4) and replace it with:

```python
    # Keep existing repos that are NOT in this run's discovered set
    # Hybrid policy: manual_override → always keep; otherwise → keep if last_seen
    # is within 30 days (scrape-absent grace window). API-confirmed-dead repos
    # (404/rename/archived) get dropped from `enrich_repos_from_github_api`
    # BEFORE they reach this function, so they tend to be old `last_seen`
    # values and naturally age out.
    AGE_OUT_DAYS = 30
    today_date = _today()
    removed_count = 0
    aged_out_count = 0
    for existing_repo in existing:
        key = existing_repo.merge_key()
        if key in discovered_keys:
            continue  # Will be handled in the discovered loop below
        if existing_repo.manual_override:
            logger.debug(f"Keeping protected repo not seen this run: {key}")
            merged.append(existing_repo)
            continue
        # Soft age-out
        if existing_repo.last_seen:
            try:
                last = date.fromisoformat(existing_repo.last_seen)
                days_since = (today_date - last).days
            except ValueError:
                days_since = AGE_OUT_DAYS + 1  # bad data → age out
        else:
            days_since = AGE_OUT_DAYS + 1  # no last_seen → age out
        if days_since <= AGE_OUT_DAYS:
            logger.debug(f"Keeping scrape-absent repo (within {AGE_OUT_DAYS}d): {key}")
            merged.append(existing_repo)
        else:
            logger.info(f"Aging out scrape-absent repo ({days_since}d since last seen): {key}")
            aged_out_count += 1
            removed_count += 1

    if removed_count:
        logger.info(
            f"Removed {removed_count} stale repo(s) from state "
            f"({aged_out_count} aged out past {AGE_OUT_DAYS}d)"
        )
```

No new import needed — Task 4 already added `from datetime import date, datetime`. The loop above uses `date.fromisoformat` directly.

- [ ] **Step 4: (Removed)** Per user decision 2026-05-20: ship everything in state to the slim JSON, including stale-but-protected entries within the 30-day window. No code change to `_write_repos_outputs`'s slim-write block. The hybrid staleness policy still operates at the persistence layer — stale entries either recover on the next scrape (updating `last_seen`) or age out after 30 days.

- [ ] **Step 5: Run the age-out tests to verify they pass**

Run: `cd ndif-citations && pytest tests/test_output_schema.py -v -k 'age_out or within_30 or manual_override_survives'`
Expected: 3 PASS.

- [ ] **Step 6: Run the full suite**

Run: `cd ndif-citations && pytest -x -q`
Expected: all pass. If `tests/test_output_merge.py` breaks because pre-existing test fixtures don't set `last_seen`, fix those tests inline by adding `last_seen=today` to the fixtures (the new behavior is correct; the test data needs updating).

- [ ] **Step 7: Commit**

```bash
cd ndif-citations
git add src/ndif_citations/output.py tests/test_output_schema.py
git commit -m "feat: hybrid staleness — 30-day age-out for scrape-absent repos

Replaces the one-shot hard purge with a soft 30-day grace window keyed
off last_seen. Repos still alive on GitHub but missing from a flaky
dependents scrape survive for up to a month; API-confirmed-dead repos
(already filtered upstream) naturally age out. manual_override always
survives. The slim JSON only ships repos observed this run."
```

---

## Task 6: `_tag_repo_type()` honors `manual_override=True`

**Files:**
- Modify: `src/ndif_citations/discover.py` (~1103 — `_tag_repo_type`)
- Modify: `src/ndif_citations/cli.py` (~156-165 — the tagging loop and linked_paper_url clear)
- Modify: `tests/test_output_schema.py` (add test)

Today the tagger runs unconditionally — a curator who hand-sets `repo_type="course"` and `manual_override=True` loses it next run. Add an early-return; also guard the cli.py loop that nulls `linked_paper_url` on course repos so manually curated links aren't wiped.

- [ ] **Step 1: Add a failing test for the override**

Append to `tests/test_output_schema.py`:

```python
from ndif_citations.discover import _tag_repo_type


def test_tag_repo_type_respects_manual_override():
    # A repo a curator hand-tagged as 'course', with manual_override=True.
    # Its natural classification (stars=100, has description, has linked paper)
    # would otherwise put it in 'research'.
    r = DiscoveredRepo(
        owner="curator", repo="manually-flagged", url="https://github.com/curator/x",
        stars=100, description="real research project", linked_paper_url="https://arxiv.org/abs/1234.5678",
        repo_type="course", manual_override=True,
    )
    result = _tag_repo_type(r, unlinked_set=set())
    assert result == "course", "manual_override must short-circuit the 7-rule tree"


def test_tag_repo_type_does_not_short_circuit_without_override():
    # Same repo without manual_override — should be re-tagged as 'research'.
    r = DiscoveredRepo(
        owner="curator", repo="not-flagged", url="https://github.com/curator/y",
        stars=100, description="real research project", linked_paper_url="https://arxiv.org/abs/1234.5678",
        repo_type="course", manual_override=False,
    )
    result = _tag_repo_type(r, unlinked_set=set())
    assert result == "research", "without manual_override, normal rules apply"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ndif-citations && pytest tests/test_output_schema.py -v -k 'tag_repo_type'`
Expected: the override test FAILS (the function re-tags `course` → `research` because rules 5 / 6 fire); the second test PASSES.

- [ ] **Step 3: Add the early-return in `_tag_repo_type`**

In `src/ndif_citations/discover.py` at line 1103, replace the function header and add the guard as the first body line:

```python
def _tag_repo_type(repo: DiscoveredRepo, unlinked_set: set[str]) -> str:
    """Determine repo_type for a single repo using a 7-rule decision tree.

    [existing docstring continues here — keep unchanged]
    """
    # Tier 0: honor curator overrides — manual_override means a human set
    # repo_type by hand and we must not re-tag it. The cli.py loop also
    # protects the course → linked_paper_url=None side effect.
    if repo.manual_override:
        return repo.repo_type

    # --- COURSE ---
    # Tier 1: forked from known course source
    ...  # rest of function unchanged
```

- [ ] **Step 4: Guard the cli.py loop's `linked_paper_url` clear**

In `src/ndif_citations/cli.py` at the tagging loop (~157-165), replace the block:

```python
        # Tag every repo (runs on the merged set for consistent cross-repo state)
        course_cleared = 0
        for repo in all_for_cross:
            repo.repo_type = _tag_repo_type(repo, unlinked_set)
            # Course repos cite many papers — none is canonical. Clear the link
            # so they neither display a 📄 badge nor cross-link to any paper.
            # Skip this side effect for curator-overridden repos: if a human
            # set both repo_type AND linked_paper_url, trust them.
            if (
                repo.repo_type == "course"
                and repo.linked_paper_url
                and not repo.manual_override
            ):
                repo.linked_paper_url = None
                repo.linked_paper_tier = None
                course_cleared += 1
        if course_cleared:
            console.print(f"  Cleared linked_paper_url on {course_cleared} course repo(s)")
```

- [ ] **Step 5: Run the tag tests to verify they pass**

Run: `cd ndif-citations && pytest tests/test_output_schema.py -v -k 'tag_repo_type'`
Expected: both PASS.

- [ ] **Step 6: Run the full suite**

Run: `cd ndif-citations && pytest -x -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
cd ndif-citations
git add src/ndif_citations/discover.py src/ndif_citations/cli.py tests/test_output_schema.py
git commit -m "feat: honor manual_override in _tag_repo_type and link-clear loop

Curator-set repo_type now sticks across runs. _tag_repo_type early-returns
on manual_override=True; the cli.py loop that nulls linked_paper_url on
course repos also skips overridden rows so manually curated links survive."
```

---

## Task 7: Broaden `COURSE_NAME_PATTERNS`

**Files:**
- Modify: `src/ndif_citations/config.py` (line 111 — `COURSE_NAME_PATTERNS`)
- Modify: `tests/test_output_schema.py` (add test)

Two real misses in today's data: `AntonKorznikov/TheiaSae` (description: "Skoltech ML course 2025 project") and `MichaelRipa/coding-exercises-metarepo` (description: "coding exercises from various courses"). Add patterns that catch both, in a way that doesn't false-positive against legitimate research repos.

- [ ] **Step 1: Add a failing test for the two known leaks**

Append to `tests/test_output_schema.py`:

```python
def test_tag_repo_type_catches_known_course_leaks():
    # Real repos that today get tagged 'experiment' but are clearly coursework.
    theia = DiscoveredRepo(
        owner="AntonKorznikov", repo="TheiaSae", url="https://github.com/AntonKorznikov/TheiaSae",
        stars=0, description="Skoltech ML course 2025 project",
    )
    metarepo = DiscoveredRepo(
        owner="MichaelRipa", repo="coding-exercises-metarepo",
        url="https://github.com/MichaelRipa/coding-exercises-metarepo",
        stars=0, description="This repository contains all my coding exercises from various courses",
    )
    assert _tag_repo_type(theia, unlinked_set=set()) == "course"
    assert _tag_repo_type(metarepo, unlinked_set=set()) == "course"


def test_tag_repo_type_does_not_misclassify_research_with_workshop_keyword():
    # Per user decision 2026-05-20: "workshop" was dropped from COURSE_NAME_PATTERNS
    # to avoid false-positive on workshop-paper research repos. Verify a research
    # repo mentioning "workshop" still tags as research.
    r = DiscoveredRepo(
        owner="real", repo="research", url="https://github.com/real/research",
        stars=200, description="ICLR 2026 workshop paper code",
        linked_paper_url="https://arxiv.org/abs/2601.00000",
    )
    assert _tag_repo_type(r, unlinked_set=set()) == "research"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ndif-citations && pytest tests/test_output_schema.py -v -k 'known_course_leaks or misclassify_research'`
Expected: the leak test FAILS; the misclassify test would also fail (or pass depending on chosen patterns).

- [ ] **Step 3: Extend `COURSE_NAME_PATTERNS`**

In `src/ndif_citations/config.py` line 111, replace:

```python
COURSE_NAME_PATTERNS: list[str] = ["ARENA", "MATS", "CBAI"]  # case-insensitive name/desc substrings
```

with:

```python
COURSE_NAME_PATTERNS: list[str] = [
    # Course-program identifiers (existing — kept as-is)
    "ARENA", "MATS", "CBAI",
    # Generic coursework markers (added 2026-05-20, see spec §3.3 audit)
    # NOTE: "workshop" intentionally omitted per user decision 2026-05-20 —
    # too many false positives against workshop-paper research repos.
    "course project", "coursework", "exercises", "capstone",
    "homework", "assignment",
]  # case-insensitive name/desc substrings
```

- [ ] **Step 4: Run the broader course tests**

Run: `cd ndif-citations && pytest tests/test_output_schema.py -v -k 'known_course_leaks or misclassify_research'`
Expected: PASS.

- [ ] **Step 5: Run the full suite + check for course-tag regressions**

Run: `cd ndif-citations && pytest -x -q`
Expected: all pass.

If `tests/test_discover_*` exercise `_tag_repo_type` with fixtures that include any of the new patterns in name/description, those fixtures may now classify differently. Read any failing test, decide whether the new classification is correct (likely yes — broaden patterns to catch real coursework), and update the test's expected value with a one-line comment referencing this PR.

- [ ] **Step 6: Commit**

```bash
cd ndif-citations
git add src/ndif_citations/config.py tests/test_output_schema.py
git commit -m "feat: broaden COURSE_NAME_PATTERNS to catch generic coursework

Adds: course project, coursework, exercises, capstone, homework,
assignment, workshop. Catches 2 known leaks (TheiaSae and
coding-exercises-metarepo) that today tag as 'experiment' despite
having course-project descriptions. Manual_override remains the escape
hatch for any false positives (e.g. a real research repo that mentions
'workshop' in its README)."
```

---

## Task 8: Update XLSX columns per spec §2.3

**Files:**
- Modify: `src/ndif_citations/output.py` (~603-651 — XLSX `github_cols` list + row builder)

Spec §2.3 columns in order: `owner, repo, url, description, stars, forks, last_commit, first_seen, language, license, topics, linked_paper_url, linked_paper_tier, readme_arxiv_ids, category, repo_type, parent_full_name, archived, is_fork, classification_reason, manual_override`. The current code emits a slightly different set / order and is missing `first_seen`, `linked_paper_tier`, `readme_arxiv_ids`, `archived`, `is_fork`.

- [ ] **Step 1: Update `github_cols` and the row builder**

In `src/ndif_citations/output.py` at ~604-614, replace the `github_cols` declaration:

```python
        github_cols = [
            "owner", "repo", "url", "description",
            "stars", "forks",
            "last_commit", "first_seen",
            "language", "license", "topics",
            "linked_paper_url", "linked_paper_tier", "readme_arxiv_ids",
            "category", "repo_type", "parent_full_name",
            "archived", "is_fork",
            "classification_reason", "manual_override",
        ]
```

And at ~626-646, replace `row_data` with:

```python
            row_data = {
                "owner": repo.owner,
                "repo": repo.repo,
                "url": repo.url,
                "description": repo.description or "",
                "stars": repo.stars,
                "forks": repo.forks,
                "last_commit": repo.last_commit.isoformat() if repo.last_commit else "",
                "first_seen": repo.first_seen or "",
                "language": repo.language or "",
                "license": repo.license or "",
                "topics": ", ".join(repo.topics),
                "linked_paper_url": repo.linked_paper_url or "",
                "linked_paper_tier": repo.linked_paper_tier if repo.linked_paper_tier is not None else "",
                "readme_arxiv_ids": ", ".join(repo.readme_arxiv_ids),
                "category": repo.category.value,
                "repo_type": repo.repo_type,
                "parent_full_name": repo.parent_full_name or "",
                "archived": repo.archived,
                "is_fork": repo.is_fork,
                "classification_reason": repo.classification_reason,
                "manual_override": repo.manual_override,
            }
```

(`has_metadata`, `has_classification`, `content_hash`, `processing_bucket` dropped per spec §2.3.)

- [ ] **Step 2: Verify by running the pipeline against existing output**

Run: `cd ndif-citations && python -m ndif_citations run --skip-papers --skip-github 2>&1 | head -30`

(`--skip-github` is intentional here — we want to test the XLSX writer against existing in-memory state without re-scraping. If the CLI doesn't support this combination cleanly, instead run the test-only invocation below.)

Alternative: write a quick smoke test that calls `write_outputs` against fixture repos and asserts the XLSX has the expected column set.

```python
def test_xlsx_github_sheet_columns(tmp_path):
    import openpyxl
    from ndif_citations.output import _write_xlsx

    repos = [DiscoveredRepo(
        owner="o", repo="r", url="https://github.com/o/r",
        stars=10, forks=2, first_seen="2026-01-01", last_seen="2026-05-20",
    )]
    _write_xlsx(papers=[], repos=repos, output_dir=tmp_path, skip_papers=True, skip_github=False)
    wb = openpyxl.load_workbook(tmp_path / "research-data.xlsx")
    ws = wb["GitHub"]
    header = [c.value for c in ws[1]]
    expected = [
        "owner", "repo", "url", "description", "stars", "forks",
        "last_commit", "first_seen", "language", "license", "topics",
        "linked_paper_url", "linked_paper_tier", "readme_arxiv_ids",
        "category", "repo_type", "parent_full_name",
        "archived", "is_fork", "classification_reason", "manual_override",
    ]
    assert header == expected
```

(`_write_xlsx` lives at `output.py:518` with signature `(papers, repos, output_dir, skip_papers=False, skip_github=False)`. `write_outputs` at line 348 does NOT take a `repos` kwarg — it composes the XLSX call internally.)

- [ ] **Step 3: Run the XLSX test**

Run: `cd ndif-citations && pytest tests/test_output_schema.py::test_xlsx_github_sheet_columns -v`
Expected: PASS.

- [ ] **Step 4: Run the full suite**

Run: `cd ndif-citations && pytest -x -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd ndif-citations
git add src/ndif_citations/output.py tests/test_output_schema.py
git commit -m "feat: align XLSX GitHub sheet with spec §2.3

Adds first_seen, linked_paper_tier, readme_arxiv_ids, archived, is_fork.
Drops content_hash, has_metadata, has_classification, processing_bucket
(no human reads these). Topics and readme_arxiv_ids now joined with
', ' instead of '|' for readability in Excel."
```

---

## Task 9: End-to-end run + acceptance verification

**Files:**
- (no code changes — manual verification step)

Walk through all 8 acceptance criteria from spec §7 against a real run.

- [ ] **Step 1: Run the pipeline against current state**

Run: `cd ndif-citations && python -m ndif_citations run --skip-papers`
Expected: clean exit. Look for log lines: "Wrote N repos to ... github-repos.json (filtered from M in full state)".

- [ ] **Step 2: Verify AC #1 — slim JSON has exactly 10 keys per entry**

Run:
```bash
cd ndif-citations
python -c "
import json
data = json.load(open('output/github-repos.json'))
expected = {'owner','repo','url','description','stars','forks','last_commit','language','linked_paper_url','is_course','is_fork','parent_full_name'}
mismatches = [(i, set(r.keys()) - expected, expected - set(r.keys())) for i, r in enumerate(data) if set(r.keys()) != expected]
print(f'Total: {len(data)}, mismatches: {len(mismatches)}')
if mismatches: print(mismatches[:3])
"
```
Expected: `mismatches: 0`.

- [ ] **Step 3: Verify AC #2 — full JSON has first_seen and last_seen on every repo**

Run:
```bash
cd ndif-citations
python -c "
import json
data = json.load(open('output/github-repos-full.json'))
missing = [r.get('owner') + '/' + r.get('repo') for r in data if not r.get('first_seen') or not r.get('last_seen')]
print(f'Total: {len(data)}, missing first/last seen: {len(missing)}')
if missing: print(missing[:5])
"
```
Expected: `missing first/last seen: 0`.

- [ ] **Step 4: Verify AC #3 — XLSX GitHub sheet matches spec, no CSV files written**

Run:
```bash
cd ndif-citations
ls output/*.csv 2>&1
python -c "
import openpyxl
wb = openpyxl.load_workbook('output/research-data.xlsx')
ws = wb['GitHub']
header = [c.value for c in ws[1]]
print('header:', header)
"
```
Expected: `ls: output/*.csv: No such file or directory` (no CSVs) and the header matches the column list from Task 8.

- [ ] **Step 5: Verify AC #4 — known course leaks now tagged**

Run:
```bash
cd ndif-citations
python -c "
import json
data = json.load(open('output/github-repos-full.json'))
for owner_repo in ['AntonKorznikov/TheiaSae', 'MichaelRipa/coding-exercises-metarepo']:
    matches = [r for r in data if f'{r[\"owner\"]}/{r[\"repo\"]}' == owner_repo]
    print(owner_repo, '->', matches[0]['repo_type'] if matches else 'NOT IN DATA')
"
```
Expected: both print `-> course` (or `NOT IN DATA` if they fell out of the dependents scrape this run — re-check on next run).

- [ ] **Step 6: Verify AC #5 — manual_override round-trip**

```bash
cd ndif-citations
# Pick any non-course repo, hand-edit github-repos-full.json to set
# repo_type="course" and manual_override=true on it, then:
python -m ndif_citations run --skip-papers
python -c "
import json
data = json.load(open('output/github-repos-full.json'))
target = [r for r in data if r.get('manual_override')]
print('manual_override repos:', [(r['owner']+'/'+r['repo'], r['repo_type']) for r in target])
"
```
Expected: the edited repo still shows `repo_type: 'course'` after the run.

- [ ] **Step 7: Verify AC #6 — partial scrape doesn't immediately delete repos**

This is hard to test without a controlled scrape failure. Instead, verify the mechanism by inspecting logs from Step 1: look for `"Aging out scrape-absent repo (Nd since last seen)"` vs `"Keeping scrape-absent repo"` lines. If your dependents page returned all expected repos, neither will fire — that's fine. The next time it returns fewer pages, the keeps-not-purges behavior will engage and you'll see it in logs.

- [ ] **Step 8: Verify AC #7 — all schema tests pass**

Run: `cd ndif-citations && pytest tests/test_output_schema.py -v`
Expected: all tests PASS (count should be ~13).

- [ ] **Step 9: Verify AC #8 — companion site PR will build cleanly**

This is for the companion site PR, not this one. Note the open follow-up in `tasks/notes-site-trim-changes.md`.

- [ ] **Step 10: Final summary commit (changelog / notes)**

```bash
cd ndif-citations
# If there's anything left to commit (e.g., an updated tests/test_output_merge.py
# fixture from Task 5), commit it now with:
git status
git add -p  # review changes interactively
git commit -m "chore: post-revamp cleanup — fixture updates and acceptance log"
```

---

## Self-Review (done before publishing)

**Spec coverage:**
- §2.1 (10-field slim JSON) → Task 3
- §2.2 (first_seen/last_seen in full JSON) → Tasks 2 + 4
- §2.3 (XLSX columns, no CSV) → Tasks 1 + 8
- §3.1 (`models.py` changes) → Tasks 2 + 3
- §3.2 (`merge_repos` stamping + hybrid staleness) → Tasks 4 + 5
- §3.3 (course-tag improvements: patterns + manual_override + parent_full_name finding) → Tasks 6 + 7. The parent_full_name "not-a-bug" finding requires no code change (per spec rev 2 §3.3).
- §3.4 (snapshot tests) → embedded across Tasks 2–8 in `tests/test_output_schema.py`
- §4 (site changes) → out of scope for this plan; tracked in `tasks/notes-site-trim-changes.md` and acceptance criterion #8
- §5 (FeaturedCode) → out of scope (site PR)
- §6 (out of scope) → respected throughout
- §7 (acceptance criteria) → Task 9 walks all 8

**Placeholder scan:** no TBDs, no "implement later", every code step has a full code block.

**Type consistency:** `first_seen` / `last_seen` typed as `Optional[str]` (ISO YYYY-MM-DD) consistently in models, merge_repos, and tests. `is_course` is a `bool` property derived from `repo_type == "course"`. `_today()` returns `date`. AGE_OUT_DAYS is `int = 30`.
