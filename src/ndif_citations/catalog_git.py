"""Commit + push the tracked catalog safely — the ONE path every writer uses (spec §5.2).

Bot writers are serialized by a GitHub Actions FIFO queue, so a rebase here only
happens when a curator's local push races a bot. Other tracked files must be clean
(nothing is ever stashed or reset away), and unpushed catalog commits from an earlier
offline/conflicted attempt are always retried. Text merges of the catalog JSON
can be semantically wrong (one paper in two buckets), so after any rebase the
merged FULL JSON is validated before the slim files are regenerated from it. On
any doubt, nothing is pushed and the local commit is kept on a conflict branch.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PAPERS_FULL = "output/research-papers-full.json"
REPOS_FULL = "output/github-repos-full.json"
SLIM_PATHS = ("output/research-papers.json", "output/github-repos.json")
IMAGES = "output/images"
CATALOG_PATHS = (PAPERS_FULL, REPOS_FULL, *SLIM_PATHS, IMAGES)


class CatalogPushError(RuntimeError):
    """The catalog could not be pushed safely (message says why / where it was saved)."""


def _git(repo: Path, *args: str, check: bool = True, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", **(env_extra or {})}
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check, env=env)


def _show_json(repo: Path, ref: str, path: str):
    r = _git(repo, "show", f"{ref}:{path}", check=False)
    return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None


def _paper_keys(full: dict | None) -> list[str]:
    from ndif_citations.models import DiscoveredPaper
    keys: list[str] = []
    for bucket in ("pending", "verified", "discarded"):
        for item in (full or {}).get(bucket, []):
            keys.append(DiscoveredPaper.model_validate(item).merge_key())
    return keys


def _paper_key_titles(full: dict | None) -> list[tuple[str, str]]:
    """(merge_key, normalized title) for every paper — titles normalized as in _paper_ids."""
    from ndif_citations.models import DiscoveredPaper
    return [(DiscoveredPaper.model_validate(item).merge_key(), " ".join((item.get("title") or "").lower().split()))
            for bucket in ("pending", "verified", "discarded") for item in (full or {}).get(bucket, [])]


def _paper_ids(full: dict | None) -> set[str]:
    """Merge keys plus normalized titles: enrichment moves a paper's key from title to
    arXiv/DOI, and that must not look like a lost paper."""
    out = set(_paper_keys(full))
    for bucket in ("pending", "verified", "discarded"):
        out |= {"t:" + " ".join((p.get("title") or "").lower().split()) for p in (full or {}).get(bucket, [])}
    return out


def _repo_keys(items: list | None) -> list[str]:
    return [f"{r['owner']}/{r['repo']}" for r in (items or [])]


def _key_problems(label: str, base: list[str], ours: list[str], theirs: list[str], merged: list[str]) -> list[str]:
    problems: list[str] = []
    base_dups = {k for k in base if base.count(k) > 1}  # ponytail: O(n²), fine at ~300 keys
    new_dups = sorted({k for k in merged if merged.count(k) > 1} - base_dups)
    if new_dups:
        problems.append(f"{label}: new duplicate keys {new_dups[:5]}")
    B, O, T, M = set(base), set(ours), set(theirs), set(merged)
    expected = (O | T) - (B - O) - (B - T)  # whatever either side deleted is gone; the rest survives
    if M != expected:
        problems.append(f"{label}: key-set mismatch (missing {sorted(expected - M)[:5]}, "
                        f"unexpected {sorted(M - expected)[:5]})")
    return problems


def validate_catalog(base: dict | None, ours: dict | None, theirs: dict | None, merged: dict) -> list[str]:
    """Problems with a merged research-papers-full.json (empty list = safe to push)."""
    if not isinstance(merged, dict) or not {"pending", "verified", "discarded"} <= set(merged):
        return ["papers: merged catalog is not the 3-bucket dict"]
    try:
        merged_keys = _paper_keys(merged)  # every item must validate — nothing silently dropped
    except Exception as e:  # noqa: BLE001
        return [f"papers: merged catalog has an invalid paper ({type(e).__name__})"]
    return _key_problems("papers", _paper_keys(base), _paper_keys(ours), _paper_keys(theirs), merged_keys)


def validate_repo_list(base, ours, theirs, merged) -> list[str]:
    """Problems with a merged github-repos-full.json (empty list = safe to push)."""
    if not isinstance(merged, list):
        return ["repos: merged repo list is not a list"]
    try:
        merged_keys = _repo_keys(merged)
    except (KeyError, TypeError):
        return ["repos: merged repo list has a malformed entry"]
    return _key_problems("repos", _repo_keys(base), _repo_keys(ours), _repo_keys(theirs), merged_keys)


def _regenerate_slim(repo: Path) -> None:
    from ndif_citations.models import PipelineRun
    from ndif_citations.output import (_write_repos_outputs, load_existing_papers,
                                       load_existing_repos, write_outputs)
    out = repo / "output"
    write_outputs(load_existing_papers(out), out, PipelineRun())
    _write_repos_outputs(load_existing_repos(out), out)


def _stage_catalog(repo: Path) -> None:
    """Stage the 4 JSONs, tracked image edits/deletions, and new images the catalog references.
    A stray untracked thumbnail is never committed."""
    _git(repo, "add", "--", PAPERS_FULL, REPOS_FULL, *SLIM_PATHS)
    if _git(repo, "ls-files", "--", IMAGES).stdout.strip():  # `add -u` errors when nothing is tracked there
        _git(repo, "add", "-u", "--", IMAGES)
    papers = json.loads((repo / PAPERS_FULL).read_text())
    items = [i for b in ("pending", "verified", "discarded") for i in papers.get(b) or []]
    items += json.loads((repo / REPOS_FULL).read_text())
    on_disk = set(os.listdir(repo / IMAGES)) if (repo / IMAGES).is_dir() else set()  # exact names
    refs = sorted({n for n in (Path(i["image"]).name for i in items if isinstance(i, dict)
                               and isinstance(i.get("image"), str) and i["image"])
                   if n in on_disk and (repo / IMAGES / n).is_file()})
    if refs:  # literal pathspecs: thumbnail names can hold [ ] * ?
        _git(repo, "add", "--", *(f"{IMAGES}/{n}" for n in refs), env_extra={"GIT_LITERAL_PATHSPECS": "1"})


def _git_or_fail(repo: Path, *args: str, what: str) -> subprocess.CompletedProcess:
    """Run git; turn any failure into a CatalogPushError the caller can show."""
    r = _git(repo, *args, check=False)
    if r.returncode != 0:
        raise CatalogPushError(f"{what} failed: {(r.stderr or r.stdout).strip()[:300]}")
    return r


def _ahead(repo: Path, upstream: str) -> int:
    return int(_git_or_fail(repo, "rev-list", "--count", f"{upstream}..HEAD", what="git rev-list").stdout)


def _resolvable(path: str) -> bool:
    """Conflicts we can resolve mechanically: generated slim files and thumbnails."""
    return path in SLIM_PATHS or path.startswith("output/images/")


def _rebase_onto(repo: Path, remote: str, branch: str) -> bool:
    """Rebase our catalog commit onto the remote and validate. False = unsafe (state restored)."""
    upstream = f"{remote}/{branch}"
    # Our commit, captured explicitly: ORIG_HEAD is stale after a no-op rebase (and absent in a fresh clone).
    pre = _git_or_fail(repo, "rev-parse", "HEAD", what="git rev-parse").stdout.strip()
    base_ref = _git_or_fail(repo, "merge-base", "HEAD", upstream, what="git merge-base").stdout.strip()
    snaps = {p: (_show_json(repo, base_ref, p), _show_json(repo, "HEAD", p), _show_json(repo, upstream, p))
             for p in (PAPERS_FULL, REPOS_FULL)}
    no_editor = {"GIT_EDITOR": "true"}
    try:
        r = _git(repo, "rebase", upstream, check=False, env_extra=no_editor)
        while r.returncode != 0:
            conflicted = _git(repo, "-c", "core.quotepath=off", "diff", "--name-only", "--diff-filter=U",
                              check=False).stdout.splitlines()  # thumbnail names have spaces / non-ASCII
            if conflicted and all(_resolvable(c) for c in conflicted):
                for c in conflicted:  # either side will do: slim files are regenerated below
                    _git(repo, "checkout", "--theirs", "--", c)
                    _git(repo, "add", "--", c)
                r = _git(repo, "rebase", "--continue", check=False, env_extra=no_editor)
                continue
            _git(repo, "rebase", "--abort", check=False)
            return False
        if _ahead(repo, upstream) == 0:  # every local commit was already upstream: HEAD is published, never amend
            return True
        problems: list[str] = []
        for path, (b, o, t) in snaps.items():
            merged = json.loads((repo / path).read_text())
            problems += (validate_catalog if path == PAPERS_FULL else validate_repo_list)(b, o, t, merged)
        if problems:
            logger.error("catalog merge rejected: %s", "; ".join(problems))
            _git(repo, "reset", "--keep", pre)  # back to our pre-rebase commit; --keep never drops edits
            return False
        _regenerate_slim(repo)
        _stage_catalog(repo)
        if _git(repo, "diff", "--cached", "--quiet", check=False).returncode != 0:
            _git(repo, "commit", "--amend", "--no-edit")
        return True
    except Exception:
        # e.g. a modify/delete conflict on an image, or unparsable merged JSON: never leave
        # the repo mid-rebase or an unvalidated rebased commit on main.
        logger.exception("catalog merge failed")
        _git(repo, "rebase", "--abort", check=False)
        _git(repo, "reset", "--keep", pre, check=False)
        return False


def _save_conflict_branch(repo: Path, remote: str, branch: str, push: bool) -> str:
    """Keep our commit on a conflict branch, then put local main back on the remote."""
    name = f"catalog-conflict-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    _git_or_fail(repo, "branch", name, "HEAD", what="saving the conflict branch")
    if push and _git(repo, "push", remote, name, check=False).returncode != 0:
        raise CatalogPushError(f"catalog conflict, and pushing branch {name!r} failed — "
                               "the commit exists only on this machine (CI uploads output/ as an artifact)")
    _git(repo, "reset", "--keep", f"{remote}/{branch}")
    return name


def _refuse_shrink(repo: Path, upstream: str) -> None:
    """The repo is the source of truth: never push a catalog that lost a big slice of its entries."""
    before_full, head_full = _show_json(repo, upstream, PAPERS_FULL), _show_json(repo, "HEAD", PAPERS_FULL)
    head_ids = _paper_ids(head_full)
    lost_papers = [k for k, t in _paper_key_titles(before_full)
                   if k not in head_ids and "t:" + t not in head_ids]
    before_repos = set(_repo_keys(_show_json(repo, upstream, REPOS_FULL)))
    lost_repos = sorted(before_repos - set(_repo_keys(_show_json(repo, "HEAD", REPOS_FULL))))
    for path, lost, total in ((PAPERS_FULL, lost_papers, len(_paper_keys(before_full))),
                              (REPOS_FULL, lost_repos, len(before_repos))):
        if len(lost) > max(5, total // 10):
            raise CatalogPushError(
                f"refusing to push: {path} would lose {len(lost)} of {total} entries (e.g. {sorted(lost)[:3]}). "
                f"Inspect with `git diff {upstream} --stat -- output/`; if the loss is intended, push it yourself with git.")


def push_catalog(repo: Path, message: str, *, push_conflict_branch: bool = False,
                 remote: str = "origin", branch: str = "main", attempts: int = 3) -> dict:
    """Commit the tracked catalog paths (if changed) and push every unpushed catalog commit."""
    repo = Path(repo)
    current = _git_or_fail(repo, "rev-parse", "--abbrev-ref", "HEAD", what="git rev-parse").stdout.strip()
    if current != branch:
        raise CatalogPushError(f"refusing to push the catalog from branch {current!r} (expected {branch!r})")
    # Pathspec exclude, not a prefix check: porcelain quotes non-ASCII paths ("output/images/DFWe-E…").
    other_dirty = _git(repo, "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)output",
                       check=False).stdout.splitlines()
    if other_dirty:
        raise CatalogPushError(f"commit or stash your other changes first: {other_dirty[:5]}")
    for rel, kind in ((PAPERS_FULL, dict), (REPOS_FULL, list)):  # never commit a missing/corrupt catalog
        try:
            ok = isinstance(json.loads((repo / rel).read_text()), kind)
        except (OSError, ValueError):
            ok = False
        if not ok:
            raise CatalogPushError(f"{rel} is missing or not a valid JSON {kind.__name__}; nothing was committed")
    _stage_catalog(repo)
    if _git(repo, "diff", "--cached", "--quiet", check=False).returncode != 0:
        _git_or_fail(repo, "commit", "-m", message, what="git commit")
    if _git(repo, "fetch", remote, branch, check=False).returncode != 0:
        raise CatalogPushError("couldn't reach GitHub — your changes are committed locally and will be "
                               "pushed the next time you push the catalog")
    if _ahead(repo, f"{remote}/{branch}") == 0:
        return {"pushed": False, "commit": None, "conflict_branch": None, "message": "no catalog changes"}
    # HEAD:main pushes every unpushed commit; code must go through a PR, never through Push catalog.
    bad = [f for f in _git_or_fail(repo, "-c", "core.quotepath=off", "diff", "--name-only",
                                   f"{remote}/{branch}...HEAD", what="git diff").stdout.splitlines()
           if not f.startswith("output/")]
    if bad:
        raise CatalogPushError(f"local {branch} has non-catalog commits — push them via a PR first: {bad[:5]}")
    last = None
    for _ in range(attempts):
        _refuse_shrink(repo, f"{remote}/{branch}")  # every attempt: a rebase can change what HEAD holds
        last = _git(repo, "push", remote, f"HEAD:{branch}", check=False)
        if last.returncode == 0:
            sha = _git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
            return {"pushed": True, "commit": sha, "conflict_branch": None, "message": f"pushed {sha}"}
        _git(repo, "fetch", remote, branch, check=False)
        if not _rebase_onto(repo, remote, branch):
            name = _save_conflict_branch(repo, remote, branch, push_conflict_branch)
            raise CatalogPushError(
                f"catalog changes conflict with {remote}/{branch}. ALL your unpushed catalog commits were moved "
                f"to branch {name!r} and local {branch} now matches {remote}/{branch}, so the catalog no longer "
                f"shows any of those edits — re-apply all of those edits (or ask a maintainer to merge the branch)."
            )
        if _ahead(repo, f"{remote}/{branch}") == 0:
            return {"pushed": False, "commit": None, "conflict_branch": None,
                    "message": f"already on {remote}/{branch}"}
    raise CatalogPushError(f"push to {remote}/{branch} kept being rejected after {attempts} rebases: "
                           f"{(last.stderr or '').strip()[:300]}")


def pull_if_clean(repo: Path) -> str:
    """Fast-forward the catalog before local curation; never touches local changes."""
    repo = Path(repo)
    if _git(repo, "rev-parse", "--is-inside-work-tree", check=False).returncode != 0:
        return "not a git checkout — skipping catalog pull"
    # Untracked files don't count: a stray thumbnail must not block every pull (and --ff-only
    # itself refuses to overwrite an untracked file).
    if _git(repo, "status", "--porcelain", "--untracked-files=no", "--", *CATALOG_PATHS,
            check=False).stdout.strip():
        return "local catalog has unpushed changes — not pulling (use Push catalog)"
    r = _git(repo, "pull", "--ff-only", check=False)
    return "catalog up to date" if r.returncode == 0 else "catalog pull failed (offline or diverged) — continuing"
