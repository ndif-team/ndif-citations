"""Preview what "Push catalog" will publish (spec §5.7).

Publishing is now: push the catalog to ndif-citations `main` (catalog_git.push_catalog);
ndif.us pulls it daily. This module only previews the change — local slim outputs vs
the last pushed state — reusing the diff helpers the Publish page already renders.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

# Slim output filenames (identical at source and destination).
_PAPERS_JSON = "research-papers.json"
_REPOS_JSON = "github-repos.json"


def _load_json_list(path: Path) -> list[dict]:
    """Load a JSON array from *path*; return [] if the file is absent."""
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON array")
    return data


def _paper_key(p: dict) -> str:
    """Stable identity for a slim paper: prefer url, fall back to title."""
    return p.get("url") or p.get("title") or ""


def _repo_key(r: dict) -> str:
    """Stable identity for a slim repo: owner/repo, fall back to url."""
    owner = r.get("owner")
    repo = r.get("repo")
    if owner and repo:
        return f"{owner}/{repo}"
    return r.get("url") or ""


def _diff_list(out_items: list[dict], target_items: list[dict], key_fn) -> dict:
    """Return {added, changed, removed} comparing out vs target by *key_fn*."""
    out_by_key = {key_fn(x): x for x in out_items}
    tgt_by_key = {key_fn(x): x for x in target_items}

    added = [out_by_key[k] for k in out_by_key if k not in tgt_by_key]
    removed = [tgt_by_key[k] for k in tgt_by_key if k not in out_by_key]
    changed = []
    for k in out_by_key:
        if k in tgt_by_key and out_by_key[k] != tgt_by_key[k]:
            # Copy + annotate with WHICH fields differ so the publish preview
            # can say "stars, description" instead of an opaque "changed".
            item = dict(out_by_key[k])
            item["_changed_fields"] = sorted(
                f
                for f in set(out_by_key[k]) | set(tgt_by_key[k])
                if out_by_key[k].get(f) != tgt_by_key[k].get(f)
            )
            changed.append(item)
    return {"added": added, "changed": changed, "removed": removed}


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo_root), "-c", "core.quotepath=off", *args],
                          capture_output=True, text=True)


def _base(repo_root: Path) -> str:
    """What the push is measured against: where local main forked from origin/main. That counts
    committed-but-unpushed edits, and never shows bot commits we haven't pulled as removals."""
    r = _git(repo_root, "merge-base", "HEAD", "origin/main")
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "HEAD"


def _base_json_list(repo_root: Path, base: str, rel: str) -> list[dict]:
    r = _git(repo_root, "show", f"{base}:{rel}")
    if r.returncode != 0 or not r.stdout.strip():
        return []
    data = json.loads(r.stdout)
    return data if isinstance(data, list) else []


def _changed_images(repo_root: Path, base: str) -> dict:
    """Images that differ from *base* (committed or not) plus untracked ones. -z: names are never quoted.
    Deletions are left out: the website copies images additively, so a deletion publishes nothing."""
    new, changed = [], []
    r = _git(repo_root, "diff", "--name-status", "--no-renames", "-z", base, "--", "output/images")
    fields = r.stdout.split("\0") if r.returncode == 0 else []
    for status, path in zip(fields[0::2], fields[1::2]):
        if status != "D":
            (new if status == "A" else changed).append(Path(path).name)
    r = _git(repo_root, "ls-files", "-o", "--exclude-standard", "-z", "--", "output/images")
    new += [Path(f).name for f in r.stdout.split("\0") if f] if r.returncode == 0 else []
    return {"new": sorted(new), "changed": sorted(changed)}


def preview(out: Path, repo_root: Path) -> dict:
    """Local slim JSON vs the merge-base with origin/main (else HEAD), in the {papers, repos, images}
    shape the UI renders."""
    out, repo_root = Path(out), Path(repo_root)
    base = _base(repo_root)
    return {
        "papers": _diff_list(_load_json_list(out / _PAPERS_JSON),
                             _base_json_list(repo_root, base, f"output/{_PAPERS_JSON}"), _paper_key),
        "repos": _diff_list(_load_json_list(out / _REPOS_JSON),
                            _base_json_list(repo_root, base, f"output/{_REPOS_JSON}"), _repo_key),
        "images": _changed_images(repo_root, base),
    }
