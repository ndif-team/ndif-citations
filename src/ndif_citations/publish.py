"""Preview what "Push catalog" will publish (spec §5.7).

Publishing is now: push the catalog to ndif-citations `main` (catalog_git.push_catalog);
ndif.us pulls it daily. This module only previews the change — local slim outputs vs
the last commit — reusing the diff helpers the Publish page already renders.
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


def _head_json_list(repo_root: Path, rel: str) -> list[dict]:
    r = subprocess.run(["git", "-C", str(repo_root), "show", f"HEAD:{rel}"], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return []
    data = json.loads(r.stdout)
    return data if isinstance(data, list) else []


def _changed_images(repo_root: Path) -> dict:
    # -uall lists each new file (not "?? output/images/"); quotepath=off keeps non-ASCII names unquoted.
    r = subprocess.run(["git", "-C", str(repo_root), "-c", "core.quotepath=off", "status", "--porcelain",
                        "-uall", "--", "output/images"], capture_output=True, text=True)
    new, changed = [], []
    for line in r.stdout.splitlines() if r.returncode == 0 else []:
        status, name = line[:2], Path(line[3:].strip()).name
        (new if status.strip() in ("??", "A") else changed).append(name)
    return {"new": sorted(new), "changed": sorted(changed)}


def preview(out: Path, repo_root: Path) -> dict:
    """Local slim JSON vs HEAD, in the {papers, repos, images} shape the UI renders."""
    out, repo_root = Path(out), Path(repo_root)
    return {
        "papers": _diff_list(_load_json_list(out / _PAPERS_JSON),
                             _head_json_list(repo_root, f"output/{_PAPERS_JSON}"), _paper_key),
        "repos": _diff_list(_load_json_list(out / _REPOS_JSON),
                            _head_json_list(repo_root, f"output/{_REPOS_JSON}"), _repo_key),
        "images": _changed_images(repo_root),
    }
