"""Publish curated slim outputs to the live NDIF website repo.

This module SUPERSEDES the site repo's manual ``sync-research-papers.mjs`` /
``sync-github-repos.mjs`` scripts. The curator drives publishing through the web
"Publish" button (server endpoints in ``server/routers/publish.py``); any such
``.mjs`` scripts remain only as a manual fallback and are NOT modified by us.

Why reimplement in Python instead of shelling out to node:
  * Testability — every step (detect / validate / diff / apply) is a pure
    function over ``Path`` args, exercised by ``tests/test_publish.py``.
  * Bug fix — the ``.mjs`` scripts copy images ADDITIVELY: they skip any image
    that already exists at the destination (``if (existsSync(dest)) continue``).
    Re-extracted thumbnails therefore never reach the live site. ``apply()``
    here FORCE-OVERWRITES an image whenever the bytes differ.

Source / destination / shape — matched exactly to the .mjs scripts:
  * Source files (the SLIM pipeline outputs, regenerated on every run/edit):
      ``out/research-papers.json``  — list of verified ``to_website_dict``s
      ``out/github-repos.json``     — list of ``to_website_dict``s
      ``out/images/{slug}.png``     — referenced as ``/images/{slug}.png`` in JSON
  * Destination (inside the production site repo, ``ndif-website``):
      ``<ndif-website>/public/data/research-papers.json``
      ``<ndif-website>/public/data/github-repos.json``
      ``<ndif-website>/public/images/{slug}.png``
  * JSON written as ``json.dumps(..., indent=2) + "\n"`` — byte-identical to
    the .mjs ``JSON.stringify(x, null, 2) + "\n"``.

Safety:
  * ``detect_target`` / ``validate_target`` accept only a dir with ``public/data``
    + ``public/images`` (a Next build-output ``out/`` dir alone is rejected). The
    production target is the sibling ``ndif-website`` project.
  * ``apply`` BACKS UP the existing destination JSON files into
    ``out/backups/<name>.<timestamp>.bak.json`` before overwriting.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

# Slim output filenames (identical at source and destination).
_PAPERS_JSON = "research-papers.json"
_REPOS_JSON = "github-repos.json"


# ---------------------------------------------------------------------------
# Target detection / validation
# ---------------------------------------------------------------------------

def validate_target(path: Path) -> bool:
    """Return True iff *path* is a usable publish target.

    Requires:
      * the path exists and is a directory,
      * it contains ``public/data/`` AND ``public/images/`` (a Next build-output
        ``out/`` dir alone is therefore not accepted).
    """
    path = Path(path)
    if not path.is_dir():
        return False
    if not (path / "public" / "data").is_dir():
        return False
    if not (path / "public" / "images").is_dir():
        return False
    return True


def detect_target(start: Path | None = None) -> Path | None:
    """Auto-detect the sibling ``ndif-website`` publish target (production ndif.us).

    Searches from the ndif-citations project root and its PARENT (the ``web-dev``
    workspace) for an ``ndif-website`` dir that passes ``validate_target`` (has
    ``public/data`` + ``public/images``; a Next ``out/`` build dir alone is not
    accepted). Returns the ``ndif-website`` dir or ``None``.
    """
    if start is None:
        # ndif-citations project root = parents[2] of this file
        # (src/ndif_citations/publish.py → .../ndif-citations).
        start = Path(__file__).resolve().parents[2]
    start = Path(start).resolve()

    # Search the project root itself and its parent (the web-dev workspace),
    # which is where ndif-website lives as a sibling.
    search_roots = [start, start.parent]
    for root in search_roots:
        candidate = root / "ndif-website"
        if validate_target(candidate):
            return candidate.resolve()
    return None


# ---------------------------------------------------------------------------
# Loading / keys
# ---------------------------------------------------------------------------

def _require_output(out: Path) -> None:
    """Raise a clear error if the slim papers output is missing."""
    src = Path(out) / _PAPERS_JSON
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found — run the pipeline first "
            f"(no slim output to publish)."
        )


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


def _image_filename(image_ref: str | None) -> str | None:
    """Strip a ``/images/`` (or ``images/``) prefix → bare filename."""
    if not image_ref:
        return None
    name = image_ref
    for prefix in ("/images/", "images/"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name or None


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def _diff_list(out_items: list[dict], target_items: list[dict], key_fn) -> dict:
    """Return {added, changed, removed} comparing out vs target by *key_fn*."""
    out_by_key = {key_fn(x): x for x in out_items}
    tgt_by_key = {key_fn(x): x for x in target_items}

    added = [out_by_key[k] for k in out_by_key if k not in tgt_by_key]
    removed = [tgt_by_key[k] for k in tgt_by_key if k not in out_by_key]
    changed = [
        out_by_key[k]
        for k in out_by_key
        if k in tgt_by_key and out_by_key[k] != tgt_by_key[k]
    ]
    return {"added": added, "changed": changed, "removed": removed}


def diff(out: Path, target: Path) -> dict:
    """Compare the slim outputs in *out* against the live files in *target*.

    Returns::

        {
          "papers": {"added": [...], "changed": [...], "removed": [...]},
          "repos":  {"added": [...], "changed": [...], "removed": [...]},
          "images": {"new": [filename, ...], "changed": [filename, ...]},
        }

    Papers are keyed by url (fallback title); repos by ``owner/repo`` (fallback
    url); changed = JSON inequality. Images are the ones REFERENCED by the slim
    papers JSON: ``new`` = absent at the target; ``changed`` = present but with
    different bytes.

    Raises ``FileNotFoundError`` if ``out/research-papers.json`` is missing.
    """
    out = Path(out)
    target = Path(target)
    _require_output(out)

    out_papers = _load_json_list(out / _PAPERS_JSON)
    out_repos = _load_json_list(out / _REPOS_JSON)
    tgt_papers = _load_json_list(target / "public" / "data" / _PAPERS_JSON)
    tgt_repos = _load_json_list(target / "public" / "data" / _REPOS_JSON)

    papers_diff = _diff_list(out_papers, tgt_papers, _paper_key)
    repos_diff = _diff_list(out_repos, tgt_repos, _repo_key)

    out_images_dir = out / "images"
    tgt_images_dir = target / "public" / "images"

    images_new: list[str] = []
    images_changed: list[str] = []
    seen: set[str] = set()
    for p in out_papers:
        fname = _image_filename(p.get("image"))
        if not fname or fname in seen:
            continue
        seen.add(fname)
        src = out_images_dir / fname
        if not src.exists():
            # Referenced but not present in the pipeline output — nothing to copy.
            continue
        dest = tgt_images_dir / fname
        if not dest.exists():
            images_new.append(fname)
        elif dest.read_bytes() != src.read_bytes():
            images_changed.append(fname)

    return {
        "papers": papers_diff,
        "repos": repos_diff,
        "images": {"new": images_new, "changed": images_changed},
    }


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def _backup(src: Path, backups_dir: Path, timestamp: str) -> Path | None:
    """Copy *src* → ``backups_dir/<name>.<timestamp>.bak.json`` if it exists."""
    if not src.exists():
        return None
    backups_dir.mkdir(parents=True, exist_ok=True)
    dest = backups_dir / f"{src.name}.{timestamp}.bak.json"
    shutil.copy2(src, dest)
    return dest


def apply(out: Path, target: Path, *, papers: bool = True, repos: bool = True) -> dict:
    """Publish the slim outputs in *out* to *target*.

    ``papers`` / ``repos`` scope the publish: ``papers`` controls
    ``research-papers.json`` **and** the referenced-image sync (images are owned
    by papers); ``repos`` controls ``github-repos.json``. Only the selected
    sections are backed up and written. Defaults publish everything.

    Steps:
      1. Back up the existing destination ``research-papers.json`` &
         ``github-repos.json`` (if present) into
         ``out/backups/<name>.<timestamp>.bak.json``.
      2. Copy ``out/research-papers.json`` & ``out/github-repos.json`` into
         ``target/public/data/`` (written with the same formatting the .mjs
         scripts produced: ``indent=2`` + trailing newline).
      3. Copy every image REFERENCED by the slim papers JSON from
         ``out/images/`` into ``target/public/images/``, FORCE-OVERWRITING when
         the bytes differ (the additive-only bug fix). Identical images are
         left untouched and not counted as overwrites.

    Returns a summary dict::

        {
          "files_written": [<dest path str>, ...],
          "images_copied": int,       # newly-created image files
          "images_overwritten": int,  # changed bytes → force-overwritten
          "images_unchanged": int,
          "images_missing": int,      # referenced but absent in out/images/
          "backups": [<backup path str>, ...],
        }

    Raises ``FileNotFoundError`` if ``out/research-papers.json`` is missing.
    """
    out = Path(out)
    target = Path(target)
    _require_output(out)

    data_dir = target / "public" / "data"
    images_dir = target / "public" / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    backups_dir = out / "backups"
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    summary: dict = {
        "files_written": [],
        "images_copied": 0,
        "images_overwritten": 0,
        "images_unchanged": 0,
        "images_missing": 0,
        "backups": [],
    }

    # 1 + 2: back up then copy each selected JSON (matching .mjs output formatting).
    _scoped = {_PAPERS_JSON: papers, _REPOS_JSON: repos}
    for name in (_PAPERS_JSON, _REPOS_JSON):
        if not _scoped[name]:
            continue
        src = out / name
        if not src.exists():
            continue
        dest = data_dir / name
        backup = _backup(dest, backups_dir, timestamp)
        if backup is not None:
            summary["backups"].append(str(backup))
        # Re-serialize through json to normalize formatting exactly like the
        # .mjs scripts (indent=2 + trailing "\n").
        payload = _load_json_list(src)
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        summary["files_written"].append(str(dest))

    # 3: copy referenced images with FORCE-OVERWRITE on byte difference (papers scope only).
    if not papers:
        return summary
    out_papers = _load_json_list(out / _PAPERS_JSON)
    seen: set[str] = set()
    for p in out_papers:
        fname = _image_filename(p.get("image"))
        if not fname or fname in seen:
            continue
        seen.add(fname)
        src = out / "images" / fname
        if not src.exists():
            summary["images_missing"] += 1
            continue
        dest = images_dir / fname
        if not dest.exists():
            shutil.copy2(src, dest)
            summary["images_copied"] += 1
        elif dest.read_bytes() != src.read_bytes():
            shutil.copy2(src, dest)  # FORCE-OVERWRITE — the bug fix.
            summary["images_overwritten"] += 1
        else:
            summary["images_unchanged"] += 1

    return summary
