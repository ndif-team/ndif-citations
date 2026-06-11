"""TDD tests for publish.py — Task 5.1 (publish curated outputs to the site).

These tests build a FAKE site target under tmp_path and a FAKE pipeline output
dir (the ``out`` arg) so they NEVER touch the real ``../ndif-website``.

Layout built per test::

    tmp/ndif-website/public/data/research-papers.json
    tmp/ndif-website/public/data/github-repos.json
    tmp/ndif-website/public/images/<slug>.png

    tmp/out/research-papers.json   (slim — list of to_website_dict)
    tmp/out/github-repos.json      (slim — list of to_website_dict)
    tmp/out/images/<slug>.png
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ndif_citations import publish


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(root: Path) -> Path:
    """Build a valid fake ndif-website target dir; return its Path."""
    site = root / "ndif-website"
    (site / "public" / "data").mkdir(parents=True)
    (site / "public" / "images").mkdir(parents=True)
    return site


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _slim_paper(title: str, url: str, image: str | None = None, desc: str = "d") -> dict:
    p = {
        "title": title,
        "authors": ["A. Author"],
        "venue": "arXiv 2024",
        "year": 2024,
        "url": url,
        "description": desc,
        "category": "uses_nnsight",
    }
    if image:
        p["image"] = image
    return p


def _slim_repo(owner: str, repo: str, stars: int = 1) -> dict:
    return {
        "owner": owner,
        "repo": repo,
        "url": f"https://github.com/{owner}/{repo}",
        "description": "r",
        "stars": stars,
        "forks": 0,
        "last_commit": None,
        "language": "Python",
        "license": None,
        "topics": [],
        "archived": False,
        "category": "uses_nnsight",
        "linked_paper_url": None,
        "linked_paper_tier": None,
        "repo_type": "experiment",
        "parent_full_name": None,
    }


def _make_out(root: Path, papers: list[dict], repos: list[dict],
              images: dict[str, bytes] | None = None) -> Path:
    out = root / "out"
    (out / "images").mkdir(parents=True)
    _write_json(out / "research-papers.json", papers)
    _write_json(out / "github-repos.json", repos)
    for name, payload in (images or {}).items():
        (out / "images" / name).write_bytes(payload)
    return out


# ---------------------------------------------------------------------------
# detect_target / validate_target
# ---------------------------------------------------------------------------

def test_detect_target_finds_valid(tmp_path: Path):
    """A sibling ndif-website/ with public/data + public/images is found."""
    project_root = tmp_path / "ndif-citations"
    project_root.mkdir()
    site = _make_target(tmp_path)  # sibling of project_root

    found = publish.detect_target(start=project_root)
    assert found is not None
    assert found.resolve() == site.resolve()


def test_detect_finds_ndif_website(tmp_path: Path):
    """ndif-website/ is the production publish target — it must be detected."""
    project_root = tmp_path / "ndif-citations"
    project_root.mkdir()
    site = tmp_path / "ndif-website"
    (site / "public" / "data").mkdir(parents=True)
    (site / "public" / "images").mkdir(parents=True)

    found = publish.detect_target(start=project_root)
    assert found is not None and found.resolve() == site.resolve()


def test_detect_skips_build_out_dir(tmp_path: Path):
    """A site with only a Next build-output dir (out/data, no public/) is not selected."""
    project_root = tmp_path / "ndif-citations"
    project_root.mkdir()
    site = tmp_path / "ndif-website"
    (site / "out" / "data").mkdir(parents=True)
    (site / "out" / "images").mkdir(parents=True)
    # NO public/data or public/images → not a valid target.

    found = publish.detect_target(start=project_root)
    assert found is None, f"detect_target must not pick a build out/ dir, got {found}"


def test_validate_target(tmp_path: Path):
    site = _make_target(tmp_path)
    assert publish.validate_target(site) is True

    # Missing images/ → invalid.
    bad = tmp_path / "broken"
    (bad / "public" / "data").mkdir(parents=True)
    assert publish.validate_target(bad) is False

    # Nonexistent → invalid.
    assert publish.validate_target(tmp_path / "nope") is False


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def test_diff_reports_added_changed_removed(tmp_path: Path):
    """out has an added paper + a changed paper; target has a removed paper."""
    ndif_us = _make_target(tmp_path)

    # Target pre-existing papers: "keep" (changed), "gone" (removed).
    _write_json(
        ndif_us / "public" / "data" / "research-papers.json",
        [
            _slim_paper("Keep", "https://x/keep", desc="OLD"),
            _slim_paper("Gone", "https://x/gone"),
        ],
    )
    _write_json(ndif_us / "public" / "data" / "github-repos.json", [_slim_repo("o", "old")])

    # out: "keep" with new description (changed), "new" (added). "gone" absent (removed).
    out = _make_out(
        tmp_path,
        papers=[
            _slim_paper("Keep", "https://x/keep", desc="NEW"),
            _slim_paper("New", "https://x/new"),
        ],
        repos=[_slim_repo("o", "new")],
    )

    d = publish.diff(out, ndif_us)

    paper_added = {p["url"] for p in d["papers"]["added"]}
    paper_changed = {p["url"] for p in d["papers"]["changed"]}
    paper_removed = {p["url"] for p in d["papers"]["removed"]}
    assert paper_added == {"https://x/new"}
    assert paper_changed == {"https://x/keep"}
    assert paper_removed == {"https://x/gone"}

    repo_added = {r["repo"] for r in d["repos"]["added"]}
    repo_removed = {r["repo"] for r in d["repos"]["removed"]}
    assert repo_added == {"new"}
    assert repo_removed == {"old"}


def test_diff_reports_images_new_and_changed(tmp_path: Path):
    """images.new = referenced+absent at target; images.changed = different bytes."""
    ndif_us = _make_target(tmp_path)
    # Pre-existing target image "stale.png" with OLD bytes.
    (ndif_us / "public" / "images" / "stale.png").write_bytes(b"OLD-BYTES")
    _write_json(ndif_us / "public" / "data" / "research-papers.json", [])
    _write_json(ndif_us / "public" / "data" / "github-repos.json", [])

    out = _make_out(
        tmp_path,
        papers=[
            _slim_paper("S", "https://x/s", image="/images/stale.png"),
            _slim_paper("F", "https://x/f", image="/images/fresh.png"),
        ],
        repos=[],
        images={"stale.png": b"NEW-BYTES", "fresh.png": b"FRESH"},
    )

    d = publish.diff(out, ndif_us)
    assert "fresh.png" in d["images"]["new"]
    assert "stale.png" in d["images"]["changed"]
    assert "stale.png" not in d["images"]["new"]


def test_diff_missing_output_errors(tmp_path: Path):
    """Missing out/research-papers.json → a clear error telling the user to run first."""
    ndif_us = _make_target(tmp_path)
    out = tmp_path / "out"
    (out / "images").mkdir(parents=True)
    # No research-papers.json written.

    with pytest.raises(FileNotFoundError) as exc:
        publish.diff(out, ndif_us)
    assert "pipeline" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# apply — the critical force-overwrite + backup test
# ---------------------------------------------------------------------------

def test_apply_backs_up_and_copies(tmp_path: Path):
    """apply() backs up existing JSON, copies slim JSON, and FORCE-OVERWRITES a
    changed image (the additive-only bug fix)."""
    ndif_us = _make_target(tmp_path)

    # Pre-existing target JSON (will be backed up + replaced).
    _write_json(
        ndif_us / "public" / "data" / "research-papers.json",
        [_slim_paper("Old", "https://x/old")],
    )
    _write_json(ndif_us / "public" / "data" / "github-repos.json", [_slim_repo("o", "old")])

    # Pre-place a STALE image at the target with old bytes — this is the bug case.
    stale_target = ndif_us / "public" / "images" / "thumb.png"
    stale_target.write_bytes(b"STALE-OLD-THUMBNAIL")

    out = _make_out(
        tmp_path,
        papers=[_slim_paper("New", "https://x/new", image="/images/thumb.png")],
        repos=[_slim_repo("o", "new")],
        images={"thumb.png": b"FRESH-NEW-THUMBNAIL"},
    )

    summary = publish.apply(out, ndif_us)

    # 1. Target JSON updated to the out/ content.
    dest_papers = json.loads((ndif_us / "public" / "data" / "research-papers.json").read_text())
    assert [p["url"] for p in dest_papers] == ["https://x/new"]
    dest_repos = json.loads((ndif_us / "public" / "data" / "github-repos.json").read_text())
    assert [r["repo"] for r in dest_repos] == ["new"]

    # 2. Backups created in out/backups/ for BOTH json files.
    backups = list((out / "backups").glob("research-papers.json.*.bak.json"))
    assert len(backups) == 1, f"expected 1 papers backup, got {backups}"
    repo_backups = list((out / "backups").glob("github-repos.json.*.bak.json"))
    assert len(repo_backups) == 1, f"expected 1 repos backup, got {repo_backups}"
    # Backup preserves the OLD content.
    backed_up = json.loads(backups[0].read_text())
    assert [p["url"] for p in backed_up] == ["https://x/old"]

    # 3. CRITICAL: the stale image was FORCE-OVERWRITTEN with the fresh bytes.
    assert stale_target.read_bytes() == b"FRESH-NEW-THUMBNAIL", (
        "additive-only bug NOT fixed: stale image was left in place"
    )

    # Summary should report the overwrite.
    assert summary["images_overwritten"] >= 1


def test_apply_copies_new_image_and_skips_identical(tmp_path: Path):
    """New images are copied; identical images are not counted as overwrites."""
    ndif_us = _make_target(tmp_path)
    _write_json(ndif_us / "public" / "data" / "research-papers.json", [])
    _write_json(ndif_us / "public" / "data" / "github-repos.json", [])

    # identical.png already at target with SAME bytes; new.png absent.
    (ndif_us / "public" / "images" / "identical.png").write_bytes(b"SAME")

    out = _make_out(
        tmp_path,
        papers=[
            _slim_paper("I", "https://x/i", image="/images/identical.png"),
            _slim_paper("N", "https://x/n", image="/images/new.png"),
        ],
        repos=[],
        images={"identical.png": b"SAME", "new.png": b"NEWIMG"},
    )

    summary = publish.apply(out, ndif_us)

    assert (ndif_us / "public" / "images" / "new.png").read_bytes() == b"NEWIMG"
    assert summary["images_copied"] >= 1       # new.png
    assert summary["images_overwritten"] == 0  # identical.png unchanged → not overwritten


def test_apply_missing_output_errors(tmp_path: Path):
    ndif_us = _make_target(tmp_path)
    out = tmp_path / "out"
    (out / "images").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        publish.apply(out, ndif_us)


def test_apply_papers_only_skips_repos(tmp_path: Path):
    out = _make_out(tmp_path, [_slim_paper("P", "http://x")], [_slim_repo("o", "r")])
    target = _make_target(tmp_path)
    # seed the repos JSON with sentinel content the papers-only apply must NOT touch
    _write_json(target / "public" / "data" / "github-repos.json", [{"sentinel": True}])
    publish.apply(out, target, papers=True, repos=False)
    assert json.loads((target / "public" / "data" / "github-repos.json").read_text()) == [{"sentinel": True}]
    assert (target / "public" / "data" / "research-papers.json").exists()


def test_apply_repos_only_skips_papers(tmp_path: Path):
    out = _make_out(tmp_path, [_slim_paper("P", "http://x")], [_slim_repo("o", "r")])
    target = _make_target(tmp_path)
    # seed the papers JSON with sentinel content the repos-only apply must NOT touch
    _write_json(target / "public" / "data" / "research-papers.json", [{"sentinel": True}])
    publish.apply(out, target, papers=False, repos=True)
    assert json.loads((target / "public" / "data" / "research-papers.json").read_text()) == [{"sentinel": True}]
    assert (target / "public" / "data" / "github-repos.json").exists()


def test_detect_target_prefers_src_layout(tmp_path: Path):
    """ndif-website 2026-06-11 layout: app in src/ (source public/ inside),
    repo-root public/ = committed build output. Publishing must target
    src/, never the build output (which `make all` clobbers)."""
    site = tmp_path / "ndif-website"
    for base in (site / "src" / "public", site / "public"):
        (base / "data").mkdir(parents=True)
        (base / "images").mkdir(parents=True)
    found = publish.detect_target(start=tmp_path / "ndif-citations")
    assert found == (site / "src").resolve()


def test_detect_target_still_accepts_flat_layout(tmp_path: Path):
    site = tmp_path / "ndif-website"
    (site / "public" / "data").mkdir(parents=True)
    (site / "public" / "images").mkdir(parents=True)
    found = publish.detect_target(start=tmp_path / "ndif-citations")
    assert found == site.resolve()
