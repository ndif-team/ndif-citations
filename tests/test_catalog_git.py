"""catalog_git against real temp git repos with a local bare remote (spec §5.2)."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ndif_citations import catalog_git
from ndif_citations.models import Bucket
from tests.conftest import make_paper

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def write_catalog(repo: Path, papers) -> None:
    out = repo / "output"
    (out / "images").mkdir(parents=True, exist_ok=True)
    data = {"pending": [], "verified": [], "discarded": []}
    for p in papers:
        data[p.bucket.value].append(p.to_full_dict())
    (out / "research-papers-full.json").write_text(json.dumps(data, indent=2, default=str))
    (out / "research-papers.json").write_text(json.dumps(
        [p.to_website_dict() for p in papers if p.bucket == Bucket.VERIFIED], indent=2))
    (out / "github-repos-full.json").write_text("[]")
    (out / "github-repos.json").write_text("[]")


def read_keys(repo: Path) -> dict[str, list[str]]:
    data = json.loads((repo / "output" / "research-papers-full.json").read_text())
    return {b: sorted(i["arxiv_id"] for i in data[b]) for b in data}


@pytest.fixture()
def remote_and_clones(tmp_path):
    """bare remote + clone A (bot) + clone B (curator), both on main with one seed commit."""
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(bare)], check=True, capture_output=True)
    a = tmp_path / "a"
    subprocess.run(["git", "clone", str(bare), str(a)], check=True, capture_output=True)
    for repo in (a,):
        git(repo, "config", "user.email", "t@t")
        git(repo, "config", "user.name", "t")
    seed = [make_paper(title="Verified V", arxiv_id="2401.00008", bucket=Bucket.VERIFIED),
            make_paper(title="Discarded D", arxiv_id="2401.00009", bucket=Bucket.DISCARDED)]
    seed += [make_paper(title=f"Paper {i}", arxiv_id=f"2401.0000{i}", bucket=Bucket.PENDING) for i in range(4)]
    write_catalog(a, seed)
    git(a, "add", "-A")
    git(a, "commit", "-m", "seed")
    git(a, "push", "origin", "main")
    b = tmp_path / "b"
    subprocess.run(["git", "clone", str(bare), str(b)], check=True, capture_output=True)
    git(b, "config", "user.email", "t@t")
    git(b, "config", "user.name", "t")
    return bare, a, b, seed


def _move(repo, papers, arxiv_id, bucket):
    for p in papers:
        if p.arxiv_id == arxiv_id:
            p.bucket = bucket
    write_catalog(repo, papers)


def _edit_venue(repo, papers, arxiv_id, venue):
    """In-place field edit: touches only that paper's lines, so races on DIFFERENT papers
    merge cleanly as text (bucket moves near empty arrays would conflict on adjacent lines)."""
    for p in papers:
        if p.arxiv_id == arxiv_id:
            p.venue = venue
    write_catalog(repo, papers)


def test_clean_push(remote_and_clones):
    _, a, _, seed = remote_and_clones
    _move(a, seed, "2401.00001", Bucket.VERIFIED)
    res = catalog_git.push_catalog(a, "data: verify 1")
    assert res["pushed"] is True and res["conflict_branch"] is None


def test_nothing_to_push(remote_and_clones):
    _, a, _, _ = remote_and_clones
    assert catalog_git.push_catalog(a, "noop")["pushed"] is False


def test_race_on_different_papers_merges(remote_and_clones):
    _, a, b, seed = remote_and_clones
    seed_b = [p.model_copy(deep=True) for p in seed]
    _edit_venue(a, seed, "2401.00000", "ICLR 2026")
    catalog_git.push_catalog(a, "bot")
    _edit_venue(b, seed_b, "2401.00003", "NeurIPS 2025")
    res = catalog_git.push_catalog(b, "curator")
    assert res["pushed"] is True
    data = json.loads((b / "output" / "research-papers-full.json").read_text())
    venues = {i["arxiv_id"]: i["venue"] for i in data["pending"]}
    assert venues["2401.00000"] == "ICLR 2026" and venues["2401.00003"] == "NeurIPS 2025"


def test_same_paper_two_buckets_leaves_conflict_branch(remote_and_clones, caplog):
    """Non-adjacent hunks (verified and discarded both non-empty) → git merges the TEXT
    cleanly into a paper in two buckets; only the validator can catch it."""
    _, a, b, seed = remote_and_clones
    seed_b = [p.model_copy(deep=True) for p in seed]
    _move(a, seed, "2401.00002", Bucket.VERIFIED)
    catalog_git.push_catalog(a, "bot publishes 2")
    _move(b, seed_b, "2401.00002", Bucket.DISCARDED)
    with pytest.raises(catalog_git.CatalogPushError, match="catalog-conflict-"):
        catalog_git.push_catalog(b, "curator discards 2")
    assert "new duplicate keys" in caplog.text
    assert any(br.strip().startswith("catalog-conflict-") for br in git(b, "branch").splitlines())
    assert git(b, "rev-parse", "HEAD") == git(b, "rev-parse", "origin/main")  # local main reset to remote


def test_unpushed_commit_is_pushed_next_time(remote_and_clones):
    bare, a, _, seed = remote_and_clones
    _edit_venue(a, seed, "2401.00001", "ICML 2026")
    git(a, "add", "-A")
    git(a, "commit", "-m", "committed while offline")   # never pushed
    res = catalog_git.push_catalog(a, "nothing new staged")
    assert res["pushed"] is True
    assert git(a, "rev-parse", "HEAD") == git(a, "rev-parse", "origin/main")


def test_refuses_when_other_tracked_files_are_dirty(remote_and_clones):
    _, a, _, seed = remote_and_clones
    (a / "README.md").write_text("tracked")
    git(a, "add", "README.md")
    git(a, "commit", "-m", "readme")
    (a / "README.md").write_text("local edit")
    _edit_venue(a, seed, "2401.00001", "ICML 2026")
    with pytest.raises(catalog_git.CatalogPushError, match="other changes"):
        catalog_git.push_catalog(a, "x")
    assert (a / "README.md").read_text() == "local edit"


def test_refuses_unpushed_non_catalog_commit(remote_and_clones):
    bare, a, _, seed = remote_and_clones
    (a / "README.md").write_text("unreviewed code")
    git(a, "add", "README.md")
    git(a, "commit", "-m", "code change")       # clean tree, but a non-catalog commit is unpushed
    _edit_venue(a, seed, "2401.00001", "ICML 2026")
    before = git(bare, "rev-parse", "main")
    with pytest.raises(catalog_git.CatalogPushError, match="non-catalog commits"):
        catalog_git.push_catalog(a, "x")
    assert git(bare, "rev-parse", "main") == before


def test_deletion_on_one_side_is_respected(remote_and_clones):
    _, a, b, seed = remote_and_clones
    seed_b = [p.model_copy(deep=True) for p in seed]
    write_catalog(a, [p for p in seed if p.arxiv_id != "2401.00000"])  # bot deletes paper 0
    catalog_git.push_catalog(a, "delete 0")
    _edit_venue(b, seed_b, "2401.00003", "ACL 2025")
    assert catalog_git.push_catalog(b, "edit 3")["pushed"] is True
    keys = read_keys(b)
    assert "2401.00000" not in sum(keys.values(), []) and "2401.00003" in keys["pending"]


def test_refuses_off_main(remote_and_clones):
    _, a, _, seed = remote_and_clones
    git(a, "checkout", "-b", "other")
    _move(a, seed, "2401.00001", Bucket.VERIFIED)
    with pytest.raises(catalog_git.CatalogPushError, match="branch 'other'"):
        catalog_git.push_catalog(a, "x")


def test_validate_catalog_rules():
    p = lambda k, b="pending": {**make_paper(arxiv_id=k).to_full_dict(), "bucket": b}  # noqa: E731
    cat = lambda *items: {"pending": [i for i in items if i["bucket"] == "pending"],  # noqa: E731
                          "verified": [i for i in items if i["bucket"] == "verified"], "discarded": []}
    base = cat(p("1"), p("2"))
    ours = cat(p("1"), p("2"), p("3"))          # we added 3
    theirs = cat(p("2"))                         # they deleted 1
    assert catalog_git.validate_catalog(base, ours, theirs, cat(p("2"), p("3"))) == []
    assert catalog_git.validate_catalog(base, ours, theirs, cat(p("1"), p("2"), p("3")))  # resurrected 1
    dup = cat(p("2"), p("2", "verified"), p("3"))
    assert any("duplicate" in m for m in catalog_git.validate_catalog(base, ours, theirs, dup))
    pre_dup = cat(p("2"), p("2", "verified"))   # duplicate already in base → tolerated
    assert catalog_git.validate_catalog(pre_dup, pre_dup, pre_dup, pre_dup) == []
