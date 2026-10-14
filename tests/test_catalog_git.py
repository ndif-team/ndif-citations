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


def remote_images(bare) -> set[str]:
    out = subprocess.run(["git", "-C", str(bare), "ls-tree", "-r", "-z", "--name-only", "main", "--", "output/images"],
                         check=True, capture_output=True, text=True).stdout
    return {f for f in out.split("\0") if f}


def _buckets_with(data: dict, arxiv_id: str) -> list[str]:
    return [b for b in data if any(i["arxiv_id"] == arxiv_id for i in data[b])]


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
    bare, a, b, seed = remote_and_clones
    seed_b = [p.model_copy(deep=True) for p in seed]
    _move(a, seed, "2401.00002", Bucket.VERIFIED)
    catalog_git.push_catalog(a, "bot publishes 2")
    bot_sha = git(bare, "rev-parse", "main")
    _move(b, seed_b, "2401.00002", Bucket.DISCARDED)
    with pytest.raises(catalog_git.CatalogPushError, match="catalog-conflict-") as exc:
        catalog_git.push_catalog(b, "curator discards 2")
    assert "new duplicate keys" in caplog.text
    conflict = [br.strip() for br in git(b, "branch").splitlines() if br.strip().startswith("catalog-conflict-")]
    assert len(conflict) == 1
    msg = str(exc.value)  # ALL unpushed commits moved, not just the conflicting edit
    assert f"ALL your unpushed catalog commits were moved to branch {conflict[0]!r}" in msg
    assert "local main now matches origin/main" in msg and "re-apply all of those edits" in msg
    assert "ask a maintainer to merge the branch" in msg
    assert git(b, "rev-parse", "HEAD") == git(b, "rev-parse", "origin/main")  # local main reset to remote
    # nothing was pushed: the remote still holds exactly the bot's commit and catalog
    assert git(bare, "rev-parse", "main") == bot_sha
    remote_cat = json.loads(git(bare, "show", "main:output/research-papers-full.json"))
    assert _buckets_with(remote_cat, "2401.00002") == ["verified"]
    saved = json.loads(git(b, "show", f"{conflict[0]}:output/research-papers-full.json"))
    assert _buckets_with(saved, "2401.00002") == ["discarded"]  # the curator's edit is kept on the branch


def test_identical_edit_already_upstream_is_not_repushed(remote_and_clones):
    """B's commit becomes empty on rebase (same edit already upstream): nothing to push, and
    the already-published upstream commit must never be amended."""
    bare, a, b, seed = remote_and_clones
    seed_b = [p.model_copy(deep=True) for p in seed]
    _edit_venue(a, seed, "2401.00001", "ICML 2026")
    catalog_git.push_catalog(a, "bot")
    bot_sha = git(bare, "rev-parse", "main")
    _edit_venue(b, seed_b, "2401.00001", "ICML 2026")
    res = catalog_git.push_catalog(b, "curator")
    assert res["pushed"] is False and res["conflict_branch"] is None
    assert git(b, "rev-parse", "HEAD") == git(b, "rev-parse", "origin/main") == bot_sha
    assert git(bare, "rev-parse", "main") == bot_sha  # no extra commit, no rewritten history


def test_thumbnail_conflict_with_space_and_non_ascii_name_auto_resolves(remote_and_clones):
    """Real thumbnail names have spaces and non-ASCII chars (git quotes those by default)."""
    bare, a, b, seed = remote_and_clones
    img = "output/images/Trigonometry for Café Models.png"
    seed[0].image = "/images/Trigonometry for Café Models.png"  # only referenced images are staged
    write_catalog(a, seed)
    (a / img).write_bytes(b"\x89PNG\x00seed")
    catalog_git.push_catalog(a, "add thumbnail")
    assert "output/images/Trigonometry for Café Models.png" in remote_images(bare)
    git(b, "pull", "--ff-only")
    (a / img).write_bytes(b"\x89PNG\x00bot")
    catalog_git.push_catalog(a, "bot re-renders thumbnail")
    (b / img).write_bytes(b"\x89PNG\x00curator")
    res = catalog_git.push_catalog(b, "curator re-renders thumbnail")
    assert res["pushed"] is True and res["conflict_branch"] is None
    assert not any("catalog-conflict-" in br for br in git(b, "branch").splitlines())
    assert git(bare, "rev-parse", "main") == git(b, "rev-parse", "HEAD")
    keys = read_keys(b)
    assert sorted(sum(keys.values(), [])) == sorted(p.arxiv_id for p in seed)  # every paper exactly once


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


@pytest.mark.parametrize("damage", ["delete", "truncate"])
def test_refuses_missing_or_corrupt_full_catalog(remote_and_clones, damage):
    bare, a, _, _ = remote_and_clones
    full = a / catalog_git.PAPERS_FULL
    if damage == "delete":
        full.unlink()
    else:
        full.write_text(full.read_text()[:40])
    remote_before, head_before = git(bare, "rev-parse", "main"), git(a, "rev-parse", "HEAD")
    with pytest.raises(catalog_git.CatalogPushError, match="research-papers-full.json"):
        catalog_git.push_catalog(a, "x")
    assert git(bare, "rev-parse", "main") == remote_before and git(a, "rev-parse", "HEAD") == head_before


def _grow_to_twelve(a, seed):
    papers = seed + [make_paper(title=f"Extra {i}", arxiv_id=f"2402.0000{i}", bucket=Bucket.PENDING) for i in range(6)]
    write_catalog(a, papers)
    assert catalog_git.push_catalog(a, "grow")["pushed"] is True
    return papers


@pytest.mark.parametrize("keep", [lambda ps: ps[6:], lambda ps: []], ids=["six-deleted", "empty-but-valid"])
def test_shrink_guard_refuses_a_gutted_catalog(remote_and_clones, keep):
    bare, a, _, seed = remote_and_clones
    papers = _grow_to_twelve(a, seed)
    write_catalog(a, keep(papers))
    remote_before = git(bare, "rev-parse", "main")
    with pytest.raises(catalog_git.CatalogPushError, match="would lose"):
        catalog_git.push_catalog(a, "gut")
    assert git(bare, "rev-parse", "main") == remote_before


def test_shrink_guard_allows_a_single_deletion(remote_and_clones):
    """A manual_add discard deletes outright; one lost entry is normal."""
    _, a, _, seed = remote_and_clones
    papers = _grow_to_twelve(a, seed)
    write_catalog(a, papers[1:])
    assert catalog_git.push_catalog(a, "drop one")["pushed"] is True


def test_shrink_guard_allows_enrichment_that_rekeys_papers(remote_and_clones):
    """Enrichment moves a paper's merge key from its title to an arXiv id: not a lost paper."""
    _, a, _, seed = remote_and_clones
    titled = [make_paper(title=f"Untitled  Work {i}", arxiv_id=None, bucket=Bucket.PENDING) for i in range(30)]
    write_catalog(a, seed + titled)
    assert catalog_git.push_catalog(a, "30 title-keyed papers")["pushed"] is True
    for i, p in enumerate(titled):
        p.arxiv_id = f"2403.{i:05d}"
    write_catalog(a, seed + titled)
    assert catalog_git.push_catalog(a, "enrich: arXiv ids")["pushed"] is True


def test_push_stages_only_referenced_images(remote_and_clones):
    bare, a, _, seed = remote_and_clones
    name = "言語 Fig [1].png"  # non-ASCII + space + glob chars: must be staged literally
    seed[0].image = f"/images/{name}"
    write_catalog(a, seed)
    (a / "output" / "images" / name).write_bytes(b"\x89PNG\x00ref")
    (a / "output" / "images" / "言語 Fig 1.png").write_bytes(b"\x89PNG\x00glob-twin")  # a glob [1] would match this
    (a / "output" / "images" / "Stray.png").write_bytes(b"\x89PNG\x00stray")  # not referenced by any entry
    assert catalog_git.push_catalog(a, "thumbnail")["pushed"] is True
    assert remote_images(bare) == {f"output/images/{name}"}
    assert (a / "output" / "images" / "Stray.png").exists()  # left alone, just never committed


def test_deleted_tracked_image_is_pushed_as_a_deletion(remote_and_clones):
    bare, a, _, seed = remote_and_clones
    seed[0].image = "/images/Old.png"
    write_catalog(a, seed)
    (a / "output" / "images" / "Old.png").write_bytes(b"\x89PNG\x00old")
    catalog_git.push_catalog(a, "thumbnail")
    assert remote_images(bare) == {"output/images/Old.png"}
    (a / "output" / "images" / "Old.png").unlink()
    seed[0].image = None
    write_catalog(a, seed)
    assert catalog_git.push_catalog(a, "drop thumbnail")["pushed"] is True
    assert remote_images(bare) == set()


def test_stray_untracked_image_does_not_block_pull(remote_and_clones):
    bare, a, b, seed = remote_and_clones
    (b / "output" / "images").mkdir(parents=True, exist_ok=True)
    (b / "output" / "images" / "Stray.png").write_bytes(b"\x89PNG\x00stray")
    _edit_venue(a, seed, "2401.00001", "ICML 2026")
    catalog_git.push_catalog(a, "bot")
    assert catalog_git.pull_if_clean(b) == "catalog up to date"
    assert git(b, "rev-parse", "HEAD") == git(bare, "rev-parse", "main")


@pytest.mark.parametrize("preset", [None, "ssh -i /k/deploy"])
def test_git_never_prompts_for_ssh_or_credentials(monkeypatch, tmp_path, preset):
    """An ssh remote with no agent/key must fail fast, not hang the app on a passphrase prompt."""
    if preset is None:
        monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)
    else:
        monkeypatch.setenv("GIT_SSH_COMMAND", preset)
    seen = {}
    monkeypatch.setattr(catalog_git.subprocess, "run", lambda *a, **kw: seen.update(kw["env"]))
    catalog_git._git(tmp_path, "status")
    assert seen["GIT_TERMINAL_PROMPT"] == "0"
    assert seen["GIT_SSH_COMMAND"] == (preset or "ssh -o BatchMode=yes")  # a user's own setting wins
