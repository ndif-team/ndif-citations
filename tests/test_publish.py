"""publish.preview: local slim JSON vs the last commit (spec §5.7)."""
import json
import shutil
import subprocess

import pytest

from ndif_citations import publish

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _repo(tmp_path, papers):
    subprocess.run(["git", "init", "-b", "main", str(tmp_path)], check=True, capture_output=True)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(tmp_path), "config", k, v], check=True)
    out = tmp_path / "output"
    (out / "images").mkdir(parents=True)
    (out / "research-papers.json").write_text(json.dumps(papers))
    (out / "github-repos.json").write_text(json.dumps([{"owner": "o", "repo": "r", "stars": 1}]))
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "seed"], check=True, capture_output=True)
    return out


def test_preview_against_head(tmp_path):
    out = _repo(tmp_path, [{"title": "A", "url": "u1"}, {"title": "B", "url": "u2"}])
    (out / "research-papers.json").write_text(json.dumps([{"title": "A2", "url": "u1"}, {"title": "C", "url": "u3"}]))
    (out / "github-repos.json").write_text(json.dumps([{"owner": "o", "repo": "r", "stars": 2}]))
    (out / "images" / "C.png").write_bytes(b"png")
    diff = publish.preview(out, tmp_path)
    assert [p["title"] for p in diff["papers"]["added"]] == ["C"]
    assert [p["title"] for p in diff["papers"]["removed"]] == ["B"]
    assert len(diff["papers"]["changed"]) == 1
    assert diff["images"]["new"] == ["C.png"]
    assert [r["_changed_fields"] for r in diff["repos"]["changed"]] == [["stars"]]


def test_preview_outside_git_treats_everything_as_added(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    (out / "research-papers.json").write_text(json.dumps([{"title": "A", "url": "u1"}]))
    (out / "github-repos.json").write_text("[]")
    assert len(publish.preview(out, tmp_path)["papers"]["added"]) == 1


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def _clone(bare, path):
    subprocess.run(["git", "clone", str(bare), str(path)], check=True, capture_output=True)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        _git(path, "config", k, v)
    return path / "output"


@pytest.fixture()
def pushed(tmp_path):
    """bare remote + a clone whose seed commit (paper A, two images) is on origin/main."""
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(bare)], check=True, capture_output=True)
    seed = tmp_path / "seed"
    out = _repo(seed, [{"title": "A", "url": "u1"}])
    (out / "images" / "Old Fig.png").write_bytes(b"old")
    (out / "images" / "Gone.png").write_bytes(b"gone")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-m", "images")
    _git(seed, "push", str(bare), "main")
    return bare


def test_preview_includes_committed_but_unpushed_changes(pushed, tmp_path):
    out = _clone(pushed, tmp_path / "c")
    (out / "research-papers.json").write_text(json.dumps([{"title": "A", "url": "u1"}, {"title": "B", "url": "u2"}]))
    _git(out.parent, "commit", "-am", "offline curation")  # committed, never pushed
    assert [p["title"] for p in publish.preview(out, out.parent)["papers"]["added"]] == ["B"]


def test_preview_ignores_bot_commits_not_yet_pulled(pushed, tmp_path):
    bot, cur = _clone(pushed, tmp_path / "bot"), _clone(pushed, tmp_path / "cur")
    (bot / "research-papers.json").write_text(json.dumps([{"title": "A", "url": "u1"}, {"title": "Bot", "url": "u9"}]))
    _git(bot.parent, "commit", "-am", "bot run")
    _git(bot.parent, "push", "origin", "main")
    _git(cur.parent, "fetch", "origin")  # origin/main moved past the curator's base
    (cur / "research-papers.json").write_text(json.dumps([{"title": "A", "url": "u1"}, {"title": "C", "url": "u3"}]))
    diff = publish.preview(cur, cur.parent)
    assert [p["title"] for p in diff["papers"]["added"]] == ["C"]
    assert diff["papers"]["removed"] == []  # the bot's paper is not something this push removes


def test_preview_images_vs_base(pushed, tmp_path):
    out = _clone(pushed, tmp_path / "c")
    img = out / "images"
    (img / "Fig one.png").write_bytes(b"new")
    _git(out.parent, "add", "output/images/Fig one.png")
    _git(out.parent, "commit", "-m", "offline thumbnail")         # committed, unpushed
    (img / "Old Fig.png").write_bytes(b"re-rendered")             # tracked, modified, uncommitted
    (img / 'New "Café" fig.png').write_bytes(b"new")             # untracked; quote + non-ASCII
    (img / "Gone.png").unlink()                                   # deletions are not listed (site images are additive)
    images = publish.preview(out, out.parent)["images"]
    assert images == {"new": ["Fig one.png", 'New "Café" fig.png'], "changed": ["Old Fig.png"]}
