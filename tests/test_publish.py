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
    (out / "github-repos.json").write_text("[]")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "seed"], check=True, capture_output=True)
    return out


def test_preview_against_head(tmp_path):
    out = _repo(tmp_path, [{"title": "A", "url": "u1"}, {"title": "B", "url": "u2"}])
    (out / "research-papers.json").write_text(json.dumps([{"title": "A2", "url": "u1"}, {"title": "C", "url": "u3"}]))
    (out / "images" / "C.png").write_bytes(b"png")
    diff = publish.preview(out, tmp_path)
    assert [p["title"] for p in diff["papers"]["added"]] == ["C"]
    assert [p["title"] for p in diff["papers"]["removed"]] == ["B"]
    assert len(diff["papers"]["changed"]) == 1
    assert diff["images"]["new"] == ["C.png"]


def test_preview_outside_git_treats_everything_as_added(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    (out / "research-papers.json").write_text(json.dumps([{"title": "A", "url": "u1"}]))
    (out / "github-repos.json").write_text("[]")
    assert len(publish.preview(out, tmp_path)["papers"]["added"]) == 1
