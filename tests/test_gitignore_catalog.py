"""The catalog is tracked; PDFs and scratch output are not (spec §5.1)."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def ignored(path: str) -> bool:
    return subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", path]).returncode == 0


@pytest.mark.parametrize("path", [
    "output/research-papers-full.json", "output/research-papers.json",
    "output/github-repos-full.json", "output/github-repos.json", "output/images/X.png",
])
def test_catalog_paths_tracked(path):
    assert not ignored(path)


@pytest.mark.parametrize("path", [
    "output/pdfs/arxiv-1.pdf", "output/backups/x.json", "output/runs/r.json",
    "output/raw/s2.json", "output/research-papers.csv", "output/research-data.xlsx",
])
def test_scratch_paths_ignored(path):
    assert ignored(path)
