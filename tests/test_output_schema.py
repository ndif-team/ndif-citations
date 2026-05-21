"""Schema and behavioral tests for the GitHub pipeline revamp (2026-05-20)."""
from __future__ import annotations

from datetime import date
from ndif_citations.models import DiscoveredRepo


def test_discoveredrepo_persists_first_and_last_seen():
    r = DiscoveredRepo(
        owner="o", repo="r", url="https://github.com/o/r",
        first_seen="2025-01-15", last_seen="2026-05-20",
    )
    dumped = r.to_full_dict()
    assert dumped["first_seen"] == "2025-01-15"
    assert dumped["last_seen"] == "2026-05-20"


SLIM_KEYS = {
    "owner", "repo", "url", "description", "stars", "forks",
    "last_commit", "language", "linked_paper_url",
    "is_course", "is_fork", "parent_full_name",
}


def test_to_website_dict_emits_exactly_the_twelve_slim_keys():
    r = DiscoveredRepo(
        owner="o", repo="r", url="https://github.com/o/r",
        description="hello", stars=10, forks=2,
        last_commit=date(2026, 1, 1), language="Python",
        linked_paper_url="https://arxiv.org/abs/1234.5678",
        repo_type="course",
        is_fork=True, parent_full_name="upstream/parent",
    )
    slim = r.to_website_dict()
    assert set(slim.keys()) == SLIM_KEYS
    assert slim["is_course"] is True
    assert slim["is_fork"] is True
    assert slim["parent_full_name"] == "upstream/parent"
    assert slim["last_commit"] == "2026-01-01"


def test_to_website_dict_is_course_false_for_research():
    r = DiscoveredRepo(owner="o", repo="r", url="https://github.com/o/r", repo_type="research")
    assert r.to_website_dict()["is_course"] is False


def test_to_website_dict_is_course_false_for_experiment():
    r = DiscoveredRepo(owner="o", repo="r", url="https://github.com/o/r", repo_type="experiment")
    assert r.to_website_dict()["is_course"] is False
