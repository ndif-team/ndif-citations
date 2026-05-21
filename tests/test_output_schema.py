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


# ---------------------------------------------------------------------------
# merge_repos stamping tests
# ---------------------------------------------------------------------------

from ndif_citations import output as output_module
from ndif_citations.output import merge_repos


def test_merge_repos_stamps_first_and_last_seen_on_new_repo(monkeypatch):
    monkeypatch.setattr(output_module, "_today", lambda: date(2026, 5, 20))
    discovered = [DiscoveredRepo(owner="o", repo="new", url="https://github.com/o/new")]
    merged = merge_repos(discovered=discovered, existing=[])
    assert len(merged) == 1
    assert merged[0].first_seen == "2026-05-20"
    assert merged[0].last_seen == "2026-05-20"


def test_merge_repos_preserves_first_seen_and_updates_last_seen_on_re_observed(monkeypatch):
    monkeypatch.setattr(output_module, "_today", lambda: date(2026, 5, 20))
    existing = [DiscoveredRepo(
        owner="o", repo="r", url="https://github.com/o/r",
        first_seen="2025-01-15", last_seen="2026-01-01",
    )]
    discovered = [DiscoveredRepo(owner="o", repo="r", url="https://github.com/o/r")]
    merged = merge_repos(discovered=discovered, existing=existing)
    target = next(m for m in merged if m.merge_key() == "o/r")
    assert target.first_seen == "2025-01-15"
    assert target.last_seen == "2026-05-20"


def test_merge_repos_backfills_first_seen_when_missing_on_existing(monkeypatch):
    monkeypatch.setattr(output_module, "_today", lambda: date(2026, 5, 20))
    existing = [DiscoveredRepo(owner="o", repo="r", url="https://github.com/o/r")]  # no first_seen
    discovered = [DiscoveredRepo(owner="o", repo="r", url="https://github.com/o/r")]
    merged = merge_repos(discovered=discovered, existing=existing)
    target = next(m for m in merged if m.merge_key() == "o/r")
    assert target.first_seen == "2026-05-20"
    assert target.last_seen == "2026-05-20"
