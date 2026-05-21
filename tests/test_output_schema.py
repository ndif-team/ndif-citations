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


def test_merge_repos_preserves_timestamps_on_protected_but_absent(monkeypatch):
    monkeypatch.setattr(output_module, "_today", lambda: date(2026, 5, 20))
    existing = [DiscoveredRepo(
        owner="o", repo="protected", url="https://github.com/o/protected",
        first_seen="2025-06-01", last_seen="2026-04-01",
        manual_override=True,
    )]
    merged = merge_repos(discovered=[], existing=existing)
    target = next(m for m in merged if m.merge_key() == "o/protected")
    assert target.first_seen == "2025-06-01"  # unchanged — no fresh observation
    assert target.last_seen == "2026-04-01"   # unchanged — no fresh observation


def test_merge_repos_keeps_scrape_absent_repo_within_30_days(monkeypatch):
    monkeypatch.setattr(output_module, "_today", lambda: date(2026, 5, 20))
    existing = [DiscoveredRepo(
        owner="o", repo="r", url="https://github.com/o/r",
        first_seen="2025-01-01", last_seen="2026-05-01",  # 19 days ago
    )]
    merged = merge_repos(discovered=[], existing=existing)
    assert any(m.merge_key() == "o/r" for m in merged), \
        "Scrape-absent repo within 30d should survive"


def test_merge_repos_ages_out_scrape_absent_repo_past_30_days(monkeypatch):
    monkeypatch.setattr(output_module, "_today", lambda: date(2026, 5, 20))
    existing = [DiscoveredRepo(
        owner="o", repo="r", url="https://github.com/o/r",
        first_seen="2025-01-01", last_seen="2026-04-01",  # 49 days ago
    )]
    merged = merge_repos(discovered=[], existing=existing)
    assert all(m.merge_key() != "o/r" for m in merged), \
        "Scrape-absent repo older than 30d should age out"


def test_merge_repos_manual_override_survives_age_out(monkeypatch):
    monkeypatch.setattr(output_module, "_today", lambda: date(2026, 5, 20))
    existing = [DiscoveredRepo(
        owner="o", repo="r", url="https://github.com/o/r",
        first_seen="2024-01-01", last_seen="2024-01-01",  # ancient
        manual_override=True,
    )]
    merged = merge_repos(discovered=[], existing=existing)
    assert any(m.merge_key() == "o/r" for m in merged), \
        "manual_override must always survive regardless of last_seen"


# ---------------------------------------------------------------------------
# _tag_repo_type manual_override tests
# ---------------------------------------------------------------------------

from ndif_citations.discover import _tag_repo_type


def test_tag_repo_type_respects_manual_override():
    # A repo a curator hand-tagged as 'course', with manual_override=True.
    # Its natural classification (stars=100, has description, has linked paper)
    # would otherwise put it in 'research'.
    r = DiscoveredRepo(
        owner="curator", repo="manually-flagged", url="https://github.com/curator/x",
        stars=100, description="real research project", linked_paper_url="https://arxiv.org/abs/1234.5678",
        repo_type="course", manual_override=True,
    )
    result = _tag_repo_type(r, unlinked_set=set())
    assert result == "course", "manual_override must short-circuit the 7-rule tree"


def test_tag_repo_type_does_not_short_circuit_without_override():
    # Same repo without manual_override — should be re-tagged as 'research'.
    r = DiscoveredRepo(
        owner="curator", repo="not-flagged", url="https://github.com/curator/y",
        stars=100, description="real research project", linked_paper_url="https://arxiv.org/abs/1234.5678",
        repo_type="course", manual_override=False,
    )
    result = _tag_repo_type(r, unlinked_set=set())
    assert result == "research", "without manual_override, normal rules apply"


def test_tag_repo_type_catches_known_course_leaks():
    # Real repos that today get tagged 'experiment' but are clearly coursework.
    theia = DiscoveredRepo(
        owner="AntonKorznikov", repo="TheiaSae", url="https://github.com/AntonKorznikov/TheiaSae",
        stars=0, description="Skoltech ML course 2025 project",
    )
    metarepo = DiscoveredRepo(
        owner="MichaelRipa", repo="coding-exercises-metarepo",
        url="https://github.com/MichaelRipa/coding-exercises-metarepo",
        stars=0, description="This repository contains all my coding exercises from various courses",
    )
    assert _tag_repo_type(theia, unlinked_set=set()) == "course"
    assert _tag_repo_type(metarepo, unlinked_set=set()) == "course"


def test_tag_repo_type_does_not_misclassify_research_with_workshop_keyword():
    # Per user decision 2026-05-20: "workshop" was dropped from COURSE_NAME_PATTERNS
    # to avoid false-positive on workshop-paper research repos. Verify a research
    # repo mentioning "workshop" still tags as research.
    r = DiscoveredRepo(
        owner="real", repo="research", url="https://github.com/real/research",
        stars=200, description="ICLR 2026 workshop paper code",
        linked_paper_url="https://arxiv.org/abs/2601.00000",
    )
    assert _tag_repo_type(r, unlinked_set=set()) == "research"


def test_xlsx_github_sheet_columns(tmp_path):
    import openpyxl
    from ndif_citations.output import _write_xlsx

    repos = [DiscoveredRepo(
        owner="o", repo="r", url="https://github.com/o/r",
        stars=10, forks=2, first_seen="2026-01-01", last_seen="2026-05-20",
    )]
    _write_xlsx(papers=[], repos=repos, output_dir=tmp_path, skip_papers=True, skip_github=False)
    wb = openpyxl.load_workbook(tmp_path / "research-data.xlsx")
    ws = wb["GitHub"]
    header = [c.value for c in ws[1]]
    expected = [
        "owner", "repo", "url", "description", "stars", "forks",
        "last_commit", "first_seen", "language", "license", "topics",
        "linked_paper_url", "linked_paper_tier", "readme_arxiv_ids",
        "category", "repo_type", "parent_full_name",
        "archived", "is_fork", "classification_reason", "manual_override",
    ]
    assert header == expected
