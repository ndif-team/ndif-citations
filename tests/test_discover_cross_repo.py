"""Tests for _unlink_shared_template_papers and _tag_repo_type (discover.py)."""
import pytest

from ndif_citations.discover import _tag_repo_type, _unlink_shared_template_papers
from ndif_citations.models import Category
from tests.conftest import make_repo


THRESHOLD = 5  # matches config.SHARED_PAPER_THRESHOLD


class TestUnlinkSharedTemplatePapers:
    def _repos_sharing_url(self, n: int, url: str = "https://arxiv.org/abs/2407.14561") -> list:
        return [
            make_repo(owner="user", repo=f"repo{i}", stars=i, linked_paper_url=url)
            for i in range(n)
        ]

    def test_cluster_at_threshold_unlinks_all_but_highest_star(self):
        repos = self._repos_sharing_url(THRESHOLD)
        unlinked = _unlink_shared_template_papers(repos)
        # Highest-star repo (stars=THRESHOLD-1) keeps its link
        high_star = max(repos, key=lambda r: r.stars or 0)
        assert high_star.linked_paper_url is not None
        # All others are unlinked
        others = [r for r in repos if r.merge_key() != high_star.merge_key()]
        for r in others:
            assert r.linked_paper_url is None
        assert len(unlinked) == THRESHOLD - 1

    def test_cluster_below_threshold_no_change(self):
        repos = self._repos_sharing_url(THRESHOLD - 1)
        unlinked = _unlink_shared_template_papers(repos)
        assert unlinked == set()
        for r in repos:
            assert r.linked_paper_url is not None

    def test_returns_merge_keys_of_unlinked_repos(self):
        repos = self._repos_sharing_url(THRESHOLD)
        unlinked = _unlink_shared_template_papers(repos)
        high_star = max(repos, key=lambda r: r.stars or 0)
        expected_keys = {r.merge_key() for r in repos if r.merge_key() != high_star.merge_key()}
        assert unlinked == expected_keys

    def test_multiple_urls_handled_independently(self):
        url_a = "https://arxiv.org/abs/2407.00001"
        url_b = "https://arxiv.org/abs/2407.00002"
        repos_a = self._repos_sharing_url(THRESHOLD, url_a)
        repos_b = self._repos_sharing_url(THRESHOLD - 1, url_b)  # below threshold
        all_repos = repos_a + repos_b
        unlinked = _unlink_shared_template_papers(all_repos)
        # Only url_a repos get unlinked
        for r in repos_b:
            assert r.linked_paper_url == url_b
        assert len(unlinked) == THRESHOLD - 1


class TestTagRepoType:
    def test_tier1_known_course_source(self):
        repo = make_repo(parent_full_name="callummcdougall/ARENA_3.0")
        assert _tag_repo_type(repo, set()) == "course"

    def test_tier2_unlinked_zero_stars_no_description(self):
        repo = make_repo(owner="u", repo="r", stars=0, description=None, linked_paper_url=None)
        unlinked = {repo.merge_key()}
        assert _tag_repo_type(repo, unlinked) == "course"

    def test_tier2_unlinked_but_has_description_not_course(self):
        repo = make_repo(stars=0, description="some description")
        unlinked = {repo.merge_key()}
        # Has description → doesn't match tier 2 → falls to experiment
        result = _tag_repo_type(repo, unlinked)
        assert result in ("research", "experiment")

    def test_tier3_course_name_pattern(self):
        repo = make_repo(repo="ARENA-solutions", description=None)
        assert _tag_repo_type(repo, set()) == "course"

    def test_tier3_mats_in_description(self):
        repo = make_repo(description="MATS scholarship project")
        assert _tag_repo_type(repo, set()) == "course"

    def test_tier4_uses_ndif_is_research(self):
        repo = make_repo(category=Category.USES_NDIF)
        assert _tag_repo_type(repo, set()) == "research"

    def test_tier5_linked_paper_is_research(self):
        repo = make_repo(linked_paper_url="https://arxiv.org/abs/2407.14561")
        assert _tag_repo_type(repo, set()) == "research"

    def test_tier6_notable_stars_and_description(self):
        repo = make_repo(stars=10, description="A real research project")
        assert _tag_repo_type(repo, set()) == "research"

    def test_tier6_high_stars_no_description_is_experiment(self):
        repo = make_repo(stars=10, description=None)
        assert _tag_repo_type(repo, set()) == "experiment"

    def test_tier7_default_experiment(self):
        repo = make_repo(stars=0, description=None)
        assert _tag_repo_type(repo, set()) == "experiment"


# ---------------------------------------------------------------------------
# TestTierPropagation (US-T3)
# ---------------------------------------------------------------------------

class TestTierPropagation:
    def _run_link(self, repos, papers):
        from ndif_citations.discover import link_repos_to_papers
        link_repos_to_papers(repos, papers)

    def test_tier1_repo_sets_paper_tier_to_1(self):
        from tests.conftest import make_paper
        paper = make_paper(arxiv_id="2407.00001", year=2024)
        repo = make_repo(
            linked_paper_url="https://arxiv.org/abs/2407.00001",
            linked_paper_tier=1,
        )
        self._run_link([repo], [paper])
        assert paper.linked_paper_tier == 1

    def test_strongest_tier_wins_when_two_repos_link_same_paper(self):
        from tests.conftest import make_paper
        paper = make_paper(arxiv_id="2407.00001", year=2024)
        repo1 = make_repo(
            owner="user1", repo="repo1",
            linked_paper_url="https://arxiv.org/abs/2407.00001",
            linked_paper_tier=1,
        )
        repo2 = make_repo(
            owner="user2", repo="repo2",
            linked_paper_url="https://arxiv.org/abs/2407.00001",
            linked_paper_tier=3,
        )
        self._run_link([repo1, repo2], [paper])
        assert paper.linked_paper_tier == 1

    def test_strongest_tier_wins_when_weaker_comes_first(self):
        from tests.conftest import make_paper
        paper = make_paper(arxiv_id="2407.00001", year=2024)
        repo_weak = make_repo(
            owner="user1", repo="repo1",
            linked_paper_url="https://arxiv.org/abs/2407.00001",
            linked_paper_tier=4,
        )
        repo_strong = make_repo(
            owner="user2", repo="repo2",
            linked_paper_url="https://arxiv.org/abs/2407.00001",
            linked_paper_tier=2,
        )
        self._run_link([repo_weak, repo_strong], [paper])
        assert paper.linked_paper_tier == 2

    def test_three_repos_tiers_4_4_2_paper_gets_2(self):
        from tests.conftest import make_paper
        paper = make_paper(arxiv_id="2407.00001", year=2024)
        repos = [
            make_repo(owner="u1", repo="r1", linked_paper_url="https://arxiv.org/abs/2407.00001", linked_paper_tier=4),
            make_repo(owner="u2", repo="r2", linked_paper_url="https://arxiv.org/abs/2407.00001", linked_paper_tier=4),
            make_repo(owner="u3", repo="r3", linked_paper_url="https://arxiv.org/abs/2407.00001", linked_paper_tier=2),
        ]
        self._run_link(repos, [paper])
        assert paper.linked_paper_tier == 2

    def test_repo_with_no_tier_does_not_set_paper_tier(self):
        from tests.conftest import make_paper
        paper = make_paper(arxiv_id="2407.00001", year=2024)
        repo = make_repo(
            linked_paper_url="https://arxiv.org/abs/2407.00001",
            linked_paper_tier=None,
        )
        self._run_link([repo], [paper])
        assert paper.linked_paper_tier is None
