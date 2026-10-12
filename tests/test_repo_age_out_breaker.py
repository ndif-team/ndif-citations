"""Mass age-out = broken scrape, not mass deletion (spec §5.4)."""
from datetime import date, timedelta

from ndif_citations.models import PipelineRun
from ndif_citations.output import merge_repos
from tests.conftest import make_repo

OLD = (date.today() - timedelta(days=60)).isoformat()


def _repos(n, prefix):
    out = []
    for i in range(n):
        r = make_repo(owner=prefix, repo=f"r{i}")
        r.last_seen = OLD
        out.append(r)
    return out


def test_breaker_skips_mass_age_out():
    existing = _repos(20, "o")
    run = PipelineRun()
    merged = merge_repos(existing[:2], existing, run=run)   # scrape returned only 2 of 20
    assert len(merged) == 20
    assert run.age_out_skipped == 18


def test_small_age_out_still_happens():
    existing = _repos(20, "o")
    run = PipelineRun()
    merged = merge_repos(existing[:17], existing, run=run)  # 3 stale <= max(5, 10%)
    assert len(merged) == 17
    assert run.age_out_skipped == 0


def test_run_argument_is_optional():
    existing = _repos(3, "o")
    assert len(merge_repos(existing, existing)) == 3
