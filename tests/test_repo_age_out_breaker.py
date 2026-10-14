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
    out[0].last_seen = date.today().isoformat()  # previous-run reference = today; the rest are 60d older
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


def test_age_out_is_relative_to_previous_run_not_today():
    def ago(days):
        return (date.today() - timedelta(days=days)).isoformat()

    a, b, c = (make_repo(owner="o", repo=n) for n in "abc")
    a.last_seen, b.last_seen, c.last_seen = ago(120), ago(125), ago(160)
    # Previous run was 120 days ago: b is 5d older (kept), c is 40d older (aged out).
    merged = merge_repos([], [a, b, c])
    assert {r.repo for r in merged} == {"a", "b"}


def _ago(days):
    return (date.today() - timedelta(days=days)).isoformat()


def test_future_dated_last_seen_does_not_age_out_the_others():
    a, b, c = (make_repo(owner="o", repo=n) for n in "abc")
    a.last_seen, b.last_seen, c.last_seen = _ago(5), _ago(6), _ago(-100)  # c: clock skew / bad edit
    merged = merge_repos([], [a, b, c])
    assert {"a", "b"} <= {r.repo for r in merged}  # reference is capped at today, not 100 days ahead


def test_one_malformed_last_seen_does_not_reset_the_reference_to_today():
    a, b, c = (make_repo(owner="o", repo=n) for n in "abc")
    a.last_seen, b.last_seen, c.last_seen = _ago(120), _ago(125), "not-a-date"
    merged = merge_repos([], [a, b, c])
    assert {r.repo for r in merged} == {"a", "b"}  # previous run = 120d ago; c (bad data) ages out
