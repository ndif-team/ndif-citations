"""Issue sync planning + gh application (spec §6.1)."""
import subprocess

import pytest
from click.testing import CliRunner

from ndif_citations.models import Bucket
from ndif_citations.review_render import marker_for, render_issue
from ndif_citations.review_sync import Gh, _run_gh, apply_sync, plan_sync, resolve_paper
from tests.conftest import make_paper

BOT = {"login": "app/github-actions"}


def issue(n, paper=None, body=None, author=BOT, labels=("review",)):
    if body is None:
        _, body, _ = render_issue(paper)
    return {"number": n, "title": f"Review: {paper.title if paper else 'x'}", "body": body,
            "labels": [{"name": l} for l in labels], "author": author}


def test_creates_issue_for_unclaimed_pending_paper():
    p = make_paper(bucket=Bucket.PENDING)
    acts = plan_sync([p], [])
    assert [a.kind for a in acts] == ["create"] and acts[0].labels[0] == "review"


def test_closes_issue_when_paper_resolved():
    p = make_paper(bucket=Bucket.PENDING)
    i = issue(7, p)
    p.bucket = Bucket.VERIFIED
    acts = plan_sync([p], [i])
    assert [(a.kind, a.issue) for a in acts] == [("close", 7)] and "verified" in acts[0].comment


def test_marker_resolves_after_key_drift():
    p = make_paper(arxiv_id=None, doi=None, title="Drifting Title", url="https://openreview.net/forum?id=abc",
                   bucket=Bucket.PENDING)
    i = issue(3, p)                       # marker key = "title:drifting title"
    p.arxiv_id = "2601.00042"             # backfilled later → key changes to arxiv:...
    acts = plan_sync([p], [i])
    assert all(a.kind != "create" for a in acts)
    assert all(a.kind != "close" for a in acts)


def test_unresolvable_marker_gets_attention_not_closed():
    gone = make_paper(arxiv_id="2501.99999", bucket=Bucket.PENDING)
    i = issue(5, gone)
    other = make_paper(arxiv_id="2401.00002", title="Other", bucket=Bucket.VERIFIED)  # non-empty catalog (amendment #8)
    acts = plan_sync([other], [i])
    assert [(a.kind, a.issue) for a in acts] == [("attention", 5)]


def test_duplicate_issues_close_newer():
    p = make_paper(bucket=Bucket.PENDING)
    acts = plan_sync([p], [issue(2, p), issue(9, p)])
    assert ("close", 9) in [(a.kind, a.issue) for a in acts]


def test_human_authored_issue_is_ignored():
    p = make_paper(bucket=Bucket.PENDING)
    other = make_paper(arxiv_id="2401.00002", title="Other", bucket=Bucket.PENDING)
    acts = plan_sync([p, other], [issue(1, other), issue(4, p, author={"login": "mallory"})])
    # the human issue doesn't count as p's issue (#1 may also get a label update; ignore it)
    assert [(a.kind, a.paper_key) for a in acts if a.kind == "create"] == [("create", p.merge_key())]


def test_markers_but_no_bot_author_aborts():
    """If gh's bot login format changes, every bot issue would look human: refuse to mass-create."""
    p = make_paper(bucket=Bucket.PENDING)
    with pytest.raises(RuntimeError, match="none is bot-authored"):
        plan_sync([p], [issue(4, p, author={"login": "github-actions-renamed"})])


def test_resolve_paper_order():
    a = make_paper(arxiv_id="1", title="A")
    b = make_paper(arxiv_id="2", title="B", url="https://x/b")
    assert resolve_paper([a, b], {"url": "https://x/b"}) is b
    assert resolve_paper([a, b], {"title": "  a "}) is a
    assert resolve_paper([a, b], None) is None


def test_apply_sync_dry_run_and_real(monkeypatch):
    calls = []
    gh = Gh(runner=lambda args, stdin=None: calls.append((args, stdin)) or "")
    p = make_paper(bucket=Bucket.PENDING)
    acts = plan_sync([p], [])
    assert apply_sync(acts, gh, dry_run=True) and calls == []
    apply_sync(acts, gh, assignees="alice,bob")
    args, stdin = calls[0]
    assert args[:2] == ["issue", "create"] and "--assignee" in args and marker_for(p) in stdin


def test_update_removes_only_stale_bot_labels():
    calls = []
    gh = Gh(runner=lambda args, stdin=None: calls.append(args) or "")
    p = make_paper(bucket=Bucket.PENDING)
    current = {"review", "why:uses-ndif", "needs-attention", "good first issue"}
    acts = plan_sync([p], [issue(6, p, labels=tuple(current))])
    apply_sync(acts, gh, current_labels={6: current})
    args = calls[0]
    assert args[:3] == ["issue", "edit", "6"]
    assert "why:uses-ndif" in args and "needs-attention" in args and "good first issue" not in args


# --- lead-approved amendments -------------------------------------------------

def _three_papers():
    return [make_paper(arxiv_id=f"2601.0000{n}", title=f"Paper {n}", bucket=Bucket.PENDING) for n in (1, 2, 3)]


def test_failed_action_is_logged_and_sync_continues():
    calls = []

    def runner(args, stdin=None):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError("gh ['issue', 'create'] failed: boom")
        return ""

    acts = plan_sync(_three_papers(), [])
    log = apply_sync(acts, Gh(runner=runner))
    assert len(calls) == 3
    assert [l.startswith("FAILED") for l in log if "FAILED" in l] == [True]
    assert "FAILED create #-: gh ['issue', 'create'] failed: boom" in log[-2] + log[-1] + "".join(log)


def test_run_gh_wraps_stderr(monkeypatch):
    def boom(*a, **k):
        raise subprocess.CalledProcessError(1, a[0], stderr="  HTTP 403 secondary rate limit \n")
    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(RuntimeError, match=r"gh \['issue', 'create'\] failed: HTTP 403 secondary rate limit$"):
        _run_gh(["issue", "create", "--title", "t"])


def test_cli_exits_1_when_an_action_fails_but_runs_the_rest(monkeypatch):
    import ndif_citations.output as output
    import ndif_citations.review_sync as rs
    from ndif_citations.cli import cli

    calls = []

    def runner(args, stdin=None):
        if args[:2] == ["issue", "list"]:
            return "[]"
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError("gh ['issue', 'create'] failed: [boom]")
        return ""

    monkeypatch.setattr(output, "load_existing_papers", lambda out: _three_papers())
    monkeypatch.setattr(rs, "Gh", lambda: Gh(runner=runner))
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    res = CliRunner().invoke(cli, ["review-sync"])
    assert len(calls) == 3 and res.exit_code == 1
    assert "[boom]" in res.output        # markup=False: brackets are printed literally


def test_title_and_body_compared_normalized():
    p = make_paper(bucket=Bucket.PENDING)
    title, body, labels = render_issue(p)
    i = issue(8, p, body=body.replace("\n", "\r\n") + "\n\n", labels=tuple(labels))
    i["title"] = title
    assert plan_sync([p], [i]) == []
    i["body"] = marker_for(p) + "\nstale"
    assert [a.kind for a in plan_sync([p], [i])] == ["update"]


def test_gh_pacing(monkeypatch):
    import ndif_citations.review_sync as rs
    slept = []
    monkeypatch.setattr(rs.time, "sleep", slept.append)
    assert Gh(runner=lambda *a, **k: "").pace == 0.0 and Gh().pace == 1.0
    gh = Gh(runner=lambda *a, **k: "")
    gh.pace = 2.0
    gh.create("t", "b", ["review"])
    gh.close(1)
    assert slept == [2.0, 2.0]


def test_empty_catalog_with_open_issues_refuses():
    p = make_paper(bucket=Bucket.PENDING)
    with pytest.raises(RuntimeError, match="catalog is empty or missing"):
        plan_sync([], [issue(1, p)])
    assert plan_sync([], []) == []


def test_list_command_is_single_unfiltered_by_author():
    calls = []
    Gh(runner=lambda args, stdin=None: calls.append(args) or "[]").list_review_issues()
    assert calls == [["issue", "list", "--label", "review", "--state", "open",
                      "--json", "number,title,body,labels,author", "--limit", "500"]]
