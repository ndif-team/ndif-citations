"""Slash commands without network (spec §6.3)."""
import json

import pytest

from ndif_citations.models import Bucket, Category
from ndif_citations.output import load_existing_papers
from ndif_citations.review_commands import ReviewContext, parse_commands, run_commands
from ndif_citations.review_render import render_issue
from tests.conftest import make_paper


def _write(out, papers):
    (out / "images").mkdir(parents=True, exist_ok=True)
    data = {"pending": [], "verified": [], "discarded": []}
    for p in papers:
        data[p.bucket.value].append(p.to_full_dict())
    (out / "research-papers-full.json").write_text(json.dumps(data, default=str))


def _ctx(out, paper, comment, *, author="github-actions[bot]", labels=("review",)):
    _, body, _ = render_issue(paper)
    return ReviewContext(out=out, issue_number=12, issue_body=body, issue_author=author,
                         issue_labels=set(labels), comment_body=comment, comment_author="alice")


def _get(out, key):
    return next(p for p in load_existing_papers(out) if p.merge_key() == key)


@pytest.fixture()
def pending(tmp_path):
    p = make_paper(title="Pending One", arxiv_id="2501.00001", bucket=Bucket.PENDING,
                   category=Category.USES_NNSIGHT, url="https://arxiv.org/abs/2501.00001")
    _write(tmp_path, [p])
    return tmp_path, p


def test_parse_commands():
    cmds = parse_commands("thanks!\n/verify uses_ndif\n  /discard dup of #3\n/" + "\n/help" * 20)
    assert [c.name for c in cmds][:3] == ["verify", "discard", "help"] and len(cmds) == 10
    assert cmds[0].args == "uses_ndif"


def test_verify_with_category(pending):
    out, p = pending
    r = run_commands(_ctx(out, p, "/verify uses_ndif"))
    got = _get(out, p.merge_key())
    assert r.changed and got.bucket == Bucket.VERIFIED and got.category == Category.USES_NDIF


def test_verify_unclassified_requires_category(tmp_path):
    p = make_paper(bucket=Bucket.PENDING, category=Category.UNCLASSIFIED)
    _write(tmp_path, [p])
    r = run_commands(_ctx(tmp_path, p, "/verify"))
    assert not r.changed and "needs a category" in r.reply
    assert _get(tmp_path, p.merge_key()).bucket == Bucket.PENDING


def test_discard_with_reason(pending):
    out, p = pending
    r = run_commands(_ctx(out, p, "/discard not about interpretability"))
    got = _get(out, p.merge_key())
    assert r.changed and got.bucket == Bucket.DISCARDED and got.reason_detail == "not about interpretability"


def test_edit_preserves_bucket(pending):
    out, p = pending
    r = run_commands(_ctx(out, p, "/edit venue: ICLR 2026"))
    got = _get(out, p.merge_key())
    assert r.changed and got.venue == "ICLR 2026" and got.bucket == Bucket.PENDING


def test_edit_rejects_unknown_field(pending):
    out, p = pending
    assert "can't edit" in run_commands(_ctx(out, p, "/edit bucket: verified")).reply


def test_evidence_stores_quote(pending):
    out, p = pending
    run_commands(_ctx(out, p, '/evidence "All interventions use NNsight remote execution."'))
    assert _get(out, p.merge_key()).curator_quotes == ["All interventions use NNsight remote execution."]


def test_marker_on_human_issue_is_ignored(pending):
    out, p = pending
    r = run_commands(_ctx(out, p, "/verify uses_ndif", author="mallory"))
    assert not r.changed and _get(out, p.merge_key()).bucket == Bucket.PENDING


def test_explicit_target_in_digest_issue(pending):
    out, p = pending
    ctx = ReviewContext(out=out, issue_number=40, issue_body="digest", issue_author="github-actions[bot]",
                        issue_labels={"digest"}, comment_body="/discard https://arxiv.org/abs/2501.00001 wrong field",
                        comment_author="alice")
    r = run_commands(ctx)
    assert r.changed and _get(out, p.merge_key()).bucket == Bucket.DISCARDED


def test_title_target_rejected(pending):
    out, p = pending
    ctx = ReviewContext(out=out, issue_number=40, issue_body="", issue_author="github-actions[bot]",
                        issue_labels={"report"}, comment_body="/discard title:pending one", comment_author="alice")
    assert "URL" in run_commands(ctx).reply


def test_unknown_command_gets_help(pending):
    out, p = pending
    assert "| Command |" in run_commands(_ctx(out, p, "/frobnicate")).reply


@pytest.mark.parametrize("comment", ["/verify", "/discard off-topic"])
def test_resolving_command_closes_review_issue(pending, comment):
    out, p = pending
    assert run_commands(_ctx(out, p, comment)).close is True


def test_edit_and_digest_commands_never_close(pending):
    out, p = pending
    assert run_commands(_ctx(out, p, "/edit venue: ICLR 2026")).close is False
    ctx = ReviewContext(out=out, issue_number=40, issue_body="digest", issue_author="github-actions[bot]",
                        issue_labels={"digest"}, comment_body="/verify https://arxiv.org/abs/2501.00001",
                        comment_author="alice")
    r = run_commands(ctx)
    assert r.changed and r.close is False


# --- Lead-approved amendments (2026-10-13) + task-body requirements -------------------------

def test_verify_reply_shows_curator_locked(pending):
    out, p = pending
    r = run_commands(_ctx(out, p, "/verify uses_ndif"))
    assert "now **verified** · uses_ndif (curator-locked)" in r.reply


def test_edit_rejects_empty_value(pending):  # #21: an empty title would change a title-keyed merge key
    out, p = pending
    r = run_commands(_ctx(out, p, "/edit title:"))
    assert not r.changed and r.reply.startswith("⚠️") and _get(out, p.merge_key()).title == p.title


@pytest.mark.parametrize("field", ["url", "github"])
def test_edit_url_requires_http_link(pending, field):  # #21: url / project_url reach ndif.us hrefs
    out, p = pending
    r = run_commands(_ctx(out, p, f"/edit {field}: javascript:alert(1)"))
    got = _get(out, p.merge_key())
    assert not r.changed and "use an http(s) link" in r.reply
    assert got.url == p.url and got.project_url == p.project_url
    assert run_commands(_ctx(out, p, f"/edit {field}: https://github.com/a/b")).changed


def test_evidence_error_reply_has_no_exception_text(pending):
    out, p = pending
    r = run_commands(_ctx(out, p, "/evidence"))
    assert not r.changed and r.reply.startswith("⚠️") and "empty quote" not in r.reply


def test_review_apply_cli_prints_reply_without_markup(pending, monkeypatch):  # #4b
    from click.testing import CliRunner

    from ndif_citations import config
    from ndif_citations.cli import cli

    out, p = pending
    _, body, _ = render_issue(p)
    (out / "issue.json").write_text(json.dumps({"number": 12, "body": body, "user": {"login": "github-actions[bot]"},
                                                "labels": [{"name": "review"}]}))
    (out / "comment.json").write_text(json.dumps({"body": "/edit venue: ICLR 2026\n/verify [/x]",
                                                  "user": {"login": "alice"}}))
    monkeypatch.setenv("GITHUB_OUTPUT", str(out / "gh_output"))
    monkeypatch.setattr(config, "get_output_dir", lambda _: out)
    res = CliRunner().invoke(cli, ["review-apply", "--kind", "command", "--issue-json", str(out / "issue.json"),
                                   "--comment-json", str(out / "comment.json"), "--out", str(out / "result.json")])
    assert res.exit_code == 0, res.output
    assert "`[/x]`" in res.output                      # printed literally, no MarkupError
    got = json.loads((out / "result.json").read_text())
    assert got["changed"] is True and got["close"] is False and "`[/x]`" in got["reply"]
    assert (out / "gh_output").read_text() == "changed=true\n"
    assert _get(out, p.merge_key()).venue == "ICLR 2026"


# --- I4: a curator's own /edit must not break the issue's marker ---

def _title_keyed(tmp_path, url):
    p = make_paper(title="Keyless Paper", arxiv_id=None, doi=None, bucket=Bucket.PENDING,
                   category=Category.USES_NNSIGHT, url=url)
    _write(tmp_path, [p])
    return p


def _sync_kinds(out, ctx):
    from ndif_citations.review_sync import plan_sync
    issue = {"number": 12, "title": "x", "body": ctx.issue_body, "labels": [{"name": "review"}],
             "author": {"login": "github-actions[bot]"}}
    return [a.kind for a in plan_sync(load_existing_papers(out), [issue])]


def test_edit_title_and_url_then_verify_in_one_comment(tmp_path):
    p = _title_keyed(tmp_path, "https://example.org/a")
    ctx = _ctx(tmp_path, p, "/edit title: Renamed Paper\n/edit url: https://example.org/b\n/verify")
    r = run_commands(ctx)
    assert r.reply.count("✅") == 3 and "can't find" not in r.reply, r.reply
    assert _get(tmp_path, "title:renamed paper").bucket == Bucket.VERIFIED
    kinds = _sync_kinds(tmp_path, ctx)
    assert "attention" not in kinds and "create" not in kinds


def test_edit_title_on_urlless_paper_keeps_issue(tmp_path):
    p = _title_keyed(tmp_path, "")
    ctx = _ctx(tmp_path, p, "/edit title: Renamed Paper")
    assert run_commands(ctx).changed
    kinds = _sync_kinds(tmp_path, ctx)
    assert "attention" not in kinds and "create" not in kinds


def test_review_apply_cli_returns_refreshed_body_only_after_edit(pending, monkeypatch):
    from click.testing import CliRunner

    from ndif_citations import config
    from ndif_citations.cli import cli

    out, p = pending
    _, body, _ = render_issue(p)
    (out / "issue.json").write_text(json.dumps({"number": 12, "body": body, "user": {"login": "github-actions[bot]"},
                                                "labels": [{"name": "review"}]}))
    monkeypatch.setattr(config, "get_output_dir", lambda _: out)

    def apply(comment):
        (out / "comment.json").write_text(json.dumps({"body": comment, "user": {"login": "alice"}}))
        res = CliRunner().invoke(cli, ["review-apply", "--kind", "command", "--issue-json", str(out / "issue.json"),
                                       "--comment-json", str(out / "comment.json"), "--out", str(out / "result.json")])
        assert res.exit_code == 0, res.output
        return json.loads((out / "result.json").read_text())

    assert "body" not in apply("/help")
    got = apply("/edit title: Pending One Renamed")
    assert "Pending One Renamed" in got["body"] and got["body"] != body
