"""Intake: S2-resolved manual add from an issue (spec §6.4)."""
import json
from types import SimpleNamespace

import pytest

from ndif_citations import manual_add, review_intake
from ndif_citations.models import Bucket
from ndif_citations.review_commands import Result, ReviewContext, run_commands
from tests.conftest import make_paper

FORM = "### Paper URL\n\nhttps://arxiv.org/abs/2503.12345\n\n### Notes\n\nuses nnsight\n"


def _write(out, papers=()):
    (out / "images").mkdir(parents=True, exist_ok=True)
    data = {"pending": [], "verified": [], "discarded": []}
    for p in papers:
        data[p.bucket.value].append(p.to_full_dict())
    (out / "research-papers-full.json").write_text(json.dumps(data, default=str))


def test_url_from_issue_form():
    assert review_intake.url_from_issue_form(FORM) == "https://arxiv.org/abs/2503.12345"
    assert review_intake.url_from_issue_form("no form here") is None


def test_s2_id_for_url():
    assert manual_add.s2_id_for_url("https://doi.org/10.18653/v1/2024.acl-long.1") == "DOI:10.18653/v1/2024.acl-long.1"
    assert manual_add.s2_id_for_url("https://arxiv.org/abs/2503.12345") == "ARXIV:2503.12345"
    assert manual_add.s2_id_for_url("https://aclanthology.org/2024.acl-long.1/") == "URL:https://aclanthology.org/2024.acl-long.1/"


def test_fetch_s2_metadata_reaches_s2(monkeypatch):
    """Guards against a swallowed NameError/ImportError turning every lookup into None."""
    import semanticscholar
    monkeypatch.setattr(semanticscholar, "SemanticScholar",
                        lambda **kw: SimpleNamespace(get_paper=lambda sid, fields=None: SimpleNamespace(title="T")))
    assert manual_add.fetch_s2_metadata("ARXIV:1").title == "T"


def test_unresolvable_url_changes_nothing(tmp_path, monkeypatch):
    _write(tmp_path)
    monkeypatch.setattr(manual_add, "resolve_seed_from_url", lambda url: None)
    assert review_intake.run_intake(tmp_path, "https://example.org/x").status == "unresolved"


def test_duplicate_reports_existing(tmp_path, monkeypatch):
    existing = make_paper(title="Known Paper", arxiv_id="2503.12345", bucket=Bucket.VERIFIED)
    _write(tmp_path, [existing])
    monkeypatch.setattr(manual_add, "resolve_seed_from_url",
                        lambda url: make_paper(title="Known Paper", arxiv_id="2503.12345"))
    r = review_intake.run_intake(tmp_path, "https://arxiv.org/abs/2503.12345")
    assert r.status == "duplicate" and r.bucket == "verified"


def test_new_paper_pending(tmp_path, monkeypatch):
    _write(tmp_path)
    seed = make_paper(title="Brand New", arxiv_id="2504.00001", bucket=Bucket.PENDING)
    monkeypatch.setattr(manual_add, "resolve_seed_from_url", lambda url: seed)
    monkeypatch.setattr(manual_add, "run_manual_add_seed",
                        lambda out, seeds, **k: SimpleNamespace(merged_papers=[seed]))
    assert review_intake.run_intake(tmp_path, "https://arxiv.org/abs/2504.00001").status == "pending"


def test_intake_from_issue_uses_form_url_only_for_maintainer_opened(tmp_path, monkeypatch):
    _write(tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "x")   # intake runs the LLM preflight; conftest stubs the live check
    seen = []
    monkeypatch.setattr(review_intake, "run_intake",
                        lambda out, url: seen.append(url) or review_intake.IntakeResult("unresolved", "", None, url))
    ctx = ReviewContext(tmp_path, 3, FORM, "alice", {"intake"}, "", "")
    review_intake.intake_from_issue(ctx)
    assert seen == ["https://arxiv.org/abs/2503.12345"]


def test_intake_waits_when_llm_unavailable(tmp_path, monkeypatch):
    _write(tmp_path)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(review_intake, "run_intake", lambda out, url: pytest.fail("ran without the LLM"))
    r = review_intake.intake_from_issue(ReviewContext(tmp_path, 3, FORM, "alice", {"intake"}, "", ""))
    assert not r.changed and "The LLM check failed" in r.reply and "LLM_API_KEY" not in r.reply


def test_discarded_match_is_duplicate(tmp_path, monkeypatch):
    existing = make_paper(title="Old Reject", arxiv_id="2501.00002", bucket=Bucket.DISCARDED)
    _write(tmp_path, [existing])
    seed = make_paper(title="Old Reject", arxiv_id="2501.00002")
    monkeypatch.setattr(manual_add, "resolve_seed_from_url", lambda url: seed)
    monkeypatch.setattr(manual_add, "find_duplicate", lambda *a, **k: None)  # force the post-merge path
    monkeypatch.setattr(manual_add, "run_manual_add_seed",
                        lambda out, seeds, **k: SimpleNamespace(merged_papers=[existing]))
    r = review_intake.run_intake(tmp_path, "https://arxiv.org/abs/2501.00002")
    assert r.status == "existing" and r.bucket == "discarded"
    res = Result()
    review_intake._reply(r, res)
    assert res.changed and res.close and "Already in the catalog as **discarded**" in res.reply  # the add rewrote it


def _zero_hit_add(tmp_path, monkeypatch):
    """The pipeline auto-discards a brand-new paper whose PDF has no NDIF/NNsight hit."""
    from ndif_citations.models import PaperReason
    _write(tmp_path)
    seed = make_paper(title="Silent Paper", arxiv_id="2504.00009", bucket=Bucket.PENDING)
    monkeypatch.setattr(manual_add, "resolve_seed_from_url", lambda url: seed)

    def fake_add(out, seeds, **k):
        p = make_paper(title="Silent Paper", arxiv_id="2504.00009", bucket=Bucket.DISCARDED,
                       reason=PaperReason.ZERO_PDF_HITS)
        _write(out, [p])
        return SimpleNamespace(merged_papers=[p])
    monkeypatch.setattr(manual_add, "run_manual_add_seed", fake_add)


def test_zero_hit_new_add_goes_to_review(tmp_path, monkeypatch):
    from ndif_citations.output import load_existing_papers
    _zero_hit_add(tmp_path, monkeypatch)
    r = review_intake.run_intake(tmp_path, "https://arxiv.org/abs/2504.00009")
    assert r.status == "pending" and r.bucket == "pending"
    [p] = load_existing_papers(tmp_path)
    assert p.bucket == Bucket.PENDING and p.reason.value == "needs_review"
    assert p.reason_detail == "added via intake: no NDIF/NNsight mention found in the PDF"
    res = Result()
    review_intake._reply(r, res)
    assert res.changed and res.close and "for review" in res.reply and "Already" not in res.reply


def test_add_command_zero_hit_is_success(tmp_path, monkeypatch):
    _zero_hit_add(tmp_path, monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "x")
    ctx = ReviewContext(tmp_path, 3, "", "alice", {"intake"}, "/add https://arxiv.org/abs/2504.00009", "bob")
    out = run_commands(ctx)
    assert out.changed and out.close and out.reply.startswith("✅ `/add`")


def test_unresolved_after_add_still_pushes(tmp_path, monkeypatch):
    _write(tmp_path)
    seed = make_paper(title="Ghost", arxiv_id="2504.00010", bucket=Bucket.PENDING)
    monkeypatch.setattr(manual_add, "resolve_seed_from_url", lambda url: seed)
    monkeypatch.setattr(manual_add, "run_manual_add_seed", lambda out, seeds, **k: SimpleNamespace(merged_papers=[]))
    r = review_intake.run_intake(tmp_path, "https://arxiv.org/abs/2504.00010")
    assert r.status == "added"
    res = Result()
    review_intake._reply(r, res)
    assert res.changed and res.close and "a maintainer should check the catalog entry" in res.reply
    monkeypatch.setenv("LLM_API_KEY", "x")
    out = run_commands(ReviewContext(tmp_path, 3, "", "alice", {"intake"}, "/add https://arxiv.org/abs/2504.00010", "bob"))
    assert out.changed and out.reply.startswith("✅ `/add`")


def test_reply_neutralizes_untrusted_text():
    r = review_intake.IntakeResult("pending", "<img src=x> [x](http://evil) @everyone", "pending", "u")
    res = Result()
    review_intake._reply(r, res)
    # every live trigger is backslash-escaped or defanged (zero-width space after @ and the scheme colon)
    assert "\\<img" in res.reply and "\\[x\\]" in res.reply
    assert "@everyone" not in res.reply and "://" not in res.reply
    res = Result()
    review_intake._reply(review_intake.IntakeResult("unresolved", "", None, "https://x.org/a`b"), res)
    assert res.reply.count("`") == 2


def test_duplicate_url_shown_only_if_safe():
    res = Result()
    review_intake._reply(review_intake.IntakeResult("duplicate", "T", "verified", "https://x.org/a"), res)
    assert "<https://x.org/a>" in res.reply
    res = Result()
    review_intake._reply(review_intake.IntakeResult("duplicate", "T", "verified", "javascript:alert(1)"), res)
    assert "javascript" not in res.reply


def test_add_command_needs_intake_label_and_explicit_url(tmp_path, monkeypatch):
    _write(tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.setattr(review_intake, "run_intake",
                        lambda out, url: review_intake.IntakeResult("pending", "P", "pending", url))
    ok = ReviewContext(tmp_path, 3, "", "alice", {"intake"}, "/add https://arxiv.org/abs/2504.00001", "bob")
    out = run_commands(ok)
    assert out.changed and "P" in out.reply
    bad = ReviewContext(tmp_path, 3, "", "alice", {"review"}, "/add https://arxiv.org/abs/2504.00001", "bob")
    assert not run_commands(bad).changed
    noarg = ReviewContext(tmp_path, 3, "", "alice", {"intake"}, "/add", "bob")
    assert "explicitly" in run_commands(noarg).reply


@pytest.mark.parametrize("url, want", [
    ("https://evil.example/doi.org/10.1234/x", "URL:https://evil.example/doi.org/10.1234/x"),
    ("https://evil.example/x?r=doi.org/&id=10.1234/abc", "URL:https://evil.example/x?r=doi.org/&id=10.1234/abc"),
    ("https://evil.example/arxiv.org/abs/2501.00001", "URL:https://evil.example/arxiv.org/abs/2501.00001"),
    ("https://doi.org/10.1145/abc?utm=1#frag", "DOI:10.1145/abc"),
    ("https://dx.doi.org/10.1145/a%2Fb/", "DOI:10.1145/a/b"),
    ("https://export.arxiv.org/abs/2501.00001", "ARXIV:2501.00001"),
    ("10.1145/abc", "DOI:10.1145/abc"),
])
def test_s2_id_for_url_anchors_the_host(url, want):
    assert manual_add.s2_id_for_url(url) == want
