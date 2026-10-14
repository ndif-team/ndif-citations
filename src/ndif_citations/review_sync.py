"""Keep exactly one bot-authored review issue open per pending paper (spec §6.1)."""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable

from ndif_citations.models import Bucket, DiscoveredPaper
from ndif_citations.review_render import BOT_LOGINS, parse_marker, render_issue

Runner = Callable[..., str]


def _norm(title: str | None) -> str:
    return " ".join((title or "").lower().split())


def resolve_paper(papers: list[DiscoveredPaper], ident: dict | None) -> DiscoveredPaper | None:
    """The paper a marker / explicit target refers to. Keys drift, so try every id."""
    if not ident:
        return None
    key, arxiv, doi = ident.get("key"), ident.get("arxiv"), (ident.get("doi") or "").lower()
    url, title = ident.get("url"), _norm(ident.get("title"))
    for test in (
        (lambda p: p.merge_key() == key) if key else None,
        (lambda p: p.arxiv_id == arxiv) if arxiv else None,
        (lambda p: (p.doi or "").lower() == doi) if doi else None,
        (lambda p: url in (p.url, p.pdf_url)) if url else None,
        (lambda p: _norm(p.title) == title) if title else None,
    ):
        if test:
            hit = next((p for p in papers if test(p)), None)
            if hit:
                return hit
    return None


@dataclass
class SyncAction:
    kind: str                 # create | update | close | attention
    issue: int | None
    paper_key: str | None = None
    title: str = ""
    body: str = ""
    labels: list[str] = field(default_factory=list)
    comment: str = ""


def _labels(issue: dict) -> set[str]:
    return {l["name"] for l in issue.get("labels", [])}


def _crlf(text: str | None) -> str:
    return (text or "").replace("\r\n", "\n").rstrip()


def plan_sync(papers: list[DiscoveredPaper], issues: list[dict]) -> list[SyncAction]:
    """Diff pending papers against open review issues (gh JSON) → actions. Pure."""
    bot_issues = sorted((i for i in issues if (i.get("author") or {}).get("login") in BOT_LOGINS),
                        key=lambda i: i["number"])
    if not bot_issues and any(parse_marker(i.get("body", "")) for i in issues):
        # Safety abort: markers exist but no issue passes the author filter, so gh's
        # bot login probably changed. Creating now would duplicate every review issue.
        # (Only triage-level maintainers can label an issue `review`, so outsiders
        # can't trigger this.)
        raise RuntimeError(
            f"open review issues carry paper markers but none is bot-authored "
            f"(expected an author login in {sorted(BOT_LOGINS)}); refusing to create duplicates"
        )
    if not papers and bot_issues:
        raise RuntimeError("catalog is empty or missing; refusing to flag every review issue")
    actions: list[SyncAction] = []
    claimed: dict[str, int] = {}
    for issue in bot_issues:
        paper = resolve_paper(papers, parse_marker(issue.get("body", "")))
        if paper is None:
            if "needs-attention" not in _labels(issue):
                actions.append(SyncAction("attention", issue["number"], comment=(
                    "I can't find this paper in the catalog anymore (merged or deleted?). "
                    "A maintainer should check and close this issue.")))
            continue
        key = paper.merge_key()
        if key in claimed:
            actions.append(SyncAction("close", issue["number"], key, comment=f"Duplicate of #{claimed[key]}."))
            continue
        claimed[key] = issue["number"]
        if paper.bucket != Bucket.PENDING:
            actions.append(SyncAction("close", issue["number"], key, comment=(
                f"Resolved: now **{paper.bucket.value}** · {paper.category.value}.")))
            continue
        title, body, labels = render_issue(paper)
        # Only the labels the bot manages count; maintainer-added labels never trigger an
        # edit, and a stale needs-attention is removed once the paper resolves again.
        managed = {l for l in _labels(issue) if l in ("review", "needs-attention") or l.startswith("why:")}
        if (_crlf(title) != _crlf(issue.get("title")) or _crlf(body) != _crlf(issue.get("body"))
                or set(labels) != managed):
            actions.append(SyncAction("update", issue["number"], key, title, body, labels))
    for p in papers:
        if p.bucket == Bucket.PENDING and p.merge_key() not in claimed:
            title, body, labels = render_issue(p)
            actions.append(SyncAction("create", None, p.merge_key(), title, body, labels))
    return actions


def _run_gh(args: list[str], stdin: str | None = None) -> str:
    try:
        return subprocess.run(["gh", *args], input=stdin, capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        # Logs only; never posted to GitHub.
        raise RuntimeError(f"gh {args[:2]} failed: {(e.stderr or '').strip()[:300]}") from e


class Gh:
    """Thin gh CLI wrapper; *runner(args, stdin=None) -> stdout* is injectable for tests."""

    def __init__(self, runner: Runner | None = None):
        self.run = runner or _run_gh
        # ponytail: fixed 1s pacing; honour Retry-After if GitHub still 403s
        self.pace = 1.0 if runner is None else 0.0

    def list_review_issues(self) -> list[dict]:
        # ponytail: >500 open review issues would look unclaimed; paginate if the queue ever gets that big
        out = self.run(["issue", "list", "--label", "review", "--state", "open",
                        "--json", "number,title,body,labels,author", "--limit", "500"])
        return json.loads(out or "[]")

    def create(self, title: str, body: str, labels: list[str], assignees: str = "") -> None:
        args = ["issue", "create", "--title", title, "--body-file", "-"]
        for l in labels:
            args += ["--label", l]
        if assignees:
            args += ["--assignee", assignees]
        self.run(args, stdin=body)
        time.sleep(self.pace)

    def edit(self, number: int, title: str, body: str, add_labels: list[str], remove_labels: list[str]) -> None:
        args = ["issue", "edit", str(number), "--title", title, "--body-file", "-"]
        for l in add_labels:
            args += ["--add-label", l]
        for l in remove_labels:
            args += ["--remove-label", l]
        self.run(args, stdin=body)
        time.sleep(self.pace)

    def comment(self, number: int, body: str) -> None:
        self.run(["issue", "comment", str(number), "--body-file", "-"], stdin=body)
        time.sleep(self.pace)

    def close(self, number: int) -> None:
        self.run(["issue", "close", str(number)])
        time.sleep(self.pace)

    def add_label(self, number: int, label: str) -> None:
        self.run(["issue", "edit", str(number), "--add-label", label])
        time.sleep(self.pace)


def apply_sync(actions: list[SyncAction], gh: Gh, *, assignees: str = "", dry_run: bool = False,
               current_labels: dict[int, set[str]] | None = None) -> list[str]:
    """Execute (or, with dry_run, only describe) the planned actions.

    A failing action is logged as ``FAILED <kind> #<n>: <err>`` and the rest still run.
    """
    log: list[str] = []
    for a in actions:
        log.append(f"{a.kind:9} #{a.issue or '-'} {a.paper_key or ''}")
        if dry_run:
            continue
        try:
            if a.kind == "create":
                gh.create(a.title, a.body, a.labels, assignees)
            elif a.kind == "update":
                # Remove only labels actually on the issue (gh errors on absent ones).
                current = (current_labels or {}).get(a.issue, set())
                remove = sorted({l for l in current if l.startswith("why:") or l == "needs-attention"} - set(a.labels))
                gh.edit(a.issue, a.title, a.body, a.labels, remove)
            elif a.kind == "close":
                gh.comment(a.issue, a.comment)
                gh.close(a.issue)
            elif a.kind == "attention":
                gh.comment(a.issue, a.comment)
                gh.add_label(a.issue, "needs-attention")
        except Exception as e:  # noqa: BLE001 - keep syncing; the CLI turns FAILED lines into exit 1
            log.append(f"FAILED {a.kind} #{a.issue or '-'}: {e}")
    return log
