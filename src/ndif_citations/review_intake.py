"""Intake: add a suggested paper by URL from a GitHub issue (spec §6.4)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ndif_citations import manual_add
from ndif_citations.digest import _SAFE_URL, inline
from ndif_citations.models import Bucket, PaperReason
from ndif_citations.review_commands import Result, ReviewContext, UserError, _llm_ready
from ndif_citations.review_sync import resolve_paper

_FORM_URL = re.compile(r"###\s*Paper URL\s*\n+\s*(https?://\S+)", re.I)


@dataclass
class IntakeResult:
    status: str            # unresolved | duplicate | existing | added | published | pending
    title: str
    bucket: str | None
    url: str


def url_from_issue_form(body: str) -> str | None:
    m = _FORM_URL.search(body or "")
    return m.group(1).strip() if m else None


def run_intake(out: Path, url: str) -> IntakeResult:
    seed = manual_add.resolve_seed_from_url(url)
    if seed is None:
        return IntakeResult("unresolved", "", None, url)
    dup = manual_add.find_duplicate(out, title=seed.title, arxiv_id=seed.arxiv_id, doi=seed.doi)
    if dup is not None:
        return IntakeResult("duplicate", dup.title, dup.bucket.value, dup.url or url)
    # From here the catalog has been written, so every status below must lead to a push.
    result = manual_add.run_manual_add_seed(out, [seed])
    paper = resolve_paper(result.merged_papers, {"url": url, "arxiv": seed.arxiv_id or "",
                                                 "doi": seed.doi or "", "title": seed.title})
    if paper is None:
        return IntakeResult("added", seed.title, None, url)
    if paper.bucket == Bucket.DISCARDED and paper.reason == PaperReason.ZERO_PDF_HITS:
        # A human asked for this paper: surface the 0 hits for review, never discard on their behalf.
        from ndif_citations.server.services import papers_svc
        papers_svc.set_bucket(out, paper.merge_key(), "pending", "needs_review",
                              "added via intake: no NDIF/NNsight mention found in the PDF")
        return IntakeResult("pending", paper.title, "pending", url)
    if paper.bucket == Bucket.DISCARDED:  # merged into a discarded paper find_duplicate missed
        return IntakeResult("existing", paper.title, "discarded", url)
    status = "published" if paper.bucket == Bucket.VERIFIED else "pending"
    return IntakeResult(status, paper.title, paper.bucket.value, url)


def _reply(r: IntakeResult, res: Result) -> None:
    if r.status == "unresolved":
        res.replies.append(f"⚠️ I couldn't look up `{r.url.replace('`', chr(39))}`. Use an arXiv, DOI, or ACL "
                           "Anthology link — or add it in the local app.")
        return
    res.changed = r.status != "duplicate"  # only a find_duplicate hit skipped the write
    res.close = True
    if r.status in ("duplicate", "existing"):
        link = f": <{r.url}>" if _SAFE_URL.fullmatch(r.url) else ""
        res.replies.append(f"ℹ️ Already in the catalog as **{r.bucket}**{link}")
    elif r.status == "added":
        res.replies.append("✅ Added — a maintainer should check the catalog entry.")
    elif r.status == "published":
        res.replies.append(f"✅ Added and auto-published: **{inline(r.title)}**. ndif.us updates within a day.")
    else:
        res.replies.append(f"✅ Added **{inline(r.title)}** for review — a review issue will be opened for it shortly.")


def intake_from_issue(ctx: ReviewContext) -> Result:
    """Issue opened by a maintainer with the intake form → use the form's URL.

    Caller must have verified the commenter/issue author has write access (catalog-apply's
    Load step); this function performs no authorization.
    """
    res = Result()
    url = url_from_issue_form(ctx.issue_body)
    if not url:
        res.replies.append("⚠️ I couldn't find a `Paper URL` in this issue.")
        return res
    try:
        _llm_ready()  # a new paper gets summarized + classified: never with silent keyword fallbacks
    except UserError as e:
        res.replies.append(f"⚠️ {e} A maintainer can retry with `/add <url>`.")
        return res
    _reply(run_intake(ctx.out, url), res)
    return res
