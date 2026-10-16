"""Render + parse GitHub review issues for pending papers (spec §6.1). Pure functions."""
from __future__ import annotations

import json
import re
from urllib.parse import quote

from ndif_citations.digest import _SAFE_URL, fence, inline, plain
from ndif_citations.models import Category, DiscoveredPaper, PaperReason

BOT_LOGINS = frozenset({"app/github-actions", "github-actions[bot]", "github-actions"})
_MARKER_START = "<!-- ndif-paper: "
_MARKER_RE = re.compile(r"<!-- ndif-paper: (\{.*?\}) -->", re.S)

WHY_TEXT = {
    "why:needs-metadata": "its metadata is incomplete (missing year, abstract, or identifier)",
    "why:needs-evidence": "no full-text PDF was found, so the bot only saw the abstract (or nothing)",
    "why:uses-ndif": "the model thinks it used NDIF — the public “NDIF-supported” badge always gets a human check",
    "why:check-label": "the model's label isn't confident enough to publish automatically",
}

HELP_TEXT = """| Command | What it does |
|---|---|
| `/verify` · `/verify uses_nnsight` | publish it (optionally fixing the label: `uses_ndif`, `uses_nnsight`, `referencing`) |
| `/discard reason…` | remove it from the catalog |
| `/pdf https://…` | attach an **open-access** PDF, re-extract evidence, re-classify |
| `/evidence the sentence that shows use` | paywalled paper? paste the key sentence (then `/verify`) |
| `/thumbnail` + a pasted screenshot | set the figure shown on ndif.us |
| `/edit title: …` | fix a field: title, authors, affiliations, venue, year, summary, url, github |
| `/add https://…` | (intake issues) add the suggested paper |
| `/retry` · `/summarize` · `/help` | re-run PDF lookup + classification · regenerate the summary · this table |

In digest/report issues, put the paper's URL (or `arxiv:<id>` / `doi:<doi>`) right after the command."""


def marker_for(paper: DiscoveredPaper) -> str:
    data = {"key": paper.merge_key(), "url": paper.url or "", "arxiv": paper.arxiv_id or "",
            "doi": paper.doi or "", "title": paper.title}
    # "--" can't appear inside an HTML comment and a raw "@" could mention someone; the JSON
    # escapes \u002d / \u0040 decode back to "-" / "@".
    blob = json.dumps(data, ensure_ascii=False).replace("--", "\\u002d\\u002d").replace("@", "\\u0040")
    return f"<!-- ndif-paper: {blob} -->"


def parse_marker(body: str) -> dict | None:
    # The bot's marker is the body's LAST line. Titles and excerpts above it are untrusted
    # and may contain marker-like text, so only the last occurrence counts.
    body = body or ""
    start = body.rfind(_MARKER_START)
    m = _MARKER_RE.match(body, start) if start >= 0 else None
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def why_label(paper: DiscoveredPaper) -> str:
    if paper.reason == PaperReason.STUB_METADATA:
        return "why:needs-metadata"
    if paper.context_source != "pdf":
        return "why:needs-evidence"
    if paper.category == Category.USES_NDIF:
        return "why:uses-ndif"
    return "why:check-label"


def _href(u: str | None) -> str | None:
    return f"<{u}>" if u and "\\" not in u and _SAFE_URL.fullmatch(u) else None


def render_issue(paper: DiscoveredPaper) -> tuple[str, str, list[str]]:
    label = why_label(paper)
    title = f"Review: {plain(paper.title)}"[:240].rstrip()
    # A locked paper's band reloads as CERTAIN (models.model_post_init), so don't present it as the model's.
    conf = "curator-locked" if paper.manual_override else f"confidence {paper.category_confidence_band.value}"
    key = plain(paper.merge_key()).replace("`", "'")
    lines = [
        f"**{inline(paper.title)}** · {inline(paper.venue or '')} {paper.year or ''} · `{key}`",
        "",
        f"**Why you're seeing this:** {WHY_TEXT[label]}.",
        f"**Model says:** {paper.category.value} · {conf} · "
        f"evidence from {inline(paper.context_source)} ({len(paper.ndif_context_windows or [])} excerpt(s))",
    ]
    if paper.reason_detail:  # e.g. "re-classified: uses_ndif → referencing"
        lines.append(f"**Note:** {inline(paper.reason_detail)}")
    for w in (paper.ndif_context_windows or [])[:3]:
        lines += ["", fence(w, 500)]
    if paper.curator_quotes:
        lines += ["", "**Curator quotes:**"] + [fence(q, 500) for q in paper.curator_quotes]
    paper_href, pdf_href = _href(paper.url), _href(paper.pdf_url)
    links = [f"[paper]({paper_href})" if paper_href else "",
             f"[PDF]({pdf_href})" if pdf_href else "",
             f"[arXiv](https://arxiv.org/abs/{quote(paper.arxiv_id, safe='/.')})" if paper.arxiv_id else "",
             f"[DOI](https://doi.org/{quote(paper.doi, safe='/')})" if paper.doi else ""]
    lines += ["", "**Links:** " + " · ".join(l for l in links if l),
              "", "<details><summary>Commands (maintainers)</summary>", "", HELP_TEXT, "", "</details>",
              "", marker_for(paper)]
    return title, "\n".join(lines) + "\n", ["review", label]
