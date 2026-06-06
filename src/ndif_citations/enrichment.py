"""Authoritative-source metadata reconciliation engine + orchestration.

Pure helpers (is_broken / reconcile_field / reconcile_paper) take already-fetched
values and decide the best one; the fetch/orchestration helpers reuse existing
query functions. See docs/superpowers/specs/2026-06-06-robust-enrichment-design.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from ndif_citations.venue import _WEAK_VENUE_RE
from ndif_citations.extract import _openalex_fetch_work
from ndif_citations.utils import extract_arxiv_id_from_url

_ELLIPSIS = ("…", "...")
_ABSTRACT_MIN = 280


def is_broken(field: str, value) -> bool:
    """True if `value` for `field` is empty/truncated/low-quality."""
    if field == "year":
        return not value
    text = (value or "").strip() if isinstance(value, str) else ""
    if not text:
        return True
    if field == "abstract":
        return text.endswith(_ELLIPSIS) or len(text) < _ABSTRACT_MIN
    if field == "authors":
        stripped = text.rstrip(". ")
        return text.endswith(_ELLIPSIS) or "…" in text or stripped.endswith("et al")
    if field == "venue":
        return bool(_WEAK_VENUE_RE.match(text))
    if field == "affiliations":
        return False  # non-empty affiliations are acceptable
    return False


SOURCE_TRUST: dict[str, int] = {
    "openalex": 4, "crossref": 3, "arxiv": 3, "s2": 3,
    "manual_add": 2, "scholar": 1, "unknown": 0,
}


@dataclass(frozen=True)
class Candidate:
    value: object
    source: str


@dataclass(frozen=True)
class Resolution:
    value: object
    source: str
    changed: bool
    low_confidence: bool


def _trust(source: str) -> int:
    return SOURCE_TRUST.get(source, 0)


def _completeness(field: str, value) -> int:
    if field in ("abstract", "authors") and isinstance(value, str):
        return len(value)
    return 1


def _score(field: str, c: Candidate) -> tuple[int, int, int]:
    # higher tuple wins: valid first, then trust, then completeness
    return (0 if is_broken(field, c.value) else 1, _trust(c.source), _completeness(field, c.value))


def reconcile_field(field: str, current: Candidate, candidates: list[Candidate],
                    low_confidence_sources: set[str] | None = None) -> Resolution:
    low_confidence_sources = low_confidence_sources or set()
    valid = [c for c in candidates if c.value not in (None, "", 0)]
    if not valid:
        return Resolution(value=current.value, source=current.source, changed=False, low_confidence=False)
    best = max(valid, key=lambda c: _score(field, c))
    # Replace the current value only if it's broken, or `best` strictly out-scores it.
    # Otherwise keep current — a non-broken value is never swapped for an equal-or-worse one.
    if is_broken(field, current.value) or _score(field, best) > _score(field, current):
        winner = best
    else:
        winner = current
    changed = winner.value != current.value
    low_conf = changed and winner.source in low_confidence_sources
    return Resolution(value=winner.value, source=winner.source, changed=changed, low_confidence=low_conf)


TITLE_MATCH_THRESHOLD = 0.90


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (t or "").lower()).strip()


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_title(a), _norm_title(b)).ratio()


def resolve_identifiers(paper) -> bool:
    """Resolve+persist missing ids. Returns True if anything was resolved.
    Sets paper._enrichment_via_title = True when an id was adopted via title.search."""
    if paper.arxiv_id or paper.doi:
        return False
    for u in (paper.url, paper.pdf_url):
        axid = extract_arxiv_id_from_url(u) if u else None
        if axid:
            paper.arxiv_id = axid
            return True
    work = _openalex_fetch_work(f"title.search:{paper.title[:100]}", by="filter")
    if not work:
        return False
    if title_similarity(paper.title, work.get("title") or "") < TITLE_MATCH_THRESHOLD:
        return False
    paper.openalex_id = work.get("id") or paper.openalex_id
    ids = work.get("ids") or {}
    doi = (ids.get("doi") or "").replace("https://doi.org/", "")
    if doi and not paper.doi:
        paper.doi = doi
    object.__setattr__(paper, "_enrichment_via_title", True)
    return True
