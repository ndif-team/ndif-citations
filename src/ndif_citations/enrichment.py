"""Authoritative-source metadata reconciliation engine + orchestration.

Pure helpers (is_broken / reconcile_field / reconcile_paper) take already-fetched
values and decide the best one; the fetch/orchestration helpers reuse existing
query functions. See docs/superpowers/specs/2026-06-06-robust-enrichment-design.md.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from ndif_citations.venue import _WEAK_VENUE_RE
from ndif_citations.extract import _openalex_fetch_work
from ndif_citations.utils import extract_arxiv_id_from_url, query_arxiv_api, rate_limit_sleep
from ndif_citations.discover import _openalex_work_to_discovered
from ndif_citations import config

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class ResolveResult:
    """Outcome of resolve_identifiers. `via_title` marks ids adopted via a
    title-search match (lower confidence) so the caller can flag downstream
    field changes for review."""
    resolved: bool
    via_title: bool = False


def _strip_doi(doi: str) -> str:
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", (doi or "").strip())


def resolve_identifiers(paper) -> ResolveResult:
    """Resolve+persist missing identifiers on `paper`. Returns a ResolveResult;
    `via_title` is True when the id was adopted via an OpenAlex title.search match."""
    if paper.arxiv_id or paper.doi:
        return ResolveResult(resolved=False)
    for u in (paper.url, paper.pdf_url):
        axid = extract_arxiv_id_from_url(u) if u else None
        if axid:
            paper.arxiv_id = axid
            return ResolveResult(resolved=True)
    work = _openalex_fetch_work(f"title.search:{(paper.title or '')[:100]}", by="filter")
    if not work:
        return ResolveResult(resolved=False)
    if title_similarity(paper.title or "", work.get("title") or "") < TITLE_MATCH_THRESHOLD:
        return ResolveResult(resolved=False)
    paper.openalex_id = work.get("id") or paper.openalex_id
    doi = _strip_doi((work.get("ids") or {}).get("doi") or "")
    if doi and not paper.doi:
        paper.doi = doi
    return ResolveResult(resolved=True, via_title=True)


_MANAGED_FIELDS = ("abstract", "authors", "affiliations", "venue", "year")


@dataclass(frozen=True)
class Record:
    source: str
    fields: dict[str, object]


def _openalex_record(paper) -> "Record | None":
    work = None
    if paper.openalex_id:
        work = _openalex_fetch_work(paper.openalex_id.replace("https://openalex.org/", ""), by="id")
    if not work and paper.arxiv_id:
        work = _openalex_fetch_work(
            f"locations.landing_page_url:https://arxiv.org/abs/{paper.arxiv_id}", by="filter")
    if not work and paper.doi:
        work = _openalex_fetch_work(f"doi:{paper.doi}", by="filter")
    if not work:
        return None
    d = _openalex_work_to_discovered(work)
    if d is None:
        return None
    return Record(source="openalex", fields={
        "abstract": d.abstract or "", "authors": d.authors or "",
        "affiliations": d.affiliations or "", "venue": d.venue or "", "year": d.year or 0,
    })


def fetch_records(paper) -> list[Record]:
    """Gather authority records for a paper (reuses existing query helpers).
    Network failure on one source must not abort the others."""
    records: list[Record] = []
    try:
        rate_limit_sleep(config.OPENALEX_RATE_LIMIT_SLEEP, "OpenAlex enrich")
        oa = _openalex_record(paper)
        if oa:
            records.append(oa)
    except Exception as e:
        logger.warning("OpenAlex enrich failed for %r: %s", (paper.title or "?")[:60], e)
    if paper.arxiv_id:
        try:
            rate_limit_sleep(0.3, "arXiv enrich")
            ax = query_arxiv_api([paper.arxiv_id]).get(paper.arxiv_id) or {}
            authors = ", ".join(ax.get("authors") or [])
            affils = ", ".join(ax.get("affiliations") or [])
            if authors or affils:
                records.append(Record(source="arxiv",
                                       fields={"authors": authors, "affiliations": affils}))
        except Exception as e:
            logger.warning("arXiv enrich failed for %s: %s", paper.arxiv_id, e)
    return records
