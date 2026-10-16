"""Core logic for adding a single paper by URL (Task 4.5).

This module contains the extracted add-by-URL business logic, decoupled from
both the CLI and the server so it can be called from either without importing
server-specific or CLI-specific modules.

Entry point: ``add_paper_by_url(out, url, *, cancel_check=None) -> dict``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def seed_from_url(url: str) -> "DiscoveredPaper":
    """Build a manual-add seed paper from a URL (arXiv ID extracted if present)."""
    from ndif_citations.models import DiscoveredPaper, DiscoverySource
    from ndif_citations.utils import extract_arxiv_id_from_url

    return DiscoveredPaper(
        title="[Pending metadata lookup]",
        url=url,
        arxiv_id=extract_arxiv_id_from_url(url),
        source=DiscoverySource.MANUAL_ADD,
    )


def seed_from_pdf(
    *, title: str, arxiv_id: str | None = None, doi: str | None = None
) -> "DiscoveredPaper":
    """Build a manual-add seed paper from user-provided fields for a PDF upload."""
    from ndif_citations.models import DiscoveredPaper, DiscoverySource

    # No url for a PDF-only seed (e.g. a paywalled paper) — left blank intentionally.
    return DiscoveredPaper(
        title=title,
        arxiv_id=arxiv_id or None,
        doi=doi or None,
        source=DiscoverySource.MANUAL_ADD,
    )


def run_manual_add_seed(out, seed_papers, *, pdf_bytes=None, cancel_check=None):
    """Synchronously enrich -> (cache pdf) -> route -> process -> finalize a seed list (no gate).

    Caches pdf_bytes AFTER enrichment (so the filename matches any resolved
    arXiv/DOI). Used by the CLI add-pdf (terminal has no interactive gate).
    Returns the orchestrator FinalizeResult.
    """
    from ndif_citations import orchestrator
    from ndif_citations.models import PipelineRun
    from ndif_citations.pdf_cache import write_pdf_to_cache
    d = orchestrator.DiscoverResult(papers=list(seed_papers), repos=[], run_stats=PipelineRun())
    e = orchestrator.enrich_stage(out, d, skip_papers=False, skip_github=True, fresh=False)
    if pdf_bytes is not None and e.papers:
        write_pdf_to_cache(e.papers[0], pdf_bytes, out)
    r = orchestrator.route_stage(out, e, skip_papers=False, skip_github=True, fresh=False)
    completed = orchestrator.process_stage(out, r, skip_papers=False, skip_github=True, cancel_check=cancel_check)
    return orchestrator.finalize_stage(out, r, d.run_stats, skip_papers=False, skip_github=True, fresh=False, completed=completed)


def find_duplicate(out: Path, *, title: str, arxiv_id: str | None = None, doi: str | None = None) -> "DiscoveredPaper | None":
    """Return an existing catalog paper matching the seed metadata, or None.

    Match precedence: exact arXiv id, exact DOI, then lenient fuzzy title
    (token_set_ratio >= 85, via utils.fuzzy_title_match).
    """
    from ndif_citations.output import load_existing_papers
    from ndif_citations.utils import fuzzy_title_match, normalize_arxiv_id

    existing = load_existing_papers(out)
    ax = normalize_arxiv_id(arxiv_id) if arxiv_id else None
    if ax:
        for p in existing:
            if p.arxiv_id and p.arxiv_id == ax:
                return p
    if doi:
        for p in existing:
            if p.doi and p.doi.lower() == doi.lower():
                return p
    if title and title.strip():
        for p in existing:
            if p.title and fuzzy_title_match(title, p.title):
                return p
    return None


_DOI_HOSTS = {"doi.org", "dx.doi.org", "www.doi.org"}
_ARXIV_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}


def s2_id_for_url(url: str) -> str:
    """Semantic Scholar paper id for a URL: DOI: > ARXIV: > URL: (S2 resolves arXiv/ACL/ACM/bioRxiv URLs).

    Ids come only from the URL's host + path (never its query or another host's path), so the
    stored url can't point at a different paper than the metadata.
    """
    from urllib.parse import unquote, urlparse

    from ndif_citations.utils import extract_arxiv_id_from_url
    u = urlparse(url)
    host = u.netloc.lower()
    if host in _DOI_HOSTS and u.path.startswith("/10."):
        return f"DOI:{unquote(u.path[1:]).rstrip('/')}"
    if url.startswith("10."):
        return f"DOI:{url}"
    if host in _ARXIV_HOSTS:
        arxiv = extract_arxiv_id_from_url(url)
        if arxiv:
            return f"ARXIV:{arxiv}"
    return f"URL:{url}"


def fetch_s2_metadata(s2_id: str):
    """S2 paper object for *s2_id*, or None on any failure."""
    try:
        from ndif_citations import config  # manual_add has no module-level config import
        from semanticscholar import SemanticScholar
        sch = SemanticScholar(api_key=config.S2_API_KEY) if config.S2_API_KEY else SemanticScholar()
        return sch.get_paper(s2_id, fields=config.S2_FIELDS)
    except Exception as exc:  # noqa: BLE001
        logger.warning("S2 lookup failed for %s: %s", s2_id, exc)
        return None


def apply_s2_metadata(paper, s2_paper) -> None:
    """Copy title/authors/abstract/venue/date/ids/OA-PDF from an S2 paper onto *paper*."""
    paper.title = getattr(s2_paper, "title", None) or paper.title
    authors_list = getattr(s2_paper, "authors", []) or []
    paper.authors = ", ".join(
        a.get("name", "") if isinstance(a, dict) else getattr(a, "name", str(a)) for a in authors_list
    )
    paper.abstract = getattr(s2_paper, "abstract", None)
    paper.venue = getattr(s2_paper, "venue", "") or ""
    pub_date_str = getattr(s2_paper, "publicationDate", None)
    if pub_date_str:
        from datetime import date
        try:
            paper.publication_date = date.fromisoformat(pub_date_str) if isinstance(pub_date_str, str) else pub_date_str
            paper.year = paper.publication_date.year
        except (ValueError, AttributeError):
            pass
    external_ids = getattr(s2_paper, "externalIds", {}) or {}
    paper.doi = external_ids.get("DOI") or paper.doi
    paper.arxiv_id = paper.arxiv_id or external_ids.get("ArXiv")
    paper.s2_paper_id = getattr(s2_paper, "paperId", None)
    open_access = getattr(s2_paper, "openAccessPdf", None)
    if open_access:
        paper.pdf_url = open_access.get("url") if isinstance(open_access, dict) else getattr(open_access, "url", None)


def resolve_seed_from_url(url: str):
    """A MANUAL_ADD seed with real metadata for *url*, or None if S2 can't resolve it."""
    from ndif_citations.models import DiscoveredPaper, DiscoverySource
    s2 = fetch_s2_metadata(s2_id_for_url(url))
    if s2 is None or not getattr(s2, "title", None):
        return None
    paper = DiscoveredPaper(title=s2.title, url=url, source=DiscoverySource.MANUAL_ADD)
    apply_s2_metadata(paper, s2)
    return paper


def add_paper_by_url(
    out: Path,
    url: str,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict:
    """Process a single paper by URL and append it to the output.

    Logic mirrors the ``add`` CLI command (Task 1.9 fix):
      1. Parse arXiv ID from URL.
      2. Build a ``DiscoveredPaper(source=MANUAL_ADD)``.
      3. Optionally look up S2 metadata (failure degrades gracefully).
      4. ``enrich_papers([paper])``
      5. ``load_existing_papers(out)``
      6. ``route_papers([paper], existing)``
      7. ``process_papers(decisions, out, cancel_check=cancel_check)``
      8. ``merge_papers(existing, processed)``
      9. ``write_outputs(merged, out, run_stats)``

    Args:
        out: Output directory (the pipeline's ``out/`` tree).
        url: URL of the paper to add (ideally an arXiv URL or any http URL).
        cancel_check: Optional callable that returns ``True`` when the caller
            has requested cancellation (forwarded to ``process_papers``).

    Returns:
        A dict with::

            {
                "added":      bool,          # True iff run_stats.new_papers > 0
                "merge_key":  str,           # paper.merge_key()
                "title":      str,
                "new_papers": int,           # run_stats.new_papers
            }
    """
    from ndif_citations.extract import enrich_papers
    from ndif_citations.models import DiscoveredPaper, DiscoverySource
    from ndif_citations.output import load_existing_papers, merge_papers, write_outputs
    from ndif_citations.process import process_papers
    from ndif_citations.router import route_papers
    from ndif_citations.utils import extract_arxiv_id_from_url

    arxiv_id = extract_arxiv_id_from_url(url)

    paper = DiscoveredPaper(
        title="[Pending metadata lookup]",
        url=url,
        arxiv_id=arxiv_id,
        source=DiscoverySource.MANUAL_ADD,
    )

    # Try to look up metadata via S2; network failure degrades gracefully.
    if arxiv_id:
        s2_paper = fetch_s2_metadata(f"ARXIV:{arxiv_id}")
        if s2_paper:
            apply_s2_metadata(paper, s2_paper)

    # Enrich → Route → Process → Merge → Write
    papers = enrich_papers([paper])
    paper = papers[0]

    existing = load_existing_papers(out)
    decisions = route_papers(papers, existing)
    processed = process_papers(decisions, out, cancel_check=cancel_check)

    merged, run_stats = merge_papers(existing, processed)
    write_outputs(merged, out, run_stats)

    return {
        "added": run_stats.new_papers > 0,
        "merge_key": paper.merge_key(),
        "title": paper.title,
        "new_papers": run_stats.new_papers,
    }
