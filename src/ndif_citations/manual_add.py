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


def find_duplicate(out, *, title: str, arxiv_id: str | None = None, doi: str | None = None):
    """Return an existing catalog paper matching the seed metadata, or None.

    Match precedence: exact arXiv id, exact DOI, then fuzzy title (rapidfuzz >= 90).
    """
    from ndif_citations.output import load_existing_papers
    from ndif_citations.utils import is_duplicate, normalize_arxiv_id

    existing = load_existing_papers(out)
    ax = normalize_arxiv_id(arxiv_id) if arxiv_id else None
    if ax:
        for p in existing:
            if p.arxiv_id and p.arxiv_id == ax:
                return p
    if doi:
        for p in existing:
            if p.doi and p.doi == doi:
                return p
    if title and title.strip():
        for p in existing:
            if p.title and is_duplicate(title, p.title, threshold=90.0):
                return p
    return None


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
    from ndif_citations import config
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
        try:
            from semanticscholar import SemanticScholar

            sch = (
                SemanticScholar(api_key=config.S2_API_KEY)
                if config.S2_API_KEY
                else SemanticScholar()
            )
            s2_paper = sch.get_paper(f"ARXIV:{arxiv_id}", fields=config.S2_FIELDS)
            if s2_paper:
                paper.title = getattr(s2_paper, "title", paper.title)
                authors_list = getattr(s2_paper, "authors", []) or []
                paper.authors = ", ".join(
                    a.get("name", "") if isinstance(a, dict) else getattr(a, "name", str(a))
                    for a in authors_list
                )
                paper.abstract = getattr(s2_paper, "abstract", None)
                paper.venue = getattr(s2_paper, "venue", "") or ""
                pub_date_str = getattr(s2_paper, "publicationDate", None)
                if pub_date_str:
                    from datetime import date

                    try:
                        if isinstance(pub_date_str, str):
                            paper.publication_date = date.fromisoformat(pub_date_str)
                        else:
                            paper.publication_date = pub_date_str
                        paper.year = paper.publication_date.year
                    except (ValueError, AttributeError):
                        pass
                external_ids = getattr(s2_paper, "externalIds", {}) or {}
                paper.doi = external_ids.get("DOI")
                paper.s2_paper_id = getattr(s2_paper, "paperId", None)
                open_access = getattr(s2_paper, "openAccessPdf", None)
                if open_access:
                    paper.pdf_url = (
                        open_access.get("url")
                        if isinstance(open_access, dict)
                        else getattr(open_access, "url", None)
                    )
        except Exception as exc:
            logger.warning("S2 lookup failed for %s: %s", url, exc)

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
