"""Phase 2: Metadata extraction and enrichment."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ndif_citations import config
from ndif_citations.models import DiscoveredPaper
from ndif_citations.utils import (
    generate_bibtex,
    query_arxiv_api,
    query_crossref,
    query_openreview_venue,
    query_s2_publication_venue,
    rate_limit_sleep,
)
from ndif_citations.venue import (
    is_preprint_sentinel,
    normalize_venue,
    resolve_venue,
)

logger = logging.getLogger(__name__)


def enrich_papers(papers: list[DiscoveredPaper], raw_dir: Path | None = None) -> list[DiscoveredPaper]:
    """Enrich papers with formatted venues, peer-review status, BibTeX, and image paths."""
    logger.info(f"Enriching metadata for {len(papers)} papers...")

    # Step 1: Collect raw venue strings from all upstream APIs into a per-paper
    # dict, then resolve each paper's canonical venue in one pass. This replaces
    # the old "first non-placeholder wins" logic that locked in long-form names.
    enrich_via_external_apis(papers)

    # Step 2: Fill remaining affiliation gaps using OpenAlex (also opportunistically
    # enriches venue from OpenAlex's primary_location.source.display_name).
    _enrich_affiliations_from_openalex(papers, raw_dir)

    # Step 3: Per-paper post-processing — peer-review, venue_type, bibtex, URL.
    for paper in papers:
        paper.peer_reviewed = detect_peer_review(paper.venue)
        paper.venue_type = detect_venue_type(paper.venue)

        if not paper.bibtex:
            paper.bibtex = generate_bibtex(
                title=paper.title,
                authors=paper.authors,
                year=paper.year,
                venue=paper.venue,
                url=paper.url,
                arxiv_id=paper.arxiv_id,
                doi=paper.doi,
            )

        paper.url = _best_url(paper)

    logger.info("Metadata enrichment complete")
    return papers


def detect_peer_review(venue: str) -> bool:
    """Detect if a paper is peer-reviewed based on its venue."""
    venue_lower = venue.lower()

    # ArXiv / preprints are not peer-reviewed
    preprints = config.KNOWN_VENUES.get("preprint_servers", [])
    for preprint in preprints:
        if preprint.lower() in venue_lower:
            return False

    # Known conferences and journals are peer-reviewed
    for conf in config.KNOWN_VENUES.get("conferences", []):
        if conf.lower() in venue_lower:
            return True

    for journal in config.KNOWN_VENUES.get("journals", []):
        if journal.lower() in venue_lower:
            return True

    # Workshops are generally peer-reviewed
    if "workshop" in venue_lower:
        return True

    # Default: unknown, assume not
    return False


def detect_venue_type(venue: str) -> str:
    """Classify venue as conference, workshop, journal, or preprint."""
    venue_lower = venue.lower()

    if any(p.lower() in venue_lower for p in config.KNOWN_VENUES.get("preprint_servers", [])):
        return "preprint"

    if "workshop" in venue_lower:
        return "workshop"

    for journal in config.KNOWN_VENUES.get("journals", []):
        if journal.lower() in venue_lower:
            return "journal"

    for conf in config.KNOWN_VENUES.get("conferences", []):
        if conf.lower() in venue_lower:
            return "conference"

    return "preprint"


def _best_url(paper: DiscoveredPaper) -> str:
    """Select the best URL for a paper. Prefer: OpenReview > arXiv > DOI > original."""
    url = paper.url

    # If we have an OpenReview URL, prefer it
    if "openreview.net" in (url or ""):
        return url

    # If arXiv, use the abstract page
    if paper.arxiv_id:
        arxiv_url = f"https://arxiv.org/abs/{paper.arxiv_id}"
        # Only override if current URL is not a better venue
        if not url or "arxiv" in url or not any(
            domain in url for domain in ["openreview.net", "aclanthology.org", "dl.acm.org"]
        ):
            return arxiv_url

    # If DOI and no URL
    if paper.doi and not url:
        return f"https://doi.org/{paper.doi}"

    return url or ""


def enrich_via_external_apis(papers: list[DiscoveredPaper]) -> None:
    """Collect raw venue signals from CrossRef/S2/OpenReview/arXiv, resolve, normalize.

    Pass 1: hit each upstream API once per paper, accumulating raw venue strings
    into a per-paper sources dict. Affiliations and author-name refresh ride along.
    Pass 2: feed all signals to `venue.resolve_venue()` which applies the priority
    cascade (DOI prefix → arXiv comment → OpenAlex → S2 → CrossRef → OpenReview →
    existing → ArXiv fallback) and `normalize_venue()` (acronym map, suffix strip).
    """
    logger.info("Enriching papers via CrossRef, S2, OpenReview, arXiv...")

    # Per-paper raw venue dict, keyed by id(paper) so we don't depend on a hashable model
    sources_by_id: dict[int, dict[str, str]] = {id(p): {} for p in papers}

    # ---- CrossRef: venue + affiliations for non-arXiv DOIs --------------------
    for paper in papers:
        doi = paper.doi or ""
        if not doi or "arxiv" in doi.lower():
            continue
        try:
            data = query_crossref(doi)
            if not data:
                continue

            container = data.get("container_title") or ""
            event = data.get("event_name") or ""
            sources_by_id[id(paper)]["crossref"] = container or event

            if not paper.affiliations:
                aff_set: set[str] = set()
                for author in data.get("authors", []):
                    for aff in author.get("affiliation", []):
                        name = aff.get("name") if isinstance(aff, dict) else str(aff)
                        if name:
                            aff_set.add(name)
                if aff_set:
                    paper.affiliations = ", ".join(sorted(aff_set))

            rate_limit_sleep(
                getattr(config, "CROSSREF_RATE_LIMIT_SLEEP", 0.2), "CrossRef"
            )
        except Exception as e:
            logger.debug(f"CrossRef enrichment failed for '{paper.title[:40]}': {e}")

    # ---- S2 publicationVenue (only when current venue is a placeholder) -------
    logger.info("Querying S2 publicationVenue for papers with placeholder venues...")
    for paper in papers:
        if not is_preprint_sentinel(paper.venue):
            continue
        if not paper.arxiv_id and not paper.doi:
            continue
        try:
            pv = query_s2_publication_venue(paper.arxiv_id, paper.doi)
            name = pv.get("name") or ""
            if name:
                sources_by_id[id(paper)]["s2"] = name
            rate_limit_sleep(
                config.S2_RATE_LIMIT_SLEEP if not config.S2_API_KEY else 0.5,
                "S2 publicationVenue",
            )
        except Exception as e:
            logger.debug(f"S2 venue lookup failed for '{paper.title[:40]}': {e}")

    # ---- OpenReview (still gated on placeholder, since search is title-fuzzy) -
    logger.info("Querying OpenReview for venue resolution...")
    for paper in papers:
        if not is_preprint_sentinel(paper.venue):
            continue
        if not paper.title:
            continue
        try:
            data = query_openreview_venue(paper.title)
            venue_name = data.get("venue") or ""
            if venue_name:
                sources_by_id[id(paper)]["openreview"] = venue_name
            rate_limit_sleep(0.3, "OpenReview")
        except Exception as e:
            logger.debug(f"OpenReview lookup failed for '{paper.title[:40]}': {e}")

    # ---- arXiv API: authors, affiliations, journal_ref, comment ---------------
    arxiv_ids = [p.arxiv_id for p in papers if p.arxiv_id]
    if arxiv_ids:
        logger.info(f"Fetching arXiv metadata for {len(arxiv_ids)} papers...")
        arxiv_data = query_arxiv_api(arxiv_ids)

        for paper in papers:
            if not paper.arxiv_id or paper.arxiv_id not in arxiv_data:
                continue
            d = arxiv_data[paper.arxiv_id]
            # Refresh author names from arXiv (authoritative, clean Unicode)
            if d.get("authors"):
                paper.authors = ", ".join(d["authors"])
            if not paper.affiliations and d.get("affiliations"):
                paper.affiliations = ", ".join(d["affiliations"])
            # Stash venue signals from arXiv comment / journal-ref
            if d.get("journal_ref"):
                sources_by_id[id(paper)]["arxiv_journal_ref"] = d["journal_ref"]
            if d.get("comment"):
                sources_by_id[id(paper)]["arxiv_comment"] = d["comment"]

    # ---- Pass 2: resolve canonical venue per paper ---------------------------
    for paper in papers:
        old = paper.venue
        new, source = resolve_venue(paper, sources_by_id[id(paper)])
        if new != old:
            logger.debug(
                f"venue: '{paper.title[:50]}' | {old!r} -> {new!r} "
                f"(source={source}, signals={list(sources_by_id[id(paper)].keys())})"
            )
        # Year reconciliation: when a high-confidence source produced an
        # explicit year (e.g. "ACL 2025" from a 2025.acl-long.X DOI), trust
        # the venue year over paper.year (which is often the arXiv upload
        # year, 6-12 months ahead of the conference proceedings).
        if source in ("doi_prefix", "arxiv_comment_parsed") and new:
            m = re.search(r"\b(20\d{2})\b", new)
            if m:
                venue_year = int(m.group(1))
                if venue_year != paper.year:
                    logger.info(
                        f"year reconciled: '{paper.title[:40]}' "
                        f"{paper.year} -> {venue_year} (via {source})"
                    )
                    paper.year = venue_year
        paper.venue = new
        paper.venue_source = source


def _extract_affiliations_from_authorships(authorships: list[dict]) -> str:
    """Extract sorted, deduplicated institution names from an OpenAlex authorships list."""
    affiliations_set: set[str] = set()
    for authorship in authorships:
        for inst in authorship.get("institutions", []):
            name = inst.get("display_name")
            if name:
                affiliations_set.add(name)
    return ", ".join(sorted(affiliations_set))


def _openalex_fetch_work(identifier: str, by: str = "id") -> dict | None:
    """Fetch a single OpenAlex work.

    Args:
        identifier: The OpenAlex work ID (e.g. "W4415020698"), or a filter string.
        by: "id" for direct fetch, "filter" for search-style filter.
    """
    import requests

    params: dict = {}
    if config.OPENALEX_EMAIL:
        params["mailto"] = config.OPENALEX_EMAIL

    try:
        if by == "id":
            resp = requests.get(
                f"{config.OPENALEX_BASE_URL}/works/{identifier}",
                params=params, timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
        else:
            params["filter"] = identifier
            params["per-page"] = 1
            resp = requests.get(
                f"{config.OPENALEX_BASE_URL}/works",
                params=params, timeout=10,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    return results[0]
    except Exception as e:
        logger.debug(f"OpenAlex fetch failed ({by}={identifier}): {e}")
    return None


def _enrich_affiliations_from_openalex(papers: list[DiscoveredPaper],
                                        raw_dir: Path | None = None) -> None:
    """For papers without affiliations, look them up in OpenAlex.

    Priority: direct openalex_id fetch > arxiv_id URL filter > doi filter > title search.
    """
    from ndif_citations.utils import rate_limit_sleep

    papers_needing_affiliations = [p for p in papers if not p.affiliations]

    if not papers_needing_affiliations:
        return

    logger.info(f"Looking up affiliations for {len(papers_needing_affiliations)} papers via OpenAlex...")

    for paper in papers_needing_affiliations:
        try:
            work: dict | None = None

            if paper.openalex_id:
                # Strip full URL prefix to get bare work ID (e.g. "W4415020698")
                work_id = paper.openalex_id.replace("https://openalex.org/", "")
                work = _openalex_fetch_work(work_id, by="id")

            if not work and paper.arxiv_id:
                work = _openalex_fetch_work(
                    f"locations.landing_page_url:https://arxiv.org/abs/{paper.arxiv_id}",
                    by="filter",
                )

            if not work and paper.doi:
                work = _openalex_fetch_work(f"doi:{paper.doi}", by="filter")

            if not work and paper.title:
                work = _openalex_fetch_work(
                    f"title.search:{paper.title[:100]}", by="filter"
                )

            if work:
                affs = _extract_affiliations_from_authorships(work.get("authorships", []))
                if affs:
                    paper.affiliations = affs

            rate_limit_sleep(config.OPENALEX_RATE_LIMIT_SLEEP, "OpenAlex affiliations")

        except Exception as e:
            logger.debug(f"Failed to get affiliations for '{paper.title}': {e}")


def check_venue_upgrades(new_papers: list[DiscoveredPaper],
                         existing_papers: list[DiscoveredPaper]) -> list[str]:
    """Check if any existing arXiv papers have been published at a conference.

    Returns list of messages about upgrades.
    """
    upgrades: list[str] = []

    existing_by_key: dict[str, DiscoveredPaper] = {}
    for p in existing_papers:
        if p.arxiv_id:
            existing_by_key[f"arxiv:{p.arxiv_id}"] = p
        if p.doi:
            existing_by_key[f"doi:{p.doi}"] = p

    for new_p in new_papers:
        key = None
        if new_p.arxiv_id:
            key = f"arxiv:{new_p.arxiv_id}"
        elif new_p.doi:
            key = f"doi:{new_p.doi}"

        if key and key in existing_by_key:
            old_p = existing_by_key[key]
            old_type = detect_venue_type(old_p.venue)
            new_type = detect_venue_type(new_p.venue)

            if old_type == "preprint" and new_type in ("conference", "workshop", "journal"):
                upgrades.append(
                    f'"{old_p.title}": {old_p.venue} → {new_p.venue}'
                )
                # Update the existing paper
                old_p.venue = new_p.venue
                old_p.peer_reviewed = detect_peer_review(new_p.venue)
                old_p.venue_type = new_type

    return upgrades
