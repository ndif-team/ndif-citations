"""Phase 2: Metadata extraction and enrichment."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


def enrich_papers(papers: list[DiscoveredPaper], raw_dir: Path | None = None) -> list[DiscoveredPaper]:
    """Enrich papers with formatted venues, peer-review status, BibTeX, and image paths."""
    logger.info(f"Enriching metadata for {len(papers)} papers...")

    for paper in papers:
        # Format venue
        paper.venue = format_venue(paper.venue, paper.year)

        # Detect peer-review status
        paper.peer_reviewed = detect_peer_review(paper.venue)
        paper.venue_type = detect_venue_type(paper.venue)

        # Generate BibTeX if missing
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

        # Determine best URL (prefer OpenReview > arXiv > DOI)
        paper.url = _best_url(paper)

    # Enrich venue + author names + affiliations via external APIs
    enrich_via_external_apis(papers)

    # Fill remaining affiliation gaps using OpenAlex
    _enrich_affiliations_from_openalex(papers, raw_dir)

    logger.info("Metadata enrichment complete")
    return papers


def format_venue(venue_raw: str, year: int) -> str:
    """Format venue string to match the website convention.

    Rules:
      - Conference papers: "ICLR 2025", "NeurIPS 2024"
      - Workshop papers: "NeurIPS 2024 Workshop on Scientific Methods..."
      - ArXiv only: "ArXiv 2025"
      - Journal: "COLM 2024"
    """
    if not venue_raw or venue_raw.strip() == "":
        return f"ArXiv {year}" if year else "ArXiv"

    venue = venue_raw.strip()

    # Already has a year at the end?
    import re
    has_year = bool(re.search(r'\b20\d{2}\b', venue))

    # Check for arXiv
    if venue.lower() in ("arxiv", "arxiv.org", ""):
        return f"ArXiv {year}" if year else "ArXiv"

    # Check for bioRxiv
    if "biorxiv" in venue.lower():
        return f"BiorXiv {year}" if year else "BiorXiv"

    # Workshop — keep full name, don't append year if already present
    if "workshop" in venue.lower():
        return venue if has_year else f"{venue} {year}" if year else venue

    # Known conferences — abbreviate if we recognize them
    known_conferences = config.KNOWN_VENUES.get("conferences", [])
    for conf in known_conferences:
        if conf.lower() in venue.lower():
            # Use the abbreviation + year
            if not has_year and year:
                return f"{venue} {year}"
            return venue

    # Default: append year if missing
    if not has_year and year:
        return f"{venue} {year}"

    return venue


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


_PREPRINT_VENUE_HINTS = (
    "arxiv", "biorxiv", "medrxiv", "ssrn", "dspace", "hal archives",
    "repository", "openreview"  # plain "OpenReview" is the preprint host, not a venue
)


def _is_placeholder_venue(venue: str) -> bool:
    """True if `venue` is empty or matches a preprint/repository placeholder."""
    if not venue or not venue.strip():
        return True
    v = venue.lower()
    return any(hint in v for hint in _PREPRINT_VENUE_HINTS)


def enrich_via_external_apis(papers: list[DiscoveredPaper]) -> None:
    """Enrich venue, author names, and affiliations via CrossRef, S2, OpenReview, arXiv.

    Step A — CrossRef: for papers with non-arXiv DOIs, fetch venue + affiliations.
    Step B — S2 publicationVenue: universal lookup by arxiv_id or doi for any paper
             whose venue is still preprint-flavoured (regardless of discovery source).
    Step C — OpenReview: catches ICLR/ICML/NeurIPS papers that S2 hasn't yet updated.
    Step D — arXiv API: batch-fetch clean author names + affiliation fallback.
    """
    logger.info("Enriching papers via CrossRef and arXiv APIs...")

    # Step A: CrossRef — venue resolution + affiliations for published papers
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
            best_venue = container or event
            # Override when the current venue is a preprint/repository placeholder
            if best_venue and _is_placeholder_venue(paper.venue):
                logger.debug(f"CrossRef venue update: '{paper.title[:40]}' -> {best_venue}")
                paper.venue = best_venue

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

    # Step B: Universal S2 publicationVenue lookup
    # Catches papers like pyvene that S2 has the conference for, but were
    # discovered via OpenAlex (so the discovery-time S2 parse never ran).
    logger.info("Querying S2 publicationVenue for papers with placeholder venues...")
    for paper in papers:
        if not _is_placeholder_venue(paper.venue):
            continue
        if not paper.arxiv_id and not paper.doi:
            continue
        try:
            pv = query_s2_publication_venue(paper.arxiv_id, paper.doi)
            name = pv.get("name") or ""
            # Skip if S2 also says it's just on arXiv
            if name and not _is_placeholder_venue(name):
                logger.debug(f"S2 publicationVenue update: '{paper.title[:40]}' -> {name}")
                paper.venue = name
            rate_limit_sleep(
                config.S2_RATE_LIMIT_SLEEP if not config.S2_API_KEY else 0.5,
                "S2 publicationVenue",
            )
        except Exception as e:
            logger.debug(f"S2 venue lookup failed for '{paper.title[:40]}': {e}")

    # Step C: OpenReview — for papers still without a real venue, fuzzy-match by title
    logger.info("Querying OpenReview for venue resolution...")
    for paper in papers:
        if not _is_placeholder_venue(paper.venue):
            continue
        if not paper.title:
            continue
        try:
            data = query_openreview_venue(paper.title)
            venue_name = data.get("venue") or ""
            if venue_name:
                logger.debug(f"OpenReview venue update: '{paper.title[:40]}' -> {venue_name}")
                paper.venue = venue_name
            rate_limit_sleep(0.3, "OpenReview")
        except Exception as e:
            logger.debug(f"OpenReview lookup failed for '{paper.title[:40]}': {e}")

    # Step D: arXiv API batch fetch — clean author names + affiliation fallback
    arxiv_ids = [p.arxiv_id for p in papers if p.arxiv_id]
    if not arxiv_ids:
        return

    logger.info(f"Fetching arXiv metadata for {len(arxiv_ids)} papers...")
    arxiv_data = query_arxiv_api(arxiv_ids)

    for paper in papers:
        if not paper.arxiv_id or paper.arxiv_id not in arxiv_data:
            continue
        d = arxiv_data[paper.arxiv_id]
        # Always refresh author names from arXiv (authoritative, clean Unicode)
        if d.get("authors"):
            paper.authors = ", ".join(d["authors"])
        # Fill affiliations if still missing (arXiv tags rarely populated, but try)
        if not paper.affiliations and d.get("affiliations"):
            paper.affiliations = ", ".join(d["affiliations"])


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
