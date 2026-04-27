"""Phase 1: Paper Discovery via Semantic Scholar, OpenAlex, and GitHub dependents."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from ndif_citations import config
from ndif_citations.models import DiscoveredPaper, DiscoverySource
from ndif_citations.utils import (
    extract_arxiv_id_from_doi,
    extract_arxiv_id_from_url,
    is_duplicate,
    looks_like_pdf_url,
    normalize_arxiv_id,
    rate_limit_sleep,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source A: Semantic Scholar
# ---------------------------------------------------------------------------

def discover_s2_citations(raw_dir: Path | None = None) -> list[DiscoveredPaper]:
    """Fetch all papers that cite the NDIF/NNsight seed paper via Semantic Scholar."""
    from semanticscholar import SemanticScholar

    logger.info("Discovering papers via Semantic Scholar...")

    kwargs = {}
    if config.S2_API_KEY:
        kwargs["api_key"] = config.S2_API_KEY

    sch = SemanticScholar(**kwargs)
    papers: list[DiscoveredPaper] = []

    try:
        # Get citations with pagination — consume iterator ONCE
        citations = sch.get_paper_citations(
            config.SEED_S2_ID,
            fields=config.S2_FIELDS,
        )

        # Collect all citation objects into a list (single API traversal)
        citation_objects = []
        for c in citations:
            cp = getattr(c, "paper", None)
            if cp:
                citation_objects.append(cp)
            rate_limit_sleep(config.S2_RATE_LIMIT_SLEEP if not config.S2_API_KEY else 0.5,
                             "S2")

        # Save raw response
        if raw_dir:
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_data = [
                {"paperId": getattr(cp, "paperId", None), "title": getattr(cp, "title", None)}
                for cp in citation_objects
            ]
            with open(raw_dir / "s2_citations_raw.json", "w") as f:
                json.dump(raw_data, f, indent=2, default=str)

        # Parse papers from the collected objects (no second API call)
        for cp in citation_objects:
            if not getattr(cp, "title", None):
                continue

            paper = _s2_paper_to_discovered(cp)
            if paper:
                papers.append(paper)

    except Exception as e:
        logger.error(f"Semantic Scholar discovery failed: {e}")

    logger.info(f"Semantic Scholar: found {len(papers)} citing papers")
    return papers


def _s2_paper_to_discovered(cp) -> Optional[DiscoveredPaper]:
    """Convert an S2 paper object to a DiscoveredPaper."""
    try:
        title = getattr(cp, "title", None)
        if not title:
            return None

        # Extract identifiers
        external_ids = getattr(cp, "externalIds", {}) or {}
        arxiv_id = external_ids.get("ArXiv")
        doi = external_ids.get("DOI")
        s2_id = getattr(cp, "paperId", None)
        # Fallback: extract arXiv ID from arXiv-format DOIs (10.48550/arXiv.XXXX.XXXXX)
        if not arxiv_id and doi:
            arxiv_id = extract_arxiv_id_from_doi(doi)

        # Authors
        authors_list = getattr(cp, "authors", []) or []
        authors = ", ".join(a.get("name", "") if isinstance(a, dict) else getattr(a, "name", str(a))
                           for a in authors_list)

        # Venue — prefer structured publicationVenue over bare venue string
        pub_venue = getattr(cp, "publicationVenue", None)
        if pub_venue:
            venue = (pub_venue.get("name") if isinstance(pub_venue, dict)
                     else getattr(pub_venue, "name", "")) or ""
        else:
            venue = getattr(cp, "venue", "") or ""

        # Publication date
        pub_date_str = getattr(cp, "publicationDate", None)
        year = 0
        pub_date = None
        if pub_date_str:
            try:
                from datetime import date
                if isinstance(pub_date_str, str):
                    pub_date = date.fromisoformat(pub_date_str)
                else:
                    pub_date = pub_date_str
                year = pub_date.year
            except (ValueError, AttributeError):
                pass

        # URL
        url = getattr(cp, "url", "") or ""
        # Prefer arXiv URL if available
        if arxiv_id and not url:
            url = f"https://arxiv.org/abs/{arxiv_id}"

        # PDF URL — validate that S2's openAccessPdf.url is a real PDF link, not a DOI redirect
        open_access = getattr(cp, "openAccessPdf", None)
        pdf_url = None
        if open_access:
            candidate = (open_access.get("url") if isinstance(open_access, dict)
                         else getattr(open_access, "url", None))
            if looks_like_pdf_url(candidate):
                pdf_url = candidate

        # Abstract
        abstract = getattr(cp, "abstract", None)

        return DiscoveredPaper(
            title=title,
            arxiv_id=normalize_arxiv_id(arxiv_id) if arxiv_id else None,
            doi=doi,
            s2_paper_id=s2_id,
            authors=authors,
            venue=venue,
            year=year,
            publication_date=pub_date,
            url=url,
            pdf_url=pdf_url,
            abstract=abstract,
            source=DiscoverySource.S2_CITATION,
        )
    except Exception as e:
        logger.warning(f"Failed to parse S2 paper: {e}")
        return None


# ---------------------------------------------------------------------------
# Source B: OpenAlex
# ---------------------------------------------------------------------------

def discover_openalex(raw_dir: Path | None = None) -> list[DiscoveredPaper]:
    """Search OpenAlex for papers mentioning nnsight/NDIF in their full text."""
    logger.info("Discovering papers via OpenAlex...")
    papers: list[DiscoveredPaper] = []
    all_raw: list[dict] = []

    for query in config.OPENALEX_SEARCH_QUERIES:
        try:
            query_papers, query_raw = _openalex_search(query)
            papers.extend(query_papers)
            all_raw.extend(query_raw)
            rate_limit_sleep(config.OPENALEX_RATE_LIMIT_SLEEP, "OpenAlex")
        except Exception as e:
            logger.warning(f"OpenAlex search failed for query '{query}': {e}")

    # Save raw
    if raw_dir and all_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)
        with open(raw_dir / "openalex_raw.json", "w") as f:
            json.dump(all_raw, f, indent=2, default=str)

    logger.info(f"OpenAlex: found {len(papers)} papers")
    return papers


def _openalex_search(query: str) -> tuple[list[DiscoveredPaper], list[dict]]:
    """Execute a single OpenAlex full-text search query.

    Returns (papers, raw_results) for both processing and raw data saving.
    """
    papers: list[DiscoveredPaper] = []
    raw_results: list[dict] = []
    params: dict = {
        "filter": f"fulltext.search:{query}",
        "per-page": 200,
        "select": "id,title,authorships,primary_location,locations,publication_date,doi,open_access,biblio,abstract_inverted_index",
    }
    if config.OPENALEX_EMAIL:
        params["mailto"] = config.OPENALEX_EMAIL

    try:
        resp = requests.get(f"{config.OPENALEX_BASE_URL}/works", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for work in data.get("results", []):
            raw_results.append(work)
            paper = _openalex_work_to_discovered(work)
            if paper:
                papers.append(paper)

    except Exception as e:
        logger.warning(f"OpenAlex API request failed: {e}")

    return papers, raw_results


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


def _openalex_work_to_discovered(work: dict) -> Optional[DiscoveredPaper]:
    """Convert an OpenAlex work to a DiscoveredPaper."""
    try:
        title = work.get("title")
        if not title:
            return None

        # Authors and affiliations
        authorships = work.get("authorships", [])
        authors = ", ".join(
            a.get("author", {}).get("display_name", "")
            for a in authorships
            if a.get("author", {}).get("display_name")
        )
        affiliations_set: set[str] = set()
        for a in authorships:
            for inst in a.get("institutions", []):
                name = inst.get("display_name")
                if name:
                    affiliations_set.add(name)
        affiliations = ", ".join(sorted(affiliations_set))

        # Date
        pub_date_str = work.get("publication_date", "")
        year = 0
        pub_date = None
        if pub_date_str:
            try:
                from datetime import date
                pub_date = date.fromisoformat(pub_date_str)
                year = pub_date.year
            except ValueError:
                pass

        # DOI
        doi_url = work.get("doi", "")
        doi = doi_url.replace("https://doi.org/", "").replace("http://doi.org/", "") if doi_url else None

        # URL
        primary_loc = work.get("primary_location", {}) or {}
        source = primary_loc.get("source", {}) or {}
        landing_url = primary_loc.get("landing_page_url", "")
        pdf_url = primary_loc.get("pdf_url")

        # Try to extract arXiv ID from URL first, then from DOI
        arxiv_id = extract_arxiv_id_from_url(landing_url) if landing_url else None
        if not arxiv_id and doi:
            arxiv_id = extract_arxiv_id_from_doi(doi)

        # Venue — prefer a non-arXiv location from the full locations[] array
        venue = ""
        for loc in work.get("locations", []):
            src = loc.get("source") or {}
            src_name = src.get("display_name") or ""
            if src_name and "arxiv" not in src_name.lower():
                venue = src_name
                # Also pick up PDF URL from this preferred location if missing
                if not pdf_url and loc.get("pdf_url"):
                    pdf_url = loc.get("pdf_url")
                break
        if not venue:
            venue = source.get("display_name", "") or ""

        # Abstract
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

        # OpenAlex ID
        openalex_id = work.get("id", "")

        return DiscoveredPaper(
            title=title,
            arxiv_id=arxiv_id,
            doi=doi if doi else None,
            openalex_id=openalex_id,
            authors=authors,
            affiliations=affiliations,
            venue=venue,
            year=year,
            publication_date=pub_date,
            url=landing_url or (f"https://doi.org/{doi}" if doi else ""),
            pdf_url=pdf_url,
            abstract=abstract if abstract else None,
            source=DiscoverySource.OPENALEX_FULLTEXT,
        )
    except Exception as e:
        logger.warning(f"Failed to parse OpenAlex work: {e}")
        return None


# ---------------------------------------------------------------------------
# Source C: GitHub Dependents
# ---------------------------------------------------------------------------

def discover_github_dependents(raw_dir: Path | None = None) -> list[DiscoveredPaper]:
    """Scrape GitHub dependents page for repos using nnsight, check for paper links."""
    logger.info("Discovering papers via GitHub dependents...")
    papers: list[DiscoveredPaper] = []
    repos: list[dict] = []

    url: str | None = config.GITHUB_DEPENDENTS_URL
    page_count = 0

    while url and page_count < 20:  # Safety limit
        try:
            headers = {
                "User-Agent": "NDIFCitationTracker/0.1 (academic research)"
            }
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Find dependent repos
            for row in soup.select("[data-test-id='dg-repo-pkg-dependent']"):
                repo_link = row.select_one("a[data-hovercard-type='repository']")
                if repo_link:
                    repo_name = repo_link.get("href", "").strip("/")
                    repos.append({
                        "name": repo_name,
                        "url": f"https://github.com/{repo_name}",
                    })

            # Find next page
            next_link = soup.select_one("a.BtnGroup-item:last-child")
            if next_link and "Next" in next_link.text:
                url = next_link.get("href")
                if url and not url.startswith("http"):
                    url = f"https://github.com{url}"
            else:
                url = None

            page_count += 1
            rate_limit_sleep(config.GITHUB_RATE_LIMIT_SLEEP, "GitHub")

        except Exception as e:
            logger.warning(f"GitHub dependents scraping failed: {e}")
            break

    # Save raw
    if raw_dir and repos:
        with open(raw_dir / "github_dependents_raw.json", "w") as f:
            json.dump(repos, f, indent=2)

    # Check each repo for arxiv links
    for repo in repos:
        try:
            arxiv_papers = _check_repo_for_papers(repo["name"])
            for p in arxiv_papers:
                p.github_repo_url = repo["url"]
            papers.extend(arxiv_papers)
            rate_limit_sleep(config.GITHUB_RATE_LIMIT_SLEEP, "GitHub")
        except Exception as e:
            logger.debug(f"Failed to check repo {repo['name']}: {e}")

    logger.info(f"GitHub dependents: {len(repos)} repos found, {len(papers)} with papers")
    return papers


def _check_repo_for_papers(repo_name: str) -> list[DiscoveredPaper]:
    """Check a GitHub repo's README for arXiv links."""
    papers: list[DiscoveredPaper] = []
    try:
        # Fetch README via GitHub API
        api_url = f"https://api.github.com/repos/{repo_name}/readme"
        headers = {
            "Accept": "application/vnd.github.raw",
            "User-Agent": "NDIFCitationTracker/0.1",
        }
        resp = requests.get(api_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []

        readme_text = resp.text

        # Find arXiv links
        arxiv_pattern = r'https?://arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})'
        matches = re.findall(arxiv_pattern, readme_text)

        for arxiv_id in set(matches):
            arxiv_id = normalize_arxiv_id(arxiv_id)
            papers.append(DiscoveredPaper(
                title=f"[From GitHub: {repo_name}]",  # Placeholder, will be enriched
                arxiv_id=arxiv_id,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                source=DiscoverySource.GITHUB_DEPENDENT,
                github_repo_url=f"https://github.com/{repo_name}",
            ))

    except Exception as e:
        logger.debug(f"Failed to fetch README for {repo_name}: {e}")

    return papers


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_papers(all_papers: list[DiscoveredPaper]) -> list[DiscoveredPaper]:
    """Deduplicate papers by arXiv ID, DOI, and title similarity.

    When duplicates are found, we prefer data from S2 > OpenAlex > GitHub,
    but merge in affiliations from OpenAlex.
    """
    logger.info(f"Deduplicating {len(all_papers)} papers...")
    seen_arxiv: dict[str, int] = {}  # arxiv_id -> index in result
    seen_doi: dict[str, int] = {}    # doi -> index in result
    result: list[DiscoveredPaper] = []

    # Sort by source priority: S2 first, then OpenAlex, then GitHub
    source_priority = {
        DiscoverySource.S2_CITATION: 0,
        DiscoverySource.OPENALEX_FULLTEXT: 1,
        DiscoverySource.GITHUB_DEPENDENT: 2,
        DiscoverySource.MANUAL_ADD: 0,
    }
    all_papers.sort(key=lambda p: source_priority.get(p.source, 3))

    for paper in all_papers:
        # Ignore exactly excluded seed papers
        if paper.title.lower().strip() in config.EXCLUDED_PAPER_TITLES:
            continue

        # Check arXiv ID
        if paper.arxiv_id and paper.arxiv_id in seen_arxiv:
            idx = seen_arxiv[paper.arxiv_id]
            _merge_paper_data(result[idx], paper)
            continue

        # Check DOI
        if paper.doi and paper.doi in seen_doi:
            idx = seen_doi[paper.doi]
            _merge_paper_data(result[idx], paper)
            continue

        # Check title similarity
        is_dup = False
        for i, existing in enumerate(result):
            if is_duplicate(paper.title, existing.title):
                _merge_paper_data(existing, paper)
                is_dup = True
                break

        if not is_dup:
            idx = len(result)
            result.append(paper)
            if paper.arxiv_id:
                seen_arxiv[paper.arxiv_id] = idx
            if paper.doi:
                seen_doi[paper.doi] = idx

    logger.info(f"After deduplication: {len(result)} unique papers")
    return result


def _merge_paper_data(primary: DiscoveredPaper, secondary: DiscoveredPaper) -> None:
    """Merge data from secondary into primary, filling gaps."""
    # Fill in missing identifiers
    if not primary.arxiv_id and secondary.arxiv_id:
        primary.arxiv_id = secondary.arxiv_id
    if not primary.doi and secondary.doi:
        primary.doi = secondary.doi
    if not primary.s2_paper_id and secondary.s2_paper_id:
        primary.s2_paper_id = secondary.s2_paper_id
    if not primary.openalex_id and secondary.openalex_id:
        primary.openalex_id = secondary.openalex_id

    # Merge affiliations (OpenAlex is better for this)
    if not primary.affiliations and secondary.affiliations:
        primary.affiliations = secondary.affiliations

    # Fill missing fields
    if not primary.abstract and secondary.abstract:
        primary.abstract = secondary.abstract
    if not primary.pdf_url and secondary.pdf_url:
        primary.pdf_url = secondary.pdf_url
    if not primary.url and secondary.url:
        primary.url = secondary.url
    if not primary.authors and secondary.authors:
        primary.authors = secondary.authors
    if not primary.venue and secondary.venue:
        primary.venue = secondary.venue
    if not primary.year and secondary.year:
        primary.year = secondary.year
    if not primary.publication_date and secondary.publication_date:
        primary.publication_date = secondary.publication_date

    # Track GitHub repo if found via GitHub
    if secondary.github_repo_url and not primary.github_repo_url:
        primary.github_repo_url = secondary.github_repo_url
