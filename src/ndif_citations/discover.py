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
from ndif_citations.models import DetailCategory, DiscoveredPaper, DiscoveredRepo, DiscoverySource
from ndif_citations.utils import (
    _fetch_repo_readme,
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

        # Venue — prefer a published (non-repository) location from locations[]
        # Repositories (arXiv, DSpace@MIT, HAL, etc.) are preprint hosts; the
        # actual conference/journal lives at type=conference|journal.
        venue = ""
        for loc in work.get("locations", []):
            src = loc.get("source") or {}
            src_name = src.get("display_name") or ""
            src_type = src.get("type") or ""
            if not src_name:
                continue
            if src_type == "repository":
                continue
            if "arxiv" in src_name.lower():
                continue
            venue = src_name
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

def discover_github_dependents(raw_dir: Path | None = None) -> list[DiscoveredRepo]:
    """Scrape GitHub dependents page for repos using nnsight.

    Page-level checkpointing: each page is persisted immediately; failed runs
    can resume from the last successful page on the next invocation.
    """
    import time
    import shutil

    checkpoint_dir = (raw_dir / "github_dependents_checkpoint") if raw_dir else None
    repos: list[dict] = []
    start_url: str | None = config.GITHUB_DEPENDENTS_URL
    start_page: int = 0

    # Resume from checkpoint if one exists
    if checkpoint_dir and checkpoint_dir.exists():
        pages = sorted(checkpoint_dir.glob("page_*.json"), key=lambda p: int(p.stem.split("_")[1]))
        if pages:
            for page_file in pages:
                with open(page_file) as f:
                    page_data = json.load(f)
                repos.extend(page_data.get("repos", []))
                start_url = page_data.get("next_url")
                start_page = int(page_file.stem.split("_")[1]) + 1
            logger.info(f"Resuming GitHub dependents scrape from checkpoint (page {start_page})")

    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    url: str | None = start_url
    page_count = start_page

    while url and page_count < 20:
        page_repos: list[dict] = []
        next_url: str | None = None
        success = False

        for attempt in range(1, 4):  # 3 retries
            try:
                headers = {"User-Agent": "NDIFCitationTracker/0.1 (academic research)"}
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                # Find dependent repos
                for row in soup.select("[data-test-id='dg-repo-pkg-dependent']"):
                    repo_link = row.select_one("a[data-hovercard-type='repository']")
                    if repo_link:
                        repo_name = repo_link.get("href", "").strip("/")
                        page_repos.append({
                            "name": repo_name,
                            "url": f"https://github.com/{repo_name}",
                        })

                # Find next page
                next_link = soup.select_one("a.BtnGroup-item:last-child")
                if next_link and "Next" in next_link.text:
                    next_url = next_link.get("href")
                    if next_url and not next_url.startswith("http"):
                        next_url = f"https://github.com{next_url}"
                else:
                    next_url = None

                success = True
                break  # Page scraped successfully

            except Exception as e:
                logger.warning(f"Page {page_count} attempt {attempt}/3 failed: {e}")
                if attempt < 3:
                    time.sleep(5 * attempt)

        if not success:
            logger.error(f"GitHub dependents scrape failed after 3 retries on page {page_count}. Checkpoint saved.")
            break  # Leave checkpoint dir in place for resumption

        repos.extend(page_repos)

        # Persist page to checkpoint
        if checkpoint_dir:
            page_file = checkpoint_dir / f"page_{page_count}.json"
            with open(page_file, "w") as f:
                json.dump({"repos": page_repos, "next_url": next_url}, f, indent=2)

        url = next_url
        page_count += 1
        rate_limit_sleep(config.GITHUB_RATE_LIMIT_SLEEP, "GitHub")

    # On successful completion: clean up checkpoint and write consolidated raw
    completed_naturally = url is None  # loop ended because no next_url
    if checkpoint_dir and checkpoint_dir.exists() and completed_naturally:
        shutil.rmtree(checkpoint_dir)
        logger.debug("Cleaned up GitHub dependents checkpoint directory")

    if raw_dir and repos and completed_naturally:
        with open(raw_dir / "github_dependents_raw.json", "w") as f:
            json.dump(repos, f, indent=2)
    elif raw_dir and repos and not completed_naturally:
        logger.info("Partial scrape saved in checkpoint; github_dependents_raw.json not written")

    # Build DiscoveredRepo shells (no API calls, no README fetching)
    discovered_repos: list[DiscoveredRepo] = []
    for repo in repos:
        try:
            owner_repo = repo["name"]
            parts = owner_repo.split("/", 1)
            if len(parts) != 2:
                continue
            owner, repo_name = parts
            discovered_repos.append(DiscoveredRepo(
                owner=owner,
                repo=repo_name,
                url=repo["url"],
                readme_arxiv_ids=[],
            ))
        except Exception as e:
            logger.debug(f"Failed to process repo {repo.get('name')}: {e}")

    logger.info(f"GitHub dependents: {len(repos)} repos found, {len(discovered_repos)} DiscoveredRepo entries")
    return discovered_repos



def enrich_repos_from_github_api(
    repos: list[DiscoveredRepo],
) -> tuple[list[DiscoveredRepo], dict[str, int]]:
    """Enrich repos with GitHub API metadata.

    Returns (kept_repos, removal_counts) where removal_counts is a dict with keys
    "404", "rename_redirect", "archived" tracking how many repos were dropped for
    each reason this run.

    Marks repos for removal if:
    - GitHub returns 404 (deleted or private)
    - GitHub redirected to a different owner/repo (renamed)
    - GitHub says the repo is archived

    Rate-limited or transport-failed repos survive with stale data (never removed
    due to transient API issues).
    """
    from ndif_citations.utils import _github_api_get
    from datetime import date as date_type

    kept: list[DiscoveredRepo] = []
    removal_counts: dict[str, int] = {"404": 0, "rename_redirect": 0, "archived": 0}

    for repo in repos:
        path = f"/repos/{repo.owner}/{repo.repo}"
        data, status = _github_api_get(path)

        rate_sleep = (
            config.GITHUB_API_RATE_LIMIT_AUTH
            if config.GITHUB_TOKEN
            else config.GITHUB_API_RATE_LIMIT_ANON
        )
        rate_limit_sleep(rate_sleep, "GitHub API")

        if status == 0:
            # Transport error — keep the repo with existing data
            kept.append(repo)
            continue

        if status == 404:
            logger.info(f"Removing {repo.owner}/{repo.repo}: 404")
            removal_counts["404"] += 1
            continue  # Drop this repo

        if status in (403, 429):
            # Rate-limited — keep repo, don't remove
            kept.append(repo)
            continue

        if data is None:
            # Unexpected non-200/404 status — keep with existing data
            kept.append(repo)
            continue

        # Check for rename (GitHub transparently redirects, but full_name differs)
        full_name = data.get("full_name", "")
        expected = f"{repo.owner}/{repo.repo}"
        if full_name and full_name.lower() != expected.lower():
            logger.info(f"Removing {repo.owner}/{repo.repo}: rename_redirect (now {full_name})")
            removal_counts["rename_redirect"] += 1
            continue  # Drop — let the new name enter as NEW next scrape

        # Check for newly archived
        if data.get("archived", False):
            logger.info(f"Removing {repo.owner}/{repo.repo}: archived")
            removal_counts["archived"] += 1
            continue  # Drop

        # Fill activity fields
        repo.description = data.get("description") or repo.description
        repo.stars = data.get("stargazers_count")
        repo.forks = data.get("forks_count")
        repo.archived = data.get("archived", False)
        repo.is_fork = data.get("fork", False)
        repo.language = data.get("language")

        license_data = data.get("license")
        if license_data:
            repo.license = license_data.get("spdx_id") or license_data.get("name")

        repo.topics = data.get("topics") or []

        pushed_at = data.get("pushed_at")
        if pushed_at:
            try:
                from datetime import datetime
                repo.last_commit = datetime.fromisoformat(
                    pushed_at.replace("Z", "+00:00")
                ).date()
            except Exception:
                pass

        repo.has_metadata = True

        # Single README fetch — extract arXiv IDs and classify NDIF usage
        readme_text = _fetch_repo_readme(repo.owner, repo.repo)
        if readme_text:
            # Extract all arXiv IDs
            arxiv_pattern = r'https?://arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})'
            matches = re.findall(arxiv_pattern, readme_text)
            seen: set[str] = set()
            for arxiv_id in matches:
                normalized = normalize_arxiv_id(arxiv_id)
                if normalized not in seen:
                    seen.add(normalized)
                    repo.readme_arxiv_ids.append(normalized)

            # NDIF keyword classification (regex + substring), with boilerplate filter.
            # We collect all match spans, then drop the ones that fall entirely inside
            # a boilerplate phrase (e.g. "join the NDIF Discord"). If any real match
            # survives, the repo is uses_ndif; otherwise default uses_nnsight.
            readme_lower = readme_text.lower()

            boilerplate_spans: list[tuple[int, int]] = []
            for phrase in config.NDIF_README_NEGATIVE_PATTERNS:
                p_lower = phrase.lower()
                start = 0
                while True:
                    idx = readme_lower.find(p_lower, start)
                    if idx == -1:
                        break
                    boilerplate_spans.append((idx, idx + len(p_lower)))
                    start = idx + 1

            def _inside_boilerplate(s: int, e: int) -> bool:
                return any(bs <= s and e <= be for bs, be in boilerplate_spans)

            real_match_found = False
            for pattern in config.NDIF_README_KEYWORDS_REGEX:
                for m in re.finditer(pattern, readme_text):
                    if not _inside_boilerplate(m.start(), m.end()):
                        real_match_found = True
                        break
                if real_match_found:
                    break
            if not real_match_found:
                for kw in config.NDIF_README_KEYWORDS_SUBSTR:
                    kw_lower = kw.lower()
                    start = 0
                    while True:
                        idx = readme_lower.find(kw_lower, start)
                        if idx == -1:
                            break
                        if not _inside_boilerplate(idx, idx + len(kw_lower)):
                            real_match_found = True
                            break
                        start = idx + 1
                    if real_match_found:
                        break

            if real_match_found:
                repo.detail_category = DetailCategory.USES_NDIF
                repo.classification_reason = "ndif_keyword_match"
            else:
                repo.detail_category = DetailCategory.USES_NNSIGHT
                repo.classification_reason = "github_dependent"

            repo.has_classification = True
        else:
            # No README available (404 or transport error). The default uses_nnsight
            # classification IS valid for nnsight dependents (per original PRD US-006:
            # "all nnsight dependents use the library by definition"), so mark
            # has_classification=True to avoid wasteful FILL_GAPS retries against the
            # same 404.
            repo.detail_category = DetailCategory.USES_NNSIGHT
            repo.classification_reason = "github_dependent"
            repo.has_classification = True

        # Capture parent_full_name from API response
        parent = data.get("parent")
        if parent:
            repo.parent_full_name = parent.get("full_name")

        repo.linked_paper_url = _detect_linked_paper(readme_text, repo.readme_arxiv_ids)

        repo.content_hash = repo.compute_content_hash()
        kept.append(repo)

    return kept, removal_counts


def _arxiv_id_year(arxiv_id: str) -> int:
    """Return the 2-digit year prefix of an arXiv ID (e.g. '2407.14561' -> 24)."""
    try:
        return int(arxiv_id.split('.')[0][:2])
    except (ValueError, IndexError):
        return 0


def _detect_linked_paper(readme_text: str, all_arxiv_ids: list[str]) -> Optional[str]:
    """Detect the repo's own linked paper from README using 5-tier priority.

    Priority (first match wins):
    1. BibTeX block → first arXiv ID found in any BibTeX entry
    2. arXiv ID under a Citation/Paper/Reference/How-to-cite section header
    3. Exactly one arXiv ID in the entire README → return it
    4. Multiple IDs → most recent post-2020 (highest YYMM prefix where year >= 20)
    5. Otherwise → None

    Pre-2020 IDs (YYMM < 2001) are only eligible if no later candidate exists.
    """
    from ndif_citations.utils import parse_readme_sections, extract_bibtex_arxiv_ids

    if not readme_text or not all_arxiv_ids:
        return None

    # Tier 1: BibTeX block
    bibtex_ids = extract_bibtex_arxiv_ids(readme_text)
    if bibtex_ids:
        return f"https://arxiv.org/abs/{bibtex_ids[0]}"

    # Tier 2: Citation/Paper/Reference section
    sections = parse_readme_sections(readme_text)
    citation_headers = {"citation", "how to cite", "how to cite us", "paper", "reference", "references", "cite"}
    arxiv_id_pattern = re.compile(r'https?://arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})')

    for header, body in sections.items():
        if any(ch in header for ch in citation_headers):
            ids_in_section = arxiv_id_pattern.findall(body)
            if ids_in_section:
                return f"https://arxiv.org/abs/{normalize_arxiv_id(ids_in_section[0])}"

    # Tier 3: Exactly one post-2020 arXiv ID in the entire README
    post_2020_all = [aid for aid in all_arxiv_ids if _arxiv_id_year(aid) >= 20]
    if len(post_2020_all) == 1:
        return f"https://arxiv.org/abs/{post_2020_all[0]}"

    # Tier 4: Multiple post-2020 IDs → most recent (highest YYMM prefix)
    if len(post_2020_all) > 1:
        best = max(post_2020_all, key=lambda aid: aid.split('.')[0])
        return f"https://arxiv.org/abs/{best}"

    # Tier 5: None (no post-2020 candidates)
    return None


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


def _unlink_shared_template_papers(repos: list[DiscoveredRepo]) -> set[str]:
    """Post-enrichment cross-repo pass: unlink template-inherited paper URLs.

    When >= config.SHARED_PAPER_THRESHOLD repos share the same linked_paper_url,
    all but the highest-star repo in that group have their linked_paper_url cleared.

    Returns a set of merge_key() values for repos that were unlinked (used by
    tagging engine for Tier-2 course detection).
    """
    from collections import Counter

    # Count occurrences of each linked_paper_url
    url_counts: Counter[str] = Counter(
        r.linked_paper_url for r in repos if r.linked_paper_url
    )

    # Find URLs at or above threshold
    template_urls = {url for url, count in url_counts.items() if count >= config.SHARED_PAPER_THRESHOLD}

    unlinked_keys: set[str] = set()

    for url in template_urls:
        group = [r for r in repos if r.linked_paper_url == url]
        # Identify the "original": highest stars (None treated as 0)
        original = max(group, key=lambda r: r.stars or 0)

        for repo in group:
            if repo.merge_key() == original.merge_key():
                continue  # Keep the original's link
            repo.linked_paper_url = None
            unlinked_keys.add(repo.merge_key())
            logger.info(f"Unlinked shared template paper for {repo.owner}/{repo.repo} (was {url})")

    return unlinked_keys


def _tag_repo_type(repo: DiscoveredRepo, unlinked_set: set[str]) -> str:
    """Determine repo_type for a single repo using a 7-rule decision tree.

    Priority (first match wins):

    COURSE — deterministic to fuzzy:
      1. parent_full_name in KNOWN_COURSE_SOURCES → course
      2. repo was unlinked by shared-paper pass AND stars==0 AND no description → course
      3. Any COURSE_NAME_PATTERNS substring in repo.repo or repo.description → course

    RESEARCH — strong signal of legitimate work:
      4. detail_category == uses_ndif → research
      5. linked_paper_url is set → research
      6. stars >= 6 AND description is non-empty → research

    EXPERIMENT (default):
      7. Everything else → experiment
    """
    # --- COURSE ---
    # Tier 1: forked from known course source
    if repo.parent_full_name and repo.parent_full_name in config.KNOWN_COURSE_SOURCES:
        return "course"

    # Tier 2: shared-paper link was unlinked AND 0 stars AND no description
    if repo.merge_key() in unlinked_set and (repo.stars or 0) == 0 and not repo.description:
        return "course"

    # Tier 3: name or description contains a course pattern
    name_and_desc = (repo.repo + " " + (repo.description or "")).lower()
    for pattern in config.COURSE_NAME_PATTERNS:
        if pattern.lower() in name_and_desc:
            return "course"

    # --- RESEARCH ---
    # Tier 4: uses NDIF infrastructure
    if repo.detail_category and repo.detail_category.value == "uses_ndif":
        return "research"

    # Tier 5: has a linked paper (survived shared-paper cleanup)
    if repo.linked_paper_url:
        return "research"

    # Tier 6: notable (stars >= 6 AND has description)
    if (repo.stars or 0) >= 6 and repo.description:
        return "research"

    # --- EXPERIMENT (default) ---
    return "experiment"


def link_repos_to_papers(
    repos: list[DiscoveredRepo],
    papers: list[DiscoveredPaper],
) -> None:
    """Cross-link repos to papers using linked_paper_url set during enrichment.

    For each repo whose linked_paper_url matches a paper's arxiv_id:
      - Sets paper.github_repo_url = repo.url (only if not already set)

    Note: linked_paper_url is now set by _detect_linked_paper() in
    enrich_repos_from_github_api(), not here.
    """
    # Build O(1) lookup: arxiv_id -> paper
    by_arxiv: dict[str, DiscoveredPaper] = {}
    for paper in papers:
        if paper.arxiv_id:
            by_arxiv[paper.arxiv_id] = paper

    for repo in repos:
        if not repo.linked_paper_url:
            continue

        # Extract arXiv ID from the linked_paper_url
        matched_id = extract_arxiv_id_from_url(repo.linked_paper_url)
        if not matched_id:
            continue

        # Set paper.github_repo_url if this arXiv ID matches a known paper
        matched_paper = by_arxiv.get(matched_id)
        if matched_paper and not matched_paper.github_repo_url:
            matched_paper.github_repo_url = repo.url
            logger.debug(
                f"Cross-linked: paper '{matched_paper.title[:50]}' "
                f"<-> repo {repo.owner}/{repo.repo}"
            )
