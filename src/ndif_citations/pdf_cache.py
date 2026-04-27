"""PDF acquisition and caching module.

Provides multi-source PDF resolution:
1. Existing PDF URL from paper metadata
2. ArXiv direct PDF construction
3. Unpaywall API lookup for open access

Caches PDFs locally with standardized naming:
- arxiv-{arxiv_id}.pdf for arXiv papers
- doi-{slugify(doi)}.pdf for DOI-identified papers
- {slugify(title[:50])}.pdf for others
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import requests

from ndif_citations.models import DiscoveredPaper
from ndif_citations.utils import (
    _crossref_pdf_link,
    extract_arxiv_id_from_url,
    query_crossref,
    query_unpaywall,
    rate_limit_sleep,
    slugify,
)

logger = logging.getLogger(__name__)


def _test_pdf_url(url: str, timeout: int = 10) -> bool:
    """Test if a PDF URL is accessible.

    Uses HEAD first; falls back to a Range GET when servers (e.g. MDPI, ACM DL)
    return 403/405 to HEAD requests.  Returns True if the response looks like a PDF.
    """
    headers = {
        "User-Agent": "NDIFCitationTracker/1.0 (academic research; https://ndif.us)"
    }
    try:
        resp = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        if resp.status_code in (403, 405):
            # Some publishers block HEAD but allow GET; fetch first 1 KB only
            resp = requests.get(
                url,
                headers={**headers, "Range": "bytes=0-1023"},
                timeout=timeout,
                allow_redirects=True,
                stream=True,
            )
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").lower()
        return "pdf" in content_type or "/pdf" in url.lower() or url.lower().endswith(".pdf")
    except Exception:
        return False


def resolve_pdf_url(paper: DiscoveredPaper) -> Optional[str]:
    """Try multiple sources to get a working PDF URL.

    Tries in order:
    1. Existing paper.pdf_url if valid
    2. ArXiv direct construction (if arxiv_id)
    3. Unpaywall API lookup (if DOI)

    Returns None if no PDF URL can be resolved.
    """
    # 1. Test existing PDF URL
    if paper.pdf_url:
        if _test_pdf_url(paper.pdf_url):
            logger.debug(f"Using existing PDF URL for '{paper.title[:50]}...'")
            return paper.pdf_url
        logger.debug(f"Existing PDF URL failed test: {paper.pdf_url}")

    # 2. ArXiv direct construction (most reliable)
    if paper.arxiv_id:
        arxiv_pdf_url = f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"
        if _test_pdf_url(arxiv_pdf_url):
            logger.debug(f"Using ArXiv direct PDF for '{paper.title[:50]}...'")
            return arxiv_pdf_url
        logger.debug(f"ArXiv PDF not accessible: {arxiv_pdf_url}")

    # 3. Unpaywall API lookup
    if paper.doi:
        logger.debug(f"Trying Unpaywall for DOI: {paper.doi}")
        unpaywall_data = query_unpaywall(paper.doi)

        if unpaywall_data.get("is_oa"):
            best_location = unpaywall_data.get("best_oa_location", {})

            # Direct PDF URL
            pdf_url = best_location.get("url_for_pdf")
            if pdf_url and _test_pdf_url(pdf_url):
                logger.debug(f"Found PDF via Unpaywall for '{paper.title[:50]}...'")
                return pdf_url

            # Try landing page for arXiv redirect
            landing_url = best_location.get("url")
            if landing_url:
                # Check if it's an arXiv paper
                arxiv_id = extract_arxiv_id_from_url(landing_url)
                if arxiv_id:
                    arxiv_pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                    if _test_pdf_url(arxiv_pdf_url):
                        logger.debug(f"Found arXiv via Unpaywall for '{paper.title[:50]}...'")
                        return arxiv_pdf_url

        logger.debug(f"No open access found via Unpaywall for DOI: {paper.doi}")

    # 4. CrossRef link array (some publishers register direct PDF links)
    if paper.doi:
        try:
            cr_data = query_crossref(paper.doi)
            pdf_link = _crossref_pdf_link(cr_data)
            if pdf_link and _test_pdf_url(pdf_link):
                logger.debug(f"Found PDF via CrossRef link for '{paper.title[:50]}'")
                return pdf_link
        except Exception:
            pass

    logger.warning(
        f"No open-access PDF found for '{paper.title[:60]}' "
        f"(doi={paper.doi}, arxiv={paper.arxiv_id}) — paywalled or unavailable"
    )
    return None


def _download_pdf(url: str, dest_path: Path, timeout: int = 30) -> Optional[Path]:
    """Download PDF from URL to destination path.

    Returns path on success, None on failure.
    """
    try:
        headers = {
            "User-Agent": "NDIFCitationTracker/0.1 (academic research; https://ndif.us)"
        }
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "").lower()
        if "text/html" in content_type and not url.endswith(".pdf"):
            logger.warning(f"Downloaded HTML instead of PDF from {url}")
            return None

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.debug(f"Downloaded PDF to {dest_path}")
        return dest_path

    except Exception as e:
        logger.warning(f"Failed to download PDF from {url}: {e}")
        return None


def get_cached_pdf(paper: DiscoveredPaper, output_dir: Path) -> Optional[Path]:
    """Get cached PDF path, downloading if necessary.

    Cache file naming:
    - arxiv-{arxiv_id}.pdf for arXiv papers
    - doi-{slug}.pdf for DOI papers
    - {slugify(title[:50])}.pdf for others

    Returns path to cached PDF, or None if unavailable.
    """
    cache_dir = output_dir / "pdfs"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine cache file name
    if paper.arxiv_id:
        cache_path = cache_dir / f"arxiv-{paper.arxiv_id}.pdf"
    elif paper.doi:
        cache_path = cache_dir / f"doi-{slugify(paper.doi)}.pdf"
    else:
        cache_path = cache_dir / f"{slugify(paper.title[:50])}.pdf"

    # Cache hit
    if cache_path.exists():
        logger.debug(f"PDF cache hit: {cache_path.name}")
        # Write back the resolved URL if paper.pdf_url is missing
        if not paper.pdf_url:
            if paper.arxiv_id:
                paper.pdf_url = f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"
            elif paper.doi:
                paper.pdf_url = f"https://doi.org/{paper.doi}"
        return cache_path

    # Cache miss - resolve and download
    pdf_url = resolve_pdf_url(paper)
    if pdf_url:
        downloaded = _download_pdf(pdf_url, cache_path)
        if downloaded:
            paper.pdf_url = pdf_url  # write back the resolved URL
            return downloaded

    # All sources exhausted — clear any stale/fake pdf_url that was stored
    if paper.pdf_url and not _test_pdf_url(paper.pdf_url):
        logger.warning(
            f"No accessible PDF found for '{paper.title[:60]}' — clearing stale pdf_url"
        )
        paper.pdf_url = None

    return None


def download_pdf(url: str, dest_dir: Path | None = None, timeout: int = 30) -> Optional[Path]:
    """Download a PDF from a URL (legacy wrapper).

    DEPRECATED: Use get_cached_pdf() instead for new code.
    """
    logger.warning("download_pdf() is deprecated, use get_cached_pdf()")
    return _download_pdf(url, dest_dir / "temp_paper.pdf" if dest_dir else Path("temp_paper.pdf"), timeout)
