"""Utility functions for PDF processing, text extraction, deduplication, and figure detection."""

from __future__ import annotations

import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Title slugification
# ---------------------------------------------------------------------------

def slugify(title: str) -> str:
    """Convert title to a filename-safe slug matching the website's convention.

    Example: "Language Models Use Trigonometry" -> "Language-Models-Use-Trigonometry"
    """
    slug = re.sub(r'[^\w\s-]', '', title)
    slug = re.sub(r'\s+', '-', slug.strip())
    return slug


# ---------------------------------------------------------------------------
# PDF handling
# ---------------------------------------------------------------------------

def download_pdf(url: str, dest_dir: Path | None = None, timeout: int = 30) -> Optional[Path]:
    """Download a PDF from a URL. Returns the local path or None on failure."""
    try:
        headers = {
            "User-Agent": "NDIFCitationTracker/0.1 (academic research; https://ndif.us)"
        }
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            logger.warning(f"Aborting download: URL points to an HTML page (likely a paywall/login): {url}")
            return None
        elif "pdf" not in content_type and not url.endswith(".pdf"):
            logger.warning(f"URL may not be a PDF (Content-Type: {content_type}): {url}")

        if dest_dir:
            dest_dir.mkdir(parents=True, exist_ok=True)
            path = dest_dir / "temp_paper.pdf"
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            path = Path(tmp.name)
            tmp.close()

        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        return path

    except Exception as e:
        logger.warning(f"Failed to download PDF from {url}: {e}")
        return None


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract full text from a PDF using pymupdf."""
    try:
        import fitz  # pymupdf

        doc = fitz.open(str(pdf_path))
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        logger.warning(f"Failed to extract text from {pdf_path}: {e}")
        return ""


def extract_ndif_context(pdf_path: Path, keywords: list[str] | None = None,
                         window: int = 500, max_excerpts: int = 5) -> str:
    """Extract text around NDIF/nnsight mentions in a PDF."""
    from ndif_citations.config import NDIF_KEYWORDS, CONTEXT_WINDOW, MAX_CONTEXT_EXCERPTS

    keywords = keywords or NDIF_KEYWORDS
    window = window or CONTEXT_WINDOW
    max_excerpts = max_excerpts or MAX_CONTEXT_EXCERPTS

    full_text = extract_text_from_pdf(pdf_path)
    if not full_text:
        return "No text could be extracted from the PDF."

    text_lower = full_text.lower()
    contexts: list[str] = []

    for kw in keywords:
        idx = 0
        while True:
            idx = text_lower.find(kw.lower(), idx)
            if idx == -1:
                break
            start = max(0, idx - window)
            end = min(len(full_text), idx + len(kw) + window)
            contexts.append(full_text[start:end])
            idx += len(kw)

    if not contexts:
        return "No direct mentions of NDIF or nnsight found in the paper text."

    return "\n---\n".join(contexts[:max_excerpts])


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def is_duplicate(title_a: str, title_b: str, threshold: float = 90.0) -> bool:
    """Check if two titles are similar enough to be considered duplicates."""
    return fuzz.ratio(title_a.lower().strip(), title_b.lower().strip()) >= threshold


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Normalize arXiv ID by stripping version suffix and URL prefix.

    Examples:
        "2407.14561v2" -> "2407.14561"
        "https://arxiv.org/abs/2407.14561" -> "2407.14561"
    """
    # Strip URL prefix
    for prefix in ["https://arxiv.org/abs/", "http://arxiv.org/abs/",
                    "https://arxiv.org/pdf/", "http://arxiv.org/pdf/"]:
        if arxiv_id.startswith(prefix):
            arxiv_id = arxiv_id[len(prefix):]
            break

    # Strip .pdf suffix
    if arxiv_id.endswith(".pdf"):
        arxiv_id = arxiv_id[:-4]

    # Strip version suffix
    arxiv_id = re.sub(r'v\d+$', '', arxiv_id)

    return arxiv_id.strip()


def extract_arxiv_id_from_url(url: str) -> Optional[str]:
    """Try to extract an arXiv ID from a URL."""
    match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})', url)
    if match:
        return normalize_arxiv_id(match.group(1))
    return None


def extract_arxiv_id_from_doi(doi: str) -> Optional[str]:
    """Extract arXiv ID from an arXiv DOI (10.48550/arXiv.XXXX.XXXXX).

    Example: "10.48550/arXiv.2504.14107" -> "2504.14107"
    """
    match = re.search(r'10\.48550/[aA]r[Xx]iv\.(\d{4}\.\d{4,5})', doi)
    if match:
        return normalize_arxiv_id(match.group(1))
    return None


def looks_like_pdf_url(url: str) -> bool:
    """Return False for URLs that are clearly DOI/landing-page redirects, not actual PDFs.

    A DOI URL (doi.org/... or dx.doi.org/...) without an explicit /pdf path segment
    almost always redirects to an HTML landing page. Reject those so pdf_cache
    doesn't store a fake pdf_url.
    """
    if not url:
        return False
    # Accept anything that ends with .pdf or explicitly contains /pdf in the path
    if url.lower().endswith(".pdf") or "/pdf" in url.lower():
        return True
    # Reject bare DOI redirects
    if re.match(r'https?://(?:dx\.)?doi\.org/', url):
        return False
    return True


# ---------------------------------------------------------------------------
# Rate limiting helper
# ---------------------------------------------------------------------------

def rate_limit_sleep(seconds: float, label: str = "") -> None:
    """Sleep for rate limiting with optional logging."""
    if seconds > 0:
        if label:
            logger.debug(f"Rate limiting ({label}): sleeping {seconds}s")
        time.sleep(seconds)


# ---------------------------------------------------------------------------
# Unpaywall API for open access PDF lookup
# ---------------------------------------------------------------------------

def query_unpaywall(doi: str) -> dict:
    """Query Unpaywall API for open access PDF location.

    Unpaywall API: https://api.unpaywall.org/v2/{doi}?email={email}
    - Free, no key required
    - Rate limit: 100,000 calls/day
    - Returns open access status and PDF URLs

    Args:
        doi: DOI to look up (e.g., "10.3390/ai7030092")

    Returns:
        Dict with structure:
        {
            "is_oa": bool,
            "oa_status": "gold|green|bronze|closed",
            "best_oa_location": {
                "url_for_pdf": str | None,
                "url": str  # landing page
            }
        }
        Returns empty dict if API call fails.
    """
    from ndif_citations import config

    email = config.UNPAYWALL_EMAIL
    if not email:
        logger.debug(f"No UNPAYWALL_EMAIL configured, skipping Unpaywall lookup for {doi}")
        return {}

    url = f"https://api.unpaywall.org/v2/{doi}"
    params = {"email": email}

    try:
        headers = {
            "User-Agent": "NDIFCitationTracker/0.1 (academic research; https://ndif.us)"
        }
        resp = requests.get(url, params=params, headers=headers, timeout=15)

        if resp.status_code == 404:
            logger.debug(f"DOI not found in Unpaywall: {doi}")
            return {}

        resp.raise_for_status()
        data = resp.json()

        result = {
            "is_oa": data.get("is_oa", False),
            "oa_status": data.get("oa_status", "closed"),
            "best_oa_location": data.get("best_oa_location", {}),
            "first_oa_location": data.get("first_oa_location", {}),
        }

        logger.debug(f"Unpaywall: {doi} -> OA status: {result['oa_status']}")

        rate_limit_sleep(config.UNPAYWALL_RATE_LIMIT_SLEEP, "Unpaywall")
        return result

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 422:
            logger.debug(f"Invalid DOI format for Unpaywall: {doi}")
        else:
            logger.warning(f"Unpaywall API error for {doi}: {e}")
        return {}
    except Exception as e:
        logger.warning(f"Failed to query Unpaywall for {doi}: {e}")
        return {}


# ---------------------------------------------------------------------------
# CrossRef API for venue resolution via DOI
# ---------------------------------------------------------------------------

def query_crossref(doi: str) -> dict:
    """Query CrossRef for metadata about a DOI.

    Returns a normalized dict with keys: container_title, event_name, publisher,
    type, authors (list of CrossRef author objects with optional affiliation arrays).
    Returns {} on any failure.
    """
    from ndif_citations import config

    try:
        url = f"https://api.crossref.org/works/{doi}"
        email = config.OPENALEX_EMAIL or "noreply@ndif.us"
        ua = f"NDIFCitationTracker/1.0 (academic research; mailto:{email})"
        resp = requests.get(url, headers={"User-Agent": ua}, timeout=10)
        if resp.status_code != 200:
            return {}
        msg = resp.json().get("message", {})
        container_titles = msg.get("container-title") or []
        event = msg.get("event") or {}
        return {
            "container_title": container_titles[0] if container_titles else "",
            "event_name": event.get("name") or "",
            "publisher": msg.get("publisher") or "",
            "type": msg.get("type") or "",
            "authors": msg.get("author") or [],
        }
    except Exception as e:
        logger.debug(f"CrossRef query failed for {doi}: {e}")
        return {}


def _crossref_pdf_link(crossref_data: dict) -> Optional[str]:
    """Extract a direct PDF URL from a CrossRef response, if one is listed."""
    # CrossRef 'link' array has objects with content-type and URL
    for link in crossref_data.get("link", []):
        ct = link.get("content-type", "")
        url = link.get("URL", "")
        if "pdf" in ct.lower() and url:
            return url
    return None


# ---------------------------------------------------------------------------
# arXiv API (Atom feed) for author names and affiliations
# ---------------------------------------------------------------------------

def query_arxiv_api(arxiv_ids: list[str]) -> dict[str, dict]:
    """Batch-query the arXiv Atom API for author names and affiliations.

    Sends requests in groups of 100 IDs (arXiv's recommended batch size).
    Returns a mapping of arxiv_id -> {"authors": [...], "affiliations": [...], "categories": [...]}.
    Affiliations are only populated when <arxiv:affiliation> tags are present.
    """
    from ndif_citations import config
    import xml.etree.ElementTree as ET

    if not arxiv_ids:
        return {}

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    results: dict[str, dict] = {}
    batch_size = getattr(config, "ARXIV_API_BATCH_SIZE", 100)
    base_url = getattr(config, "ARXIV_API_BASE_URL", "https://export.arxiv.org/api/query")

    for i in range(0, len(arxiv_ids), batch_size):
        batch = arxiv_ids[i : i + batch_size]
        try:
            resp = requests.get(
                base_url,
                params={"id_list": ",".join(batch), "max_results": len(batch)},
                timeout=30,
                headers={"User-Agent": "NDIFCitationTracker/1.0 (academic research; https://ndif.us)"},
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.text)

            for entry in root.findall("atom:entry", ns):
                # arXiv ID is in <id>http://arxiv.org/abs/XXXX.XXXXX</id>
                id_el = entry.find("atom:id", ns)
                if id_el is None or not id_el.text:
                    continue
                arxiv_id = normalize_arxiv_id(id_el.text.strip())

                authors: list[str] = []
                affiliations: list[str] = []
                aff_set: set[str] = set()
                for author_el in entry.findall("atom:author", ns):
                    name_el = author_el.find("atom:name", ns)
                    if name_el is not None and name_el.text:
                        authors.append(name_el.text.strip())
                    for aff_el in author_el.findall("arxiv:affiliation", ns):
                        if aff_el.text and aff_el.text.strip():
                            aff_set.add(aff_el.text.strip())
                affiliations = sorted(aff_set)

                cats: list[str] = []
                for cat_el in entry.findall("atom:category", ns):
                    term = cat_el.get("term")
                    if term:
                        cats.append(term)

                results[arxiv_id] = {
                    "authors": authors,
                    "affiliations": affiliations,
                    "categories": cats,
                }

        except Exception as e:
            logger.warning(f"arXiv API batch query failed (batch starting {batch[0]}): {e}")

        if i + batch_size < len(arxiv_ids):
            rate_limit_sleep(
                getattr(config, "ARXIV_API_RATE_LIMIT_SLEEP", 3.0), "arXiv API"
            )

    return results


# ---------------------------------------------------------------------------
# Figure detection helpers
# ---------------------------------------------------------------------------

MECH_INTERP_TIER_1 = [
    "circuit", "crosscoder", "sae", "steering", "patching",
    "sparse autoencoder", "attribution", "faithfulness",
    "ablation", "activation patching", "logit lens"
]

MECH_INTERP_TIER_2 = [
    "attention", "latents", "neurons", "features", "probing",
    "residual stream", "mlp", "induction", "transformer"
]

MECH_INTERP_TIER_3 = [
    "overview", "architecture", "pipeline", "framework",
    "method", "approach", "structure", "diagram",
    "workflow", "design", "illustration"
]

MECH_INTERP_REJECT = [
    "logo", "header", "banner", "icon", "avatar", "institution",
    "university", "conference", "sponsor", "qr", "barcode"
]

CAPTION_PATTERN = re.compile(
    r'(?:^|\n)(Figure\s+(\d+)[:.]?\s*([^\n]{0,150})|Fig\.?\s+(\d+)[:.]?\s*([^\n]{0,150}))',
    re.IGNORECASE
)


def score_mech_interp_caption(caption_text: str, page_num: int, figure_num: int) -> float:
    """Score a caption based on mechanistic interpretability tiers and page decay."""
    caption_lower = caption_text.lower()
    score = 0.0

    # Penalize reject keywords heavily
    for kw in MECH_INTERP_REJECT:
        if kw in caption_lower:
            score -= 20.0

    # Tiered scoring
    for kw in MECH_INTERP_TIER_1:
        if kw in caption_lower:
            score += 30.0
    for kw in MECH_INTERP_TIER_2:
        if kw in caption_lower:
            score += 20.0
    for kw in MECH_INTERP_TIER_3:
        if kw in caption_lower:
            score += 10.0

    # Standard penalty for deep pages (decay by 2 points per page)
    # Allows earlier figures to win ties.
    score -= (page_num * 2.0)
    
    # Priority for early figures (architecture/overview diagrams usually Fig 1/2)
    if figure_num in [1, 2]:
        score += 35.0
    elif figure_num == 3:
        score += 15.0
        
    return max(score, 0.0)


def extract_captions_from_page(page_text: str, page_num: int = 0) -> list[tuple[int, str, float]]:
    """Extract figure captions from page text.
    
    Returns list of (figure_number, caption_text, score) tuples.
    """
    captions = []
    for match in CAPTION_PATTERN.finditer(page_text):
        num_str = match.group(2) or match.group(4)
        if num_str:
            try:
                figure_num = int(num_str)
                full_caption = match.group(1).strip()
                score = score_mech_interp_caption(full_caption, page_num, figure_num)
                captions.append((figure_num, full_caption, score))
            except ValueError:
                continue
    return captions


def get_section_for_page(page_text: str) -> str:
    """Detect what section a page belongs to based on text content."""
    text_lower = page_text[:2000].lower()

    # Section detection patterns
    if re.search(r'\b(abstract)\b', text_lower):
        return "abstract"
    if re.search(r'\b(introduction)\b', text_lower):
        return "introduction"
    if re.search(r'\b(related work|background|prior work)\b', text_lower):
        return "related"
    if re.search(r'\b(method|methodology|approach|model|architecture)\b', text_lower):
        return "method"
    if re.search(r'\b(experiment|evaluation|result)\b', text_lower):
        return "results"

    return "unknown"


def calculate_image_score(
    width: int,
    height: int,
    has_caption: bool,
    caption_score: int,
    section: str,
    page_num: int
) -> float:
    """Calculate a score for an image candidate.

    Higher scores = better candidate for a representative figure.
    """
    score = 0.0

    # Base: size matters but avoid extremes
    area = width * height
    if area < 15000:  # Too small (like icons)
        return 0
    if area > 500000:  # Too large (likely full-page banner)
        score -= 20

    # Prefer moderate sizes (likely actual figures)
    if 50000 < area < 300000:
        score += 10

    # Aspect ratio: prefer reasonable ratios
    ratio = max(width, height) / max(min(width, height), 1)
    if ratio > 5:  # Extremely wide/tall
        score -= 15
    elif ratio > 3:
        score -= 5
    else:
        score += 5

    # Caption quality
    if has_caption:
        score += 20 + caption_score

    # Section preference: method/architecture sections are best
    if section == "method":
        score += 25
    elif section == "results":
        score += 15
    elif section == "introduction":
        score += 5

    # Page preference: early pages often have overview figures
    if page_num <= 2:
        score += 5

    return score


# ---------------------------------------------------------------------------
# BibTeX generation
# ---------------------------------------------------------------------------

def generate_bibtex(title: str, authors: str, year: int, venue: str,
                    url: str, arxiv_id: str | None = None,
                    doi: str | None = None) -> str:
    """Generate a BibTeX entry from structured metadata."""
    # Create a cite key from first author's last name + year
    first_author = authors.split(",")[0].strip() if authors else "unknown"
    last_name = first_author.split()[-1].lower() if first_author else "unknown"
    # Clean last name for cite key
    cite_key = re.sub(r'[^a-z]', '', last_name) + str(year)

    # Make title word for uniqueness
    title_word = re.sub(r'[^a-z]', '', title.split()[0].lower()) if title else ""
    cite_key = f"{cite_key}{title_word}"

    lines = [f"@article{{{cite_key},"]
    lines.append(f'  title={{{title}}},')
    lines.append(f'  author={{{authors}}},')
    lines.append(f'  year={{{year}}},')
    if venue:
        lines.append(f'  journal={{{venue}}},')
    if url:
        lines.append(f'  url={{{url}}},')
    if doi:
        lines.append(f'  doi={{{doi}}},')
    if arxiv_id:
        lines.append(f'  eprint={{{arxiv_id}}},')
        lines.append('  archivePrefix={arXiv},')
    lines.append("}")

    return "\n".join(lines)
