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


# ---------------------------------------------------------------------------
# Affiliation extraction from PDF text (heuristic, no LLM)
# ---------------------------------------------------------------------------

_AFFILIATION_ORG_KW = re.compile(
    r'\b(University|Universidade|Universität|Université|Università|Univ\.|'
    r'Institute|Institut|Instituto|'
    r'College|Lab(?:oratory)?|Laboratoire|Labs?\b|'
    r'Research|Corporation|Corp\.|Inc\.|Ltd|GmbH|'
    r'Foundation|Center|Centre|Centro|'
    r'School|Department|Dept\.|'
    r'Polytechnic|Polytechnique|'
    r'Hospital|Medical|'
    r'Academy|Hochschule|'
    r'OpenAI|Anthropic|DeepMind|Google|Meta|Microsoft|NVIDIA|Apple|'
    r'IBM|Amazon|Adobe|Intel|Salesforce|Hugging\s*Face|Cohere|LinkedIn|'
    r'CNRS|MILA|Mila\b|MIT\b|CMU\b|UCL\b|ETH\b|EPFL\b|KAIST\b|UCLA\b|UCSD\b|UCSF|UCB\b|NYU\b|'
    r'Northeastern|Stanford|Harvard|Princeton|Yale|Cornell|Columbia|Berkeley|Tsinghua|Peking|'
    r'EleutherAI|Eleuther|Redwood|Apollo|MATS|Surge|Reality\s+Labs|Allen\s+(Institute|AI)|FAIR\b|'
    r'Tübingen|Tubingen|Heidelberg|Munich|München|Edinburgh|Oxford|Cambridge|Toronto|McGill|Imperial|'
    r'Sorbonne|Bocconi|Politecnico|Stuttgart|Karlsruhe|Saarland|Aachen|Hamburg|Heilbronn|Ansbach|'
    r'Helmholtz|Fraunhofer|Max\s+Planck|Saint\s+Exupéry|'
    r'Transluce|COAI|Independent\s+Researcher|Metagov)',
    re.IGNORECASE,
)

_AFFILIATION_NOISE = [
    # Bug E fix: anchor to start so "University of Freiburg♡, Microsoft Research♣"
    # isn't rejected just because "Equal contribution∗" appears elsewhere on the page.
    re.compile(r'^\s*(Equal\s+contributions?|Equally\s+contributed)', re.IGNORECASE),
    re.compile(r'Work\s+done\s+(?:during|while)', re.IGNORECASE),
    re.compile(r'^\s*(?:Code|Correspondence|Preprint|Under\s+review|Published\s+as)', re.IGNORECASE),
    re.compile(r'\S+@\S+'),
    re.compile(r'^\s*\{[^}]+@', re.IGNORECASE),  # email groups
    re.compile(r'^(?:github|http|https|www)\.', re.IGNORECASE),
]


def _affil_clean(s: str) -> str:
    """Strip emails, brace-groups, leading/trailing markers, normalize whitespace."""
    s = re.sub(r'\S+@\S+', '', s)
    s = re.sub(r'\{[^}]+\}', '', s)
    s = re.sub(r'^\s*[\d*†‡§¶♠♢♣♡⋆✠]+\s*', '', s)
    # Bug A fix: strip trailing "Correspondence to:", "Preprint.", "Under review" etc.
    s = re.sub(r'\s+(?:Correspondence|Preprint|Under\s+review|Published\s+as)[\s\S]*$',
               '', s, flags=re.IGNORECASE)
    # Strip trailing suffix markers left over from the affiliation body
    s = re.sub(r'[\s♡♣♢♠⋆✠*†‡§¶,;.:]+$', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.rstrip(',;.: ').strip()


def _affil_looks_valid(s: str) -> bool:
    if not s or not (4 <= len(s) <= 200):
        return False
    if any(p.search(s) for p in _AFFILIATION_NOISE):
        return False
    # Bug B fix: reject compound entries e.g. "1Stanford 2MIT 3Apple" that
    # contain 2+ internal digit-letter patterns — these should have been split.
    if len(re.findall(r'(?:^|\s)\d{1,2}\s*[A-Z]', s)) >= 2:
        return False
    return bool(_AFFILIATION_ORG_KW.search(s))


def _affil_fix_hyphens(text: str) -> str:
    """Join hyphenated line breaks: 'Col-\\nlege' -> 'College'."""
    return re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)


def _affil_dedupe(affs: list[str]) -> list[str]:
    seen, out = set(), []
    for a in affs:
        key = a.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(a)
    return out


def _affil_parse_marker_block(block: str) -> list[str]:
    """Parse '1Stanford 2MIT' or '†Stanford ‡Apple' into clean entries."""
    if not block:
        return []
    parts = re.split(r'(?<!\w)([\d†‡§¶♠♢♣♡⋆✠*]{1,2})\s*(?=[A-Z])', block)
    affs: list[str] = []
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        chunk = parts[i + 1]
        # Cut at next sentence/marker that ends affiliations
        chunk = re.split(
            r'(?:\.\s+(?:Correspondence|Preprint|Under review|Code|Published|We|This)\b)',
            chunk, maxsplit=1,
        )[0]
        cleaned = _affil_clean(chunk)
        if _affil_looks_valid(cleaned):
            affs.append(cleaned)
    return _affil_dedupe(affs)


def _affil_find_footnote_block(text: str) -> Optional[str]:
    """Find the bottom-of-page-1 affiliation footnote (ICML/NeurIPS template)."""
    end_match = re.search(
        r'(Correspondence to:|Preprint\.|Under review|Code for all|Published as a conference)',
        text,
    )
    if not end_match:
        return None
    end_idx = end_match.start()
    window = text[max(0, end_idx - 1500):end_idx]
    if not re.search(r'\d{1,2}\s*[A-Z][^\d]{6,}', window):
        return None
    first = re.search(r'\d{1,2}\s*[A-Z]', window)
    return window[first.start():] if first else window


def _affil_parse_suffix_markers(block: str) -> list[str]:
    """Handle 'Lab Name♡, Other Lab♣' format (ICML suffix-marker legend).

    Some papers list affiliations inline using trailing Unicode symbols
    (e.g. ♡♣♢♠) instead of leading digit prefixes. Split on commas and
    strip the trailing marker from each token.
    """
    if not block:
        return []
    suffix_re = re.compile(r'([A-Z][^,\n]+?)\s*[♡♣♢♠⋆✠*†‡§¶]+(?=\s*,|\s*\n|\s*$)')
    matches = [m.group(1).strip() for m in suffix_re.finditer(block)]
    if len(matches) < 2:
        return []
    return _affil_dedupe([m for m in matches if _affil_looks_valid(m)])


def _affil_parse_inline_block(text: str, authors: str) -> list[str]:
    """Parse affiliations between authors and 'Abstract' (ACL/EMNLP style)."""
    head = text[:2500]
    abs_match = re.search(r'(?im)^\s*(Abstract|ABSTRACT)\b', head)
    if not abs_match:
        return []
    abs_idx = abs_match.start()
    surnames = [a.split()[-1] for a in (authors.split(',')[:5]) if a.split()]
    max_end = 0
    for s in surnames:
        for m in re.finditer(re.escape(s), head[:abs_idx]):
            if m.end() > max_end:
                max_end = m.end()
    if not max_end:
        return []
    block = head[max_end:abs_idx]

    # Bug B fix: try prefix-marker parsing first; if it succeeds, trust it and stop.
    # Don't fall through to line-capture — that produces duplicate compound entries.
    prefix_affs = _affil_parse_marker_block(block)
    if prefix_affs:
        return prefix_affs

    # Bug D fix: try suffix-marker format (Lab♡, Other♣) before plain-line fallback.
    suffix_affs = _affil_parse_suffix_markers(block)
    if suffix_affs:
        return suffix_affs

    # Plain-line fallback: each line is a single affiliation (no marker splitting).
    line_affs = []
    for line in block.split('\n'):
        cleaned = _affil_clean(line)
        if _affil_looks_valid(cleaned):
            line_affs.append(cleaned)
    return _affil_dedupe(line_affs)


def _affil_block_aware(doc) -> Optional[str]:
    """Use PyMuPDF block layout to find footer affiliation blocks (IEEE template)."""
    try:
        blocks = doc[0].get_text('blocks')
        if not blocks:
            return None
        blocks = sorted(blocks, key=lambda b: b[1])
        abs_y = None
        for b in blocks:
            if 'Abstract' in b[4][:60] or 'ABSTRACT' in b[4][:60]:
                abs_y = b[1]
                break
        candidates = []
        for b in blocks:
            x0, y0, x1, y1, text, *_ = b
            if abs_y is not None and y0 < abs_y - 10:
                continue
            if (y1 - y0) < 80 and re.search(r'\d\s*[A-Z][^\d]{4,}', text) and _AFFILIATION_ORG_KW.search(text):
                candidates.append(text)
        return '\n'.join(candidates) if candidates else None
    except Exception:
        return None


def extract_affiliations_from_pdf(pdf_path: Path, authors: str = "") -> list[str]:
    """Extract author affiliations from a PDF using pure heuristics (no LLM).

    Tries three strategies in order:
      1. Block-aware footer (IEEE template)
      2. Footnote block anchored on 'Correspondence to:' / 'Preprint.' (ICML/NeurIPS)
      3. Inline block between authors and 'Abstract' (ACL/EMNLP)

    Returns a deduplicated list of affiliation strings; empty list if none found.
    """
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        logger.debug(f"Failed to open PDF for affiliation extraction: {e}")
        return []

    try:
        text = doc[0].get_text()
        if len(text) < 500 and len(doc) > 1:
            text += '\n' + doc[1].get_text()
        text = _affil_fix_hyphens(text)

        # Strategy 1: block-aware footer (IEEE template)
        block_text = _affil_block_aware(doc)
        if block_text:
            affs = _affil_parse_marker_block(block_text)
            if affs:
                return affs

        # Strategy 2: anchored footnote block (ICML/NeurIPS)
        # Bug C fix: only return here if parsing actually found entries;
        # if footnote block was detected but empty, fall through to inline.
        footnote = _affil_find_footnote_block(text)
        if footnote:
            affs = _affil_parse_marker_block(footnote)
            if affs:
                return affs

        # Strategy 3: inline block (ACL/EMNLP) — also catches suffix-marker format
        return _affil_parse_inline_block(text, authors)
    except Exception as e:
        logger.debug(f"Affiliation extraction failed for {pdf_path.name}: {e}")
        return []
    finally:
        try:
            doc.close()
        except Exception:
            pass


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


def query_s2_publication_venue(arxiv_id: str | None = None,
                                 doi: str | None = None) -> dict:
    """Query Semantic Scholar for a paper's publicationVenue.

    Returns {} on failure. On success: {"name": str, "type": str}.
    Useful for papers discovered via OpenAlex/GitHub that S2 also indexes —
    S2 often has accurate conference/journal venue data even when OpenAlex
    is still showing the arXiv preprint.
    """
    from ndif_citations import config

    if not arxiv_id and not doi:
        return {}

    paper_ref = f"ARXIV:{arxiv_id}" if arxiv_id else f"DOI:{doi}"
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_ref}"
    headers = {"User-Agent": "NDIFCitationTracker/1.0 (academic research; https://ndif.us)"}
    if config.S2_API_KEY:
        headers["x-api-key"] = config.S2_API_KEY

    try:
        resp = requests.get(
            url,
            params={"fields": "publicationVenue,venue"},
            headers=headers,
            timeout=10,
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
        pv = data.get("publicationVenue") or {}
        if pv:
            return {
                "name": pv.get("name") or "",
                "type": pv.get("type") or "",
            }
        # Fallback to bare venue string
        venue = data.get("venue") or ""
        return {"name": venue, "type": ""} if venue else {}
    except Exception as e:
        logger.debug(f"S2 publicationVenue query failed for {paper_ref}: {e}")
        return {}


def query_openreview_venue(title: str, min_title_match: int = 90) -> dict:
    """Query OpenReview for a paper's accepted-venue metadata (ICLR/ICML/NeurIPS/etc).

    OpenReview's public search has limited coverage and inconsistent indexing,
    so we fuzzy-match the title against the top-25 v1 search results and only
    accept matches above `min_title_match` (default 90/100 rapidfuzz ratio).

    Returns {} on no confident match. On match: {"venue": str, "venueid": str}.
    """
    if not title:
        return {}
    headers = {"User-Agent": "NDIFCitationTracker/1.0 (academic research; https://ndif.us)"}
    try:
        resp = requests.get(
            "https://api.openreview.net/notes/search",
            params={"term": title, "limit": 25, "source": "forum"},
            headers=headers,
            timeout=10,
        )
        if resp.status_code != 200:
            return {}
        notes = resp.json().get("notes", []) or []
        target = title.lower().strip()
        best_score, best_note = 0, None
        for note in notes:
            nt = (note.get("content", {}) or {}).get("title", "") or ""
            score = fuzz.ratio(target, nt.lower().strip())
            if score > best_score:
                best_score, best_note = score, note
        if best_note is None or best_score < min_title_match:
            return {}
        content = best_note.get("content", {}) or {}
        venue = content.get("venue") or ""
        venueid = content.get("venueid") or ""
        if not venue or "submitted" in str(venue).lower():
            return {}
        return {"venue": str(venue), "venueid": str(venueid)}
    except Exception as e:
        logger.debug(f"OpenReview query failed for '{title[:60]}': {e}")
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
