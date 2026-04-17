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

        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type and not url.endswith(".pdf"):
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
# Figure detection helpers
# ---------------------------------------------------------------------------

FIGURE_QUALITY_KEYWORDS = [
    "overview", "architecture", "pipeline", "framework", "model",
    "method", "approach", "structure", "diagram", "schematic",
    "workflow", "system", "network", "design", "illustration"
]

FIGURE_REJECT_KEYWORDS = [
    "logo", "header", "banner", "icon", "avatar", "institution",
    "university", "conference", "sponsor", "qr", "barcode"
]

CAPTION_PATTERN = re.compile(
    r'(Figure\s+(\d+)[:.]?\s*([^\n]{0,150})|Fig\.?\s+(\d+)[:.]?\s*([^\n]{0,150}))',
    re.IGNORECASE
)


def score_caption_quality(caption_text: str) -> int:
    """Score a caption based on presence of quality keywords."""
    caption_lower = caption_text.lower()
    score = 0
    for kw in FIGURE_QUALITY_KEYWORDS:
        if kw in caption_lower:
            score += 10
    for kw in FIGURE_REJECT_KEYWORDS:
        if kw in caption_lower:
            score -= 20
    return max(score, 0)


def extract_captions_from_page(page_text: str) -> list[tuple[int, str, int]]:
    """Extract figure captions from page text."""
    captions = []
    for match in CAPTION_PATTERN.finditer(page_text):
        num_str = match.group(2) or match.group(4)
        if num_str:
            try:
                figure_num = int(num_str)
                full_caption = match.group(0)
                score = score_caption_quality(full_caption)
                captions.append((figure_num, full_caption.strip(), score))
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
