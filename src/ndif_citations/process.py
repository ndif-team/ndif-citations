"""Phase 3: Content processing — LLM summaries, category classification, thumbnails."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ndif_citations import config
from ndif_citations.models import DetailCategory, DiscoveredPaper
from ndif_citations.utils import (
    download_pdf,
    extract_ndif_context,
    rate_limit_sleep,
    slugify,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def _get_llm_client():
    """Create an OpenAI-compatible client."""
    import openai

    if not config.LLM_API_KEY:
        logger.warning("LLM_API_KEY not set — LLM features will use fallbacks")
        return None

    return openai.OpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
    )


# ---------------------------------------------------------------------------
# 3a. Summary generation
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = """You summarize research paper abstracts for a public-facing website. \
Write 1-3 sentences. Be qualitative, not quantitative — focus on what the paper does and \
why it matters. Do not include specific numbers, percentages, or benchmark scores. \
Keep it accessible to a technical but non-specialist audience. \
Do not start with "This paper" — vary your openings."""


def generate_summary(paper: DiscoveredPaper) -> str:
    """Generate an LLM summary from the paper's abstract."""
    if not paper.abstract:
        logger.warning(f"No abstract for '{paper.title}' — skipping summary")
        return ""

    client = _get_llm_client()
    if not client:
        # Fallback: first 2 sentences of abstract
        return _fallback_summary(paper.abstract)

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Summarize this abstract:\n\n{paper.abstract}"},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        summary = response.choices[0].message.content.strip()
        rate_limit_sleep(config.LLM_RATE_LIMIT_SLEEP, "LLM summary")
        return summary

    except Exception as e:
        logger.warning(f"LLM summary failed for '{paper.title}': {e}")
        return _fallback_summary(paper.abstract)


def _fallback_summary(abstract: str) -> str:
    """Fallback: extract first 2 sentences from abstract."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', abstract.strip())
    return " ".join(sentences[:2]).strip()


# ---------------------------------------------------------------------------
# 3b. Category classification
# ---------------------------------------------------------------------------

CATEGORY_SYSTEM_PROMPT = """Classify how this paper relates to NDIF/NNsight based on the context excerpts.

Reply with ONLY one of these exact strings:
- "uses_ndif" — if the paper uses NDIF infrastructure (e.g., hosted models on NDIF, ran experiments on NDIF cluster)
- "uses_nnsight" — if the paper uses the nnsight Python library in their experiments/code (e.g., "import nnsight", nnsight.trace, uses nnsight as backend) but does NOT use NDIF infrastructure
- "referencing" — if the paper only mentions/cites NDIF or nnsight without actively using either

Evidence guide:
- "hosted on NDIF", "NDIF cluster", "ndif.us" in methodology → "uses_ndif"
- "import nnsight", "nnsight.trace", code snippets using nnsight API, "nnsight backend" → "uses_nnsight"
- Related work mentions, surveys listing NDIF, comparisons without actual usage → "referencing"

Reply with ONLY the category string, nothing else."""


def classify_category(paper: DiscoveredPaper, output_dir: Path,
                      pdf_path: Path | None = None) -> tuple[DetailCategory, float]:
    """Classify a paper's relationship to NDIF/NNsight.

    Args:
        paper: The paper to classify.
        output_dir: Working directory for temp files.
        pdf_path: Pre-downloaded PDF path (avoids re-downloading).

    Returns (category, confidence) where confidence is 0.0-1.0.
    """
    # Try to extract context from PDF (use pre-downloaded if available)
    context = ""
    if pdf_path and pdf_path.exists():
        context = extract_ndif_context(pdf_path)

    # If no PDF context, try using abstract
    if not context or "No direct mentions" in context:
        if paper.abstract:
            # Check abstract for keywords
            abstract_lower = paper.abstract.lower()
            has_mention = any(kw.lower() in abstract_lower for kw in config.NDIF_KEYWORDS)
            if has_mention:
                context = paper.abstract
            else:
                # No mentions at all — default to referencing
                logger.info(f"No NDIF/nnsight mentions found for '{paper.title}' — defaulting to 'referencing'")
                return DetailCategory.REFERENCING, 0.5
        else:
            return DetailCategory.REFERENCING, 0.3

    # Use LLM to classify
    client = _get_llm_client()
    if not client:
        return _fallback_classification(context), 0.4

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": CATEGORY_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Paper title: {paper.title}\n\n"
                    f"Context excerpts mentioning NDIF/nnsight:\n{context}"
                )},
            ],
            temperature=0.1,
            max_tokens=20,
        )
        raw = response.choices[0].message.content.strip().lower()
        rate_limit_sleep(config.LLM_RATE_LIMIT_SLEEP, "LLM classify")

        # Parse response
        if "uses_ndif" in raw:
            return DetailCategory.USES_NDIF, 0.85
        elif "uses_nnsight" in raw:
            return DetailCategory.USES_NNSIGHT, 0.85
        elif "referencing" in raw:
            return DetailCategory.REFERENCING, 0.85
        else:
            logger.warning(f"Unexpected LLM classification '{raw}' for '{paper.title}' — defaulting to referencing")
            return DetailCategory.REFERENCING, 0.5

    except Exception as e:
        logger.warning(f"LLM classification failed for '{paper.title}': {e}")
        return _fallback_classification(context), 0.4


def _fallback_classification(context: str) -> DetailCategory:
    """Fallback classification without LLM — rule-based."""
    context_lower = context.lower()

    # Strong signals for uses_ndif
    ndif_signals = ["hosted on ndif", "ndif cluster", "ndif.us", "ndif infrastructure"]
    if any(s in context_lower for s in ndif_signals):
        return DetailCategory.USES_NDIF

    # Strong signals for uses_nnsight
    nnsight_signals = ["import nnsight", "nnsight.trace", "from nnsight", "nnsight backend",
                       "nnsight library", "using nnsight"]
    if any(s in context_lower for s in nnsight_signals):
        return DetailCategory.USES_NNSIGHT

    return DetailCategory.REFERENCING


# ---------------------------------------------------------------------------
# 3c. Thumbnail extraction (smart figure detection - no LLM)
# ---------------------------------------------------------------------------

def extract_thumbnail(paper: DiscoveredPaper, output_dir: Path,
                    pdf_path: Path | None = None) -> Optional[str]:
    """Extract the best thumbnail image from a paper's PDF using smart figure detection.

    Scans all pages, detects captions, scores candidates based on quality keywords,
    section context, and size. Returns the saved image path or None.
    """
    if not pdf_path or not pdf_path.exists():
        logger.info(f"No PDF available for '{paper.title}' — skipping thumbnail")
        return None

    try:
        import fitz
        from ndif_citations import utils

        doc = fitz.open(str(pdf_path))
        candidates = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Get page text and detect section
            page_text = page.get_text()
            section = utils.get_section_for_page(page_text)

            # Extract captions from this page
            captions = utils.extract_captions_from_page(page_text)

            # Get images on this page
            images = page.get_images(full=True)
            if not images:
                continue

            # Heuristic: if captions with quality keywords exist
            has_quality_caption = any(score >= 10 for _, _, score in captions)
            best_caption_score = max((score for _, _, score in captions), default=0)

            for img_info in images:
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    width = base_image["width"]
                    height = base_image["height"]

                    # Skip tiny images (logos, icons)
                    if width < 80 or height < 80:
                        continue

                    # Skip extremely large images (likely full-page scans)
                    if width * height > 1000000:
                        continue

                    # Calculate score
                    score = utils.calculate_image_score(
                        width=width,
                        height=height,
                        has_caption=has_quality_caption,
                        caption_score=best_caption_score,
                        section=section,
                        page_num=page_num
                    )

                    if score > 0:
                        candidates.append({
                            'image': base_image,
                            'score': score,
                            'width': width,
                            'height': height,
                            'page_num': page_num,
                            'section': section,
                            'has_caption': has_quality_caption,
                        })

                except Exception:
                    continue

        doc.close()

        if not candidates:
            logger.info(f"No suitable figure candidates found in '{paper.title}'")
            return None

        # Sort by score (descending) and pick best
        candidates.sort(key=lambda x: x['score'], reverse=True)
        best = candidates[0]

        logger.info(f"Selected image from '{paper.title}': page {best['page_num']+1}, "
                    f"score={best['score']:.1f}, size={best['width']}x{best['height']}")

        # Save the image
        slug = slugify(paper.title)
        image_filename = f"{slug}.png"
        image_path = output_dir / "images" / image_filename

        # Ensure output directory exists
        image_parent = image_path.parent
        image_parent.mkdir(parents=True, exist_ok=True)

        # Convert to PNG if needed
        img_data = best['image']["image"]
        img_ext = best['image'].get("ext", "png")

        if img_ext != "png":
            try:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(img_data))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_data = buf.getvalue()
            except Exception as e:
                logger.debug(f"Image conversion failed, saving as-is: {e}")

        with open(image_path, "wb") as f:
            f.write(img_data)

        return f"/images/{image_filename}"

    except Exception as e:
        logger.warning(f"Thumbnail extraction failed for '{paper.title}': {e}")
        return None


# ---------------------------------------------------------------------------
# Process all papers
# ---------------------------------------------------------------------------

def process_papers(papers: list[DiscoveredPaper], output_dir: Path,
                   skip_llm: bool = False) -> list[DiscoveredPaper]:
    """Run Phase 3 on all papers: summaries, classification, thumbnails.

    Args:
        papers: List of discovered papers to process.
        output_dir: Where to save images and temp files.
        skip_llm: If True, skip LLM calls (for discover-only mode).
    """
    logger.info(f"Processing {len(papers)} papers (skip_llm={skip_llm})...")
    (output_dir / "pdfs").mkdir(parents=True, exist_ok=True)

    for i, paper in enumerate(papers):
        logger.info(f"[{i+1}/{len(papers)}] Processing: {paper.title[:60]}...")

        # Summary
        if not paper.description and not skip_llm:
            paper.description = generate_summary(paper)

        # Download PDF once for both classification and thumbnail extraction
        pdf_path = None
        pdf_url = paper.pdf_url
        if not pdf_url and paper.arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"

        if pdf_url and not skip_llm:
            pdf_path = download_pdf(pdf_url, dest_dir=output_dir / "pdfs")

        # Category classification (uses shared PDF)
        if not paper.manual_override and not skip_llm:
            category, confidence = classify_category(paper, output_dir, pdf_path=pdf_path)
            paper.detail_category = category
            paper.category_confidence = confidence

        # Thumbnail (uses shared PDF)
        image_filename = f"{slugify(paper.title)}.png"
        image_path = output_dir / "images" / image_filename

        if not image_path.exists() and not skip_llm:
            extracted = extract_thumbnail(paper, output_dir, pdf_path=pdf_path)
            if extracted:
                paper.image = extracted

        # Clean up shared PDF
        if pdf_path:
            try:
                pdf_path.unlink()
            except OSError:
                pass

    # Clean up pdfs directory
    pdfs_dir = output_dir / "pdfs"
    if pdfs_dir.exists():
        for f in pdfs_dir.iterdir():
            try:
                f.unlink()
            except OSError:
                pass
        try:
            pdfs_dir.rmdir()
        except OSError:
            pass

    logger.info("Processing complete")
    return papers
