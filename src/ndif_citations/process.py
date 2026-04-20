"""Phase 3: Content processing — LLM summaries, category classification, thumbnails."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ndif_citations import config
from ndif_citations.models import DetailCategory, DiscoveredPaper
from ndif_citations.router import ProcessingBucket, RoutingDecision
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

CATEGORY_SYSTEM_PROMPT = """Classify how this paper relates to NDIF/NNsight based on the evidence provided.

Reply with ONLY one of these exact strings:
- "uses_ndif" — verifiable evidence the paper actively uses NDIF infrastructure (e.g., hosted models on NDIF cluster, experiments run on ndif.us)
- "uses_nnsight" — verifiable evidence of nnsight library usage (e.g., import nnsight, code using nnsight API) but NOT NDIF infrastructure
- "referencing" — mentions or cites NDIF/nnsight in related work, but no evidence of active use
- "unclassified" — insufficient evidence to determine relationship

Evidence guide:
- "hosted on NDIF", "NDIF cluster", "ndif.us" in methodology → "uses_ndif"
- "import nnsight", "nnsight.trace", code snippets using nnsight API → "uses_nnsight"
- Related work mentions, surveys listing NDIF → "referencing"
- No direct mentions found in text → "unclassified"

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
        context = extract_ndif_context(pdf_path, window=config.CONTEXT_WINDOW)

    # If no PDF context, try using abstract
    if not context or "No direct mentions" in context:
        if paper.abstract:
            # Check abstract for keywords
            abstract_lower = paper.abstract.lower()
            has_mention = any(kw.lower() in abstract_lower for kw in config.NDIF_KEYWORDS)
            if has_mention:
                context = paper.abstract
            else:
                # No mentions at all — mark as UNCLASSIFIED (be honest about evidence)
                logger.info(f"No NDIF/nnsight mentions found for '{paper.title}' — marking as UNCLASSIFIED")
                return DetailCategory.UNCLASSIFIED, 0.0
        else:
            # No PDF and no abstract — cannot verify mentions
            logger.info(f"No PDF or abstract for '{paper.title}' — cannot verify mentions, marking as UNCLASSIFIED")
            return DetailCategory.UNCLASSIFIED, 0.0

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
                    f"Evidence of NDIF/nnsight usage:\n"
                    f"{context if context and 'No direct mentions' not in context else 'No direct mentions found in available text.'}"
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
        elif "unclassified" in raw:
            return DetailCategory.UNCLASSIFIED, 0.0
        else:
            logger.warning(f"Unexpected LLM classification '{raw}' for '{paper.title}' — defaulting to unclassified")
            return DetailCategory.UNCLASSIFIED, 0.0

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

_surya_model = None
_surya_processor = None

def get_layout_predictor_args():
    """Lazily load the Surya model and processor to save startup memory."""
    global _surya_model, _surya_processor
    if _surya_model is None:
        from surya.model.detection.model import load_model, load_processor
        from surya.settings import settings
        # Load the layout checkpoint instead of the default detector checkpoint
        _surya_model = load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
        _surya_processor = load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    return _surya_model, _surya_processor


def extract_thumbnail(paper: DiscoveredPaper, output_dir: Path,
                    pdf_path: Path | None = None) -> Optional[str]:
    """Extract the best thumbnail image from a paper's PDF using PyMuPDF and Surya.

    Implementation:
    1. Scan all pages with fast regex text matches to find all captions.
    2. Score the captions using Mech-Interp keyword tiers + page penalization.
    3. Rasterize ONLY the winning page.
    4. Run Surya Layout Predictor on that single page to find the exact Figure bounding box.
    5. Crop and save.
    """
    if not pdf_path or not pdf_path.exists():
        logger.info(f"No PDF available for '{paper.title}' — skipping thumbnail")
        return None

    try:
        import fitz
        from ndif_citations import utils
        from PIL import Image

        doc = fitz.open(str(pdf_path))
        all_captions = []

        # Step 1 & 2: Fast heuristic caption discovery
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            drawings = page.get_drawings()
            images = page.get_images()
            # Dense vector networks or raster images required for a figure to exist
            if len(drawings) < 20 and len(images) == 0:
                continue
                
            page_text = page.get_text()
            captions = utils.extract_captions_from_page(page_text, page_num=page_num)
            for c in captions:
                # c is (figure_num, text, score)
                all_captions.append({
                    "page_num": page_num,
                    "text": c[1],
                    "score": c[2]
                })

        doc.close()

        # Filter out 0-score or completely rejected captions
        valid_captions = [c for c in all_captions if c["score"] > 0]
        if not valid_captions:
            logger.info(f"No suitable figure candidates found in '{paper.title}'. Capturing title area as fallback.")
            doc = fitz.open(str(pdf_path))
            page = doc[0]
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            # Fallback to top ~35% of the first page (Title, Author, Abstract area)
            hero_img = img.crop((0, 0, img.width, min(img.height, int(img.height * 0.35))))
            
            slug = slugify(paper.title)
            image_filename = f"{slug}.png"
            image_path = output_dir / "images" / image_filename
            image_path.parent.mkdir(parents=True, exist_ok=True)
            hero_img.save(image_path, format="PNG")
            return f"/images/{image_filename}"

        # Sort descending by score, pick the best
        valid_captions.sort(key=lambda x: x["score"], reverse=True)
        best_candidate = valid_captions[0]
        target_page_num = best_candidate["page_num"]
        target_text = best_candidate["text"]

        logger.info(f"Selected candidate from '{paper.title}': page {target_page_num+1}, "
                    f"score={best_candidate['score']:.1f}")

        # Step 3: Rasterize the winning page
        doc = fitz.open(str(pdf_path))
        target_page = doc[target_page_num]
        
        # Determine the Y coordinate of the caption in fitz space (dpi=72)
        target_y_fitz = target_page.rect.height  # default to bottom
        rects = target_page.search_for(target_text[:40])
        if rects:
            target_y_fitz = rects[0].y0

        # Render page to PIL Image at 150 DPI
        render_dpi = 150
        scale_factor = render_dpi / 72.0
        target_y_px = target_y_fitz * scale_factor

        pix = target_page.get_pixmap(dpi=render_dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()

        # Step 4: Run Surya on the single targeted page
        from surya.layout import batch_layout_detection
        model, processor = get_layout_predictor_args()
        
        # Surya expects a list of images.
        predictions = batch_layout_detection([img], model, processor)
        page_pred = predictions[0]

        # Find all Figure/Picture bounding boxes
        fig_boxes = [b.polygon for b in page_pred.bboxes if b.label in ['Figure', 'Picture']]
        
        if not fig_boxes:
            logger.warning(f"Surya found no Figure/Picture elements on page {target_page_num+1}. Falling back to default crop.")
            # Fallback: Capture a generous portion above the caption
            crop_box = [0, max(0, target_y_px - int(img.height * 0.4)), img.width, min(img.height, target_y_px + 50)]
        else:
            # Surya polygon is [ [x0, y0], [x1, y1], [x2, y2], [x3, y3] ]. Convert to bbox [x0, y0, x1, y1].
            # Using [min_x, min_y, max_x, max_y]
            rect_boxes = []
            for poly in fig_boxes:
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                rect_boxes.append([min(xs), min(ys), max(xs), max(ys)])

            # Step 5: Match the Figure box(es) sitting exactly above the caption
            # Surya sometimes fragments a unified diagram into multiple distinct boxes.
            # We cluster all boxes that are vertically above the caption and physically close to each other.
            
            # Filter boxes to only those above or near the caption's top edge (with small 50px leniency)
            valid_boxes = [b for b in rect_boxes if b[3] <= target_y_px + 50]
            valid_boxes.sort(key=lambda b: b[3], reverse=True) # closest to caption first
            
            if not valid_boxes:
                # Capture a generous portion above the caption if no boxes qualify
                crop_box = [0, max(0, target_y_px - int(img.height * 0.4)), img.width, min(img.height, target_y_px + 50)]
            else:
                cluster = [valid_boxes[0]]
                
                def bdist(b1, b2):
                    # Manhattan distance edge-to-edge
                    dx = max(0, b2[0] - b1[2], b1[0] - b2[2])
                    dy = max(0, b2[1] - b1[3], b1[1] - b2[3])
                    return dx + dy
                
                added = True
                while added:
                    added = False
                    for b in valid_boxes:
                        if b in cluster: continue
                        # Check if box is close to any box already in our cluster (250px threshold handles large column/vertical gaps in diagrams)
                        for c in cluster:
                            if bdist(b, c) < 250:
                                cluster.append(b)
                                added = True
                                break

                # Encompassing box with 10px buffer
                crop_box = [
                    max(0, min(b[0] for b in cluster) - 10),
                    max(0, min(b[1] for b in cluster) - 10),
                    min(img.width, max(b[2] for b in cluster) + 10),
                    min(img.height, max(b[3] for b in cluster) + 10)
                ]

        # Crop the image
        hero_img = img.crop(crop_box)

        # Save the image
        slug = slugify(paper.title)
        image_filename = f"{slug}.png"
        image_path = output_dir / "images" / image_filename

        # Ensure output directory exists
        image_parent = image_path.parent
        image_parent.mkdir(parents=True, exist_ok=True)

        hero_img.save(image_path, format="PNG")
        return f"/images/{image_filename}"

    except Exception as e:
        logger.warning(f"Thumbnail extraction failed for '{paper.title}': {e}")
        return None


# ---------------------------------------------------------------------------
# Process all papers
# ---------------------------------------------------------------------------

def process_papers(
    decisions: list[RoutingDecision],
    output_dir: Path,
    skip_llm: bool = False
) -> list[DiscoveredPaper]:
    """Process papers based on routing decisions (selective processing).

    Args:
        decisions: Routing decisions from the Early Router.
        output_dir: Where to save images and PDF cache.
        skip_llm: If True, skip LLM calls entirely.

    Returns:
        List of processed DiscoveredPaper objects.
    """
    from ndif_citations.pdf_cache import get_cached_pdf

    logger.info(f"Processing {len(decisions)} papers (skip_llm={skip_llm})...")

    # Ensure cache directories exist
    (output_dir / "pdfs").mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    results: list[DiscoveredPaper] = []

    for i, decision in enumerate(decisions):
        paper = decision.paper
        bucket = decision.bucket
        needs = decision.processing_needed

        logger.info(f"[{i+1}/{len(decisions)}] {bucket.value}: {paper.title[:60]}...")

        # SKIP and PROTECTED: just copy as-is
        if bucket in (ProcessingBucket.SKIP, ProcessingBucket.PROTECTED):
            if bucket == ProcessingBucket.PROTECTED and decision.existing_paper:
                # For protected, merge in any new identifiers but preserve processed fields
                if decision.existing_paper.s2_paper_id and not paper.s2_paper_id:
                    paper.s2_paper_id = decision.existing_paper.s2_paper_id
                if decision.existing_paper.openalex_id and not paper.openalex_id:
                    paper.openalex_id = decision.existing_paper.openalex_id
            results.append(paper)
            continue

        # Get PDF path if needed for classification or thumbnail
        pdf_path = None
        if needs.get("classify") or needs.get("thumbnail"):
            pdf_path = get_cached_pdf(paper, output_dir)
            if not pdf_path:
                logger.warning(f"Could not get PDF for '{paper.title[:50]}...'")

        # Summary generation
        if needs.get("summary") and not skip_llm:
            paper.description = generate_summary(paper)
            paper.has_summary = bool(paper.description)

        # Category classification
        if needs.get("classify") and not skip_llm:
            if pdf_path and pdf_path.exists():
                category, confidence = classify_category(paper, output_dir, pdf_path=pdf_path)
                paper.detail_category = category
                paper.category_confidence = confidence
                paper.has_classification = category != DetailCategory.UNCLASSIFIED
            else:
                # No PDF available - mark as UNCLASSIFIED
                paper.detail_category = DetailCategory.UNCLASSIFIED
                paper.category_confidence = 0.0
                paper.has_classification = False

        # Thumbnail extraction
        if needs.get("thumbnail"):
            if pdf_path and pdf_path.exists():
                image_filename = f"{slugify(paper.title)}.png"
                image_path = output_dir / "images" / image_filename

                if not image_path.exists():
                    extracted = extract_thumbnail(paper, output_dir, pdf_path=pdf_path)
                    if extracted:
                        paper.image = extracted

            paper.has_thumbnail = bool(paper.image)

        results.append(paper)

    logger.info("Processing complete")
    return results


def process_papers_legacy(
    papers: list[DiscoveredPaper],
    output_dir: Path,
    skip_llm: bool = False
) -> list[DiscoveredPaper]:
    """DEPRECATED: Legacy processing without router.

    Use process_papers() with RoutingDecisions instead.
    """
    logger.warning("process_papers_legacy is deprecated. Use process_papers with routing decisions.")

    # Convert papers to NEW routing decisions
    from ndif_citations.router import route_papers

    existing: list[DiscoveredPaper] = []
    decisions = route_papers(papers, existing)

    return process_papers(decisions, output_dir, skip_llm)
