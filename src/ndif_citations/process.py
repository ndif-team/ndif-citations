"""Phase 3: Content processing — LLM summaries, category classification, thumbnails."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from ndif_citations import config
from ndif_citations.models import (
    Bucket, Category, Confidence, DiscoveredPaper, DiscoveredRepo, PaperReason,
)
from ndif_citations.router import ProcessingBucket, RepoRoutingDecision, RoutingDecision
from ndif_citations.utils import (
    _fetch_repo_readme,
    download_pdf,
    extract_ndif_context,
    extract_text_from_pdf,
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
# Confidence band derivation (pure function)
# ---------------------------------------------------------------------------

def _compute_confidence_band(
    *,
    signal: str,
    linked_paper_tier: Optional[int],
    surviving_window_count: int,
    context_source: str,
    category: Category,
) -> Confidence:
    """Map classification metadata to a Confidence band.

    Rules (first match wins):
      1. UNCLASSIFIED                                          -> NONE
      2. signal == pre_filter:negative_evidence                -> CERTAIN
      3. signal in (pre_filter:comparison_table,
                    pre_filter:acks_only_thank_you)            -> MEDIUM
      4. signal == keyword_fallback                            -> LOW
      5. signal == llm AND (linked_paper_tier <= 2
                            OR surviving_window_count >= 2)    -> HIGH
      6. signal == llm AND (single window OR abstract-only)    -> MEDIUM
      7. anything else                                         -> LOW (defensive)
    """
    if category == Category.UNCLASSIFIED:
        return Confidence.NONE
    if signal == "pre_filter:negative_evidence":
        return Confidence.CERTAIN
    if signal in (
        "pre_filter:comparison_table",
        "pre_filter:acks_only_thank_you",
    ):
        return Confidence.MEDIUM
    if signal == "keyword_fallback":
        return Confidence.LOW
    if signal == "llm":
        if linked_paper_tier is not None and linked_paper_tier <= 2:
            return Confidence.HIGH
        if surviving_window_count >= 2:
            return Confidence.HIGH
        return Confidence.MEDIUM
    return Confidence.LOW


# ---------------------------------------------------------------------------
# 3a. Summary generation
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = """You summarize research paper abstracts for a public-facing website. \
Write 1-3 sentences. Be qualitative, not quantitative — focus on what the paper does and \
why it matters. Do not include specific numbers, percentages, or benchmark scores. \
Keep it accessible to a technical but non-specialist audience. \
Do not start with "This paper" — vary your openings."""


def _has_usable_abstract(paper: DiscoveredPaper) -> bool:
    """Return True if the paper has a non-trivial abstract suitable for LLM summarization."""
    if not paper.abstract:
        return False
    return len(paper.abstract.strip().strip(",").strip()) >= 20


def generate_summary(paper: DiscoveredPaper) -> str:
    """Generate an LLM summary from the paper's abstract."""
    if not _has_usable_abstract(paper):
        raw = (paper.abstract or "")[:40]
        logger.warning(
            f"Skipping LLM summary for '{paper.title}' — "
            f"abstract is {'missing' if not paper.abstract else 'too short or malformed'} "
            f"(value: {raw!r})"
        )
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

# ---------------------------------------------------------------------------
# 3b-i. Pre-filter helpers (no LLM needed for these patterns)
# ---------------------------------------------------------------------------

_NEGATIVE_RE = re.compile(
    r'removing\s+(?:the\s+)?(?:dependency\s+on\s+)?(?:nnsight|ndif)'
    r'|removed?\s+(?:the\s+)?(?:nnsight|ndif)(?:\s+dependency)?'
    r'|rather\s+than\s+(?:nnsight|ndif)'
    r'|instead\s+of\s+(?:nnsight|ndif)'
    r'|without\s+(?:using\s+)?(?:nnsight|ndif)'
    r'|no\s+longer\s+(?:uses?|using)\s+(?:nnsight|ndif)'
    r'|alternative\s+to\s+(?:nnsight|ndif)'
    r'|compared\s+to\s+(?:alternatives?\s+like\s+)?(?:nnsight|ndif)',
    re.IGNORECASE,
)

_TABLE_CHARS_RE = re.compile(r'[✓✗∼]')

_ACK_THANK_RE = re.compile(
    r'\b(?:thank|acknowledge|grateful|supported\s+by|funded\s+by|provided\s+by)\b'
    r'.{0,60}'
    r'\b(?:ndif|nnsight|national\s+deep\s+inference)\b',
    re.IGNORECASE | re.DOTALL,
)

_IMPL_CONFIRM_RE = re.compile(
    r'(?:'
    r'\b(?:implement|use[sd]?|using|appl(?:y|ied)|ran|conduct(?:ed)?|perform)[^.]{0,60}\b(?:nnsight|ndif)\b'
    r'|'
    r'\b(?:nnsight|ndif)\b(?:[^.]|et\s+al\.|i\.e\.|e\.g\.|\.(?!\s+[A-Z])){0,200}\b(?:is|was|we)\s+used\b'
    r'|'
    r'\b(?:nnsight|ndif)\b[^.]{0,100}to\s+implement\b'
    r')',
    re.IGNORECASE | re.DOTALL,
)


def _has_negative_evidence(window: str) -> bool:
    """Return True if the context window explicitly states nnsight/NDIF was removed or not used."""
    return bool(_NEGATIVE_RE.search(window))


def _context_is_comparison_table(window: str) -> bool:
    """Return True if the context window looks like a capability-comparison table row."""
    return len(_TABLE_CHARS_RE.findall(window)) >= 3


def _is_ack_only_thank_you(window: str) -> bool:
    """Return True if window is an acknowledgment thank-you without implementation confirmation."""
    return bool(_ACK_THANK_RE.search(window)) and not bool(_IMPL_CONFIRM_RE.search(window))


def _apply_prefilters(windows: list[str], paper) -> tuple[list[str], Optional[str]]:
    """Filter context windows through pre-filter rules.

    Returns (surviving_windows, signal) where signal is set only if ALL windows were filtered.
    Applies negative evidence first, then comparison-table, then acks-only.
    """
    surviving = []
    first_signal: Optional[str] = None

    for window in windows:
        if _has_negative_evidence(window):
            if first_signal is None:
                first_signal = "pre_filter:negative_evidence"
        elif _context_is_comparison_table(window):
            if first_signal is None:
                first_signal = "pre_filter:comparison_table"
        elif _is_ack_only_thank_you(window):
            if first_signal is None:
                first_signal = "pre_filter:acks_only_thank_you"
        else:
            surviving.append(window)

    if not surviving:
        return [], first_signal
    return surviving, None


# ---------------------------------------------------------------------------
# 3b-ii. Classification prompt + LLM call
# ---------------------------------------------------------------------------

UNIFIED_PROMPT = """Classify how this paper relates to NDIF/NNsight based on the evidence provided.

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

LIBRARY_PROMPT = """Classify how this paper uses the NNsight library based on the evidence provided.
The paper mentions only NNsight (the library) — not NDIF infrastructure.

Reply with ONLY one of these exact strings:
- "uses_nnsight" — verifiable evidence of nnsight library usage (e.g., import nnsight, nnsight.trace, code using nnsight API)
- "referencing" — mentions or cites NNsight in related work, but no verifiable active use
- "unclassified" — insufficient evidence to determine relationship

Reply with ONLY the category string, nothing else."""

INFRASTRUCTURE_PROMPT = """Classify how this paper uses NDIF infrastructure based on the evidence provided.
The paper mentions only NDIF (the inference fabric) — not the NNsight library specifically.

Reply with ONLY one of these exact strings:
- "uses_ndif" — verifiable evidence the paper uses NDIF infrastructure (e.g., hosted models on NDIF cluster, experiments run on ndif.us)
- "referencing" — mentions or cites NDIF in related work, but no verifiable active use
- "unclassified" — insufficient evidence to determine relationship

Reply with ONLY the category string, nothing else."""

# Keep legacy alias so existing code referencing CATEGORY_SYSTEM_PROMPT still works
CATEGORY_SYSTEM_PROMPT = UNIFIED_PROMPT

# Keyword families for prompt routing
_LIBRARY_KEYWORDS = frozenset({"nnsight", "import nnsight", "nnsight.net", "nnsight.trace", "from nnsight"})
_INFRA_KEYWORDS = frozenset({"ndif", "ndif.us", "national deep inference", "ndif cluster", "hosted on ndif"})


def _select_classification_prompt(context: str) -> str:
    """Select the appropriate classification prompt based on which keyword families matched.

    Returns LIBRARY_PROMPT if only library keywords matched,
    INFRASTRUCTURE_PROMPT if only infrastructure keywords matched,
    UNIFIED_PROMPT for mixed matches or empty context.
    """
    if not context or "No direct mentions" in context:
        return UNIFIED_PROMPT

    lower = context.lower()
    has_library = any(kw in lower for kw in _LIBRARY_KEYWORDS)
    has_infra = any(kw in lower for kw in _INFRA_KEYWORDS)

    if has_library and not has_infra:
        return LIBRARY_PROMPT
    if has_infra and not has_library:
        return INFRASTRUCTURE_PROMPT
    return UNIFIED_PROMPT


_TIER_DESCRIPTORS = {
    1: "BibTeX block in README",
    2: "Citation/Paper section heading",
}

_TIER_AUGMENTATION_TEMPLATE = """
TIER {tier} CROSS-LINK EVIDENCE:
This paper is the declared paper of a GitHub repository (via {descriptor}).
Repositories that nnsight is a dependency of and that explicitly cite a paper
overwhelmingly use that library. Weight this evidence heavily. The classification
should be uses_nnsight or uses_ndif (whichever the keyword evidence supports);
referencing is rare in this case."""


def _augment_prompt_with_tier(prompt: str, tier: Optional[int]) -> str:
    """Append a tier-context block for Tier 1/2 cross-links. Returns prompt unchanged otherwise."""
    if tier is None or tier > 2:
        return prompt
    descriptor = _TIER_DESCRIPTORS.get(tier, f"tier {tier} link")
    return prompt + _TIER_AUGMENTATION_TEMPLATE.format(tier=tier, descriptor=descriptor)


def classify_category(
    paper: DiscoveredPaper, output_dir: Path,
    pdf_path: Path | None = None,
) -> tuple[Category, float, Confidence]:
    """Classify a paper's relationship to NDIF/NNsight.

    Returns (category, float_confidence, band). The float is derived from
    the band via _BAND_TO_FLOAT for backwards-compat callers. Sets
    paper.unclassified_reason on UNCLASSIFIED outcomes and
    paper.classification_signal to the label used by _compute_confidence_band.
    """
    from ndif_citations.models import _BAND_TO_FLOAT  # local import to avoid cycle

    context = ""
    context_source = "none"

    if pdf_path and pdf_path.exists():
        context = extract_ndif_context(pdf_path, window=config.CONTEXT_WINDOW)
        logger.debug(f"PDF text extracted for '{paper.title}': context length={len(context)}")

    # If no PDF context, try using abstract
    if not context or "No direct mentions" in context:
        if paper.abstract:
            abstract_lower = paper.abstract.lower()
            has_mention = any(kw.lower() in abstract_lower for kw in config.NDIF_KEYWORDS)
            if has_mention:
                context = paper.abstract
                context_source = "abstract"
                logger.debug(f"Using abstract as context for '{paper.title}'")
            else:
                logger.info(f"No NDIF/nnsight mentions found for '{paper.title}' — UNCLASSIFIED")
                paper.unclassified_reason = "no_keywords_anywhere"
                paper.classification_signal = "no_keywords"
                band = Confidence.NONE
                return Category.UNCLASSIFIED, _BAND_TO_FLOAT[band], band
        else:
            logger.info(f"No extractable evidence for '{paper.title}' — UNCLASSIFIED")
            paper.unclassified_reason = "no_evidence_extractable"
            paper.classification_signal = "no_evidence"
            band = Confidence.NONE
            return Category.UNCLASSIFIED, _BAND_TO_FLOAT[band], band
    else:
        context_source = "pdf"

    logger.debug(
        f"Classifying '{paper.title}': context_source={context_source}, "
        f"context_length={len(context)}"
    )

    # Pre-filters: split context into windows and filter each one
    windows = context.split("\n---\n")
    surviving_windows, prefilter_signal = _apply_prefilters(windows, paper)
    surviving_window_count = len(surviving_windows)

    if not surviving_windows:
        # All windows eliminated by pre-filter — classify without LLM
        paper.classification_signal = prefilter_signal
        logger.debug(
            f"Pre-filter '{prefilter_signal}' matched all windows for '{paper.title}' — REFERENCING"
        )
        band = _compute_confidence_band(
            signal=prefilter_signal or "pre_filter:unknown",
            linked_paper_tier=paper.linked_paper_tier,
            surviving_window_count=0,
            context_source=context_source,
            category=Category.REFERENCING,
        )
        return Category.REFERENCING, _BAND_TO_FLOAT[band], band

    # Some or all windows survived — rebuild context for LLM
    context = "\n---\n".join(surviving_windows)

    # Use LLM to classify
    client = _get_llm_client()
    if not client:
        logger.debug(f"No LLM client — using fallback keyword rules for '{paper.title}'")
        cat = _fallback_classification(context)
        paper.classification_signal = "keyword_fallback"
        band = _compute_confidence_band(
            signal="keyword_fallback",
            linked_paper_tier=paper.linked_paper_tier,
            surviving_window_count=surviving_window_count,
            context_source=context_source,
            category=cat,
        )
        return cat, _BAND_TO_FLOAT[band], band

    try:
        system_prompt = _augment_prompt_with_tier(
            _select_classification_prompt(context),
            paper.linked_paper_tier,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Paper title: {paper.title}\n\n"
                f"Evidence of NDIF/nnsight usage:\n"
                f"{context if context and 'No direct mentions' not in context else 'No direct mentions found in available text.'}"
            )},
        ]
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=20,
        )
        raw = response.choices[0].message.content.strip().lower()
        logger.debug(f"LLM raw reply for '{paper.title}': {raw!r}")
        rate_limit_sleep(config.LLM_RATE_LIMIT_SLEEP, "LLM classify")

        # Parse response
        if "uses_ndif" in raw:
            cat = Category.USES_NDIF
        elif "uses_nnsight" in raw:
            cat = Category.USES_NNSIGHT
        elif "referencing" in raw:
            cat = Category.REFERENCING
        elif "unclassified" in raw:
            paper.unclassified_reason = "llm_returned_unclassified"
            paper.classification_signal = "llm"
            band = Confidence.NONE
            return Category.UNCLASSIFIED, _BAND_TO_FLOAT[band], band
        else:
            logger.warning(f"Unexpected LLM classification '{raw}' for '{paper.title}' — UNCLASSIFIED")
            paper.unclassified_reason = "llm_unparseable"
            paper.classification_signal = "llm"
            band = Confidence.NONE
            return Category.UNCLASSIFIED, _BAND_TO_FLOAT[band], band

        paper.classification_signal = "llm"
        band = _compute_confidence_band(
            signal="llm",
            linked_paper_tier=paper.linked_paper_tier,
            surviving_window_count=surviving_window_count,
            context_source=context_source,
            category=cat,
        )
        return cat, _BAND_TO_FLOAT[band], band

    except Exception as e:
        logger.warning(f"LLM classification failed for '{paper.title}': {e}")
        cat = _fallback_classification(context)
        paper.classification_signal = "keyword_fallback"
        band = _compute_confidence_band(
            signal="keyword_fallback",
            linked_paper_tier=paper.linked_paper_tier,
            surviving_window_count=surviving_window_count,
            context_source=context_source,
            category=cat,
        )
        return cat, _BAND_TO_FLOAT[band], band


def _fallback_classification(context: str) -> Category:
    """Fallback classification without LLM — rule-based."""
    context_lower = context.lower()

    # Strong signals for uses_ndif
    ndif_signals = ["hosted on ndif", "ndif cluster", "ndif.us", "ndif infrastructure"]
    if any(s in context_lower for s in ndif_signals):
        return Category.USES_NDIF

    # Strong signals for uses_nnsight
    nnsight_signals = ["import nnsight", "nnsight.trace", "from nnsight", "nnsight backend",
                       "nnsight library", "using nnsight"]
    if any(s in context_lower for s in nnsight_signals):
        return Category.USES_NNSIGHT

    return Category.REFERENCING


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
# Bucket decision helpers (US-D2, US-D3)
# ---------------------------------------------------------------------------

def _decide_bucket(paper: DiscoveredPaper) -> tuple[Bucket, Optional[PaperReason]]:
    """Return (bucket, reason) for a paper based on ordered demotion rules.

    Rules apply IN ORDER; first match wins:
    1. stub_metadata     — year==0 OR no usable abstract OR no identifier
    2. unclassified_*    — Category.UNCLASSIFIED
    3. medium_confidence — band == MEDIUM
    4. low_confidence    — band == LOW

    Papers with band CERTAIN or HIGH pass through to VERIFIED.
    """
    has_any_identifier = bool(
        paper.arxiv_id or paper.doi or paper.s2_paper_id
        or paper.openalex_id or paper.url
    )
    if (
        paper.year == 0
        or not _has_usable_abstract(paper)
        or not has_any_identifier
    ):
        return Bucket.PENDING, PaperReason.STUB_METADATA

    if paper.category == Category.UNCLASSIFIED:
        if paper.unclassified_reason in ("no_keywords_anywhere", "no_evidence_extractable"):
            return Bucket.PENDING, PaperReason.UNCLASSIFIED_NO_KEYWORDS
        return Bucket.PENDING, PaperReason.UNCLASSIFIED_LLM

    if paper.category_confidence_band == Confidence.MEDIUM:
        return Bucket.PENDING, PaperReason.MEDIUM_CONFIDENCE

    if paper.category_confidence_band == Confidence.LOW:
        return Bucket.PENDING, PaperReason.LOW_CONFIDENCE

    return Bucket.VERIFIED, None


def _check_discard_zero_pdf_hits(paper: DiscoveredPaper, pdf_path: Optional[Path]) -> bool:
    """Return True and mutate paper if PDF has ≥1000 chars and zero keyword matches.

    Only fires when a valid PDF was extracted. Short/broken PDFs (< 1000 chars) are
    not discarded — they may be scan failures, not genuine zero-match papers.
    """
    if not pdf_path or not pdf_path.exists():
        return False

    try:
        text = extract_text_from_pdf(pdf_path)
    except Exception:
        return False

    if len(text) < 1000:
        return False

    text_lower = text.lower()
    for kw in config.NDIF_KEYWORDS:
        if kw.lower() in text_lower:
            return False

    # All keywords checked, zero matches in a substantial PDF
    kw_list = ", ".join(config.NDIF_KEYWORDS[:5]) + ("..." if len(config.NDIF_KEYWORDS) > 5 else "")
    paper.bucket = Bucket.DISCARDED
    paper.reason = PaperReason.ZERO_PDF_HITS
    paper.reason_detail = f"Extracted {len(text)} chars; 0 matches for any of [{kw_list}]."
    return True


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

        # SKIP and PROTECTED: just copy as-is, preserving all override fields
        if bucket in (ProcessingBucket.SKIP, ProcessingBucket.PROTECTED):
            if bucket == ProcessingBucket.PROTECTED and decision.existing_paper:
                existing = decision.existing_paper
                # Carry over new identifiers (safe, non-destructive)
                if existing.s2_paper_id and not paper.s2_paper_id:
                    paper.s2_paper_id = existing.s2_paper_id
                if existing.openalex_id and not paper.openalex_id:
                    paper.openalex_id = existing.openalex_id
                # Carry over bucket/category/reason — manual_override freezes these
                paper.bucket = existing.bucket
                paper.category = existing.category
                paper.reason = existing.reason
                paper.reason_detail = existing.reason_detail
                paper.description = existing.description
            results.append(paper)
            continue

        # FILL_GAPS on a manual_override paper: hydrate the discovered paper
        # with the curated existing state so the fields we don't refill
        # (everything except has_* gaps) persist across runs.
        is_protected_manual = (
            bucket == ProcessingBucket.FILL_GAPS
            and decision.existing_paper is not None
            and decision.existing_paper.manual_override
        )
        if is_protected_manual:
            existing = decision.existing_paper
            paper.title = existing.title or paper.title
            paper.authors = existing.authors or paper.authors
            paper.affiliations = existing.affiliations or paper.affiliations
            paper.venue = existing.venue or paper.venue
            paper.year = existing.year or paper.year
            paper.description = existing.description or paper.description
            paper.category = existing.category
            paper.category_confidence = existing.category_confidence
            paper.category_confidence_band = existing.category_confidence_band
            paper.bucket = existing.bucket
            paper.reason = existing.reason
            paper.reason_detail = existing.reason_detail
            paper.image = existing.image or paper.image
            paper.project_url = existing.project_url or paper.project_url
            paper.url = existing.url or paper.url
            paper.pdf_url = existing.pdf_url or paper.pdf_url
            paper.manual_override = True

        # Get PDF path if needed for classification, thumbnail, or affiliations
        pdf_path = None
        if needs.get("classify") or needs.get("thumbnail") or needs.get("affiliations"):
            pdf_path = get_cached_pdf(paper, output_dir)
            if not pdf_path:
                logger.warning(f"Could not get PDF for '{paper.title[:50]}...'")

        # Summary generation — guarded for manual_override
        if needs.get("summary") and not skip_llm:
            if is_protected_manual and paper.description:
                logger.debug(f"FILL_GAPS guard: keeping curated description for '{paper.title[:50]}'")
            else:
                paper.description = generate_summary(paper)
        paper.has_summary = bool(paper.description)

        # Discard check: zero PDF keyword hits (US-D2)
        if needs.get("classify") and pdf_path:
            if _check_discard_zero_pdf_hits(paper, pdf_path):
                logger.info(f"Discarded: {paper.title[:60]} (zero_pdf_hits, {paper.reason_detail})")
                results.append(paper)
                continue

        # Category classification — guarded for manual_override
        if needs.get("classify") and not skip_llm:
            if is_protected_manual and paper.category != Category.UNCLASSIFIED:
                logger.debug(f"FILL_GAPS guard: keeping curated category for '{paper.title[:50]}'")
            else:
                effective_pdf = pdf_path if pdf_path and pdf_path.exists() else None
                category, confidence, band = classify_category(paper, output_dir, pdf_path=effective_pdf)
                paper.category = category
                paper.category_confidence = confidence
                paper.category_confidence_band = band
                paper.has_classification = category != Category.UNCLASSIFIED
                paper.bucket, paper.reason = _decide_bucket(paper)

        # Thumbnail extraction — guarded for manual_override
        if needs.get("thumbnail"):
            if is_protected_manual and paper.image:
                logger.debug(f"FILL_GAPS guard: keeping curated image for '{paper.title[:50]}'")
            elif pdf_path and pdf_path.exists():
                image_filename = f"{slugify(paper.title)}.png"
                image_path = output_dir / "images" / image_filename
                if not image_path.exists():
                    extracted = extract_thumbnail(paper, output_dir, pdf_path=pdf_path)
                    if extracted:
                        paper.image = extracted
        paper.has_thumbnail = bool(paper.image)

        # Affiliation extraction from PDF (heuristic, no LLM)
        if needs.get("affiliations") and not paper.affiliations and pdf_path and pdf_path.exists():
            from ndif_citations.utils import extract_affiliations_from_pdf
            try:
                affs = extract_affiliations_from_pdf(pdf_path, paper.authors or "")
                if affs:
                    paper.affiliations = ", ".join(affs)
                    logger.debug(f"Extracted affiliations: {paper.affiliations[:80]}")
            except Exception as e:
                logger.debug(f"Affiliation extraction failed: {e}")
        paper.has_affiliations = bool(paper.affiliations)

        results.append(paper)

    logger.info("Processing complete")
    return results


def classify_repo(repo: "DiscoveredRepo") -> None:
    """Classify a repo's NDIF usage via keyword-only README scan (no LLM).

    Default: uses_nnsight (all nnsight dependents use the library by definition).
    Upgrade to uses_ndif if README mentions any NDIF_README_KEYWORDS.
    Mutates repo in place.
    """
    from ndif_citations.models import Category, DiscoveredRepo

    # Default — all GitHub dependents use nnsight
    repo.category = Category.USES_NNSIGHT
    repo.classification_reason = "github_dependent"

    # Fetch README and keyword-scan for NDIF usage
    readme_text = _fetch_repo_readme(repo.owner, repo.repo)
    if readme_text:
        matched = False
        for pattern in config.NDIF_README_KEYWORDS_REGEX:
            if re.search(pattern, readme_text):
                matched = True
                break
        if not matched:
            readme_lower = readme_text.lower()
            for kw in config.NDIF_README_KEYWORDS_SUBSTR:
                if kw.lower() in readme_lower:
                    matched = True
                    break
        if matched:
            repo.category = Category.USES_NDIF
            repo.classification_reason = "ndif_keyword_match"
            logger.debug(
                f"Repo {repo.owner}/{repo.repo} upgraded to uses_ndif (ndif_keyword_match)"
            )

    repo.has_classification = True


def process_repos(decisions: list[RepoRoutingDecision]) -> list[DiscoveredRepo]:
    """Phase 3 for repos: apply classification per routing bucket.

    NEW / REPROCESS: recompute content_hash (classification already done in enrichment)
    FILL_GAPS: classify_repo() only if has_classification is False (enrichment fallback)
    SKIP / PROTECTED: pass through unchanged
    """
    results: list[DiscoveredRepo] = []
    total = len(decisions)

    for i, decision in enumerate(decisions):
        repo = decision.repo
        bucket = decision.bucket

        try:
            if bucket in (ProcessingBucket.NEW, ProcessingBucket.REPROCESS):
                logger.info(
                    f"[{i + 1}/{total}] Repo {repo.owner}/{repo.repo} "
                    f"(bucket: {bucket.value}, already classified in enrichment)"
                )
                repo.content_hash = repo.compute_content_hash()

            elif bucket == ProcessingBucket.FILL_GAPS:
                logger.info(
                    f"[{i + 1}/{total}] Fill-gaps repo {repo.owner}/{repo.repo}"
                )
                if not repo.has_classification:
                    classify_repo(repo)  # fallback for repos that enrichment missed
                repo.content_hash = repo.compute_content_hash()

            elif bucket == ProcessingBucket.SKIP:
                logger.debug(
                    f"[{i + 1}/{total}] Skipping repo {repo.owner}/{repo.repo}"
                )
                # Carry over existing state from routing decision
                if decision.existing_repo:
                    repo = decision.existing_repo

            elif bucket == ProcessingBucket.PROTECTED:
                logger.debug(
                    f"[{i + 1}/{total}] Protected repo {repo.owner}/{repo.repo}"
                )
                if decision.existing_repo:
                    repo = decision.existing_repo

        except Exception as e:
            logger.warning(
                f"Failed to process repo {repo.owner}/{repo.repo}: {e}"
            )

        results.append(repo)

    logger.info(f"Repo processing complete: {len(results)} repos processed")
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
