"""Early Router: Decision layer before expensive LLM processing.

Routes papers to buckets based on comparison with existing database:
- NEW: Not in DB (full processing)
- REPROCESS: Hash changed or venue changed (full reprocessing)
- FILL_GAPS: Hash same, missing fields (partial processing)
- SKIP: Hash same, all complete (copy as-is)
- PROTECTED: manual_override=True (never touch)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ndif_citations.models import DiscoveredPaper

logger = logging.getLogger(__name__)


class ProcessingBucket(str, Enum):
    """Routing buckets for paper processing decisions."""
    NEW = "new"
    REPROCESS = "reprocess"  # Hash or venue changed
    FILL_GAPS = "fill_gaps"   # Same hash, missing fields
    SKIP = "skip"             # Unchanged, complete
    PROTECTED = "protected"   # Manual override set


@dataclass
class RoutingDecision:
    """Result of routing a single paper."""
    paper: DiscoveredPaper
    bucket: ProcessingBucket
    existing_paper: DiscoveredPaper | None
    processing_needed: dict[str, bool]  # {"summary": True, "classify": False, ...}


def _all_true() -> dict[str, bool]:
    """All processing flags True."""
    return {"summary": True, "classify": True, "thumbnail": True, "affiliations": True}


def _all_false() -> dict[str, bool]:
    """All processing flags False."""
    return {"summary": False, "classify": False, "thumbnail": False, "affiliations": False}


def _detect_venue_type(venue: str) -> str:
    """Detect venue type from venue string.

    Returns: "preprint", "conference", "journal", "workshop", or "unknown"
    """
    venue_lower = venue.lower()

    # ArXiv preprints
    if "arxiv" in venue_lower:
        return "preprint"

    # Known conference indicators
    conference_indicators = [
        "proceedings", "conference", "conf", "symposium",
        "neurips", "icml", "iclr", "aaai", "acl", "emnlp",
        "workshop", "coling", "naacl"
    ]
    for indicator in conference_indicators:
        if indicator in venue_lower:
            return "conference"

    # Journal indicators
    journal_indicators = [
        "journal", "transactions", "letters", "review",
        "nature", "science", "ieee", "acm"
    ]
    for indicator in journal_indicators:
        if indicator in venue_lower:
            return "journal"

    return "unknown"


def _is_venue_upgrade(existing_venue: str, new_venue: str) -> bool:
    """Check if venue changed from preprint → conference/journal.

    arXiv → Conference/Journal is a venue upgrade (camera-ready version).
    """
    existing_type = _detect_venue_type(existing_venue)
    new_type = _detect_venue_type(new_venue)

    if existing_type == "preprint" and new_type in ("conference", "journal"):
        return True
    return False


def _route_single_paper(
    paper: DiscoveredPaper,
    by_arxiv: dict[str, DiscoveredPaper],
    by_doi: dict[str, DiscoveredPaper],
    by_hash: dict[str, DiscoveredPaper]
) -> RoutingDecision:
    """Route a single paper to appropriate bucket.

    Logic:
    1. Match by arXiv ID, DOI, or content hash
    2. If no match → NEW
    3. If manual_override → PROTECTED
    4. If venue upgrade (arXiv → conf/journal) → REPROCESS
    5. If hash changed → REPROCESS
    6. If missing fields → FILL_GAPS
    7. If untouched → SKIP
    """
    # Match by identifiers
    existing: Optional[DiscoveredPaper] = None

    if paper.arxiv_id and paper.arxiv_id in by_arxiv:
        existing = by_arxiv[paper.arxiv_id]
    elif paper.doi and paper.doi in by_doi:
        existing = by_doi[paper.doi]
    else:
        # Compute hash if not already set
        paper_hash = paper.content_hash or paper.compute_hash()
        existing = by_hash.get(paper_hash)

    # NEW: Not in DB
    if not existing:
        logger.debug(f"NEW: {paper.title[:50]}...")
        return RoutingDecision(paper, ProcessingBucket.NEW, None, _all_true())

    # PROTECTED: Manual override
    if existing.manual_override:
        logger.debug(f"PROTECTED: {paper.title[:50]}...")
        return RoutingDecision(paper, ProcessingBucket.PROTECTED, existing, _all_false())

    # Check for venue upgrade BEFORE hash check
    # (arXiv → conference/journal is a real publication event)
    if _is_venue_upgrade(existing.venue, paper.venue):
        logger.debug(f"REPROCESS (venue upgrade): {paper.title[:50]}... ({existing.venue} -> {paper.venue})")
        return RoutingDecision(paper, ProcessingBucket.REPROCESS, existing, _all_true())

    # Check content hash for actual content changes
    paper_hash = paper.content_hash or paper.compute_hash()
    if existing.content_hash != paper_hash:
        logger.debug(f"REPROCESS (content changed): {paper.title[:50]}...")
        return RoutingDecision(paper, ProcessingBucket.REPROCESS, existing, _all_true())

    # Same content - check for missing fields
    needs = {
        "summary": not existing.has_summary,
        "classify": not existing.has_classification,
        "thumbnail": not existing.has_thumbnail,
        "affiliations": not existing.has_affiliations,
    }

    if any(needs.values()):
        logger.debug(f"FILL_GAPS: {paper.title[:50]}... (needs: {needs})")
        return RoutingDecision(paper, ProcessingBucket.FILL_GAPS, existing, needs)

    # Truly unchanged and complete
    logger.debug(f"SKIP: {paper.title[:50]}...")
    return RoutingDecision(paper, ProcessingBucket.SKIP, existing, _all_false())


def route_papers(
    discovered: list[DiscoveredPaper],
    existing: list[DiscoveredPaper]
) -> list[RoutingDecision]:
    """Route all discovered papers against existing database.

    Builds O(1) lookup tables for efficient matching, then routes each paper.

    Args:
        discovered: Papers found in this run
        existing: Papers from previous runs (loaded from database)

    Returns:
        RoutingDecision for each discovered paper
    """
    logger.info(f"Routing {len(discovered)} discovered papers against {len(existing)} existing...")

    # Build O(1) lookup tables
    by_arxiv: dict[str, DiscoveredPaper] = {}
    by_doi: dict[str, DiscoveredPaper] = {}
    by_hash: dict[str, DiscoveredPaper] = {}

    for paper in existing:
        if paper.arxiv_id:
            by_arxiv[paper.arxiv_id] = paper
        if paper.doi:
            by_doi[paper.doi] = paper
        if paper.content_hash:
            by_hash[paper.content_hash] = paper

    # Route each paper
    decisions: list[RoutingDecision] = []
    for paper in discovered:
        decision = _route_single_paper(paper, by_arxiv, by_doi, by_hash)

        # Track bucket on paper for debugging
        paper.processing_bucket = decision.bucket.value

        decisions.append(decision)

    # Log summary
    bucket_counts: dict[str, int] = {}
    for d in decisions:
        bucket_counts[d.bucket.value] = bucket_counts.get(d.bucket.value, 0) + 1

    logger.info(f"Routing complete: {bucket_counts}")

    return decisions


def get_bucket_summary(decisions: list[RoutingDecision]) -> dict[str, int]:
    """Get counts per bucket for reporting."""
    counts: dict[str, int] = {}
    for d in decisions:
        counts[d.bucket.value] = counts.get(d.bucket.value, 0) + 1
    return counts
