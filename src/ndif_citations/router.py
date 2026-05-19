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

from ndif_citations.models import DiscoveredPaper, DiscoveredRepo

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


@dataclass
class RepoRoutingDecision:
    """Result of routing a single GitHub repo."""
    repo: DiscoveredRepo
    bucket: ProcessingBucket
    existing_repo: DiscoveredRepo | None


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

    # PROTECTED: Manual override — but allow fill-gaps for empty fields so the
    # pipeline can backfill description / thumbnail / affiliations without
    # overwriting curated values. process.py guards each write to skip
    # non-empty fields on manual_override papers.
    if existing.manual_override:
        needs = {
            "summary":      not existing.has_summary,
            "classify":     not existing.has_classification,
            "thumbnail":    not existing.has_thumbnail,
            "affiliations": not existing.has_affiliations,
        }
        if any(needs.values()):
            logger.debug(f"PROTECTED->FILL_GAPS: {paper.title[:50]}... (needs: {needs})")
            return RoutingDecision(paper, ProcessingBucket.FILL_GAPS, existing, needs)
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


def route_repos(
    discovered: list[DiscoveredRepo],
    existing: list[DiscoveredRepo],
) -> list[RepoRoutingDecision]:
    """Route all discovered repos against existing database.

    Uses merge_key() (owner/repo string) for O(1) lookup.
    Buckets:
    - NEW: Not in DB
    - REPROCESS: content_hash changed (description, last_commit, or archived changed)
    - FILL_GAPS: Same hash, but has_classification is False
    - SKIP: Hash same, classification complete
    - PROTECTED: manual_override=True (note: does NOT protect against staleness removal)
    """
    logger.info(f"Routing {len(discovered)} repos against {len(existing)} existing...")

    # Build O(1) lookup by merge_key
    by_key: dict[str, DiscoveredRepo] = {r.merge_key(): r for r in existing}

    decisions: list[RepoRoutingDecision] = []
    for repo in discovered:
        key = repo.merge_key()
        existing_repo = by_key.get(key)

        if existing_repo is None:
            bucket = ProcessingBucket.NEW
        elif existing_repo.manual_override:
            bucket = ProcessingBucket.PROTECTED
        elif existing_repo.content_hash != repo.content_hash:
            bucket = ProcessingBucket.REPROCESS
        elif not existing_repo.has_classification:
            bucket = ProcessingBucket.FILL_GAPS
        else:
            bucket = ProcessingBucket.SKIP

        repo.processing_bucket = bucket.value
        decisions.append(RepoRoutingDecision(repo=repo, bucket=bucket, existing_repo=existing_repo))

    bucket_counts: dict[str, int] = {}
    for d in decisions:
        bucket_counts[d.bucket.value] = bucket_counts.get(d.bucket.value, 0) + 1
    logger.info(f"Repo routing complete: {bucket_counts}")

    return decisions


def get_bucket_summary(decisions: list[RoutingDecision]) -> dict[str, int]:
    """Get counts per bucket for reporting."""
    counts: dict[str, int] = {}
    for d in decisions:
        counts[d.bucket.value] = counts.get(d.bucket.value, 0) + 1
    return counts
